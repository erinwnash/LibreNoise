#include "librenoise/filter.hpp"
#include <cmath>
#include <vector>
#include <algorithm>
#include <array>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace librenoise {

namespace {

// =========================================================================
// 8-point Discrete Cosine Transform (Orthonormal DCT-II)
// =========================================================================
inline void dct8_1d(const float in[8], float out[8]) {
    constexpr float s0 = 0.35355339f; // 1 / sqrt(8)
    constexpr float s = 0.5f;          // sqrt(2 / 8)

    for (int k = 0; k < 8; ++k) {
        float sum = 0.0f;
        float factor = (k == 0) ? s0 : s;
        for (int n = 0; n < 8; ++n) {
            sum += in[n] * std::cos((3.141592653589793f / 8.0f) * (n + 0.5f) * k);
        }
        out[k] = factor * sum;
    }
}

inline void idct8_1d(const float in[8], float out[8]) {
    constexpr float s0 = 0.35355339f;
    constexpr float s = 0.5f;

    for (int n = 0; n < 8; ++n) {
        float sum = s0 * in[0];
        for (int k = 1; k < 8; ++k) {
            sum += s * in[k] * std::cos((3.141592653589793f / 8.0f) * (n + 0.5f) * k);
        }
        out[n] = sum;
    }
}

void dct8x8_2d(const float in[64], float out[64]) {
    float temp[64];
    for (int r = 0; r < 8; ++r) dct8_1d(&in[r * 8], &temp[r * 8]);
    for (int c = 0; c < 8; ++c) {
        float col_in[8], col_out[8];
        for (int r = 0; r < 8; ++r) col_in[r] = temp[r * 8 + c];
        dct8_1d(col_in, col_out);
        for (int r = 0; r < 8; ++r) out[r * 8 + c] = col_out[r];
    }
}

void idct8x8_2d(const float in[64], float out[64]) {
    float temp[64];
    for (int c = 0; c < 8; ++c) {
        float col_in[8], col_out[8];
        for (int r = 0; r < 8; ++r) col_in[r] = in[r * 8 + c];
        idct8_1d(col_in, col_out);
        for (int r = 0; r < 8; ++r) temp[r * 8 + c] = col_out[r];
    }
    for (int r = 0; r < 8; ++r) idct8_1d(&temp[r * 8], &out[r * 8]);
}

void temporal_dct_forward(const float* in, float* out, int K) {
    if (K <= 1) { out[0] = in[0]; return; }
    const float alpha0 = std::sqrt(1.0f / static_cast<float>(K));
    const float alpha = std::sqrt(2.0f / static_cast<float>(K));
    const float pi = 3.141592653589793f;

    for (int k = 0; k < K; ++k) {
        float factor = (k == 0) ? alpha0 : alpha;
        float sum = 0.0f;
        for (int n = 0; n < K; ++n) {
            sum += in[n] * std::cos((pi / static_cast<float>(K)) * (n + 0.5f) * k);
        }
        out[k] = factor * sum;
    }
}

void temporal_dct_inverse(const float* in, float* out, int K) {
    if (K <= 1) { out[0] = in[0]; return; }
    const float alpha0 = std::sqrt(1.0f / static_cast<float>(K));
    const float alpha = std::sqrt(2.0f / static_cast<float>(K));
    const float pi = 3.141592653589793f;

    for (int n = 0; n < K; ++n) {
        float sum = alpha0 * in[0];
        for (int k = 1; k < K; ++k) {
            sum += alpha * in[k] * std::cos((pi / static_cast<float>(K)) * (n + 0.5f) * k);
        }
        out[n] = sum;
    }
}

// Motion-compensated temporal accumulation helper
ImagePlane32F compute_temporal_average(
    const std::vector<const ImagePlane32F*>& temporal_planes,
    size_t ref_index,
    const std::vector<const MotionField*>& motion_fields,
    const NoiseLevelFunction& nlf
) {
    const int32_t w = temporal_planes[ref_index]->width();
    const int32_t h = temporal_planes[ref_index]->height();
    ImagePlane32F avg(w, h);

    #pragma omp parallel for schedule(dynamic)
    for (int32_t y = 0; y < h; ++y) {
        for (int32_t x = 0; x < w; ++x) {
            float ref_val = temporal_planes[ref_index]->at(x, y);
            float sigma = nlf.sigma_at(ref_val);
            float trust_limit = 3.5f * sigma;

            float sum = ref_val;
            float weight_sum = 1.0f;

            for (size_t f = 0; f < temporal_planes.size(); ++f) {
                if (f == ref_index) continue;
                int32_t dx = 0, dy = 0;
                bool trusted = true;
                if (motion_fields[f]) {
                    auto mv = motion_fields[f]->get_vector_at_pixel(x, y);
                    dx = mv.dx;
                    dy = mv.dy;
                    trusted = mv.trusted;
                }
                if (!trusted) continue;

                int32_t cx = std::clamp(x + dx, 0, w - 1);
                int32_t cy = std::clamp(y + dy, 0, h - 1);
                float cand_val = temporal_planes[f]->at(cx, cy);

                if (std::abs(ref_val - cand_val) <= trust_limit) {
                    sum += cand_val;
                    weight_sum += 1.0f;
                }
            }
            avg.at(x, y) = sum / weight_sum;
        }
    }
    return avg;
}

} // anonymous namespace

// =========================================================================
// Engine 1: 3D Collaborative Transform Filter (BM3D lineage)
// =========================================================================
void CollaborativePatchFilter3D::process_plane(
    const std::vector<const ImagePlane32F*>& temporal_planes,
    size_t ref_index,
    const std::vector<const MotionField*>& motion_fields,
    const NoiseLevelFunction& nlf,
    const NoisePsd& psd,
    const FilterParams& params,
    ImagePlane32F& output_plane
) {
    (void)psd;
    const int32_t w = output_plane.width();
    const int32_t h = output_plane.height();
    const int32_t ps = params.patch_size; // 8
    const int32_t step = params.step_size; // e.g. 3
    const size_t total_pixels = static_cast<size_t>(w * h);

    ImagePlane32F numerator(w, h);
    ImagePlane32F denominator(w, h);
    numerator.fill(0.0f);
    denominator.fill(0.0f);

    const ImagePlane32F* ref_plane = temporal_planes[ref_index];

    #pragma omp parallel
    {
        ImagePlane32F local_num(w, h);
        ImagePlane32F local_den(w, h);
        local_num.fill(0.0f);
        local_den.fill(0.0f);

        #pragma omp for schedule(dynamic)
        for (int32_t y = 0; y <= h - ps; y += step) {
            for (int32_t x = 0; x <= w - ps; x += step) {
                float ref_patch[64];
                for (int32_t py = 0; py < ps; ++py) {
                    for (int32_t px = 0; px < ps; ++px) {
                        ref_patch[py * 8 + px] = ref_plane->at(x + px, y + py);
                    }
                }

                float center_intensity = ref_patch[27];
                float sigma = nlf.sigma_at(center_intensity);
                float hard_threshold = params.hard_thresh_lambda * sigma;

                struct GroupMember {
                    size_t frame_idx;
                    int32_t patch_x;
                    int32_t patch_y;
                    float patch[64];
                };
                std::vector<GroupMember> group;
                group.reserve(temporal_planes.size());

                GroupMember ref_member{ref_index, x, y, {}};
                std::copy(ref_patch, ref_patch + 64, ref_member.patch);
                group.push_back(ref_member);

                for (size_t f = 0; f < temporal_planes.size(); ++f) {
                    if (f == ref_index) continue;

                    int32_t disp_x = 0, disp_y = 0;
                    bool trusted = true;
                    if (motion_fields[f]) {
                        auto mv = motion_fields[f]->get_vector_at_pixel(x + ps / 2, y + ps / 2);
                        disp_x = mv.dx;
                        disp_y = mv.dy;
                        trusted = mv.trusted;
                    }
                    if (!trusted) continue;

                    int32_t cand_x = std::clamp(x + disp_x, 0, w - ps);
                    int32_t cand_y = std::clamp(y + disp_y, 0, h - ps);

                    GroupMember cand_member{f, cand_x, cand_y, {}};
                    for (int32_t py = 0; py < ps; ++py) {
                        for (int32_t px = 0; px < ps; ++px) {
                            cand_member.patch[py * 8 + px] = temporal_planes[f]->at(cand_x + px, cand_y + py);
                        }
                    }

                    float dist = 0.0f;
                    for (int32_t i = 0; i < 64; ++i) {
                        float diff = ref_patch[i] - cand_member.patch[i];
                        dist += diff * diff;
                    }
                    dist /= 64.0f;

                    if (dist <= 4.0f * (sigma * sigma + 1e-6f)) {
                        group.push_back(cand_member);
                        if (group.size() >= static_cast<size_t>(params.max_group_size)) break;
                    }
                }

                int K = static_cast<int>(group.size());

                // Spatial 2D DCT
                std::vector<std::array<float, 64>> spatial_coeffs(K);
                for (int k = 0; k < K; ++k) dct8x8_2d(group[k].patch, spatial_coeffs[k].data());

                // Temporal 1D DCT
                std::vector<std::array<float, 64>> transform_3d(K);
                for (int coeff_idx = 0; coeff_idx < 64; ++coeff_idx) {
                    float temp_in[16], temp_out[16];
                    for (int k = 0; k < K; ++k) temp_in[k] = spatial_coeffs[k][coeff_idx];
                    temporal_dct_forward(temp_in, temp_out, K);
                    for (int k = 0; k < K; ++k) transform_3d[k][coeff_idx] = temp_out[k];
                }

                // Collaborative Thresholding
                int non_zero_count = 0;
                for (int k = 0; k < K; ++k) {
                    for (int coeff_idx = 0; coeff_idx < 64; ++coeff_idx) {
                        if (k == 0 && coeff_idx == 0) { non_zero_count++; continue; }
                        float val = transform_3d[k][coeff_idx];
                        if (std::abs(val) < hard_threshold) {
                            transform_3d[k][coeff_idx] = 0.0f;
                        } else {
                            non_zero_count++;
                        }
                    }
                }

                // Inverse 1D DCT
                for (int coeff_idx = 0; coeff_idx < 64; ++coeff_idx) {
                    float temp_in[16], temp_out[16];
                    for (int k = 0; k < K; ++k) temp_in[k] = transform_3d[k][coeff_idx];
                    temporal_dct_inverse(temp_in, temp_out, K);
                    for (int k = 0; k < K; ++k) spatial_coeffs[k][coeff_idx] = temp_out[k];
                }

                // Inverse 2D IDCT
                float filtered_ref_patch[64];
                idct8x8_2d(spatial_coeffs[0].data(), filtered_ref_patch);

                float weight = 1.0f / (static_cast<float>(std::max(1, non_zero_count)));
                for (int32_t py = 0; py < ps; ++py) {
                    for (int32_t px = 0; px < ps; ++px) {
                        local_num.at(x + px, y + py) += filtered_ref_patch[py * 8 + px] * weight;
                        local_den.at(x + px, y + py) += weight;
                    }
                }
            }
        }

        #pragma omp critical
        {
            float* num_ptr = numerator.data();
            float* den_ptr = denominator.data();
            const float* lnum_ptr = local_num.data();
            const float* lden_ptr = local_den.data();
            for (size_t i = 0; i < total_pixels; ++i) {
                num_ptr[i] += lnum_ptr[i];
                den_ptr[i] += lden_ptr[i];
            }
        }
    }

    for (int32_t y = 0; y < h; ++y) {
        for (int32_t x = 0; x < w; ++x) {
            float d = denominator.at(x, y);
            output_plane.at(x, y) = (d > 1e-7f) ? (numerator.at(x, y) / d) : ref_plane->at(x, y);
        }
    }
}

// =========================================================================
// Engine 2: 5-Band Multi-Scale Wavelet / Pyramid Filter
// =========================================================================
namespace {

ImagePlane32F downsample2x(const ImagePlane32F& src) {
    int32_t sw = src.width();
    int32_t sh = src.height();
    int32_t dw = std::max(1, sw / 2);
    int32_t dh = std::max(1, sh / 2);
    ImagePlane32F dst(dw, dh);

    #pragma omp parallel for schedule(dynamic)
    for (int32_t y = 0; y < dh; ++y) {
        int32_t sy = y * 2;
        int32_t sy1 = std::min(sh - 1, sy + 1);
        for (int32_t x = 0; x < dw; ++x) {
            int32_t sx = x * 2;
            int32_t sx1 = std::min(sw - 1, sx + 1);
            dst.at(x, y) = 0.25f * (src.at(sx, sy) + src.at(sx1, sy) + src.at(sx, sy1) + src.at(sx1, sy1));
        }
    }
    return dst;
}

ImagePlane32F upsample2x(const ImagePlane32F& src, int32_t target_w, int32_t target_h) {
    ImagePlane32F dst(target_w, target_h);
    int32_t sw = src.width();
    int32_t sh = src.height();

    #pragma omp parallel for schedule(dynamic)
    for (int32_t y = 0; y < target_h; ++y) {
        float v = (static_cast<float>(y) + 0.5f) * (static_cast<float>(sh) / static_cast<float>(target_h)) - 0.5f;
        int32_t y0 = std::clamp(static_cast<int32_t>(std::floor(v)), 0, sh - 1);
        int32_t y1 = std::clamp(y0 + 1, 0, sh - 1);
        float fy = v - static_cast<float>(y0);

        for (int32_t x = 0; x < target_w; ++x) {
            float u = (static_cast<float>(x) + 0.5f) * (static_cast<float>(sw) / static_cast<float>(target_w)) - 0.5f;
            int32_t x0 = std::clamp(static_cast<int32_t>(std::floor(u)), 0, sw - 1);
            int32_t x1 = std::clamp(x0 + 1, 0, sw - 1);
            float fx = u - static_cast<float>(x0);

            float p00 = src.at(x0, y0);
            float p10 = src.at(x1, y0);
            float p01 = src.at(x0, y1);
            float p11 = src.at(x1, y1);

            dst.at(x, y) = (1.0f - fx) * (1.0f - fy) * p00 +
                           fx * (1.0f - fy) * p10 +
                           (1.0f - fx) * fy * p01 +
                           fx * fy * p11;
        }
    }
    return dst;
}

} // anonymous namespace

void Wavelet5BandFilter::process_plane(
    const std::vector<const ImagePlane32F*>& temporal_planes,
    size_t ref_index,
    const std::vector<const MotionField*>& motion_fields,
    const NoiseLevelFunction& nlf,
    const NoisePsd& psd,
    const FilterParams& params,
    ImagePlane32F& output_plane
) {
    (void)psd;
    // Step 1: Temporal Motion-Compensated baseline
    ImagePlane32F current = compute_temporal_average(temporal_planes, ref_index, motion_fields, nlf);

    const int32_t num_levels = 5;
    std::vector<ImagePlane32F> pyramid(num_levels + 1);
    std::vector<ImagePlane32F> details(num_levels);

    pyramid[0] = current;

    // Decomposition into 5 frequency octaves
    for (int l = 0; l < num_levels; ++l) {
        pyramid[l + 1] = downsample2x(pyramid[l]);
        ImagePlane32F up = upsample2x(pyramid[l + 1], pyramid[l].width(), pyramid[l].height());

        details[l] = ImagePlane32F(pyramid[l].width(), pyramid[l].height());
        int band = std::clamp(4 - l, 0, 4); // 4 = Very High down to 0 = Very Low
        float band_mult = params.band_weights[band];

        int32_t pw = pyramid[l].width();
        int32_t ph = pyramid[l].height();

        #pragma omp parallel for schedule(dynamic)
        for (int32_t y = 0; y < ph; ++y) {
            for (int32_t x = 0; x < pw; ++x) {
                float raw_diff = pyramid[l].at(x, y) - up.at(x, y);
                float base_val = up.at(x, y);
                float sigma = nlf.sigma_at(base_val);

                // Hard shrinkage on detail band (preserves true edge contrast without bias)
                float thresh = params.wavelet_soft_threshold * sigma * band_mult;
                float filtered = (std::abs(raw_diff) < thresh) ? 0.0f : raw_diff;
                details[l].at(x, y) = filtered;
            }
        }
    }

    // Reconstruction
    ImagePlane32F recon = pyramid[num_levels];
    for (int l = num_levels - 1; l >= 0; --l) {
        ImagePlane32F up = upsample2x(recon, details[l].width(), details[l].height());
        recon = ImagePlane32F(details[l].width(), details[l].height());

        int32_t rw = details[l].width();
        int32_t rh = details[l].height();

        #pragma omp parallel for schedule(dynamic)
        for (int32_t y = 0; y < rh; ++y) {
            for (int32_t x = 0; x < rw; ++x) {
                recon.at(x, y) = up.at(x, y) + details[l].at(x, y);
            }
        }
    }

    output_plane = recon;
}

// =========================================================================
// Engine 3: Accelerated Non-Local Means (NLM)
// =========================================================================
void NonLocalMeansFilter::process_plane(
    const std::vector<const ImagePlane32F*>& temporal_planes,
    size_t ref_index,
    const std::vector<const MotionField*>& motion_fields,
    const NoiseLevelFunction& nlf,
    const NoisePsd& psd,
    const FilterParams& params,
    ImagePlane32F& output_plane
) {
    (void)psd;
    // Step 1: Temporal Motion-Compensated baseline
    ImagePlane32F temp_accum = compute_temporal_average(temporal_planes, ref_index, motion_fields, nlf);

    const int32_t w = temp_accum.width();
    const int32_t h = temp_accum.height();
    const int32_t r_search = params.nlm_search_radius; // 6
    const int32_t r_patch = params.nlm_patch_radius;   // 2 -> 5x5 patch
    const float patch_norm = 1.0f / static_cast<float>((2 * r_patch + 1) * (2 * r_patch + 1));

    #pragma omp parallel for schedule(dynamic)
    for (int32_t y = 0; y < h; ++y) {
        for (int32_t x = 0; x < w; ++x) {
            float center_val = temp_accum.at(x, y);
            float sigma = nlf.sigma_at(center_val);
            float h_sq = params.nlm_h_factor * sigma * sigma * 2.0f + 1e-6f;

            float weight_sum = 0.0f;
            float val_sum = 0.0f;

            int32_t min_sy = std::max(r_patch, y - r_search);
            int32_t max_sy = std::min(h - 1 - r_patch, y + r_search);
            int32_t min_sx = std::max(r_patch, x - r_search);
            int32_t max_sx = std::min(w - 1 - r_patch, x + r_search);

            for (int32_t sy = min_sy; sy <= max_sy; ++sy) {
                for (int32_t sx = min_sx; sx <= max_sx; ++sx) {
                    float dist = 0.0f;
                    for (int32_t py = -r_patch; py <= r_patch; ++py) {
                        for (int32_t px = -r_patch; px <= r_patch; ++px) {
                            int32_t rx = std::clamp(x + px, 0, w - 1);
                            int32_t ry = std::clamp(y + py, 0, h - 1);
                            float diff = temp_accum.at(rx, ry) - temp_accum.at(sx + px, sy + py);
                            dist += diff * diff;
                        }
                    }
                    dist *= patch_norm;

                    float w_val = std::exp(-std::max(0.0f, dist - 2.0f * sigma * sigma) / h_sq);
                    val_sum += temp_accum.at(sx, sy) * w_val;
                    weight_sum += w_val;
                }
            }

            output_plane.at(x, y) = (weight_sum > 1e-6f) ? (val_sum / weight_sum) : center_val;
        }
    }
}

// =========================================================================
// Engine 4: 3D Frequency-Domain Spectral Thresholding (DFTTest family)
// =========================================================================
void FrequencyDomainFilter3D::process_plane(
    const std::vector<const ImagePlane32F*>& temporal_planes,
    size_t ref_index,
    const std::vector<const MotionField*>& motion_fields,
    const NoiseLevelFunction& nlf,
    const NoisePsd& psd,
    const FilterParams& params,
    ImagePlane32F& output_plane
) {
    (void)psd;
    // Step 1: Temporal motion-compensated pre-estimate
    ImagePlane32F temp_accum = compute_temporal_average(temporal_planes, ref_index, motion_fields, nlf);

    const int32_t w = temp_accum.width();
    const int32_t h = temp_accum.height();
    const int32_t bs = 8;
    const int32_t step = 4; // Overlapping sliding blocks

    ImagePlane32F numerator(w, h);
    ImagePlane32F denominator(w, h);
    numerator.fill(0.0f);
    denominator.fill(0.0f);

    #pragma omp parallel
    {
        ImagePlane32F local_num(w, h);
        ImagePlane32F local_den(w, h);
        local_num.fill(0.0f);
        local_den.fill(0.0f);

        #pragma omp for schedule(dynamic)
        for (int32_t y = 0; y <= h - bs; y += step) {
            for (int32_t x = 0; x <= w - bs; x += step) {
                float block[64];
                for (int32_t py = 0; py < bs; ++py) {
                    for (int32_t px = 0; px < bs; ++px) {
                        block[py * 8 + px] = temp_accum.at(x + px, y + py);
                    }
                }

                float center_intensity = block[27];
                float sigma = nlf.sigma_at(center_intensity);

                // 2D DCT Transform
                float spectrum[64];
                dct8x8_2d(block, spectrum);

                // 5-Band Radial Frequency Mapping and Spectral Shrinkage
                for (int v = 0; v < 8; ++v) {
                    for (int u = 0; u < 8; ++u) {
                        if (u == 0 && v == 0) continue; // Keep DC

                        float r = std::sqrt(static_cast<float>(u * u + v * v));
                        int band = 2; // Mid default
                        if (r < 1.5f) band = 0;      // Very Low
                        else if (r < 3.0f) band = 1; // Low
                        else if (r < 5.0f) band = 2; // Mid
                        else if (r < 7.5f) band = 3; // High
                        else band = 4;               // Very High

                        float coeff = spectrum[v * 8 + u];
                        float pwr = coeff * coeff;
                        float noise_pwr = sigma * sigma * params.band_weights[band];

                        // Wiener empirical spectral shrinkage
                        float gain = std::max(0.0f, (pwr - noise_pwr) / (pwr + 1e-7f));
                        spectrum[v * 8 + u] = coeff * gain;
                    }
                }

                // Inverse 2D IDCT
                float filtered_block[64];
                idct8x8_2d(spectrum, filtered_block);

                for (int32_t py = 0; py < bs; ++py) {
                    for (int32_t px = 0; px < bs; ++px) {
                        local_num.at(x + px, y + py) += filtered_block[py * 8 + px];
                        local_den.at(x + px, y + py) += 1.0f;
                    }
                }
            }
        }

        #pragma omp critical
        {
            float* num_ptr = numerator.data();
            float* den_ptr = denominator.data();
            const float* lnum_ptr = local_num.data();
            const float* lden_ptr = local_den.data();
            size_t total = static_cast<size_t>(w * h);
            for (size_t i = 0; i < total; ++i) {
                num_ptr[i] += lnum_ptr[i];
                den_ptr[i] += lden_ptr[i];
            }
        }
    }

    for (int32_t y = 0; y < h; ++y) {
        for (int32_t x = 0; x < w; ++x) {
            float d = denominator.at(x, y);
            output_plane.at(x, y) = (d > 1e-7f) ? (numerator.at(x, y) / d) : temp_accum.at(x, y);
        }
    }
}

// =========================================================================
// Unified Filter Dispatcher
// =========================================================================
void FilterDispatcher::process(
    const std::vector<VideoFrame32F>& temporal_frames,
    size_t ref_index,
    const std::vector<MotionField>& motion_fields,
    const NoiseProfile& profile,
    const FilterParams& params,
    VideoFrame32F& output_frame
) {
    if (temporal_frames.empty() || ref_index >= temporal_frames.size()) return;

    output_frame = VideoFrame32F(
        temporal_frames[ref_index].width,
        temporal_frames[ref_index].height,
        temporal_frames[ref_index].channel_count(),
        temporal_frames[ref_index].transfer
    );

    std::unique_ptr<ISpatialEngine> engine;
    switch (params.engine) {
        case DenoiseEngine::Wavelet5Band:
            engine = std::make_unique<Wavelet5BandFilter>();
            break;
        case DenoiseEngine::NonLocalMeans:
            engine = std::make_unique<NonLocalMeansFilter>();
            break;
        case DenoiseEngine::FrequencyDomain3D:
            engine = std::make_unique<FrequencyDomainFilter3D>();
            break;
        case DenoiseEngine::Collaborative3D:
        default:
            engine = std::make_unique<CollaborativePatchFilter3D>(params);
            break;
    }

    for (size_t c = 0; c < output_frame.channel_count(); ++c) {
        std::vector<const ImagePlane32F*> planes;
        planes.reserve(temporal_frames.size());
        for (const auto& tf : temporal_frames) planes.push_back(&tf.channels[c]);

        std::vector<const MotionField*> mf_ptrs;
        mf_ptrs.reserve(motion_fields.size());
        for (const auto& mf : motion_fields) mf_ptrs.push_back(&mf);

        const auto& nlf = profile.nlf_per_channel[std::min(c, size_t{2})];
        engine->process_plane(planes, ref_index, mf_ptrs, nlf, profile.psd, params, output_frame.channels[c]);
    }
}

} // namespace librenoise
