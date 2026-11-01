#include "librenoise/filter.hpp"
#include <cmath>
#include <vector>
#include <algorithm>

namespace librenoise {

namespace {

// 1D 8-point Discrete Cosine Transform (Orthonormal DCT-II)
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

// 1D 8-point Inverse Discrete Cosine Transform (Orthonormal IDCT-II)
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

// 2D 8x8 DCT
void dct8x8_2d(const float in[64], float out[64]) {
    float temp[64];
    for (int r = 0; r < 8; ++r) {
        dct8_1d(&in[r * 8], &temp[r * 8]);
    }
    for (int c = 0; c < 8; ++c) {
        float col_in[8];
        float col_out[8];
        for (int r = 0; r < 8; ++r) col_in[r] = temp[r * 8 + c];
        dct8_1d(col_in, col_out);
        for (int r = 0; r < 8; ++r) out[r * 8 + c] = col_out[r];
    }
}

// 2D 8x8 IDCT
void idct8x8_2d(const float in[64], float out[64]) {
    float temp[64];
    for (int c = 0; c < 8; ++c) {
        float col_in[8];
        float col_out[8];
        for (int r = 0; r < 8; ++r) col_in[r] = in[r * 8 + c];
        idct8_1d(col_in, col_out);
        for (int r = 0; r < 8; ++r) temp[r * 8 + c] = col_out[r];
    }
    for (int r = 0; r < 8; ++r) {
        idct8_1d(&temp[r * 8], &out[r * 8]);
    }
}

// Arbitrary-length orthonormal 1D DCT for temporal axis of length K (1 <= K <= 16)
void temporal_dct_forward(const float* in, float* out, int K) {
    if (K <= 1) {
        out[0] = in[0];
        return;
    }
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
    if (K <= 1) {
        out[0] = in[0];
        return;
    }
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

} // anonymous namespace

void CollaborativePatchFilter3D::process_plane(
    const std::vector<const ImagePlane32F*>& temporal_planes,
    size_t ref_index,
    const std::vector<const MotionField*>& motion_fields,
    const NoiseLevelFunction& nlf,
    ImagePlane32F& output_plane
) {
    const int32_t w = output_plane.width();
    const int32_t h = output_plane.height();
    const int32_t ps = params_.patch_size; // 8
    const int32_t step = params_.step_size; // e.g. 3 or 4

    ImagePlane32F numerator(w, h);
    ImagePlane32F denominator(w, h);
    numerator.fill(0.0f);
    denominator.fill(0.0f);

    const ImagePlane32F* ref_plane = temporal_planes[ref_index];

    // Slide reference patch across frame
    for (int32_t y = 0; y <= h - ps; y += step) {
        for (int32_t x = 0; x <= w - ps; x += step) {
            float ref_patch[64];
            for (int32_t py = 0; py < ps; ++py) {
                for (int32_t px = 0; px < ps; ++px) {
                    ref_patch[py * 8 + px] = ref_plane->at(x + px, y + py);
                }
            }

            // Estimate local noise sigma
            float center_intensity = ref_patch[27];
            float sigma = nlf.sigma_at(center_intensity);
            float hard_threshold = params_.hard_thresh_lambda * sigma;

            // Grouping: reference patch is always group[0]
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

            // Group matching across temporal neighbors
            for (size_t f = 0; f < temporal_planes.size(); ++f) {
                if (f == ref_index) continue;

                const auto* mf = motion_fields[f];
                int32_t disp_x = 0;
                int32_t disp_y = 0;
                bool trusted = true;

                if (mf) {
                    auto mv = mf->get_vector_at_pixel(x + ps / 2, y + ps / 2);
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

                // Check distance
                float dist = 0.0f;
                for (int32_t i = 0; i < 64; ++i) {
                    float diff = ref_patch[i] - cand_member.patch[i];
                    dist += diff * diff;
                }
                dist /= 64.0f;

                // Residual trust threshold: allow up to 4 * sigma^2
                if (dist <= 4.0f * (sigma * sigma + 1e-6f)) {
                    group.push_back(cand_member);
                    if (group.size() >= static_cast<size_t>(params_.max_group_size)) break;
                }
            }

            int K = static_cast<int>(group.size());

            // 1. Spatial 2D DCT for each patch in group
            std::vector<std::array<float, 64>> spatial_coeffs(K);
            for (int k = 0; k < K; ++k) {
                dct8x8_2d(group[k].patch, spatial_coeffs[k].data());
            }

            // 2. Temporal 1D DCT along group axis K
            std::vector<std::array<float, 64>> transform_3d(K);
            for (int coeff_idx = 0; coeff_idx < 64; ++coeff_idx) {
                float temp_in[16];
                float temp_out[16];
                for (int k = 0; k < K; ++k) temp_in[k] = spatial_coeffs[k][coeff_idx];
                temporal_dct_forward(temp_in, temp_out, K);
                for (int k = 0; k < K; ++k) transform_3d[k][coeff_idx] = temp_out[k];
            }

            // 3. Collaborative Hard Thresholding (Pass 1)
            int non_zero_count = 0;
            for (int k = 0; k < K; ++k) {
                for (int coeff_idx = 0; coeff_idx < 64; ++coeff_idx) {
                    // Do not threshold the global DC coefficient
                    if (k == 0 && coeff_idx == 0) {
                        non_zero_count++;
                        continue;
                    }
                    float val = transform_3d[k][coeff_idx];
                    if (std::abs(val) < hard_threshold) {
                        transform_3d[k][coeff_idx] = 0.0f;
                    } else {
                        non_zero_count++;
                    }
                }
            }

            // 4. Inverse Temporal 1D DCT
            for (int coeff_idx = 0; coeff_idx < 64; ++coeff_idx) {
                float temp_in[16];
                float temp_out[16];
                for (int k = 0; k < K; ++k) temp_in[k] = transform_3d[k][coeff_idx];
                temporal_dct_inverse(temp_in, temp_out, K);
                for (int k = 0; k < K; ++k) spatial_coeffs[k][coeff_idx] = temp_out[k];
            }

            // 5. Inverse Spatial 2D IDCT of Reference Patch (k = 0)
            float filtered_ref_patch[64];
            idct8x8_2d(spatial_coeffs[0].data(), filtered_ref_patch);

            // 6. Aggregation
            float weight = 1.0f / (static_cast<float>(std::max(1, non_zero_count)));
            for (int32_t py = 0; py < ps; ++py) {
                for (int32_t px = 0; px < ps; ++px) {
                    numerator.at(x + px, y + py) += filtered_ref_patch[py * 8 + px] * weight;
                    denominator.at(x + px, y + py) += weight;
                }
            }
        }
    }

    // Canvas normalization
    for (int32_t y = 0; y < h; ++y) {
        for (int32_t x = 0; x < w; ++x) {
            float d = denominator.at(x, y);
            if (d > 1e-7f) {
                output_plane.at(x, y) = numerator.at(x, y) / d;
            } else {
                output_plane.at(x, y) = ref_plane->at(x, y);
            }
        }
    }
}

void CollaborativePatchFilter3D::process(
    const std::vector<VideoFrame32F>& temporal_frames,
    size_t ref_index,
    const std::vector<MotionField>& motion_fields,
    const NoiseProfile& profile,
    VideoFrame32F& output_frame
) {
    if (temporal_frames.empty() || ref_index >= temporal_frames.size()) return;

    output_frame = VideoFrame32F(
        temporal_frames[ref_index].width,
        temporal_frames[ref_index].height,
        temporal_frames[ref_index].channel_count(),
        temporal_frames[ref_index].transfer
    );

    for (size_t c = 0; c < output_frame.channel_count(); ++c) {
        std::vector<const ImagePlane32F*> planes;
        planes.reserve(temporal_frames.size());
        for (const auto& tf : temporal_frames) {
            planes.push_back(&tf.channels[c]);
        }

        std::vector<const MotionField*> mf_ptrs;
        mf_ptrs.reserve(motion_fields.size());
        for (const auto& mf : motion_fields) {
            mf_ptrs.push_back(&mf);
        }

        const auto& nlf = profile.nlf_per_channel[std::min(c, size_t{2})];
        process_plane(planes, ref_index, mf_ptrs, nlf, output_frame.channels[c]);
    }
}

} // namespace librenoise
