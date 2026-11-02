#include "librenoise/noise_model.hpp"
#include "librenoise/colorspace.hpp"
#include <cmath>
#include <numeric>
#include <algorithm>

namespace librenoise {

namespace {

inline float box4_sample(const ImagePlane32F& plane, int32_t x, int32_t y) {
    const int32_t w = plane.width();
    const int32_t h = plane.height();
    float sum = 0.0f;
    for (int32_t dy = 0; dy < 4; ++dy) {
        int32_t py = std::clamp(y + dy, 0, h - 1);
        for (int32_t dx = 0; dx < 4; ++dx) {
            int32_t px = std::clamp(x + dx, 0, w - 1);
            sum += plane.at(px, py);
        }
    }
    return sum * (1.0f / 16.0f);
}

void dct8x8_block(const float in[64], float out[64]) {
    static float mat[8][8];
    static bool init = false;
    if (!init) {
        constexpr float s0 = 0.35355339059327376f;
        constexpr float s = 0.5f;
        constexpr double pi = 3.14159265358979323846;
        for (int k = 0; k < 8; ++k) {
            float factor = (k == 0) ? s0 : s;
            for (int n = 0; n < 8; ++n) {
                mat[k][n] = factor * static_cast<float>(std::cos((pi / 8.0) * (n + 0.5) * k));
            }
        }
        init = true;
    }

    float tmp[64];
    for (int r = 0; r < 8; ++r) {
        for (int k = 0; k < 8; ++k) {
            float sum = 0.0f;
            for (int n = 0; n < 8; ++n) sum += mat[k][n] * in[r * 8 + n];
            tmp[r * 8 + k] = sum;
        }
    }
    for (int c = 0; c < 8; ++c) {
        for (int k = 0; k < 8; ++k) {
            float sum = 0.0f;
            for (int m = 0; m < 8; ++m) sum += mat[k][m] * tmp[m * 8 + c];
            out[k * 8 + c] = sum;
        }
    }
}

void detrended_mean_and_var(const ImagePlane32F& plane, int32_t bx, int32_t by, int32_t ps, float& out_mean, float& out_var) {
    const float center = 0.5f * static_cast<float>(ps - 1);
    const float count = static_cast<float>(ps * ps);

    float mean = 0.0f;
    for (int32_t py = 0; py < ps; ++py) {
        for (int32_t px = 0; px < ps; ++px) {
            mean += plane.at(bx + px, by + py);
        }
    }
    mean /= count;

    float sx = 0.0f, sy = 0.0f, denom = 0.0f;
    for (int32_t py = 0; py < ps; ++py) {
        float dv = static_cast<float>(py) - center;
        for (int32_t px = 0; px < ps; ++px) {
            float du = static_cast<float>(px) - center;
            float d = plane.at(bx + px, by + py) - mean;
            sx += du * d;
            sy += dv * d;
            denom += du * du;
        }
    }
    float gx = (denom > 1e-6f) ? (sx / denom) : 0.0f;
    float gy = (denom > 1e-6f) ? (sy / denom) : 0.0f;

    float rss = 0.0f;
    for (int32_t py = 0; py < ps; ++py) {
        float dv = static_cast<float>(py) - center;
        for (int32_t px = 0; px < ps; ++px) {
            float du = static_cast<float>(px) - center;
            float res = (plane.at(bx + px, by + py) - mean) - (gx * du + gy * dv);
            rss += res * res;
        }
    }

    out_mean = mean;
    out_var = std::max(1e-8f, rss / std::max(1.0f, count - 3.0f));
}

} // anonymous namespace

NoiseProfile NoiseProfiler::auto_estimate(const VideoFrame32F& frame_in, const ProfileParams& params) {
    // Noise is measured in the same Y/Cb/Cr working space the denoise engines operate in.
    const VideoFrame32F frame = (frame_in.channel_count() >= 3) ? colorspace::to_ycc(frame_in) : frame_in;
    NoiseProfile profile;
    profile.profile_name = "AutoEstimated";
    profile.transfer = frame.transfer;

    if (!frame.is_valid()) {
        return profile;
    }

    const int32_t w = frame.width;
    const int32_t h = frame.height;
    const int32_t ps = std::clamp(params.patch_size, 8, std::max(8, std::min(w, h) / 2));
    const int32_t stride = std::max(ps / 2, std::min(w, h) / 48);

    struct PatchLocation {
        int32_t x = 0;
        int32_t y = 0;
        float grad_norm = 0.0f;
        float mean_val = 0.0f;
    };

    std::vector<PatchLocation> buckets[4];
    std::vector<PatchLocation> all_unclipped;

    // Scan frame for low-structure candidate patches using 4x4 box-smoothed gradients,
    // rejecting clipped-black film mattes / letterbox bars and stratifying across tonal zones.
    const size_t num_ch = std::min(size_t{3}, frame.channel_count());
    const auto& ch0 = frame.channels[0];
    const auto& ch1 = frame.channels[std::min(size_t{1}, num_ch - 1)];
    const auto& ch2 = frame.channels[std::min(size_t{2}, num_ch - 1)];
    const int32_t margin = std::min(8, std::max(0, (std::min(w, h) - ps) / 2));

    for (int32_t y = margin; y + ps <= h - margin; y += stride) {
        for (int32_t x = margin; x + ps <= w - margin; x += stride) {
            float grad_sum = 0.0f;
            float mean_sum = 0.0f;
            float min_lum = 1e9f;
            float max_lum = -1e9f;
            int gcount = 0;
            int zero_count = 0;

            for (int32_t py = 0; py <= ps - 4; py += 4) {
                for (int32_t px = 0; px <= ps - 4; px += 4) {
                    float c0 = box4_sample(ch0, x + px, y + py);
                    float c1 = box4_sample(ch1, x + px, y + py);
                    float c2 = box4_sample(ch2, x + px, y + py);
                    (void)c1; (void)c2;
                    if (c0 <= 1e-4f) zero_count++; // clipped-black matte (luma only; chroma may be negative)

                    float lum = c0;
                    mean_sum += lum;
                    min_lum = std::min(min_lum, lum);
                    max_lum = std::max(max_lum, lum);

                    float gx = box4_sample(ch0, x + px + 2, y + py) - box4_sample(ch0, x + px - 2, y + py);
                    float gy = box4_sample(ch0, x + px, y + py + 2) - box4_sample(ch0, x + px, y + py - 2);
                    grad_sum += std::abs(gx) + std::abs(gy);
                    gcount++;
                }
            }

            if (gcount == 0) continue;
            float mean_v = mean_sum / static_cast<float>(gcount);
            float raw_grad = grad_sum / static_cast<float>(gcount);
            float norm_denom = std::max(0.06f, mean_v);
            float rel_grad = raw_grad / norm_denom;
            float rel_span = (max_lum - min_lum) / norm_denom;
            float score_metric = rel_grad + 0.35f * rel_span;

            // Skip clipped-black letterbox / Cineon dmin mattes unless the entire image is near-zero
            if (zero_count == 0 && mean_v >= 0.005f) {
                PatchLocation pl{x, y, score_metric, mean_v};
                if (rel_span < 0.35f) {
                    int b_idx = (mean_v < 0.05f) ? 0 : (mean_v < 0.15f) ? 1 : (mean_v < 0.45f) ? 2 : 3;
                    buckets[b_idx].push_back(pl);
                }
                all_unclipped.push_back(pl);
            } else if (all_unclipped.empty()) {
                all_unclipped.push_back({x, y, score_metric + 10.0f, mean_v});
            }
        }
    }

    auto cmp_patch = [](const PatchLocation& a, const PatchLocation& b) {
        return a.grad_norm < b.grad_norm;
    };

    std::vector<PatchLocation> flat_patches;
    const float rel_smooth_thresh = std::max(params.flatness_threshold * 8.0f, 0.24f);
    const size_t per_bucket_cap = std::max<size_t>(4, static_cast<size_t>(params.max_patches) / 4);

    for (int b = 0; b < 4; ++b) {
        std::sort(buckets[b].begin(), buckets[b].end(), cmp_patch);
        size_t added = 0;
        for (const auto& cand : buckets[b]) {
            if (cand.grad_norm < rel_smooth_thresh) {
                flat_patches.push_back(cand);
                if (++added >= per_bucket_cap) break;
            }
        }
    }

    if (flat_patches.size() < 6 && !all_unclipped.empty()) {
        std::sort(all_unclipped.begin(), all_unclipped.end(), cmp_patch);
        for (const auto& cand : all_unclipped) {
            flat_patches.push_back(cand);
            if (flat_patches.size() >= 12) break;
        }
    }

    for (size_t c = 0; c < 3; ++c) {
        size_t src_c = std::min(c, num_ch - 1);
        const auto& plane = frame.channels[src_c];

        if (!flat_patches.empty()) {
            struct PatchSample {
                int32_t x;
                int32_t y;
                float mean;
                float var;
                float norm_var;
            };
            std::vector<PatchSample> samples;
            samples.reserve(flat_patches.size());
            std::vector<float> norm_vars;
            norm_vars.reserve(flat_patches.size());

            for (const auto& loc : flat_patches) {
                float mean = 0.0f, var = 0.0f;
                detrended_mean_and_var(plane, loc.x, loc.y, ps, mean, var);
                if (c > 0) { // noise level is a function of LUMA brightness for every channel
                    float dummy_var = 0.0f;
                    detrended_mean_and_var(frame.channels[0], loc.x, loc.y, ps, mean, dummy_var);
                }
                float nv = var / std::max(0.03f, mean);
                samples.push_back({loc.x, loc.y, mean, var, nv});
                norm_vars.push_back(nv);
            }

            std::sort(norm_vars.begin(), norm_vars.end());
            float med_nv = norm_vars[norm_vars.size() / 2];
            float max_allowed_nv = med_nv * 2.2f + 1e-7f;

            float sum_x = 0.0f, sum_y = 0.0f, sum_xy = 0.0f, sum_xx = 0.0f;
            std::vector<float> inlier_vars;
            inlier_vars.reserve(samples.size());

            std::array<double, 5> band_energy{0.0, 0.0, 0.0, 0.0, 0.0};
            std::array<int64_t, 5> band_counts{0, 0, 0, 0, 0};
            double total_ac_energy = 0.0;
            int64_t total_ac_bins = 0;

            for (const auto& sp : samples) {
                if (sp.norm_var > max_allowed_nv && inlier_vars.size() >= 3) continue;
                inlier_vars.push_back(sp.var);
                sum_x += sp.mean;
                sum_y += sp.var;
                sum_xy += sp.mean * sp.var;
                sum_xx += sp.mean * sp.mean;

                // Measure planar-detrended 8x8 2D DCT PSD inside verified flat inlier patch
                for (int32_t by = sp.y; by + 8 <= sp.y + ps; by += 8) {
                    for (int32_t bx = sp.x; bx + 8 <= sp.x + ps; bx += 8) {
                        float block[64];
                        float bmean = 0.0f;
                        for (int py = 0; py < 8; ++py) {
                            for (int px = 0; px < 8; ++px) {
                                float v = plane.at(bx + px, by + py);
                                block[py * 8 + px] = v;
                                bmean += v;
                            }
                        }
                        bmean /= 64.0f;
                        float sx = 0.0f, sy = 0.0f;
                        for (int py = 0; py < 8; ++py) {
                            float dv = static_cast<float>(py) - 3.5f;
                            for (int px = 0; px < 8; ++px) {
                                float du = static_cast<float>(px) - 3.5f;
                                float d = block[py * 8 + px] - bmean;
                                sx += du * d;
                                sy += dv * d;
                            }
                        }
                        float gx = sx / 336.0f;
                        float gy = sy / 336.0f;
                        for (int py = 0; py < 8; ++py) {
                            float dv = static_cast<float>(py) - 3.5f;
                            for (int px = 0; px < 8; ++px) {
                                float du = static_cast<float>(px) - 3.5f;
                                block[py * 8 + px] = (block[py * 8 + px] - bmean) - (gx * du + gy * dv);
                            }
                        }

                        float spec[64];
                        dct8x8_block(block, spec);
                        for (int v = 0; v < 8; ++v) {
                            for (int u = 0; u < 8; ++u) {
                                if (u == 0 && v == 0) continue;
                                float r = std::sqrt(static_cast<float>(u * u + v * v));
                                int band = (r < 1.5f) ? 0 : (r < 3.0f) ? 1 : (r < 5.0f) ? 2 : (r < 7.5f) ? 3 : 4;
                                double pwr = static_cast<double>(spec[v * 8 + u]) * static_cast<double>(spec[v * 8 + u]);
                                band_energy[band] += pwr;
                                band_counts[band]++;
                                total_ac_energy += pwr;
                                total_ac_bins++;
                            }
                        }
                    }
                }
            }

            float n = std::max(1.0f, static_cast<float>(inlier_vars.size()));
            float avg_mean = std::max(0.02f, sum_x / n);
            std::sort(inlier_vars.begin(), inlier_vars.end());
            float median_var = inlier_vars.empty() ? 1e-6f : inlier_vars[inlier_vars.size() / 2];
            float avg_var = std::min(sum_y / n, median_var * 1.35f);
            float denom = (n * sum_xx - sum_x * sum_x);

            // Require genuine dynamic range spread across inlier flat patches to fit linear slope;
            // otherwise split variance cleanly at avg_mean so Var(avg_mean) == avg_var.
            if (inlier_vars.size() >= 5 && denom > 1e-3f * n * n) {
                float raw_slope = (n * sum_xy - sum_x * sum_y) / denom;
                if (raw_slope > 0.0f) {
                    float slope = std::clamp(raw_slope, 1e-7f, avg_var * 1.8f / avg_mean);
                    float intercept = std::max(1e-8f, (sum_y / n) - slope * (sum_x / n));
                    profile.nlf_per_channel[c].a = slope;
                    profile.nlf_per_channel[c].b = intercept;
                } else {
                    profile.nlf_per_channel[c].a = avg_var * 0.4f / avg_mean;
                    profile.nlf_per_channel[c].b = avg_var * 0.6f;
                }
            } else {
                profile.nlf_per_channel[c].a = avg_var * 0.5f / avg_mean;
                profile.nlf_per_channel[c].b = avg_var * 0.5f;
            }

            if (total_ac_bins > 0 && total_ac_energy > 1e-14) {
                double mean_ac_pwr = total_ac_energy / static_cast<double>(total_ac_bins);
                for (size_t b = 0; b < 5; ++b) {
                    double b_pwr = (band_counts[b] > 0) ? (band_energy[b] / static_cast<double>(band_counts[b])) : mean_ac_pwr;
                    float sigma_scale = static_cast<float>(std::sqrt(std::max(0.0, b_pwr / mean_ac_pwr)));
                    profile.psd.octave_weights[c][b] = std::clamp(sigma_scale, 0.3f, 4.0f);
                }
            }
        } else {
            profile.nlf_per_channel[c].a = 0.0005f;
            profile.nlf_per_channel[c].b = 0.0001f;
        }
    }

    return profile;
}

NoiseProfile NoiseProfiler::estimate_from_roi(const VideoFrame32F& frame, const Rect& roi) {
    NoiseProfile profile;
    profile.profile_name = "UserSampled";
    if (!frame.is_valid() || roi.empty()) return profile;

    const int32_t rx = std::clamp(roi.x, 0, frame.width - 1);
    const int32_t ry = std::clamp(roi.y, 0, frame.height - 1);
    const int32_t rw = std::clamp(roi.width, 1, frame.width - rx);
    const int32_t rh = std::clamp(roi.height, 1, frame.height - ry);

    for (size_t c = 0; c < frame.channel_count(); ++c) {
        const auto& plane = frame.channels[c];
        float sum = 0.0f;
        float sum_sq = 0.0f;
        for (int32_t y = 0; y < rh; ++y) {
            for (int32_t x = 0; x < rw; ++x) {
                float v = plane.at(rx + x, ry + y);
                sum += v;
                sum_sq += v * v;
            }
        }
        float count = static_cast<float>(rw * rh);
        float mean = sum / count;
        float var = std::max(1e-8f, (sum_sq / count) - (mean * mean));

        // Point estimate
        profile.nlf_per_channel[c].b = var;
        profile.nlf_per_channel[c].a = var * 0.05f;
    }

    return profile;
}

void NoiseProfiler::apply_variance_stabilization(ImagePlane32F& plane, const NoiseLevelFunction& nlf) {
    const float a = std::max(1e-7f, nlf.a);
    const float b = std::max(1e-7f, nlf.b);
    const float inv_a = 1.0f / a;
    const float constant_term = (3.0f / 8.0f) * a * a + b;

    float* data = plane.data();
    const size_t total_pixels = static_cast<size_t>(plane.stride() * plane.height());

    for (size_t i = 0; i < total_pixels; ++i) {
        float x = std::max(0.0f, data[i]);
        // Generalized Anscombe: 2 * sqrt(x + (3/8)*a + b/a)
        data[i] = 2.0f * std::sqrt(std::max(0.0f, x + constant_term * inv_a));
    }
}

void NoiseProfiler::invert_variance_stabilization(ImagePlane32F& plane, const NoiseLevelFunction& nlf) {
    const float a = std::max(1e-7f, nlf.a);
    const float b = std::max(1e-7f, nlf.b);
    const float inv_a = 1.0f / a;
    const float constant_term = (3.0f / 8.0f) * a * a + b;

    float* data = plane.data();
    const size_t total_pixels = static_cast<size_t>(plane.stride() * plane.height());

    for (size_t i = 0; i < total_pixels; ++i) {
        float y = data[i];
        float val = (y * 0.5f) * (y * 0.5f) - constant_term * inv_a;
        data[i] = std::max(0.0f, val);
    }
}

float NoiseProfiler::compute_exposure_drift(const ImagePlane32F& current, const ImagePlane32F& next) {
    if (current.width() != next.width() || current.height() != next.height()) {
        return 1.0f;
    }

    double sum_curr = 0.0;
    double sum_next = 0.0;
    const int32_t step = 4; // Subsampled for fast drift check

    int64_t count = 0;
    for (int32_t y = 0; y < current.height(); y += step) {
        for (int32_t x = 0; x < current.width(); x += step) {
            sum_curr += current.at(x, y);
            sum_next += next.at(x, y);
            count++;
        }
    }

    if (count == 0 || sum_curr <= 1e-5) return 1.0f;
    return static_cast<float>(sum_next / sum_curr);
}

} // namespace librenoise
