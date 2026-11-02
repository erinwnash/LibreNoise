#include "librenoise/roi_analyzer.hpp"
#include "librenoise/colorspace.hpp"
#include <cmath>
#include <algorithm>
#include <vector>
#include <numeric>

namespace librenoise {

namespace {

inline float sample_box4(const ImagePlane32F& plane, int x, int y) {
    const int w = plane.width();
    const int h = plane.height();
    float sum = 0.0f;
    for (int dy = 0; dy < 4; ++dy) {
        int py = std::clamp(y + dy, 0, h - 1);
        for (int dx = 0; dx < 4; ++dx) {
            int px = std::clamp(x + dx, 0, w - 1);
            sum += plane.at(px, py);
        }
    }
    return sum * (1.0f / 16.0f);
}

// Evaluates structural edge gradient energy inside an ROI using 4x4 box-smoothed samples
// so sensor grain does not masquerade as image structure.
float evaluate_patch_gradient(const ImagePlane32F& plane, int x, int y, int w, int h) {
    float grad_sum = 0.0f;
    int count = 0;

    for (int py = y + 2; py < y + h - 4; py += 4) {
        for (int px = x + 2; px < x + w - 4; px += 4) {
            float dx = (sample_box4(plane, px + 2, py) - sample_box4(plane, px - 2, py)) * 0.25f;
            float dy = (sample_box4(plane, px, py + 2) - sample_box4(plane, px, py - 2)) * 0.25f;
            grad_sum += std::sqrt(dx * dx + dy * dy);
            count++;
        }
    }
    return count > 0 ? (grad_sum / count) : 1e9f;
}

// Computes planar-detrended variance of a B x B block
float compute_detrended_block_variance(const ImagePlane32F& plane, int bx, int by, int bs) {
    const float center = 0.5f * static_cast<float>(bs - 1);
    const float count = static_cast<float>(bs * bs);

    float mean = 0.0f;
    for (int py = 0; py < bs; ++py) {
        for (int px = 0; px < bs; ++px) {
            mean += plane.at(bx + px, by + py);
        }
    }
    mean /= count;

    float sum_ux = 0.0f;
    float sum_vy = 0.0f;
    float denom_u = 0.0f;
    for (int py = 0; py < bs; ++py) {
        float dv = static_cast<float>(py) - center;
        for (int px = 0; px < bs; ++px) {
            float du = static_cast<float>(px) - center;
            float val = plane.at(bx + px, by + py) - mean;
            sum_ux += du * val;
            sum_vy += dv * val;
            denom_u += du * du;
        }
    }

    float gx = (denom_u > 1e-6f) ? (sum_ux / denom_u) : 0.0f;
    float gy = (denom_u > 1e-6f) ? (sum_vy / denom_u) : 0.0f;

    float rss = 0.0f;
    for (int py = 0; py < bs; ++py) {
        float dv = static_cast<float>(py) - center;
        for (int px = 0; px < bs; ++px) {
            float du = static_cast<float>(px) - center;
            float res = (plane.at(bx + px, by + py) - mean) - (gx * du + gy * dv);
            rss += res * res;
        }
    }

    float dof = std::max(1.0f, count - 3.0f);
    return std::max(1e-9f, rss / dof);
}

// Computes structural range & variance disparity across all 16x16 sub-blocks in an ROI
// so any pole, fold, or edge passing anywhere through the box is immediately penalized.
void evaluate_roi_stationarity(
    const ImagePlane32F& plane, int x, int y, int w, int h,
    float& out_mean_span, float& out_var_disparity, float& out_clipped_fraction
) {
    const int bs = std::min(16, std::min(w, h));
    const int step = std::max(8, bs);
    float min_bmean = 1e9f;
    float max_bmean = -1e9f;
    std::vector<float> bvars;
    int zero_count = 0;
    int total_samples = 0;

    for (int by = y; by <= y + h - bs; by += step) {
        for (int bx = x; bx <= x + w - bs; bx += step) {
            float bsum = 0.0f;
            for (int py = 0; py < bs; py += 2) {
                for (int px = 0; px < bs; px += 2) {
                    float v = plane.at(bx + px, by + py);
                    bsum += v;
                    if (v <= 1e-5f) zero_count++;
                    total_samples++;
                }
            }
            float bmean = bsum / static_cast<float>((bs / 2) * (bs / 2));
            min_bmean = std::min(min_bmean, bmean);
            max_bmean = std::max(max_bmean, bmean);
            bvars.push_back(compute_detrended_block_variance(plane, bx, by, bs));
        }
    }

    out_mean_span = (max_bmean > min_bmean) ? (max_bmean - min_bmean) : 0.0f;
    out_clipped_fraction = (total_samples > 0) ? (static_cast<float>(zero_count) / static_cast<float>(total_samples)) : 0.0f;

    if (bvars.size() >= 2) {
        std::sort(bvars.begin(), bvars.end());
        float med = bvars[bvars.size() / 2];
        float p90 = bvars[(bvars.size() * 9) / 10];
        out_var_disparity = (med > 1e-8f) ? ((p90 - med) / med) : 0.0f;
    } else {
        out_var_disparity = 0.0f;
    }
}

void dct8x8_forward(const float in[64], float out[64]) {
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

} // anonymous namespace

AnalysisRoi NoiseRoiAnalyzer::find_optimal_roi(
    const VideoFrame32F& frame,
    int32_t preferred_size,
    int32_t min_size
) {
    AnalysisRoi best_roi;
    if (!frame.is_valid()) return best_roi;

    const int32_t img_w = frame.width;
    const int32_t img_h = frame.height;
    const VideoFrame32F luma_frame = (frame.channel_count() >= 3) ? colorspace::to_ycc(frame) : frame;
    const auto& y_plane = luma_frame.channels[0];

    int32_t box_w = std::min(preferred_size, img_w - 4);
    int32_t box_h = std::min(preferred_size, img_h - 4);
    if (box_w < min_size || box_h < min_size) {
        box_w = std::max(32, std::min(min_size, img_w));
        box_h = std::max(32, std::min(min_size, img_h));
    }

    best_roi.x = (img_w - box_w) / 2;
    best_roi.y = (img_h - box_h) / 2;
    best_roi.width = box_w;
    best_roi.height = box_h;

    float best_score = -1e9f;
    const int step = std::max(16, box_w / 2);

    for (int32_t y = 8; y <= img_h - box_h - 8; y += step) {
        for (int32_t x = 8; x <= img_w - box_w - 8; x += step) {
            // Mean intensity check
            float mean_val = 0.0f;
            int count = 0;
            for (int py = y; py < y + box_h; py += 4) {
                for (int px = x; px < x + box_w; px += 4) {
                    mean_val += y_plane.at(px, py);
                    count++;
                }
            }
            mean_val = count > 0 ? (mean_val / count) : 0.5f;

            // Reject clipped-black matte / letterbox or blown super-whites, while supporting scene-linear shadows (0.008+)
            if (mean_val < 0.008f || mean_val > 2.5f) continue;

            float mean_span = 0.0f, disparity = 0.0f, clipped_frac = 0.0f;
            evaluate_roi_stationarity(y_plane, x, y, box_w, box_h, mean_span, disparity, clipped_frac);
            if (clipped_frac > 0.02f) continue;

            float grad = evaluate_patch_gradient(y_plane, x, y, box_w, box_h);
            float norm_scale = std::max(0.05f, mean_val);
            float rel_grad = grad / norm_scale;
            float rel_span = mean_span / norm_scale;

            // Perceptual midtone preference (~0.04 to 0.45 scene-linear = 24% to 70% sRGB)
            float perc = std::sqrt(std::clamp(mean_val, 0.0f, 1.0f));
            float exposure_score = 1.0f - std::abs(perc - 0.42f) * 0.8f;

            // Score heavily penalizes any structural edge, curtain fold, or block-mean span inside the ROI
            float score = exposure_score * 25.0f - rel_grad * 450.0f - rel_span * 120.0f - disparity * 15.0f;

            if (score > best_score) {
                best_score = score;
                best_roi.x = x;
                best_roi.y = y;
                best_roi.width = box_w;
                best_roi.height = box_h;
            }
        }
    }

    return best_roi;
}

RoiAnalysisResult NoiseRoiAnalyzer::analyze_roi(
    const VideoFrame32F& frame_rgb,
    const AnalysisRoi& roi
) {
    RoiAnalysisResult result;
    // Measure in the Y/Cb/Cr working space used by the denoise engines.
    const VideoFrame32F frame = (frame_rgb.channel_count() >= 3) ? colorspace::to_ycc(frame_rgb) : frame_rgb;
    result.roi = roi;

    if (!frame.is_valid() || !roi.is_valid(frame.width, frame.height)) {
        result.assessment = "Invalid Region";
        return result;
    }

    const auto& y_plane = frame.channels[0];
    const auto& cb_plane = (frame.channel_count() > 1) ? frame.channels[1] : frame.channels[0];
    const auto& cr_plane = (frame.channel_count() > 2) ? frame.channels[2] : frame.channels[0];
    const ImagePlane32F* planes[3] = {&y_plane, &cb_plane, &cr_plane};

    // Compute per-channel mean intensity
    auto compute_roi_mean = [&](const ImagePlane32F& plane) {
        double sum = 0.0;
        int count = 0;
        for (int y = roi.y; y < roi.y + roi.height; ++y) {
            for (int x = roi.x; x < roi.x + roi.width; ++x) {
                sum += plane.at(x, y);
                count++;
            }
        }
        return count > 0 ? static_cast<float>(sum / count) : 0.5f;
    };

    float mean_y = compute_roi_mean(y_plane);
    float mean_cb = compute_roi_mean(cb_plane);
    float mean_cr = compute_roi_mean(cr_plane);
    result.mean_intensity_y = mean_y;

    // Multi-scale planar-detrended block variance estimation:
    // Cancels smooth lighting gradients while capturing correlated cinema/Bayer grain,
    // using median across blocks to reject stray dust or edge outliers.
    auto estimate_channel_sigma = [&](const ImagePlane32F& plane) {
        std::vector<float> block_vars;
        const int bs = (roi.width >= 16 && roi.height >= 16) ? 16 : 8;
        const int step = std::max(4, bs / 2);

        if (roi.width >= bs && roi.height >= bs) {
            for (int by = roi.y; by <= roi.y + roi.height - bs; by += step) {
                for (int bx = roi.x; bx <= roi.x + roi.width - bs; bx += step) {
                    block_vars.push_back(compute_detrended_block_variance(plane, bx, by, bs));
                }
            }
        }

        // 1px Laplacian fallback / floor
        float diff_sum = 0.0f;
        int dcount = 0;
        for (int y = roi.y + 1; y < roi.y + roi.height - 1; ++y) {
            for (int x = roi.x + 1; x < roi.x + roi.width - 1; ++x) {
                float center = plane.at(x, y);
                float lap = (plane.at(x + 1, y) + plane.at(x - 1, y) +
                             plane.at(x, y + 1) + plane.at(x, y - 1)) * 0.25f - center;
                diff_sum += lap * lap;
                dcount++;
            }
        }
        float lap_var = (dcount > 0) ? ((diff_sum / dcount) * 0.8f) : 1e-6f;

        if (!block_vars.empty()) {
            std::sort(block_vars.begin(), block_vars.end());
            float median_var = block_vars[block_vars.size() / 2];
            return std::sqrt(std::max(median_var, lap_var * 0.5f));
        }
        return std::sqrt(std::max(1e-8f, lap_var));
    };

    result.sigma_y = estimate_channel_sigma(y_plane);
    result.sigma_cb = estimate_channel_sigma(cb_plane);
    result.sigma_cr = estimate_channel_sigma(cr_plane);

    // Fit heteroscedastic model per channel: sigma^2(I) = a * I + b
    float var_y = result.sigma_y * result.sigma_y;
    // All three channels look their noise level up against LUMA intensity.
    result.nlf_y.a = var_y * 0.5f / std::max(0.05f, mean_y);
    result.nlf_y.b = var_y * 0.5f;

    float var_cb = result.sigma_cb * result.sigma_cb;
    (void)mean_cb; (void)mean_cr;
    result.nlf_cb.a = var_cb * 0.5f / std::max(0.05f, mean_y);
    result.nlf_cb.b = var_cb * 0.5f;

    float var_cr = result.sigma_cr * result.sigma_cr;
    result.nlf_cr.a = var_cr * 0.5f / std::max(0.05f, mean_y);
    result.nlf_cr.b = var_cr * 0.5f;

    // Measure the 5-band 2D DCT noise spectrum of each channel independently (sigma_scale relative to
    // that channel's own mean AC amplitude). Chroma noise is typically far lower-frequency than luma.
    for (int ch = 0; ch < 3; ++ch) {
        std::array<double, 5> band_energy{0.0, 0.0, 0.0, 0.0, 0.0};
        std::array<int64_t, 5> band_counts{0, 0, 0, 0, 0};
        double total_ac_energy = 0.0;
        int64_t total_ac_bins = 0;
        const auto& pl = *planes[ch];
        for (int by = roi.y; by <= roi.y + roi.height - 8; by += 8) {
            for (int bx = roi.x; bx <= roi.x + roi.width - 8; bx += 8) {
                float patch[64];
                float mean = 0.0f;
                for (int py = 0; py < 8; ++py) {
                    for (int px = 0; px < 8; ++px) {
                        float v = pl.at(bx + px, by + py);
                        patch[py * 8 + px] = v;
                        mean += v;
                    }
                }
                mean /= 64.0f;

                // Detrend linear plane before 2D DCT so smooth ramps don't bias Band 0
                float sx = 0.0f, sy = 0.0f;
                for (int py = 0; py < 8; ++py) {
                    float dv = static_cast<float>(py) - 3.5f;
                    for (int px = 0; px < 8; ++px) {
                        float du = static_cast<float>(px) - 3.5f;
                        float d = patch[py * 8 + px] - mean;
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
                        patch[py * 8 + px] = (patch[py * 8 + px] - mean) - (gx * du + gy * dv);
                    }
                }

                float spec[64];
                dct8x8_forward(patch, spec);

                for (int v = 0; v < 8; ++v) {
                    for (int u = 0; u < 8; ++u) {
                        if (u == 0 && v == 0) continue; // Skip DC
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

        if (total_ac_bins > 0 && total_ac_energy > 1e-14) {
            double mean_ac_pwr = total_ac_energy / static_cast<double>(total_ac_bins);
            for (int i = 0; i < 5; ++i) {
                double band_mean = (band_counts[i] > 0) ? (band_energy[i] / static_cast<double>(band_counts[i])) : mean_ac_pwr;
                float sigma_scale = static_cast<float>(std::sqrt(std::max(0.0, band_mean / mean_ac_pwr)));
                // Wide clamp: strongly correlated (blurred / compressed) noise has a very uneven spectrum
                result.measured_psd_ch[static_cast<size_t>(ch)][static_cast<size_t>(i)] = std::clamp(sigma_scale, 0.3f, 4.0f);
            }
        }
    }
    result.measured_psd = result.measured_psd_ch[0];
    // Uniformity and quality score (normalized for both scene-linear and display-referred plates)
    float grad = evaluate_patch_gradient(y_plane, roi.x, roi.y, roi.width, roi.height);
    float mean_span = 0.0f, disparity = 0.0f, clipped_frac = 0.0f;
    evaluate_roi_stationarity(y_plane, roi.x, roi.y, roi.width, roi.height, mean_span, disparity, clipped_frac);

    float norm_scale = std::max(0.06f, result.mean_intensity_y);
    float rel_span = std::max(0.0f, (mean_span - 1.8f * result.sigma_y) / norm_scale);
    float rel_grad = std::max(0.0f, (grad - 0.35f * result.sigma_y) / norm_scale);

    float raw_score = 100.0f * std::exp(-rel_grad * 18.0f - rel_span * 3.5f - disparity * 0.35f);
    if (result.mean_intensity_y < 0.006f || clipped_frac > 0.05f) {
        raw_score *= 0.4f; // Penalize clipped black / letterbox matte
    }
    result.uniformity_score = std::clamp(raw_score, 0.0f, 99.9f);

    if (result.uniformity_score >= 80.0f) {
        result.assessment = "Excellent (High Reliability)";
    } else if (result.uniformity_score >= 60.0f) {
        result.assessment = "Good (Suitable for Production)";
    } else if (result.uniformity_score >= 40.0f) {
        result.assessment = "Acceptable (Low Contrast Textures Present)";
    } else {
        result.assessment = "Unsuitable (Edges or Strong Texture Detected)";
    }

    return result;
}

} // namespace librenoise
