#include "librenoise/roi_analyzer.hpp"
#include <cmath>
#include <algorithm>
#include <vector>
#include <numeric>

namespace librenoise {

namespace {

// Evaluates edge gradient energy inside an ROI on a single plane
float evaluate_patch_gradient(const ImagePlane32F& plane, int x, int y, int w, int h) {
    float grad_sum = 0.0f;
    int count = 0;

    for (int py = y + 1; py < y + h - 1; py += 2) {
        for (int px = x + 1; px < x + w - 1; px += 2) {
            float dx = (plane.at(px + 1, py) - plane.at(px - 1, py)) * 0.5f;
            float dy = (plane.at(px, py + 1) - plane.at(px, py - 1)) * 0.5f;
            grad_sum += std::sqrt(dx * dx + dy * dy);
            count++;
        }
    }
    return count > 0 ? (grad_sum / count) : 1e9f;
}

// Computes variance consistency across 4 sub-quadrants to verify spatial stationarity
float evaluate_quadrant_disparity(const ImagePlane32F& plane, int x, int y, int w, int h) {
    int half_w = w / 2;
    int half_h = h / 2;
    float quad_var[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    int qx[4] = {x, x + half_w, x, x + half_w};
    int qy[4] = {y, y, y + half_h, y + half_h};

    for (int q = 0; q < 4; ++q) {
        float mean = 0.0f;
        int count = 0;
        for (int py = qy[q]; py < qy[q] + half_h; py += 2) {
            for (int px = qx[q]; px < qx[q] + half_w; px += 2) {
                mean += plane.at(px, py);
                count++;
            }
        }
        mean = count > 0 ? (mean / count) : 0.0f;

        float var = 0.0f;
        for (int py = qy[q]; py < qy[q] + half_h; py += 2) {
            for (int px = qx[q]; px < qx[q] + half_w; px += 2) {
                float d = plane.at(px, py) - mean;
                var += d * d;
            }
        }
        quad_var[q] = count > 0 ? (var / count) : 0.0f;
    }

    float avg_var = (quad_var[0] + quad_var[1] + quad_var[2] + quad_var[3]) * 0.25f;
    float max_diff = 0.0f;
    for (int q = 0; q < 4; ++q) {
        max_diff = std::max(max_diff, std::abs(quad_var[q] - avg_var));
    }
    return (avg_var > 1e-8f) ? (max_diff / avg_var) : 0.0f;
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
    const auto& y_plane = frame.channels[0];

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

            // Reject clipped regions
            if (mean_val < 0.03f || mean_val > 0.95f) continue;

            float grad = evaluate_patch_gradient(y_plane, x, y, box_w, box_h);
            float disparity = evaluate_quadrant_disparity(y_plane, x, y, box_w, box_h);

            // Exposure bonus for midtones (0.15 to 0.70)
            float exposure_score = 1.0f - std::abs(mean_val - 0.40f) * 1.5f;

            // Score: penalizes edges heavily, rewards uniform noise and good exposure
            float score = exposure_score * 50.0f - grad * 1000.0f - disparity * 30.0f;

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
    const VideoFrame32F& frame,
    const AnalysisRoi& roi
) {
    RoiAnalysisResult result;
    result.roi = roi;

    if (!frame.is_valid() || !roi.is_valid(frame.width, frame.height)) {
        result.assessment = "Invalid Region";
        return result;
    }

    const auto& y_plane = frame.channels[0];
    const auto& cb_plane = (frame.channel_count() > 1) ? frame.channels[1] : frame.channels[0];
    const auto& cr_plane = (frame.channel_count() > 2) ? frame.channels[2] : frame.channels[0];

    // Compute mean intensity
    float y_sum = 0.0f;
    int count = 0;
    for (int y = roi.y; y < roi.y + roi.height; ++y) {
        for (int x = roi.x; x < roi.x + roi.width; ++x) {
            y_sum += y_plane.at(x, y);
            count++;
        }
    }
    result.mean_intensity_y = count > 0 ? (y_sum / count) : 0.5f;

    // High-pass noise variance estimation (Laplacian kernel avoids smooth gradients)
    auto estimate_channel_sigma = [&](const ImagePlane32F& plane) {
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
        // Normalize Laplacian variance scaling factor sqrt(2)
        return (dcount > 0) ? (std::sqrt(diff_sum / dcount) * 1.4142f) : 0.01f;
    };

    result.sigma_y = estimate_channel_sigma(y_plane);
    result.sigma_cb = estimate_channel_sigma(cb_plane);
    result.sigma_cr = estimate_channel_sigma(cr_plane);

    // Fit heteroscedastic model: sigma^2(I) = a * I + b
    float var_y = result.sigma_y * result.sigma_y;
    result.nlf_y.a = var_y * 0.6f / std::max(0.05f, result.mean_intensity_y);
    result.nlf_y.b = var_y * 0.4f;

    float var_cb = result.sigma_cb * result.sigma_cb;
    result.nlf_cb.a = var_cb * 0.6f / std::max(0.05f, result.mean_intensity_y);
    result.nlf_cb.b = var_cb * 0.4f;

    float var_cr = result.sigma_cr * result.sigma_cr;
    result.nlf_cr.a = var_cr * 0.6f / std::max(0.05f, result.mean_intensity_y);
    result.nlf_cr.b = var_cr * 0.4f;

    // Measure 5-band frequency spectrum across 8x8 blocks
    std::array<float, 5> band_energy{0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    int block_count = 0;

    for (int by = roi.y; by <= roi.y + roi.height - 8; by += 8) {
        for (int bx = roi.x; bx <= roi.x + roi.width - 8; bx += 8) {
            float block_mean = 0.0f;
            for (int py = 0; py < 8; ++py) {
                for (int px = 0; px < 8; ++px) {
                    block_mean += y_plane.at(bx + px, by + py);
                }
            }
            block_mean /= 64.0f;

            for (int py = 0; py < 8; ++py) {
                for (int px = 0; px < 8; ++px) {
                    float val = y_plane.at(bx + px, by + py) - block_mean;
                    float r = std::sqrt(static_cast<float>(px * px + py * py));
                    int band = (r < 1.5f) ? 0 : (r < 3.0f) ? 1 : (r < 5.0f) ? 2 : (r < 7.5f) ? 3 : 4;
                    band_energy[band] += val * val;
                }
            }
            block_count++;
        }
    }

    if (block_count > 0) {
        float total = 0.0f;
        for (int i = 0; i < 5; ++i) {
            band_energy[i] /= static_cast<float>(block_count * 12.8f);
            total += band_energy[i];
        }
        float norm_factor = (total > 1e-7f) ? (5.0f / total) : 1.0f;
        for (int i = 0; i < 5; ++i) {
            result.measured_psd[i] = std::clamp(band_energy[i] * norm_factor, 0.2f, 3.0f);
        }
    }

    // Uniformity and quality score
    float grad = evaluate_patch_gradient(y_plane, roi.x, roi.y, roi.width, roi.height);
    float disparity = evaluate_quadrant_disparity(y_plane, roi.x, roi.y, roi.width, roi.height);

    float raw_score = 100.0f * std::exp(-grad * 120.0f - disparity * 0.8f);
    if (result.mean_intensity_y < 0.05f || result.mean_intensity_y > 0.95f) {
        raw_score *= 0.5f; // Penalize near-black or near-white
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
