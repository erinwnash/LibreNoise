#include "librenoise/noise_model.hpp"
#include <cmath>
#include <numeric>
#include <algorithm>

namespace librenoise {

NoiseProfile NoiseProfiler::auto_estimate(const VideoFrame32F& frame, const ProfileParams& params) {
    NoiseProfile profile;
    profile.profile_name = "AutoEstimated";
    profile.transfer = frame.transfer;

    if (!frame.is_valid()) {
        return profile;
    }

    const int32_t w = frame.width;
    const int32_t h = frame.height;
    const int32_t ps = params.patch_size;

    struct PatchStats {
        float mean = 0.0f;
        float variance = 0.0f;
        float grad_norm = 0.0f;
    };

    std::vector<PatchStats> flat_patches;

    // Scan frame for low-gradient candidate patches
    const auto& luma = frame.channels[0];
    for (int32_t y = ps; y + ps < h; y += ps) {
        for (int32_t x = ps; x + ps < w; x += ps) {
            float sum = 0.0f;
            float sum_sq = 0.0f;
            float grad_sum = 0.0f;

            for (int32_t py = 0; py < ps; ++py) {
                for (int32_t px = 0; px < ps; ++px) {
                    float val = luma.at(x + px, y + py);
                    sum += val;
                    sum_sq += val * val;

                    // Simple gradient
                    float gx = luma.at(x + px + 1, y + py) - luma.at(x + px - 1, y + py);
                    float gy = luma.at(x + px, y + py + 1) - luma.at(x + px, y + py - 1);
                    grad_sum += std::abs(gx) + std::abs(gy);
                }
            }

            float count = static_cast<float>(ps * ps);
            float mean = sum / count;
            float var = std::max(0.0f, (sum_sq / count) - (mean * mean));
            float mean_grad = grad_sum / count;

            if (mean_grad < params.flatness_threshold) {
                flat_patches.push_back({mean, var, mean_grad});
            }
        }
    }

    // Sort by lowest gradient / variance
    std::sort(flat_patches.begin(), flat_patches.end(), [](const PatchStats& a, const PatchStats& b) {
        return a.grad_norm < b.grad_norm;
    });

    if (flat_patches.size() > static_cast<size_t>(params.max_patches)) {
        flat_patches.resize(static_cast<size_t>(params.max_patches));
    }

    // Robust linear fit for Var(I) = a*I + b if sufficient patches found
    if (flat_patches.size() >= 5) {
        float sum_x = 0.0f, sum_y = 0.0f, sum_xy = 0.0f, sum_xx = 0.0f;
        for (const auto& p : flat_patches) {
            sum_x += p.mean;
            sum_y += p.variance;
            sum_xy += p.mean * p.variance;
            sum_xx += p.mean * p.mean;
        }
        float n = static_cast<float>(flat_patches.size());
        float denom = (n * sum_xx - sum_x * sum_x);
        if (std::abs(denom) > 1e-7f) {
            float slope = (n * sum_xy - sum_x * sum_y) / denom;
            float intercept = (sum_y - slope * sum_x) / n;
            profile.nlf_per_channel[0].a = std::clamp(slope, 1e-6f, 0.05f);
            profile.nlf_per_channel[0].b = std::clamp(intercept, 1e-7f, 0.01f);
        }
    } else {
        // Fallback default
        profile.nlf_per_channel[0].a = 0.0005f;
        profile.nlf_per_channel[0].b = 0.00005f;
    }

    // Chroma channels typically have lower shot noise slope but cross-coupling
    profile.nlf_per_channel[1] = profile.nlf_per_channel[0];
    profile.nlf_per_channel[2] = profile.nlf_per_channel[0];

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
