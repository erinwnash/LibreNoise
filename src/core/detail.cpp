#include "librenoise/detail.hpp"
#include <cmath>
#include <random>
#include <algorithm>

namespace librenoise {

void DetailEngine::restore_details(
    const ImagePlane32F& raw_plane,
    ImagePlane32F& denoised_plane,
    const NoiseLevelFunction& nlf,
    float recovery_strength
) {
    if (recovery_strength <= 0.001f) return;

    const int32_t w = raw_plane.width();
    const int32_t h = raw_plane.height();

    for (int32_t y = 1; y < h - 1; ++y) {
        for (int32_t x = 1; x < w - 1; ++x) {
            float raw_val = raw_plane.at(x, y);
            float den_val = denoised_plane.at(x, y);
            float residual = raw_val - den_val;

            // Local gradient in denoised plane to detect true structural edges
            float gx = denoised_plane.at(x + 1, y) - denoised_plane.at(x - 1, y);
            float gy = denoised_plane.at(x, y + 1) - denoised_plane.at(x, y - 1);
            float grad_mag = std::sqrt(gx * gx + gy * gy);

            // Threshold against expected noise
            float sigma = nlf.sigma_at(den_val);
            if (grad_mag > 1.5f * sigma) {
                // Structural edge region: selectively re-inject high-frequency micro-contrast
                float boost = std::clamp(recovery_strength * (grad_mag / (grad_mag + sigma)), 0.0f, 0.5f);
                denoised_plane.at(x, y) = std::clamp(den_val + boost * residual, 0.0f, 1.0f);
            }
        }
    }
}

void DetailEngine::inject_calibrated_grain(
    ImagePlane32F& plane,
    const NoiseLevelFunction& nlf,
    float grain_amount,
    uint32_t seed
) {
    if (grain_amount <= 0.001f) return;

    const int32_t w = plane.width();
    const int32_t h = plane.height();

    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    for (int32_t y = 0; y < h; ++y) {
        for (int32_t x = 0; x < w; ++x) {
            float val = plane.at(x, y);
            float sigma = nlf.sigma_at(val);
            float noise_sample = dist(rng) * sigma * grain_amount;
            plane.at(x, y) = std::clamp(val + noise_sample, 0.0f, 1.0f);
        }
    }
}

ImagePlane32F DetailEngine::compute_residual(
    const ImagePlane32F& raw_plane,
    const ImagePlane32F& denoised_plane
) {
    const int32_t w = raw_plane.width();
    const int32_t h = raw_plane.height();
    ImagePlane32F residual(w, h);

    for (int32_t y = 0; y < h; ++y) {
        for (int32_t x = 0; x < w; ++x) {
            residual.at(x, y) = raw_plane.at(x, y) - denoised_plane.at(x, y);
        }
    }

    return residual;
}

} // namespace librenoise
