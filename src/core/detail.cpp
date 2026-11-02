#include "librenoise/detail.hpp"
#include <cmath>
#include <algorithm>

namespace librenoise {

namespace {
inline uint32_t hash_u32(uint32_t v) {
    v ^= v >> 16; v *= 0x7feb352dU; v ^= v >> 15; v *= 0x846ca68bU; v ^= v >> 16;
    return v;
}
inline float hash_gaussian(uint32_t x, uint32_t y, uint32_t seed) {
    const uint32_t h1 = hash_u32(x * 0x9E3779B1U ^ hash_u32(y + seed * 0x85EBCA6BU));
    const uint32_t h2 = hash_u32(h1 ^ 0xC2B2AE35U);
    const float u1 = (static_cast<float>(h1 >> 8) + 0.5f) * (1.0f / 16777216.0f);
    const float u2 = static_cast<float>(h2 >> 8) * (1.0f / 16777216.0f);
    return std::sqrt(-2.0f * std::log(u1)) * std::cos(6.28318530718f * u2);
}
} // namespace


void DetailEngine::restore_details(
    const ImagePlane32F& raw_plane,
    ImagePlane32F& denoised_plane,
    const NoiseLevelFunction& nlf,
    float recovery_strength
) {
    if (recovery_strength <= 0.001f) return;

    const int32_t w = raw_plane.width();
    const int32_t h = raw_plane.height();

    // Gradients are read from an unmodified snapshot so the result is independent of visit order
    // (and rows can run in parallel).
    const ImagePlane32F snapshot = denoised_plane;

    #pragma omp parallel for schedule(static)
    for (int32_t y = 1; y < h - 1; ++y) {
        const float* raw_row = raw_plane.data() + static_cast<ptrdiff_t>(y) * raw_plane.stride();
        const float* up = snapshot.data() + static_cast<ptrdiff_t>(y - 1) * snapshot.stride();
        const float* mid = snapshot.data() + static_cast<ptrdiff_t>(y) * snapshot.stride();
        const float* dn = snapshot.data() + static_cast<ptrdiff_t>(y + 1) * snapshot.stride();
        float* dst = denoised_plane.data() + static_cast<ptrdiff_t>(y) * denoised_plane.stride();
        for (int32_t x = 1; x < w - 1; ++x) {
            const float den_val = mid[x];
            const float gx = (mid[x + 1] - mid[x - 1]) * 0.5f;
            const float gy = (dn[x] - up[x]) * 0.5f;
            const float sigma = nlf.sigma_at(den_val);
            // Compare squared magnitudes first: skips the sqrt on the (majority) flat pixels
            const float gm2 = gx * gx + gy * gy;
            const float thr = 1.8f * sigma;
            if (gm2 > thr * thr) {
                const float grad_mag = std::sqrt(gm2);
                const float residual = raw_row[x] - den_val;
                const float capped_res = std::clamp(residual, -1.5f * sigma, 1.5f * sigma);
                const float boost = std::clamp(recovery_strength * (grad_mag / (grad_mag + 2.0f * sigma)), 0.0f, 0.5f);
                dst[x] = std::max(0.0f, den_val + boost * capped_res);
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

    // Counter-based Gaussian (hash + Box-Muller): same N(0,1) statistics as a sequential RNG, but
    // every pixel is independent, so rows can be generated in parallel and frames are repeatable.
    #pragma omp parallel for schedule(static)
    for (int32_t y = 0; y < h; ++y) {
        float* row = plane.data() + static_cast<ptrdiff_t>(y) * plane.stride();
        for (int32_t x = 0; x < w; ++x) {
            const float val = row[x];
            const float sigma = nlf.sigma_at(val);
            row[x] = std::max(0.0f, val + hash_gaussian(static_cast<uint32_t>(x), static_cast<uint32_t>(y), seed) * sigma * grain_amount);
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
