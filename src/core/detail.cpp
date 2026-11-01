#include "librenoise/detail.hpp"
#include <cmath>
#include <algorithm>
#include <thread>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace librenoise {

namespace {

inline int resolve_detail_threads() noexcept {
    int hw = static_cast<int>(std::thread::hardware_concurrency());
    if (hw <= 0) hw = 8;
    return std::clamp((hw * 3) / 4, 2, 16);
}

inline float hash_gaussian_cpu(uint32_t x, uint32_t y, uint32_t seed) noexcept {
    uint32_t h1 = (x * 73856093u) ^ (y * 19349663u) ^ (seed * 83492791u + 0x9E3779B9u);
    h1 ^= h1 >> 16; h1 *= 0x85ebca6bu; h1 ^= h1 >> 13; h1 *= 0xc2b2ae35u; h1 ^= h1 >> 16;

    uint32_t h2 = (x * 2654435761u) ^ (y * 2246822519u) ^ (seed * 3266489917u + 0x7F4A7C15u);
    h2 ^= h2 >> 16; h2 *= 0x85ebca6bu; h2 ^= h2 >> 13; h2 *= 0xc2b2ae35u; h2 ^= h2 >> 16;

    float u1 = std::max(1e-6f, static_cast<float>(h1 & 0x00FFFFFFu) * (1.0f / 16777216.0f));
    float u2 = static_cast<float>(h2 & 0x00FFFFFFu) * (6.283185307f / 16777216.0f);
    return std::sqrt(-2.0f * std::log(u1)) * std::cos(u2);
}

} // anonymous namespace

void DetailEngine::restore_details(
    const ImagePlane32F& raw_plane,
    ImagePlane32F& denoised_plane,
    const NoiseLevelFunction& nlf,
    float recovery_strength
) {
    if (recovery_strength <= 0.001f) return;

    const int32_t w = raw_plane.width();
    const int32_t h = raw_plane.height();
    if (w < 3 || h < 3) return;

    ImagePlane32F base_plane = denoised_plane;
    const int n_threads = resolve_detail_threads();

    #pragma omp parallel for schedule(static) num_threads(n_threads)
    for (int32_t y = 1; y < h - 1; ++y) {
        const float* raw_row = &raw_plane.at(0, y);
        const float* base_prev = &base_plane.at(0, y - 1);
        const float* base_curr = &base_plane.at(0, y);
        const float* base_next = &base_plane.at(0, y + 1);
        float* dst_row = &denoised_plane.at(0, y);

        for (int32_t x = 1; x < w - 1; ++x) {
            float raw_val = raw_row[x];
            float den_val = base_curr[x];
            float residual = raw_val - den_val;

            float gx = (base_curr[x + 1] - base_curr[x - 1]) * 0.5f;
            float gy = (base_next[x] - base_prev[x]) * 0.5f;
            float grad_sq = gx * gx + gy * gy;

            float sigma = nlf.sigma_at(den_val);
            if (grad_sq > (1.8f * 1.8f) * sigma * sigma) {
                float grad_mag = std::sqrt(grad_sq);
                float capped_res = std::clamp(residual, -1.5f * sigma, 1.5f * sigma);
                float boost = std::clamp(recovery_strength * (grad_mag / (grad_mag + 2.0f * sigma)), 0.0f, 0.5f);
                dst_row[x] = std::max(0.0f, den_val + boost * capped_res);
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
    const int n_threads = resolve_detail_threads();

    #pragma omp parallel for schedule(static) num_threads(n_threads)
    for (int32_t y = 0; y < h; ++y) {
        float* row = &plane.at(0, y);
        for (int32_t x = 0; x < w; ++x) {
            float val = row[x];
            float sigma = nlf.sigma_at(val);
            float noise_sample = hash_gaussian_cpu(static_cast<uint32_t>(x), static_cast<uint32_t>(y), seed) * sigma * grain_amount;
            row[x] = std::max(0.0f, val + noise_sample);
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
    const int n_threads = resolve_detail_threads();

    #pragma omp parallel for schedule(static) num_threads(n_threads)
    for (int32_t y = 0; y < h; ++y) {
        for (int32_t x = 0; x < w; ++x) {
            residual.at(x, y) = raw_plane.at(x, y) - denoised_plane.at(x, y);
        }
    }

    return residual;
}

} // namespace librenoise
