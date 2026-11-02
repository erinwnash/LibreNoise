#pragma once

#include "types.hpp"
#include "noise_model.hpp"

namespace librenoise {

struct DetailParams {
    float detail_recovery = 0.15f; // Edge-protected micro-contrast recovery [0.0, 1.0]
    float grain_amount = 0.05f; // Calibrated synthetic grain re-injection [0.0, 1.0]
    uint32_t grain_seed = 1337; // Random seed for grain generation (host should add the frame number so grain is not a fixed pattern) (host should add the frame number so grain is not a fixed pattern)
};

class DetailEngine {
public:
    /**
     * @brief Restore high-frequency micro-contrast along edges without boosting noise.
     * Extracts structural gradients from the residual (raw - denoised) and selectively blends them back.
     */
    static void restore_details(
        const ImagePlane32F& raw_plane,
        ImagePlane32F& denoised_plane,
        const NoiseLevelFunction& nlf,
        float recovery_strength
    );

    /**
     * @brief Inject calibrated synthetic grain matching the camera noise profile
     * to eliminate the waxy/plastic appearance on skin and flat surfaces.
     */
    static void inject_calibrated_grain(
        ImagePlane32F& plane,
        const NoiseLevelFunction& nlf,
        float grain_amount,
        uint32_t seed,
        const ImagePlane32F* guide = nullptr // Luma plane used to evaluate the NLF for chroma planes
    );

    /**
     * @brief Compute the residual image: Residual = Raw - Denoised.
     * Useful for visual inspection and metric evaluation in the CLI harness.
     */
    static ImagePlane32F compute_residual(
        const ImagePlane32F& raw_plane,
        const ImagePlane32F& denoised_plane
    );
};

} // namespace librenoise
