#pragma once

#include "types.hpp"
#include <array>
#include <string>
#include <vector>

namespace librenoise {

/**
 * @brief Heteroscedastic Signal-Dependent Noise Level Function (NLF).
 * Models variance as a linear function of intensity: Var(I) = a * I + b
 * where 'a' corresponds to photon shot noise gain and 'b' corresponds to thermal/read noise.
 */
struct NoiseLevelFunction {
    float a = 0.0001f; // Shot noise slope (gain)
    float b = 0.00001f; // Read noise pedestal

    float variance_at(float intensity) const noexcept {
        return std::max(1e-8f, a * std::max(0.0f, intensity) + b);
    }

    float sigma_at(float intensity) const noexcept {
        return std::sqrt(variance_at(intensity));
    }
};

/**
 * @brief Frequency and channel-resolved noise power spectrum density (PSD).
 * Decomposes noise into 5 spatial frequency bands:
 *   [0] Very Low, [1] Low, [2] Mid, [3] High, [4] Very High
 * across 3 decorrelated channels: [0] Y, [1] Cb, [2] Cr.
 */
struct NoisePsd {
    static constexpr size_t NUM_OCTAVES = 5;
    static constexpr size_t NUM_CHANNELS = 3;

    // sigma_scale[channel][octave] relative to base sigma
    std::array<std::array<float, NUM_OCTAVES>, NUM_CHANNELS> octave_weights{};

    NoisePsd() {
        for (size_t c = 0; c < NUM_CHANNELS; ++c) {
            for (size_t o = 0; o < NUM_OCTAVES; ++o) {
                octave_weights[c][o] = 1.0f;
            }
        }
    }
};

/**
 * @brief Complete sensor noise profile.
 */
struct NoiseProfile {
    std::string profile_name = "Generic_Default";
    std::string camera_model = "Unknown";
    int32_t iso = 800;
    ColorTransfer transfer = ColorTransfer::Linear;

    std::array<NoiseLevelFunction, 3> nlf_per_channel; // Y, Cb, Cr
    NoisePsd psd;
    bool has_fixed_pattern = false;
    bool remove_compression_artifacts = true; // Separate processing for compression/blocking artifacts
    ImagePlane32F fixed_pattern_map;

    NoiseProfile() = default;
};

/**
 * @brief Noise Profiler subsystem.
 */
class NoiseProfiler {
public:
    struct ProfileParams {
        int32_t patch_size = 32;
        int32_t max_patches = 100;
        float flatness_threshold = 0.02f;
    };

    /**
     * @brief Automatically detect uniform/featureless regions and build a calibrated profile.
     */
    static NoiseProfile auto_estimate(const VideoFrame32F& frame, const ProfileParams& params = ProfileParams{});

    /**
     * @brief Estimate noise parameters from a manually designated user ROI.
     */
    static NoiseProfile estimate_from_roi(const VideoFrame32F& frame, const Rect& roi);

    /**
     * @brief Apply generalized Anscombe variance-stabilizing transform:
     * Transforms heteroscedastic noise Var(I) = a*I + b into homoscedastic noise ~ N(0, 1).
     */
    static void apply_variance_stabilization(ImagePlane32F& plane, const NoiseLevelFunction& nlf);

    /**
     * @brief Invert the generalized Anscombe transform back to linear space.
     */
    static void invert_variance_stabilization(ImagePlane32F& plane, const NoiseLevelFunction& nlf);

    /**
     * @brief Compute global luminance drift between two consecutive frames for dissolve/fade detection.
     */
    static float compute_exposure_drift(const ImagePlane32F& current, const ImagePlane32F& next);
};

} // namespace librenoise
