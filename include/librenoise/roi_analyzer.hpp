#pragma once

#include "types.hpp"
#include "noise_model.hpp"
#include <cstdint>
#include <string>

namespace librenoise {

struct AnalysisRoi {
    int32_t x = 0;
    int32_t y = 0;
    int32_t width = 128;
    int32_t height = 128;

    bool is_valid(int32_t img_w, int32_t img_h) const noexcept {
        return (width >= 32 && height >= 32 &&
                x >= 0 && y >= 0 &&
                x + width <= img_w && y + height <= img_h);
    }
};

struct RoiAnalysisResult {
    AnalysisRoi roi;
    float uniformity_score = 0.0f; // 0.0 to 100.0%
    std::string assessment;        // "Excellent", "Good", "Acceptable", "Unsuitable (Edges Detected)"
    
    // Measured Noise Parameters
    float sigma_y = 0.0f;
    float sigma_cb = 0.0f;
    float sigma_cr = 0.0f;
    float mean_intensity_y = 0.5f;

    // 5-Band Measured Frequency Spectrum (relative to baseline)
    std::array<float, 5> measured_psd{1.0f, 1.0f, 1.0f, 1.0f, 1.0f}; // luma (== measured_psd_ch[0])
    // Per-channel spectrum [Y, Cb, Cr][band]; each row is relative to that channel's own noise level
    std::array<std::array<float, 5>, 3> measured_psd_ch{{{1.0f, 1.0f, 1.0f, 1.0f, 1.0f}, {1.0f, 1.0f, 1.0f, 1.0f, 1.0f}, {1.0f, 1.0f, 1.0f, 1.0f, 1.0f}}};

    // Resolved NLF
    NoiseLevelFunction nlf_y;
    NoiseLevelFunction nlf_cb;
    NoiseLevelFunction nlf_cr;
};

class NoiseRoiAnalyzer {
public:
    /**
     * @brief Automatically sweeps the frame to find the largest, most uniform,
     * detail-free patch with healthy exposure (no clipping).
     */
    static AnalysisRoi find_optimal_roi(
        const VideoFrame32F& frame,
        int32_t preferred_size = 128,
        int32_t min_size = 64
    );

    /**
     * @brief Analyzes a selected rectangular patch of the frame to measure noise
     * variance, frequency spectrum, and uniformity quality.
     */
    static RoiAnalysisResult analyze_roi(
        const VideoFrame32F& frame,
        const AnalysisRoi& roi
    );
};

} // namespace librenoise
