#pragma once

#include "types.hpp"
#include "noise_model.hpp"
#include <vector>

namespace librenoise {

struct MotionVector {
    int16_t dx = 0; // Integer displacement X
    int16_t dy = 0; // Integer displacement Y
    float sub_x = 0.0f; // Subpixel refinement [-0.5, 0.5]
    float sub_y = 0.0f; // Subpixel refinement [-0.5, 0.5]
    float residual = 0.0f; // Mean squared residual error
    bool trusted = true; // Trust flag based on noise profile variance threshold
};

/**
 * @brief 2D grid of motion vectors computed between reference frame t and target frame t+k.
 */
class MotionField {
public:
    MotionField() = default;
    MotionField(int32_t width_blocks, int32_t height_blocks, int32_t block_size)
        : width_blocks_(width_blocks), height_blocks_(height_blocks), block_size_(block_size),
          vectors_(static_cast<size_t>(width_blocks * height_blocks)) {}

    int32_t width_blocks() const noexcept { return width_blocks_; }
    int32_t height_blocks() const noexcept { return height_blocks_; }
    int32_t block_size() const noexcept { return block_size_; }

    MotionVector& at(int32_t bx, int32_t by) noexcept {
        return vectors_[static_cast<size_t>(by * width_blocks_ + bx)];
    }
    const MotionVector& at(int32_t bx, int32_t by) const noexcept {
        return vectors_[static_cast<size_t>(by * width_blocks_ + bx)];
    }

    MotionVector get_vector_at_pixel(int32_t px, int32_t py) const noexcept {
        int32_t bx = std::clamp(px / block_size_, 0, width_blocks_ - 1);
        int32_t by = std::clamp(py / block_size_, 0, height_blocks_ - 1);
        return at(bx, by);
    }

private:
    int32_t width_blocks_ = 0;
    int32_t height_blocks_ = 0;
    int32_t block_size_ = 16;
    std::vector<MotionVector> vectors_;
};

class MotionEstimator {
public:
    struct MatchParams {
        int32_t block_size = 16;
        int32_t search_radius = 16;
        float trust_sigma_multiplier = 2.5f; // Threshold for residual trust
        float cut_detection_threshold = 0.35f; // SAD ratio indicating scene cut
        bool enable_subpixel = true;
        bool early_exit = true; // Early termination if center match is below noise floor
    };

    /**
     * @brief Estimate dense block motion field from reference frame to candidate frame.
     * Incorporates gain compensation when exposure drift is detected.
     */
    static MotionField estimate(
        const ImagePlane32F& ref_luma,
        const ImagePlane32F& target_luma,
        const NoiseLevelFunction& nlf,
        float exposure_drift_ratio = 1.0f,
        const MatchParams& params = MatchParams{}
    );

    /**
     * @brief Detect whether target frame is across a scene cut relative to reference.
     * When a cut is detected, the temporal window resets to prevent ghosting.
     */
    static bool is_scene_cut(
        const ImagePlane32F& ref_luma,
        const ImagePlane32F& target_luma,
        float threshold = 0.35f
    );
};

} // namespace librenoise
