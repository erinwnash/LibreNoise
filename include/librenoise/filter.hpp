#pragma once

#include "types.hpp"
#include "noise_model.hpp"
#include "alignment.hpp"
#include <vector>

namespace librenoise {

struct FilterParams {
    int32_t patch_size = 8; // N1 = 8x8 patches
    int32_t max_group_size = 16; // Maximum matched patches in 3D group
    int32_t step_size = 3; // Sliding step between reference patches
    int32_t search_window = 16; // Spatial search window
    float hard_thresh_lambda = 2.7f; // Pass 1 threshold multiplier (lambda_3D)
    float temporal_weight = 1.0f; // Global temporal strength slider [0.0, 2.0]
    float spatial_weight = 1.0f; // Global spatial strength slider [0.0, 2.0]
    bool enable_wiener_pass = true; // Two-pass BM structure
    bool luma_guided_chroma = true; // Use luma patch geometry for chroma
};

/**
 * @brief Clean-room 3D Collaborative Spatiotemporal Shrinkage Engine.
 * Fully compliant with SPEC-CR-001; completely independent of TAU source code.
 */
class CollaborativePatchFilter3D {
public:
    explicit CollaborativePatchFilter3D(const FilterParams& params = FilterParams{})
        : params_(params) {}

    /**
     * @brief Execute full 3D spatiotemporal collaborative filtering over reference frame
     * using motion-aligned temporal neighbor frames.
     * 
     * @param temporal_frames Array of temporal frames centered at index (e.g. [-2, -1, 0, +1, +2])
     * @param ref_index Index of the reference frame to be denoised (e.g. 2 for 5 frames)
     * @param motion_fields Motion fields from ref_index to each temporal frame
     * @param profile Sensor noise profile
     * @param output_frame Destination denoised frame
     */
    void process(
        const std::vector<VideoFrame32F>& temporal_frames,
        size_t ref_index,
        const std::vector<MotionField>& motion_fields,
        const NoiseProfile& profile,
        VideoFrame32F& output_frame
    );

    /**
     * @brief Process single plane using collaborative 3D shrinkage.
     */
    void process_plane(
        const std::vector<const ImagePlane32F*>& temporal_planes,
        size_t ref_index,
        const std::vector<const MotionField*>& motion_fields,
        const NoiseLevelFunction& nlf,
        ImagePlane32F& output_plane
    );

private:
    FilterParams params_;
};

} // namespace librenoise
