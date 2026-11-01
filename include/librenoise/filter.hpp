#pragma once

#include "types.hpp"
#include "noise_model.hpp"
#include "alignment.hpp"
#include <vector>
#include <array>

namespace librenoise {

enum class DenoiseEngine {
    Collaborative3D = 0,    // 3D Transform Shrinkage (Highest Fidelity)
    Wavelet5Band = 1,       // 5-Band Biorthogonal Wavelet Decomposition (Fastest / Real-time)
    NonLocalMeans = 2,      // Accelerated Non-Local Means (Best for Repetitive Textures)
    FrequencyDomain3D = 3   // 3D Spectral Window Shrinkage (Best for Sensor Banding & Chroma)
};

enum class QualityPreset {
    Fast = 0,       // Real-time timeline playback (step_size = 6, fast chroma)
    Balanced = 1,   // Production grading (step_size = 4, fast chroma)
    HighQuality = 2 // Master export (step_size = 3, full redundancy)
};

struct FilterParams {
    DenoiseEngine engine = DenoiseEngine::Collaborative3D;
    QualityPreset quality = QualityPreset::Balanced;

    // Performance & Optimization Toggles
    bool fast_dct = true;            // Use unrolled AAN butterfly (0 trig calls)
    bool fast_chroma = true;         // Subsample chroma patch evaluations by 2x
    bool early_exit_motion = true;   // Terminate block matching early when error < noise floor

    // Per-Band Strengths [0: Very Low, 1: Low, 2: Mid, 3: High, 4: Very High]
    std::array<float, 5> band_weights{1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    std::array<float, 5> tonal_lut{1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    float chroma_scale_cb = 1.0f;
    float chroma_scale_cr = 1.0f;
    int32_t channel_index = 0;

    // Collaborative 3D Parameters
    int32_t patch_size = 8;
    int32_t max_group_size = 16;
    int32_t step_size = 4;           // Derived from QualityPreset, can be overridden
    int32_t search_window = 16;
    float hard_thresh_lambda = 2.7f;
    float temporal_weight = 1.0f;
    float spatial_weight = 1.0f;
    bool enable_wiener_pass = true;
    bool luma_guided_chroma = true;

    // 5-Band Wavelet Parameters
    int32_t wavelet_levels = 5;
    float wavelet_soft_threshold = 2.2f;

    // Non-Local Means Parameters
    int32_t nlm_search_radius = 6;
    int32_t nlm_patch_radius = 2;
    float nlm_h_factor = 1.0f;

    // Frequency-Domain 3D (DFTTest) Parameters
    int32_t dft_block_size = 16;
    int32_t dft_overlap = 8;
    float dft_threshold_factor = 2.5f;
};

/**
 * @brief Base Interface for Candidate Spatial/Spatiotemporal Engines.
 */
class ISpatialEngine {
public:
    virtual ~ISpatialEngine() = default;
    virtual void process_plane(
        const std::vector<const ImagePlane32F*>& temporal_planes,
        size_t ref_index,
        const std::vector<const MotionField*>& motion_fields,
        const NoiseLevelFunction& nlf,
        const NoisePsd& psd,
        const FilterParams& params,
        ImagePlane32F& output_plane
    ) = 0;
};

// Candidate Engine 1: 3D Collaborative Transform (BM3D style)
class CollaborativePatchFilter3D : public ISpatialEngine {
public:
    explicit CollaborativePatchFilter3D(const FilterParams& params = FilterParams{})
        : params_(params) {}

    void process_plane(
        const std::vector<const ImagePlane32F*>& temporal_planes,
        size_t ref_index,
        const std::vector<const MotionField*>& motion_fields,
        const NoiseLevelFunction& nlf,
        const NoisePsd& psd,
        const FilterParams& params,
        ImagePlane32F& output_plane
    ) override;

private:
    FilterParams params_;
};

// Candidate Engine 2: 5-Band Biorthogonal Wavelet Filter
class Wavelet5BandFilter : public ISpatialEngine {
public:
    void process_plane(
        const std::vector<const ImagePlane32F*>& temporal_planes,
        size_t ref_index,
        const std::vector<const MotionField*>& motion_fields,
        const NoiseLevelFunction& nlf,
        const NoisePsd& psd,
        const FilterParams& params,
        ImagePlane32F& output_plane
    ) override;
};

// Candidate Engine 3: Accelerated Non-Local Means (NLM)
class NonLocalMeansFilter : public ISpatialEngine {
public:
    void process_plane(
        const std::vector<const ImagePlane32F*>& temporal_planes,
        size_t ref_index,
        const std::vector<const MotionField*>& motion_fields,
        const NoiseLevelFunction& nlf,
        const NoisePsd& psd,
        const FilterParams& params,
        ImagePlane32F& output_plane
    ) override;
};

// Candidate Engine 4: 3D Frequency-Domain Spectral Thresholding (DFTTest lineage)
class FrequencyDomainFilter3D : public ISpatialEngine {
public:
    void process_plane(
        const std::vector<const ImagePlane32F*>& temporal_planes,
        size_t ref_index,
        const std::vector<const MotionField*>& motion_fields,
        const NoiseLevelFunction& nlf,
        const NoisePsd& psd,
        const FilterParams& params,
        ImagePlane32F& output_plane
    ) override;
};

/**
 * @brief Unified Filter Dispatcher managing all candidate engines.
 */
class FilterDispatcher {
public:
    static void process(
        const std::vector<VideoFrame32F>& temporal_frames,
        size_t ref_index,
        const std::vector<MotionField>& motion_fields,
        const NoiseProfile& profile,
        const FilterParams& params,
        VideoFrame32F& output_frame
    );
};

} // namespace librenoise
