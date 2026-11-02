#pragma once

#define LIBRENOISE_VERSION_MAJOR 0
#define LIBRENOISE_VERSION_MINOR 1
#define LIBRENOISE_VERSION_PATCH 0
#define LIBRENOISE_VERSION_STRING "0.1.0-alpha"

#include "types.hpp"
#include "noise_model.hpp"
#include "alignment.hpp"
#include "filter.hpp"
#include "detail.hpp"
#include "colorspace.hpp"

namespace librenoise {

/**
 * @brief High-level facade executing the complete 5-stage denoising pipeline.
 */
class VideoDenoiser {
public:
    struct Config {
        FilterParams filter_params{};
        MotionEstimator::MatchParams match_params{};
        DetailParams detail_params{};
        int32_t temporal_radius = 2; // Radius 2 -> 5 frames total
        int32_t reference_index = -1; // Index of the frame being denoised inside the window (-1 = centre)
    };

    explicit VideoDenoiser(const Config& config = Config{}) : config_(config) {}

    /**
     * @brief Denoise reference frame given a sequence of temporal neighbors.
     * @param temporal_window Sequence of frames of size 2 * temporal_radius + 1.
     * @param profile Sensor noise profile (if empty, auto-estimation is performed).
     * @return Fully filtered and detail-restored frame.
     */
    VideoFrame32F process_window(
        const std::vector<VideoFrame32F>& temporal_window,
        const NoiseProfile* profile = nullptr
    );

    /**
     * @brief Same as process_window() but converts 	emporal_window to the Y/Cb/Cr working space in
     * place instead of copying it (saves RAM on large frames). The frames are converted back to
     * RGB before returning, so the caller's data is restored (to within float round-off).
     *
     * Frames are expected as RGB. The returned frame is RGB. Noise profiles are in the Y/Cb/Cr domain.
     */
    VideoFrame32F process_window_inplace(
        std::vector<VideoFrame32F>& temporal_window,
        const NoiseProfile* profile = nullptr
    );

    const Config& config() const noexcept { return config_; }
    Config& config() noexcept { return config_; }

private:
    Config config_;
};

} // namespace librenoise
