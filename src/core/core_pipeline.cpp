#include "librenoise/librenoise.hpp"
#include <iostream>

namespace librenoise {

VideoFrame32F VideoDenoiser::process_window(
    const std::vector<VideoFrame32F>& temporal_window,
    const NoiseProfile* profile
) {
    if (temporal_window.empty()) {
        return VideoFrame32F();
    }

    const size_t total_frames = temporal_window.size();
    const size_t ref_idx = total_frames / 2;
    const auto& ref_frame = temporal_window[ref_idx];

    if (!ref_frame.is_valid()) {
        return VideoFrame32F();
    }

    // Stage 1: Noise profile resolution
    NoiseProfile active_profile;
    if (profile && !profile->profile_name.empty()) {
        active_profile = *profile;
    } else {
        active_profile = NoiseProfiler::auto_estimate(ref_frame);
    }

    // Stage 2: Motion alignment across window
    std::vector<MotionField> motion_fields(total_frames);
    const auto& ref_luma = ref_frame.channels[0];

    for (size_t i = 0; i < total_frames; ++i) {
        if (i == ref_idx) {
            motion_fields[i] = MotionField(); // Identity
            continue;
        }

        const auto& cand_luma = temporal_window[i].channels[0];

        // Hard cut detection: if a cut occurs, do not use this candidate
        if (MotionEstimator::is_scene_cut(ref_luma, cand_luma, config_.match_params.cut_detection_threshold)) {
            continue;
        }

        // Exposure drift calculation for dissolves/fades
        float drift_ratio = NoiseProfiler::compute_exposure_drift(ref_luma, cand_luma);

        motion_fields[i] = MotionEstimator::estimate(
            ref_luma,
            cand_luma,
            active_profile.nlf_per_channel[0],
            drift_ratio,
            config_.match_params
        );
    }

    // Stage 3: Core 3D Collaborative Spatiotemporal Shrinkage
    CollaborativePatchFilter3D filter(config_.filter_params);
    VideoFrame32F denoised_frame;
    filter.process(temporal_window, ref_idx, motion_fields, active_profile, denoised_frame);

    // Stage 4: Detail Polish & Calibrated Grain Re-injection
    for (size_t c = 0; c < denoised_frame.channel_count(); ++c) {
        const auto& nlf = active_profile.nlf_per_channel[std::min(c, size_t{2})];

        if (config_.detail_params.detail_recovery > 0.0f) {
            DetailEngine::restore_details(
                ref_frame.channels[c],
                denoised_frame.channels[c],
                nlf,
                config_.detail_params.detail_recovery
            );
        }

        if (config_.detail_params.grain_amount > 0.0f) {
            DetailEngine::inject_calibrated_grain(
                denoised_frame.channels[c],
                nlf,
                config_.detail_params.grain_amount,
                config_.detail_params.grain_seed + static_cast<uint32_t>(c)
            );
        }
    }

    return denoised_frame;
}

} // namespace librenoise
