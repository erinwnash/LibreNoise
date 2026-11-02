#include "librenoise/librenoise.hpp"
#include "librenoise/colorspace.hpp"
#include <iostream>

namespace librenoise {

VideoFrame32F VideoDenoiser::process_window(
    const std::vector<VideoFrame32F>& temporal_window,
    const NoiseProfile* profile
) {
    std::vector<VideoFrame32F> working = temporal_window;
    return process_window_inplace(working, profile);
}

VideoFrame32F VideoDenoiser::process_window_inplace(
    std::vector<VideoFrame32F>& temporal_window,
    const NoiseProfile* profile
) {
    if (temporal_window.empty()) {
        return VideoFrame32F();
    }

    const size_t total_frames = temporal_window.size();
    const size_t ref_idx = (config_.reference_index >= 0 && static_cast<size_t>(config_.reference_index) < total_frames)
                               ? static_cast<size_t>(config_.reference_index) : total_frames / 2;

    if (!temporal_window[ref_idx].is_valid()) {
        return VideoFrame32F();
    }

    const bool to_working_space = temporal_window[ref_idx].channel_count() >= 3;

    // Stage 1: Noise profile resolution. Profiles are expressed in the Y/Cb/Cr working space;
    // auto-estimation measures the RGB reference frame directly (it converts internally).
    NoiseProfile active_profile;
    if (profile && (!profile->profile_name.empty() ||
                    profile->nlf_per_channel[0].a > 0.0f ||
                    profile->nlf_per_channel[0].b > 0.0f)) {
        active_profile = *profile;
    } else {
        active_profile = NoiseProfiler::auto_estimate(temporal_window[ref_idx]);
    }

    // Convert the whole window to the working colour space in place (restored before returning)
    if (to_working_space) {
        for (auto& f : temporal_window) {
            if (f.channel_count() >= 3) f = colorspace::to_ycc(f);
        }
    }
    auto restore_rgb = [&]() {
        if (!to_working_space) return;
        for (auto& f : temporal_window) {
            if (f.channel_count() >= 3) f = colorspace::to_rgb(f);
        }
    };
    const auto& ref_frame = temporal_window[ref_idx];

    // Stage 2: Motion alignment across window
    std::vector<MotionField> motion_fields(total_frames);
    const auto& ref_luma = ref_frame.channels[0];

    float max_band_scale = 1.0f;
    for (size_t b = 0; b < 5; ++b) {
        max_band_scale = std::max(max_band_scale,
            active_profile.psd.octave_weights[0][b] * config_.filter_params.band_weights[b]);
    }

    for (size_t i = 0; i < total_frames; ++i) {
        if (config_.filter_params.is_aborted()) {
            restore_rgb();
            return temporal_window[ref_idx];
        }
        if (i == ref_idx) {
            motion_fields[i] = MotionField(); // Identity
            continue;
        }

        const auto& cand_luma = temporal_window[i].channels[0];

        // Hard cut detection: if a cut occurs, do not use this candidate
        if (MotionEstimator::is_scene_cut(ref_luma, cand_luma, config_.match_params.cut_detection_threshold)) {
            motion_fields[i].is_scene_cut = true;
            continue;
        }

        // Exposure drift calculation for dissolves/fades
        float drift_ratio = NoiseProfiler::compute_exposure_drift(ref_luma, cand_luma);

        auto match_params = config_.match_params;
        match_params.early_exit = config_.filter_params.early_exit_motion;
        match_params.max_threads = config_.filter_params.max_threads;
        match_params.trust_sigma_multiplier *= max_band_scale;

        motion_fields[i] = MotionEstimator::estimate(
            ref_luma,
            cand_luma,
            active_profile.nlf_per_channel[0],
            drift_ratio,
            match_params
        );
    }

    if (config_.filter_params.is_aborted()) {
        restore_rgb();
        return temporal_window[ref_idx];
    }

    // Stage 3: Core Filtering via Selected Engine Dispatcher
    VideoFrame32F denoised_frame;
    FilterDispatcher::process(
        temporal_window,
        ref_idx,
        motion_fields,
        active_profile,
        config_.filter_params,
        denoised_frame
    );

    if (config_.filter_params.is_aborted()) {
        restore_rgb();
        return temporal_window[ref_idx];
    }

    // Stage 4: Detail Polish & Calibrated Grain Re-injection (luma detail recovery; grain on every channel)
    const bool has_chroma = to_working_space && denoised_frame.channel_count() >= 3;
    const ImagePlane32F luma_guide = has_chroma ? denoised_frame.channels[0] : ImagePlane32F();
    for (size_t c = 0; c < denoised_frame.channel_count(); ++c) {
        const auto& nlf = active_profile.nlf_per_channel[std::min(c, size_t{2})];

        if (config_.detail_params.detail_recovery > 0.0f && (c == 0 || !has_chroma)) {
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
                config_.detail_params.grain_seed + static_cast<uint32_t>(c),
                (has_chroma && c > 0) ? &luma_guide : nullptr
            );
        }
    }

    restore_rgb();
    if (has_chroma) denoised_frame = colorspace::to_rgb(denoised_frame);
    return denoised_frame;
}

} // namespace librenoise
