#include "openfx/ofxCore.h"
#include "openfx/ofxProperty.h"
#include "openfx/ofxParam.h"
#include "openfx/ofxImageEffect.h"
#include "openfx/ofxGPURender.h"
#include "librenoise/librenoise.hpp"
#include "librenoise/gpu_denoiser.hpp"
#include "librenoise/roi_analyzer.hpp"
#include "librenoise/profile_curve.hpp"
#include "librenoise/profile_io.hpp"
#include "ofx_interact.hpp"

#if defined(LIBRENOISE_HAS_CUDA) && (LIBRENOISE_HAS_CUDA != 0)
#include <cuda_runtime.h>
#endif

#include <cstring>
#include <algorithm>
#include <vector>
#include <cmath>
#include <iostream>
#include <string>
#include <optional>
#include <thread>

#if defined(_WIN32) || defined(__WIN32__)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#define OFX_EXPORT __declspec(dllexport)
#else
#include <pthread.h>
#define OFX_EXPORT __attribute__((visibility("default")))
#endif

// Host Suites with external linkage so ofx_interact can access them
OfxHost* g_host = nullptr;
const OfxPropertySuiteV1* g_propSuite = nullptr;
const OfxParameterSuiteV1* g_paramSuite = nullptr;
const OfxImageEffectSuiteV1* g_effectSuite = nullptr;
const OfxInteractSuiteV1* g_interactSuite = nullptr;

namespace {

// Tooltip helper (shown by the host when hovering a parameter)
void set_hint(OfxPropertySetHandle props, const char* text) {
    g_propSuite->propSetString(props, kOfxParamPropHint, 0, text);
}

// Parameter Group Names
constexpr const char* kGroupProfile           = "groupProfile";
constexpr const char* kGroupDenoise           = "groupDenoise";
constexpr const char* kGroupFrequency         = "groupFrequency";
constexpr const char* kGroupTonal             = "groupTonal";
constexpr const char* kGroupDetail            = "groupDetail";
constexpr const char* kGroupPerformance       = "groupPerformance";

// Profiling & ROI Parameters
constexpr const char* kParamAnalysisBoxCorner1 = "analysisBoxCorner1";
constexpr const char* kParamAnalysisBoxCorner2 = "analysisBoxCorner2";
constexpr const char* kParamAutoProfile        = "autoProfile";
constexpr const char* kParamAutoDetectArea     = "autoDetectArea";
constexpr const char* kParamBuildProfile       = "buildProfile";
constexpr const char* kParamProfileQuality     = "profileQuality";
constexpr const char* kParamProfileAssessment  = "profileAssessment";
constexpr const char* kParamProfileDevice      = "profileDevice";
constexpr const char* kParamProfileMode        = "profileMode";
constexpr const char* kParamProfilePath        = "profilePath";
constexpr const char* kParamLoadProfile        = "loadProfile";
constexpr const char* kParamSaveProfile        = "saveProfile";

// Primary Processing Engine Controls
constexpr const char* kParamDenoiseEngine      = "denoiseEngine";
constexpr const char* kParamAccelerationMode   = "accelerationMode";
constexpr const char* kParamQualityPreset      = "qualityPreset";
constexpr const char* kParamTemporalRadius     = "temporalRadius";
constexpr const char* kParamTemporalStrength   = "temporalStrength";
constexpr const char* kParamSpatialStrength    = "spatialStrength";
constexpr const char* kParamLumaScale          = "lumaScale";
constexpr const char* kParamChromaCr           = "chromaCr";
constexpr const char* kParamChromaCb           = "chromaCb";
constexpr const char* kParamMixWithOriginal     = "mixWithOriginal";

// 5-Band Strengths
constexpr const char* kParamBandVeryLow        = "bandVeryLow";
constexpr const char* kParamBandLow            = "bandLow";
constexpr const char* kParamBandMid            = "bandMid";
constexpr const char* kParamBandHigh           = "bandHigh";
constexpr const char* kParamBandVeryHigh       = "bandVeryHigh";

// Tonal Equalizer
constexpr const char* kParamTonalShadows       = "tonalShadows";
constexpr const char* kParamTonalDarkMid       = "tonalDarkMid";
constexpr const char* kParamTonalMidtones      = "tonalMidtones";
constexpr const char* kParamTonalLightMid      = "tonalLightMid";
constexpr const char* kParamTonalHighlights    = "tonalHighlights";

// Detail & Grain
constexpr const char* kParamDetailRecovery     = "detailRecovery";
constexpr const char* kParamGrainAmount        = "grainAmount";
constexpr const char* kParamPreviewMode        = "previewMode";

// Calibration Persistence (Secret Parameters)
constexpr const char* kParamCalibrated          = "profileCalibrated";
constexpr const char* kParamNlfSlopeY           = "nlfSlopeY";
constexpr const char* kParamNlfInterceptY       = "nlfInterceptY";
constexpr const char* kParamNlfSlopeCb          = "nlfSlopeCb";
constexpr const char* kParamNlfInterceptCb      = "nlfInterceptCb";
constexpr const char* kParamNlfSlopeCr          = "nlfSlopeCr";
constexpr const char* kParamNlfInterceptCr      = "nlfInterceptCr";

// Performance Toggles
constexpr const char* kParamFastChroma         = "fastChroma";
constexpr const char* kParamFastDct            = "fastDct";
constexpr const char* kParamEarlyExitMotion    = "earlyExitMotion";

// Host background worker state & OS-native CPU render serialization (0 MSVCP140 ABI dependency)
int g_isBackgroundHost = 0;

struct CpuRenderLock {
#if defined(_WIN32) || defined(__WIN32__)
    SRWLOCK lock = SRWLOCK_INIT;
    void acquire() noexcept { AcquireSRWLockExclusive(&lock); }
    void release() noexcept { ReleaseSRWLockExclusive(&lock); }
#else
    pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
    void acquire() noexcept { pthread_mutex_lock(&lock); }
    void release() noexcept { pthread_mutex_unlock(&lock); }
#endif
};
static CpuRenderLock g_cpuRenderLock;

struct ScopedCpuRenderLock {
    CpuRenderLock& m;
    explicit ScopedCpuRenderLock(CpuRenderLock& l) noexcept : m(l) { m.acquire(); }
    ~ScopedCpuRenderLock() noexcept { m.release(); }
};

inline int compute_polite_cpu_threads() noexcept {
    int hw = static_cast<int>(std::thread::hardware_concurrency());
    if (hw <= 0) hw = 8;
    if (g_isBackgroundHost != 0) {
        // Multi-worker FrameServer / render-farm child process: share cores cleanly across workers
        return std::clamp(hw / 4, 2, 6);
    }
    // Interactive host GUI: keep 25% of logical cores free for UI, viewer, & FrameServer broker
    return std::clamp((hw * 3) / 4, 2, 16);
}

inline bool is_ofx_aborted(OfxImageEffectHandle effect) noexcept {
    return (effect != nullptr && g_effectSuite != nullptr &&
            g_effectSuite->abort != nullptr && g_effectSuite->abort(effect) == 1);
}

bool ofx_abort_trampoline(void* user_data) {
    return is_ofx_aborted(reinterpret_cast<OfxImageEffectHandle>(user_data));
}

// Action: Load
OfxStatus action_load() {
    if (!g_host || !g_host->host || !g_host->fetchSuite) return kOfxStatErrFatal;
    g_propSuite     = static_cast<const OfxPropertySuiteV1*>(g_host->fetchSuite(g_host->host, kOfxPropertySuite, 1));
    g_paramSuite    = static_cast<const OfxParameterSuiteV1*>(g_host->fetchSuite(g_host->host, kOfxParameterSuite, 1));
    g_effectSuite   = static_cast<const OfxImageEffectSuiteV1*>(g_host->fetchSuite(g_host->host, kOfxImageEffectSuite, 1));
    g_interactSuite = static_cast<const OfxInteractSuiteV1*>(g_host->fetchSuite(g_host->host, kOfxInteractSuite, 1));

    if (!g_propSuite || !g_paramSuite || !g_effectSuite) {
        return kOfxStatErrMissingHostFeature;
    }
    g_propSuite->propGetInt(g_host->host, kOfxImageEffectHostPropIsBackground, 0, &g_isBackgroundHost);
    return kOfxStatOK;
}

// Helper struct and pixel conversion utilities for arbitrary OpenFX host image layouts
struct OfxImgInfo {
    void* data = nullptr;
    int rowBytes = 0;
    int bounds[4] = {0, 0, 0, 0};
    char* bitDepth = nullptr;
    char* components = nullptr;
    int channels = 4;

    int width() const { return bounds[2] - bounds[0]; }
    int height() const { return bounds[3] - bounds[1]; }
    bool valid() const { return data != nullptr && width() > 0 && height() > 0 && rowBytes != 0; }
};

OfxImgInfo query_ofx_image(OfxPropertySetHandle img) {
    OfxImgInfo info;
    if (!img || !g_propSuite) return info;
    g_propSuite->propGetPointer(img, kOfxImagePropData, 0, &info.data);
    g_propSuite->propGetInt(img, kOfxImagePropRowBytes, 0, &info.rowBytes);
    g_propSuite->propGetIntN(img, kOfxImagePropBounds, 4, info.bounds);
    g_propSuite->propGetString(img, kOfxImageEffectPropPixelDepth, 0, &info.bitDepth);
    g_propSuite->propGetString(img, kOfxImageEffectPropComponents, 0, &info.components);
    if (info.components) {
        if (std::strcmp(info.components, kOfxImageComponentRGB) == 0) info.channels = 3;
        else if (std::strcmp(info.components, kOfxImageComponentAlpha) == 0) info.channels = 1;
        else info.channels = 4;
    }
    return info;
}

inline void sample_ofx_pixel_rgba(const OfxImgInfo& info, int world_x, int world_y, float out_rgba[4]) {
    int sx = std::clamp(world_x, info.bounds[0], info.bounds[2] - 1) - info.bounds[0];
    int sy = std::clamp(world_y, info.bounds[1], info.bounds[3] - 1) - info.bounds[1];
    const auto* row_ptr = reinterpret_cast<const uint8_t*>(info.data) +
                          static_cast<ptrdiff_t>(sy) * static_cast<ptrdiff_t>(info.rowBytes);
    const int ch = info.channels;

    out_rgba[0] = 0.0f; out_rgba[1] = 0.0f; out_rgba[2] = 0.0f; out_rgba[3] = 1.0f;

    if (info.bitDepth && std::strcmp(info.bitDepth, kOfxBitDepthFloat) == 0) {
        const auto* p = reinterpret_cast<const float*>(row_ptr) + sx * ch;
        if (ch >= 3) {
            out_rgba[0] = p[0]; out_rgba[1] = p[1]; out_rgba[2] = p[2];
            out_rgba[3] = (ch >= 4) ? p[3] : 1.0f;
        } else if (ch == 1) {
            out_rgba[0] = out_rgba[1] = out_rgba[2] = p[0];
        }
    } else if (info.bitDepth && std::strcmp(info.bitDepth, kOfxBitDepthShort) == 0) {
        const auto* p = reinterpret_cast<const uint16_t*>(row_ptr) + sx * ch;
        constexpr float inv16 = 1.0f / 65535.0f;
        if (ch >= 3) {
            out_rgba[0] = p[0] * inv16; out_rgba[1] = p[1] * inv16; out_rgba[2] = p[2] * inv16;
            out_rgba[3] = (ch >= 4) ? (p[3] * inv16) : 1.0f;
        } else if (ch == 1) {
            out_rgba[0] = out_rgba[1] = out_rgba[2] = p[0] * inv16;
        }
    } else if (info.bitDepth && std::strcmp(info.bitDepth, kOfxBitDepthByte) == 0) {
        const auto* p = row_ptr + sx * ch;
        constexpr float inv8 = 1.0f / 255.0f;
        if (ch >= 3) {
            out_rgba[0] = p[0] * inv8; out_rgba[1] = p[1] * inv8; out_rgba[2] = p[2] * inv8;
            out_rgba[3] = (ch >= 4) ? (p[3] * inv8) : 1.0f;
        } else if (ch == 1) {
            out_rgba[0] = out_rgba[1] = out_rgba[2] = p[0] * inv8;
        }
    }
}

inline float sample_ofx_alpha(const OfxImgInfo& info, int world_x, int world_y) noexcept {
    if (info.channels < 4) return 1.0f;
    int sx = std::clamp(world_x, info.bounds[0], info.bounds[2] - 1) - info.bounds[0];
    int sy = std::clamp(world_y, info.bounds[1], info.bounds[3] - 1) - info.bounds[1];
    const auto* row_ptr = reinterpret_cast<const uint8_t*>(info.data) +
                          static_cast<ptrdiff_t>(sy) * static_cast<ptrdiff_t>(info.rowBytes);
    if (info.bitDepth && std::strcmp(info.bitDepth, kOfxBitDepthFloat) == 0) {
        return (reinterpret_cast<const float*>(row_ptr) + sx * info.channels)[3];
    }
    if (info.bitDepth && std::strcmp(info.bitDepth, kOfxBitDepthShort) == 0) {
        return static_cast<float>((reinterpret_cast<const uint16_t*>(row_ptr) + sx * info.channels)[3]) * (1.0f / 65535.0f);
    }
    if (info.bitDepth && std::strcmp(info.bitDepth, kOfxBitDepthByte) == 0) {
        return static_cast<float>((row_ptr + sx * info.channels)[3]) * (1.0f / 255.0f);
    }
    return 1.0f;
}

// Action: Describe
OfxStatus action_describe(OfxImageEffectHandle effect) {
    OfxPropertySetHandle effectProps = nullptr;
    g_effectSuite->getPropertySet(effect, &effectProps);

    // Identity and Labels
    g_propSuite->propSetString(effectProps, kOfxPropLabel, 0, "LibreNoise");
    g_propSuite->propSetString(effectProps, kOfxPropShortLabel, 0, "LibreNoise");
    g_propSuite->propSetString(effectProps, kOfxPropLongLabel, 0, "LibreNoise Video Denoiser");
    g_propSuite->propSetString(effectProps, kOfxPropPluginDescription, 0,
        "Production-grade open-source temporal and spatial video denoiser with interactive profiling.");

    // Contexts supported: Filter & General
    g_propSuite->propSetString(effectProps, kOfxImageEffectPropSupportedContexts, 0, kOfxImageEffectContextFilter);
    g_propSuite->propSetString(effectProps, kOfxImageEffectPropSupportedContexts, 1, kOfxImageEffectContextGeneral);

    // Pixel depths
    g_propSuite->propSetString(effectProps, kOfxImageEffectPropSupportedPixelDepths, 0, kOfxBitDepthFloat);
    g_propSuite->propSetString(effectProps, kOfxImageEffectPropSupportedPixelDepths, 1, kOfxBitDepthShort);
    g_propSuite->propSetString(effectProps, kOfxImageEffectPropSupportedPixelDepths, 2, kOfxBitDepthByte);

    // Full-frame untiled spatial/motion analysis & single render call per frame (OpenMP/CUDA internal threading)
    g_propSuite->propSetInt(effectProps, kOfxImageEffectPropSupportsTiles, 0, 0);
    g_propSuite->propSetInt(effectProps, kOfxImageEffectPluginPropHostFrameThreading, 0, 0);
    g_propSuite->propSetString(effectProps, kOfxImageEffectPluginRenderThreadSafety, 0, kOfxImageEffectRenderInstanceSafe);

    // Request temporal frame access
    g_propSuite->propSetInt(effectProps, kOfxImageEffectPropTemporalClipAccess, 0, 1);

    // OpenFX 1.5 GPU Render Support (Track B - CUDA native)
    g_propSuite->propSetString(effectProps, kOfxImageEffectPropCudaRenderSupported, 0, "true");
    g_propSuite->propSetString(effectProps, kOfxImageEffectPropCudaStreamSupported, 0, "true");
    g_propSuite->propSetString(effectProps, kOfxImageEffectPropOpenCLRenderSupported, 0, "false");

    // Viewer Overlay Interact Entry (Track B / Interactive Parity)
    g_propSuite->propSetPointer(effectProps, kOfxImageEffectPluginPropOverlayInteractV1, 0,
        reinterpret_cast<void*>(libre_noise_overlay_interact_entry));

    return kOfxStatOK;
}

// Action: DescribeInContext
OfxStatus action_describe_in_context(OfxImageEffectHandle effect, OfxPropertySetHandle inArgs) {
    (void)inArgs;
    // Clips: Define Output and Source (untiled full-frame buffers)
    OfxPropertySetHandle outputProps = nullptr;
    g_effectSuite->clipDefine(effect, kOfxImageEffectOutputClipName, &outputProps);
    g_propSuite->propSetString(outputProps, kOfxImageEffectPropSupportedComponents, 0, kOfxImageComponentRGBA);
    g_propSuite->propSetString(outputProps, kOfxImageEffectPropSupportedComponents, 1, kOfxImageComponentRGB);
    g_propSuite->propSetInt(outputProps, kOfxImageEffectPropSupportsTiles, 0, 0);

    OfxPropertySetHandle sourceProps = nullptr;
    g_effectSuite->clipDefine(effect, kOfxImageEffectSimpleSourceClipName, &sourceProps);
    g_propSuite->propSetString(sourceProps, kOfxImageEffectPropSupportedComponents, 0, kOfxImageComponentRGBA);
    g_propSuite->propSetString(sourceProps, kOfxImageEffectPropSupportedComponents, 1, kOfxImageComponentRGB);
    g_propSuite->propSetInt(sourceProps, kOfxImageEffectPropTemporalClipAccess, 0, 1);
    g_propSuite->propSetInt(sourceProps, kOfxImageEffectPropSupportsTiles, 0, 0);

    // Overlay interact descriptor hook for context
    OfxPropertySetHandle effectProps = nullptr;
    g_effectSuite->getPropertySet(effect, &effectProps);
    if (effectProps) {
        g_propSuite->propSetPointer(effectProps, kOfxImageEffectPluginPropOverlayInteractV1, 0,
            reinterpret_cast<void*>(libre_noise_overlay_interact_entry));
    }

    // Parameter Set
    OfxParamSetHandle paramSet = nullptr;
    g_effectSuite->getParamSet(effect, &paramSet);
    OfxPropertySetHandle pProps = nullptr;

    // =========================================================================
    // Group 1: Noise Profile & Analysis
    // =========================================================================
    g_paramSuite->paramDefine(paramSet, kOfxParamTypeGroup, kGroupProfile, &pProps);
    set_hint(pProps, "Teach the denoiser what the noise in this clip looks like. A good profile matters more than any slider below.");
    g_propSuite->propSetString(pProps, kOfxPropLabel, 0, "Noise Profile & Analysis");

    // Analysis Box: Double2D Corner 1 and Corner 2
    g_paramSuite->paramDefine(paramSet, kOfxParamTypeDouble2D, kParamAnalysisBoxCorner1, &pProps);
    set_hint(pProps, "Top-left corner of the noise sample area. Place the box on a flat, featureless region (sky, wall, out-of-focus background).");
    g_propSuite->propSetString(pProps, kOfxPropLabel, 0, "Region Top-Left");
    g_propSuite->propSetString(pProps, kOfxParamPropDoubleType, 0, kOfxParamDoubleTypeXYAbsolute);
    g_propSuite->propSetString(pProps, kOfxParamPropParent, 0, kGroupProfile);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDefault, 0, 100.0);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDefault, 1, 100.0);

    g_paramSuite->paramDefine(paramSet, kOfxParamTypeDouble2D, kParamAnalysisBoxCorner2, &pProps);
    set_hint(pProps, "Bottom-right corner of the noise sample area. The box should contain no edges or texture.");
    g_propSuite->propSetString(pProps, kOfxPropLabel, 0, "Region Bottom-Right");
    g_propSuite->propSetString(pProps, kOfxParamPropDoubleType, 0, kOfxParamDoubleTypeXYAbsolute);
    g_propSuite->propSetString(pProps, kOfxParamPropParent, 0, kGroupProfile);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDefault, 0, 260.0);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDefault, 1, 260.0);

    // Auto Profile Button
    g_paramSuite->paramDefine(paramSet, kOfxParamTypePushButton, kParamAutoProfile, &pProps);
    set_hint(pProps, "Analyse the current frame, automatically find flat areas and build the noise profile. Quickest way to start.");
    g_propSuite->propSetString(pProps, kOfxPropLabel, 0, "Auto Profile (Search & Analyze)");
    g_propSuite->propSetString(pProps, kOfxParamPropParent, 0, kGroupProfile);

    // Auto Detect Area Button
    g_paramSuite->paramDefine(paramSet, kOfxParamTypePushButton, kParamAutoDetectArea, &pProps);
    set_hint(pProps, "Search the frame for the flattest, most noise-only region and move the sample box there.");
    g_propSuite->propSetString(pProps, kOfxPropLabel, 0, "Auto Detect Area");
    g_propSuite->propSetString(pProps, kOfxParamPropParent, 0, kGroupProfile);

    // Build Profile from Selection
    g_paramSuite->paramDefine(paramSet, kOfxParamTypePushButton, kParamBuildProfile, &pProps);
    set_hint(pProps, "Measure noise inside the sample box and store it as this clip's noise profile (Luma, Cb and Cr measured separately).");
    g_propSuite->propSetString(pProps, kOfxPropLabel, 0, "Build Profile from Selection");
    g_propSuite->propSetString(pProps, kOfxParamPropParent, 0, kGroupProfile);

    // Quality Meter (Read-only string readout)
    g_paramSuite->paramDefine(paramSet, kOfxParamTypeString, kParamProfileQuality, &pProps);
    set_hint(pProps, "How reliable the current profile looks (based on how flat the sampled area was).");
    g_propSuite->propSetString(pProps, kOfxPropLabel, 0, "Profile Quality");
    g_propSuite->propSetString(pProps, kOfxParamPropParent, 0, kGroupProfile);
    g_propSuite->propSetString(pProps, kOfxParamPropDefault, 0, "Uncalibrated");

    // Profile Assessment
    g_paramSuite->paramDefine(paramSet, kOfxParamTypeString, kParamProfileAssessment, &pProps);
    set_hint(pProps, "Plain-language verdict on the profile quality and what to do if it is poor.");
    g_propSuite->propSetString(pProps, kOfxPropLabel, 0, "Assessment");
    g_propSuite->propSetString(pProps, kOfxParamPropParent, 0, kGroupProfile);
    g_propSuite->propSetString(pProps, kOfxParamPropDefault, 0, "Click Auto Profile or drag box on viewer");

    // Device & Mode Metadata
    g_paramSuite->paramDefine(paramSet, kOfxParamTypeString, kParamProfileDevice, &pProps);
    set_hint(pProps, "Optional label for the camera / sensor this profile belongs to (saved with the .lnp file).");
    g_propSuite->propSetString(pProps, kOfxPropLabel, 0, "Camera / Device");
    g_propSuite->propSetString(pProps, kOfxParamPropParent, 0, kGroupProfile);
    g_propSuite->propSetString(pProps, kOfxParamPropDefault, 0, "Generic Digital Camera");

    g_paramSuite->paramDefine(paramSet, kOfxParamTypeString, kParamProfileMode, &pProps);
    set_hint(pProps, "Optional label for the ISO / shooting mode this profile was made for (saved with the .lnp file).");
    g_propSuite->propSetString(pProps, kOfxPropLabel, 0, "Mode Description");
    g_propSuite->propSetString(pProps, kOfxParamPropParent, 0, kGroupProfile);
    g_propSuite->propSetString(pProps, kOfxParamPropDefault, 0, "Standard ISO Profile");

    // Profile File Path & Load / Save
    std::string defaultProfilePath = librenoise::ProfileSerializer::get_default_profiles_directory() + "/profile.lnp";
    g_paramSuite->paramDefine(paramSet, kOfxParamTypeString, kParamProfilePath, &pProps);
    set_hint(pProps, "Location of the .lnp noise-profile file used by the Load / Save buttons.");
    g_propSuite->propSetString(pProps, kOfxPropLabel, 0, "Profile File (.lnp)");
    g_propSuite->propSetString(pProps, kOfxParamPropParent, 0, kGroupProfile);
    g_propSuite->propSetString(pProps, kOfxParamPropDefault, 0, defaultProfilePath.c_str());

    g_paramSuite->paramDefine(paramSet, kOfxParamTypePushButton, kParamLoadProfile, &pProps);
    set_hint(pProps, "Load a saved noise profile (.lnp) from the Profile File path and apply it to this clip.");
    g_propSuite->propSetString(pProps, kOfxPropLabel, 0, "Load Profile (.lnp)...");
    g_propSuite->propSetString(pProps, kOfxParamPropParent, 0, kGroupProfile);

    g_paramSuite->paramDefine(paramSet, kOfxParamTypePushButton, kParamSaveProfile, &pProps);
    set_hint(pProps, "Save the current noise profile, band and tonal settings to the Profile File path for reuse on matching footage.");
    g_propSuite->propSetString(pProps, kOfxPropLabel, 0, "Save Profile (.lnp)...");
    g_propSuite->propSetString(pProps, kOfxParamPropParent, 0, kGroupProfile);

    // =========================================================================
    // Group 2: Denoise Controls
    // =========================================================================
    g_paramSuite->paramDefine(paramSet, kOfxParamTypeGroup, kGroupDenoise, &pProps);
    set_hint(pProps, "Main noise reduction controls.");
    g_propSuite->propSetString(pProps, kOfxPropLabel, 0, "Denoise Controls");

    // Engine Choice
    g_paramSuite->paramDefine(paramSet, kOfxParamTypeChoice, kParamDenoiseEngine, &pProps);
    set_hint(pProps, "Algorithm used to remove noise. 3D Collaborative: best quality. Wavelet: fastest. Non-Local Means: good on repetitive texture. Frequency-Domain 3D: good on banding and chroma blotches.");
    g_propSuite->propSetString(pProps, kOfxPropLabel, 0, "Denoise Engine");
    g_propSuite->propSetString(pProps, kOfxParamPropParent, 0, kGroupDenoise);
    g_propSuite->propSetString(pProps, kOfxParamPropChoiceOption, 0, "3D Collaborative Transform (Highest Fidelity)");
    g_propSuite->propSetString(pProps, kOfxParamPropChoiceOption, 1, "5-Band Wavelet Decomposition (Fastest / Real-time)");
    g_propSuite->propSetString(pProps, kOfxParamPropChoiceOption, 2, "Non-Local Means (Best for Fabrics & Pores)");
    g_propSuite->propSetString(pProps, kOfxParamPropChoiceOption, 3, "Frequency-Domain 3D (Best for Sensor Banding)");
    g_propSuite->propSetInt(pProps, kOfxParamPropDefault, 0, 0);

    // Hardware Acceleration Mode (Track B)
    g_paramSuite->paramDefine(paramSet, kOfxParamTypeChoice, kParamAccelerationMode, &pProps);
    set_hint(pProps, "Auto uses the GPU (CUDA) when available and falls back to CPU. Force CPU is slower but is the reference-quality path.");
    g_propSuite->propSetString(pProps, kOfxPropLabel, 0, "Hardware Acceleration");
    g_propSuite->propSetString(pProps, kOfxParamPropParent, 0, kGroupDenoise);
    g_propSuite->propSetString(pProps, kOfxParamPropChoiceOption, 0, "Auto (GPU if available, CPU fallback)");
    g_propSuite->propSetString(pProps, kOfxParamPropChoiceOption, 1, "Force GPU (CUDA sm_86+)");
    g_propSuite->propSetString(pProps, kOfxParamPropChoiceOption, 2, "Force CPU (Multi-core AVX2)");
    g_propSuite->propSetInt(pProps, kOfxParamPropDefault, 0, 0);

    // Quality Preset (Track A)
    g_paramSuite->paramDefine(paramSet, kOfxParamTypeChoice, kParamQualityPreset, &pProps);
    set_hint(pProps, "Speed / quality trade-off. Fast for scrubbing, Balanced for grading, High Quality for final renders (slowest, most thorough).");
    g_propSuite->propSetString(pProps, kOfxPropLabel, 0, "Quality Preset");
    g_propSuite->propSetString(pProps, kOfxParamPropParent, 0, kGroupDenoise);
    g_propSuite->propSetString(pProps, kOfxParamPropChoiceOption, 0, "Fast (Scrubbing / Realtime)");
    g_propSuite->propSetString(pProps, kOfxParamPropChoiceOption, 1, "Balanced (Production Grade)");
    g_propSuite->propSetString(pProps, kOfxParamPropChoiceOption, 2, "High Quality (Master Export)");
    g_propSuite->propSetInt(pProps, kOfxParamPropDefault, 0, 1);

    // Temporal Radius (Integer 0 to 4)
    g_paramSuite->paramDefine(paramSet, kOfxParamTypeInteger, kParamTemporalRadius, &pProps);
    set_hint(pProps, "Number of neighbouring frames on each side used to average noise over time. 0 = spatial only. Higher removes more noise on static areas but costs time and can ghost on fast motion.");
    g_propSuite->propSetString(pProps, kOfxPropLabel, 0, "Temporal Radius");
    g_propSuite->propSetString(pProps, kOfxParamPropParent, 0, kGroupDenoise);
    g_propSuite->propSetInt(pProps, kOfxParamPropDefault, 0, 2);
    g_propSuite->propSetInt(pProps, kOfxParamPropMin, 0, 0);
    g_propSuite->propSetInt(pProps, kOfxParamPropMax, 0, 4);

    // Temporal Strength
    g_paramSuite->paramDefine(paramSet, kOfxParamTypeDouble, kParamTemporalStrength, &pProps);
    set_hint(pProps, "How strongly frames are blended over time. 1.0 = standard; lower reduces ghosting/trails on moving subjects; higher cleans static areas harder.");
    g_propSuite->propSetString(pProps, kOfxPropLabel, 0, "Temporal Strength");
    g_propSuite->propSetString(pProps, kOfxParamPropParent, 0, kGroupDenoise);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDefault, 0, 1.0);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDisplayMin, 0, 0.0);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDisplayMax, 0, 2.0);

    // Spatial Strength
    g_paramSuite->paramDefine(paramSet, kOfxParamTypeDouble, kParamSpatialStrength, &pProps);
    set_hint(pProps, "Master amount of spatial (within-frame) noise reduction applied to Luma and Chroma together. 1.0 = remove exactly the profiled noise; raise if noise remains, lower to keep more texture.");
    g_propSuite->propSetString(pProps, kOfxPropLabel, 0, "Spatial Strength");
    g_propSuite->propSetString(pProps, kOfxParamPropParent, 0, kGroupDenoise);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDefault, 0, 1.0);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDisplayMin, 0, 0.0);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDisplayMax, 0, 2.0);

    // Luminance (brightness) noise reduction amount
    g_paramSuite->paramDefine(paramSet, kOfxParamTypeDouble, kParamLumaScale, &pProps);
    g_propSuite->propSetString(pProps, kOfxPropLabel, 0, "Luma (Y) Amount");
    g_propSuite->propSetString(pProps, kOfxParamPropParent, 0, kGroupDenoise);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDefault, 0, 1.0);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDisplayMin, 0, 0.0);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDisplayMax, 0, 3.0);
    set_hint(pProps, "Noise reduction amount for brightness (Luma / Y) noise. Raise to remove more grain/speckle; lower to keep fine detail and a film-like texture. Multiplies the Spatial Strength.");

    // Chrominance Scaling
    g_paramSuite->paramDefine(paramSet, kOfxParamTypeDouble, kParamChromaCr, &pProps);
    set_hint(pProps, "Noise reduction amount for red-vs-green colour noise (Cr). Raise to remove red/cyan colour speckle; colour noise can usually be pushed much harder than Luma.");
    g_propSuite->propSetString(pProps, kOfxPropLabel, 0, "Chroma Cr (Red-Green) Amount");
    g_propSuite->propSetString(pProps, kOfxParamPropParent, 0, kGroupDenoise);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDefault, 0, 1.0);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDisplayMin, 0, 0.0);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDisplayMax, 0, 3.0);

    g_paramSuite->paramDefine(paramSet, kOfxParamTypeDouble, kParamChromaCb, &pProps);
    set_hint(pProps, "Noise reduction amount for blue-vs-yellow colour noise (Cb). Raise to remove blue/yellow colour speckle and blotches.");
    g_propSuite->propSetString(pProps, kOfxPropLabel, 0, "Chroma Cb (Blue-Yellow) Amount");
    g_propSuite->propSetString(pProps, kOfxParamPropParent, 0, kGroupDenoise);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDefault, 0, 1.0);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDisplayMin, 0, 0.0);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDisplayMax, 0, 3.0);

    // Mix With Original
    g_paramSuite->paramDefine(paramSet, kOfxParamTypeDouble, kParamMixWithOriginal, &pProps);
    set_hint(pProps, "Blend some of the original (noisy) image back in. 0 = fully denoised, 1 = original. Useful to restore natural texture.");
    g_propSuite->propSetString(pProps, kOfxPropLabel, 0, "Mix with Original");
    g_propSuite->propSetString(pProps, kOfxParamPropParent, 0, kGroupDenoise);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDefault, 0, 0.0);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDisplayMin, 0, 0.0);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDisplayMax, 0, 1.0);

    // =========================================================================
    // Group 3: Frequency Equalizer
    // =========================================================================
    g_paramSuite->paramDefine(paramSet, kOfxParamTypeGroup, kGroupFrequency, &pProps);
    set_hint(pProps, "Choose how hard to filter noise at each spatial size. Very Low = large soft blotches; Very High = finest pixel-level grain. 1.0 on every slider removes exactly the measured noise.");
    g_propSuite->propSetString(pProps, kOfxPropLabel, 0, "Frequency Equalizer (noise size)");

    g_paramSuite->paramDefine(paramSet, kOfxParamTypeDouble, kParamBandVeryLow, &pProps);
    set_hint(pProps, "Noise at the coarsest scale: large, soft colour/brightness blotches (about 8+ pixels across). Raise to remove blotchy low-frequency noise (risk: flattens gradients). 1.0 = as measured, 0 = leave untouched.");
    g_propSuite->propSetString(pProps, kOfxPropLabel, 0, "Freq: Very Low (large blotches)");
    g_propSuite->propSetString(pProps, kOfxParamPropParent, 0, kGroupFrequency);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDefault, 0, 1.0);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDisplayMin, 0, 0.0);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDisplayMax, 0, 3.0);

    g_paramSuite->paramDefine(paramSet, kOfxParamTypeDouble, kParamBandLow, &pProps);
    set_hint(pProps, "Noise at a coarse scale (about 4-8 pixels): soft mottling and compression blotching. Raise to smooth mottling; lower to protect broad detail.");
    g_propSuite->propSetString(pProps, kOfxPropLabel, 0, "Freq: Low (mottling)");
    g_propSuite->propSetString(pProps, kOfxParamPropParent, 0, kGroupFrequency);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDefault, 0, 1.0);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDisplayMin, 0, 0.0);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDisplayMax, 0, 3.0);

    g_paramSuite->paramDefine(paramSet, kOfxParamTypeDouble, kParamBandMid, &pProps);
    set_hint(pProps, "Noise at a medium scale (about 2-4 pixels): the bulk of typical sensor / film grain. The most important band: raise to remove more grain.");
    g_propSuite->propSetString(pProps, kOfxPropLabel, 0, "Freq: Mid (main grain)");
    g_propSuite->propSetString(pProps, kOfxParamPropParent, 0, kGroupFrequency);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDefault, 0, 1.0);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDisplayMin, 0, 0.0);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDisplayMax, 0, 3.0);

    g_paramSuite->paramDefine(paramSet, kOfxParamTypeDouble, kParamBandHigh, &pProps);
    set_hint(pProps, "Noise at a fine scale (about 1.5-2 pixels): crisp grain that overlaps fine detail. Raise to clean it; lower to keep sharpness and texture.");
    g_propSuite->propSetString(pProps, kOfxPropLabel, 0, "Freq: High (fine grain)");
    g_propSuite->propSetString(pProps, kOfxParamPropParent, 0, kGroupFrequency);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDefault, 0, 1.0);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDisplayMin, 0, 0.0);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDisplayMax, 0, 3.0);

    g_paramSuite->paramDefine(paramSet, kOfxParamTypeDouble, kParamBandVeryHigh, &pProps);
    set_hint(pProps, "Noise at the finest scale (single pixels): pixel-level speckle. Raise to remove speckle; also affects the sharpest image detail.");
    g_propSuite->propSetString(pProps, kOfxPropLabel, 0, "Freq: Very High (pixel speckle)");
    g_propSuite->propSetString(pProps, kOfxParamPropParent, 0, kGroupFrequency);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDefault, 0, 1.0);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDisplayMin, 0, 0.0);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDisplayMax, 0, 3.0);

    // =========================================================================
    // Group 4: Tonal Equalizer
    // =========================================================================
    g_paramSuite->paramDefine(paramSet, kOfxParamTypeGroup, kGroupTonal, &pProps);
    set_hint(pProps, "Choose how hard to filter noise in different brightness ranges. Noise is usually strongest in shadows, so shadows default higher than highlights.");
    g_propSuite->propSetString(pProps, kOfxPropLabel, 0, "Tonal Equalizer (brightness)");

    g_paramSuite->paramDefine(paramSet, kOfxParamTypeDouble, kParamTonalShadows, &pProps);
    set_hint(pProps, "Noise reduction multiplier in the darkest tones (0-20% brightness). Raise to clean noisy shadows.");
    g_propSuite->propSetString(pProps, kOfxPropLabel, 0, "Tonal: Shadows (0-20%)");
    g_propSuite->propSetString(pProps, kOfxParamPropParent, 0, kGroupTonal);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDefault, 0, 1.30);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDisplayMin, 0, 0.0);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDisplayMax, 0, 3.0);

    g_paramSuite->paramDefine(paramSet, kOfxParamTypeDouble, kParamTonalDarkMid, &pProps);
    set_hint(pProps, "Noise reduction multiplier in dark midtones (20-40% brightness).");
    g_propSuite->propSetString(pProps, kOfxPropLabel, 0, "Tonal: Dark Midtones (20-40%)");
    g_propSuite->propSetString(pProps, kOfxParamPropParent, 0, kGroupTonal);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDefault, 0, 1.15);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDisplayMin, 0, 0.0);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDisplayMax, 0, 3.0);

    g_paramSuite->paramDefine(paramSet, kOfxParamTypeDouble, kParamTonalMidtones, &pProps);
    set_hint(pProps, "Noise reduction multiplier in midtones (40-60% brightness).");
    g_propSuite->propSetString(pProps, kOfxPropLabel, 0, "Tonal: Midtones (40-60%)");
    g_propSuite->propSetString(pProps, kOfxParamPropParent, 0, kGroupTonal);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDefault, 0, 1.00);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDisplayMin, 0, 0.0);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDisplayMax, 0, 3.0);

    g_paramSuite->paramDefine(paramSet, kOfxParamTypeDouble, kParamTonalLightMid, &pProps);
    set_hint(pProps, "Noise reduction multiplier in light midtones (60-80% brightness).");
    g_propSuite->propSetString(pProps, kOfxPropLabel, 0, "Tonal: Light Midtones (60-80%)");
    g_propSuite->propSetString(pProps, kOfxParamPropParent, 0, kGroupTonal);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDefault, 0, 0.90);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDisplayMin, 0, 0.0);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDisplayMax, 0, 3.0);

    g_paramSuite->paramDefine(paramSet, kOfxParamTypeDouble, kParamTonalHighlights, &pProps);
    set_hint(pProps, "Noise reduction multiplier in the brightest tones (80-100% brightness). Keep low to preserve highlight detail.");
    g_propSuite->propSetString(pProps, kOfxPropLabel, 0, "Tonal: Highlights (80-100%)");
    g_propSuite->propSetString(pProps, kOfxParamPropParent, 0, kGroupTonal);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDefault, 0, 0.80);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDisplayMin, 0, 0.0);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDisplayMax, 0, 3.0);

    // =========================================================================
    // Group 5: Detail & Grain Recovery
    // =========================================================================
    g_paramSuite->paramDefine(paramSet, kOfxParamTypeGroup, kGroupDetail, &pProps);
    set_hint(pProps, "Put back fine detail and natural grain after denoising.");
    g_propSuite->propSetString(pProps, kOfxPropLabel, 0, "Detail & Grain Recovery");

    g_paramSuite->paramDefine(paramSet, kOfxParamTypeDouble, kParamDetailRecovery, &pProps);
    set_hint(pProps, "Recovers fine detail the filter removed along with the noise (edges, texture) without bringing the noise back. 0 = off.");
    g_propSuite->propSetString(pProps, kOfxPropLabel, 0, "Detail Recovery");
    g_propSuite->propSetString(pProps, kOfxParamPropParent, 0, kGroupDetail);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDefault, 0, 0.15);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDisplayMin, 0, 0.0);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDisplayMax, 0, 1.0);

    g_paramSuite->paramDefine(paramSet, kOfxParamTypeDouble, kParamGrainAmount, &pProps);
    set_hint(pProps, "Add fresh, uniform film-like grain after denoising so heavy cleaning does not look plastic. 0 = none.");
    g_propSuite->propSetString(pProps, kOfxPropLabel, 0, "Grain Amount");
    g_propSuite->propSetString(pProps, kOfxParamPropParent, 0, kGroupDetail);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDefault, 0, 0.0);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDisplayMin, 0, 0.0);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDisplayMax, 0, 0.5);

    g_paramSuite->paramDefine(paramSet, kOfxParamTypeChoice, kParamPreviewMode, &pProps);
    set_hint(pProps, "Denoised Output shows the result. Residual shows only what was removed: it should look like pure noise with no picture structure; visible edges mean detail is being damaged.");
    g_propSuite->propSetString(pProps, kOfxPropLabel, 0, "Preview Mode");
    g_propSuite->propSetString(pProps, kOfxParamPropParent, 0, kGroupDetail);
    g_propSuite->propSetString(pProps, kOfxParamPropChoiceOption, 0, "Denoised Output");
    g_propSuite->propSetString(pProps, kOfxParamPropChoiceOption, 1, "Residual (Noise Only)");
    g_propSuite->propSetInt(pProps, kOfxParamPropDefault, 0, 0);

    // =========================================================================
    // Group 6: Performance & Optimizations
    // =========================================================================
    g_paramSuite->paramDefine(paramSet, kOfxParamTypeGroup, kGroupPerformance, &pProps);
    set_hint(pProps, "Speed optimisations. Leave on unless troubleshooting.");
    g_propSuite->propSetString(pProps, kOfxPropLabel, 0, "Performance & Optimizations");

    g_paramSuite->paramDefine(paramSet, kOfxParamTypeBoolean, kParamFastChroma, &pProps);
    set_hint(pProps, "Filter colour channels at reduced density. Faster with little visible loss since colour noise is smooth.");
    g_propSuite->propSetString(pProps, kOfxPropLabel, 0, "Fast Chroma Subsampling");
    g_propSuite->propSetString(pProps, kOfxParamPropParent, 0, kGroupPerformance);
    g_propSuite->propSetInt(pProps, kOfxParamPropDefault, 0, 1);

    g_paramSuite->paramDefine(paramSet, kOfxParamTypeBoolean, kParamFastDct, &pProps);
    set_hint(pProps, "Use the optimised butterfly transform on the CPU. Same result, faster.");
    g_propSuite->propSetString(pProps, kOfxPropLabel, 0, "Fast Butterfly DCT");
    g_propSuite->propSetString(pProps, kOfxParamPropParent, 0, kGroupPerformance);
    g_propSuite->propSetInt(pProps, kOfxParamPropDefault, 0, 1);

    g_paramSuite->paramDefine(paramSet, kOfxParamTypeBoolean, kParamEarlyExitMotion, &pProps);
    set_hint(pProps, "Stop motion searching early when a match is already good enough. Faster; disable only if motion tracking looks wrong.");
    g_propSuite->propSetString(pProps, kOfxPropLabel, 0, "Early-Exit Motion Check");
    g_propSuite->propSetString(pProps, kOfxParamPropParent, 0, kGroupPerformance);
    g_propSuite->propSetInt(pProps, kOfxParamPropDefault, 0, 1);

    // =========================================================================
    // Secret Statistical State Parameters (Stored with Host Project)
    // =========================================================================
    auto define_secret_double = [&](const char* name, double default_val) {
        OfxPropertySetHandle sp = nullptr;
        g_paramSuite->paramDefine(paramSet, kOfxParamTypeDouble, name, &sp);
        g_propSuite->propSetInt(sp, kOfxParamPropSecret, 0, 1);
        g_propSuite->propSetDouble(sp, kOfxParamPropDefault, 0, default_val);
    };

    define_secret_double(kParamCalibrated, 0.0);
    define_secret_double(kParamNlfSlopeY, 0.00035);
    define_secret_double(kParamNlfInterceptY, 0.00003);
    define_secret_double(kParamNlfSlopeCb, 0.00030);
    define_secret_double(kParamNlfInterceptCb, 0.000025);
    define_secret_double(kParamNlfSlopeCr, 0.00030);
    define_secret_double(kParamNlfInterceptCr, 0.000025);

    return kOfxStatOK;
}

// Action: GetFramesNeeded
OfxStatus action_get_frames_needed(OfxImageEffectHandle effect, OfxPropertySetHandle inArgs, OfxPropertySetHandle outArgs) {
    double time = 0.0;
    if (inArgs) {
        g_propSuite->propGetDouble(inArgs, kOfxPropTime, 0, &time);
    }

    OfxParamSetHandle paramSet = nullptr;
    g_effectSuite->getParamSet(effect, &paramSet);

    int radius = 2;
    if (paramSet) {
        OfxParamHandle radParam = nullptr;
        g_paramSuite->paramGetHandle(paramSet, kParamTemporalRadius, &radParam, nullptr);
        if (radParam) {
            g_paramSuite->paramGetValueAtTime(radParam, time, &radius);
        }
    }

    double range[2];
    range[0] = time - radius;
    range[1] = time + radius;

    std::string propName = std::string("OfxImageClipPropFrameRange_") + kOfxImageEffectSimpleSourceClipName;
    g_propSuite->propSetDoubleN(outArgs, propName.c_str(), 2, range);

    return kOfxStatOK;
}

// Action: Instance Changed (Button Clicks and Interactive Actions)
OfxStatus action_instance_changed(OfxImageEffectHandle effect, OfxPropertySetHandle inArgs, OfxPropertySetHandle outArgs) {
    (void)outArgs;
    if (!inArgs || !g_paramSuite || !g_propSuite || !g_effectSuite) return kOfxStatReplyDefault;

    char* propType = nullptr;
    g_propSuite->propGetString(inArgs, kOfxPropType, 0, &propType);
    if (!propType || std::strcmp(propType, kOfxTypeParameter) != 0) {
        return kOfxStatReplyDefault;
    }

    char* paramName = nullptr;
    g_propSuite->propGetString(inArgs, kOfxPropName, 0, &paramName);
    if (!paramName) return kOfxStatReplyDefault;

    double changeTime = 0.0;
    g_propSuite->propGetDouble(inArgs, kOfxPropTime, 0, &changeTime);

    OfxParamSetHandle paramSet = nullptr;
    g_effectSuite->getParamSet(effect, &paramSet);
    if (!paramSet) return kOfxStatReplyDefault;

    // Helper lambda to fetch the current source frame along with its pixel origin
    struct FetchedFrame {
        librenoise::VideoFrame32F frame;
        int origin_x = 0;
        int origin_y = 0;
    };
    auto fetch_current_frame = [&]() -> std::optional<FetchedFrame> {
        OfxImageClipHandle srcClip = nullptr;
        if (g_effectSuite->clipGetHandle(effect, kOfxImageEffectSimpleSourceClipName, &srcClip, nullptr) != kOfxStatOK || !srcClip) {
            return std::nullopt;
        }
        OfxPropertySetHandle srcImg = nullptr;
        if (g_effectSuite->clipGetImage(srcClip, changeTime, nullptr, &srcImg) != kOfxStatOK || !srcImg) {
            return std::nullopt;
        }

        OfxImgInfo info = query_ofx_image(srcImg);
        if (!info.valid()) {
            g_effectSuite->clipReleaseImage(srcImg);
            return std::nullopt;
        }

        int width = info.width();
        int height = info.height();
        const int cpuThreads = compute_polite_cpu_threads();
        FetchedFrame result{librenoise::VideoFrame32F(width, height), info.bounds[0], info.bounds[1]};
        #pragma omp parallel for schedule(static) num_threads(cpuThreads)
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                float rgba[4];
                sample_ofx_pixel_rgba(info, info.bounds[0] + x, info.bounds[1] + y, rgba);
                result.frame.channels[0].at(x, y) = rgba[0];
                result.frame.channels[1].at(x, y) = rgba[1];
                result.frame.channels[2].at(x, y) = rgba[2];
            }
        }
        g_effectSuite->clipReleaseImage(srcImg);
        return result;
    };

    // Auto Profile or Auto Detect Area
    if (std::strcmp(paramName, kParamAutoProfile) == 0 || std::strcmp(paramName, kParamAutoDetectArea) == 0) {
        auto optFrame = fetch_current_frame();
        if (optFrame.has_value()) {
            const auto& frame = optFrame->frame;
            auto roi = librenoise::NoiseRoiAnalyzer::find_optimal_roi(frame);

            OfxParamHandle boxP1 = nullptr, boxP2 = nullptr;
            double bx1 = static_cast<double>(optFrame->origin_x + roi.x);
            double by1 = static_cast<double>(optFrame->origin_y + roi.y);
            double bx2 = static_cast<double>(optFrame->origin_x + roi.x + roi.width);
            double by2 = static_cast<double>(optFrame->origin_y + roi.y + roi.height);

            if (g_paramSuite->paramGetHandle(paramSet, kParamAnalysisBoxCorner1, &boxP1, nullptr) == kOfxStatOK && boxP1) {
                g_paramSuite->paramSetValue(boxP1, bx1, by1);
                g_paramSuite->paramSetValueAtTime(boxP1, changeTime, bx1, by1);
            }
            if (g_paramSuite->paramGetHandle(paramSet, kParamAnalysisBoxCorner2, &boxP2, nullptr) == kOfxStatOK && boxP2) {
                g_paramSuite->paramSetValue(boxP2, bx2, by2);
                g_paramSuite->paramSetValueAtTime(boxP2, changeTime, bx2, by2);
            }

            if (std::strcmp(paramName, kParamAutoProfile) == 0) {
                auto result = librenoise::NoiseRoiAnalyzer::analyze_roi(frame, roi);

                OfxParamHandle qP = nullptr, aP = nullptr;
                if (g_paramSuite->paramGetHandle(paramSet, kParamProfileQuality, &qP, nullptr) == kOfxStatOK) {
                    std::string qStr = std::to_string(static_cast<int>(std::round(result.uniformity_score))) + "%";
                    g_paramSuite->paramSetValue(qP, qStr.c_str());
                }
                if (g_paramSuite->paramGetHandle(paramSet, kParamProfileAssessment, &aP, nullptr) == kOfxStatOK) {
                    g_paramSuite->paramSetValue(aP, result.assessment.c_str());
                }

                // Update frequency bands
                OfxParamHandle bVL = nullptr, bL = nullptr, bM = nullptr, bH = nullptr, bVH = nullptr;
                if (g_paramSuite->paramGetHandle(paramSet, kParamBandVeryLow, &bVL, nullptr) == kOfxStatOK)
                    g_paramSuite->paramSetValue(bVL, static_cast<double>(result.measured_psd[0]));
                if (g_paramSuite->paramGetHandle(paramSet, kParamBandLow, &bL, nullptr) == kOfxStatOK)
                    g_paramSuite->paramSetValue(bL, static_cast<double>(result.measured_psd[1]));
                if (g_paramSuite->paramGetHandle(paramSet, kParamBandMid, &bM, nullptr) == kOfxStatOK)
                    g_paramSuite->paramSetValue(bM, static_cast<double>(result.measured_psd[2]));
                if (g_paramSuite->paramGetHandle(paramSet, kParamBandHigh, &bH, nullptr) == kOfxStatOK)
                    g_paramSuite->paramSetValue(bH, static_cast<double>(result.measured_psd[3]));
                if (g_paramSuite->paramGetHandle(paramSet, kParamBandVeryHigh, &bVH, nullptr) == kOfxStatOK)
                    g_paramSuite->paramSetValue(bVH, static_cast<double>(result.measured_psd[4]));

                auto set_secret_val = [&](const char* sname, double sval) {
                    OfxParamHandle sh = nullptr;
                    if (g_paramSuite->paramGetHandle(paramSet, sname, &sh, nullptr) == kOfxStatOK && sh) {
                        g_paramSuite->paramSetValue(sh, sval);
                    }
                };
                set_secret_val(kParamCalibrated, 1.0);
                set_secret_val(kParamNlfSlopeY, static_cast<double>(result.nlf_y.a));
                set_secret_val(kParamNlfInterceptY, static_cast<double>(result.nlf_y.b));
                set_secret_val(kParamNlfSlopeCb, static_cast<double>(result.nlf_cb.a));
                set_secret_val(kParamNlfInterceptCb, static_cast<double>(result.nlf_cb.b));
                set_secret_val(kParamNlfSlopeCr, static_cast<double>(result.nlf_cr.a));
                set_secret_val(kParamNlfInterceptCr, static_cast<double>(result.nlf_cr.b));
            }
        }
        return kOfxStatOK;
    }

    // Build Profile from Selection
    if (std::strcmp(paramName, kParamBuildProfile) == 0) {
        auto optFrame = fetch_current_frame();
        if (optFrame.has_value()) {
            const auto& frame = optFrame->frame;
            double coords[4] = {0, 0, 128, 128};
            OfxParamHandle boxP1 = nullptr, boxP2 = nullptr;
            if (g_paramSuite->paramGetHandle(paramSet, kParamAnalysisBoxCorner1, &boxP1, nullptr) == kOfxStatOK && boxP1) {
                g_paramSuite->paramGetValue(boxP1, &coords[0], &coords[1]);
            }
            if (g_paramSuite->paramGetHandle(paramSet, kParamAnalysisBoxCorner2, &boxP2, nullptr) == kOfxStatOK && boxP2) {
                g_paramSuite->paramGetValue(boxP2, &coords[2], &coords[3]);
            }
            int32_t x1 = static_cast<int32_t>(std::round(std::min(coords[0], coords[2]))) - optFrame->origin_x;
            int32_t y1 = static_cast<int32_t>(std::round(std::min(coords[1], coords[3]))) - optFrame->origin_y;
            int32_t x2 = static_cast<int32_t>(std::round(std::max(coords[0], coords[2]))) - optFrame->origin_x;
            int32_t y2 = static_cast<int32_t>(std::round(std::max(coords[1], coords[3]))) - optFrame->origin_y;
            x1 = std::clamp(x1, 0, std::max(0, frame.width - 16));
            y1 = std::clamp(y1, 0, std::max(0, frame.height - 16));
            int32_t rw = std::clamp(x2 - x1, 16, frame.width - x1);
            int32_t rh = std::clamp(y2 - y1, 16, frame.height - y1);
            librenoise::AnalysisRoi roi{x1, y1, rw, rh};

            auto result = librenoise::NoiseRoiAnalyzer::analyze_roi(frame, roi);

            OfxParamHandle qP = nullptr, aP = nullptr;
            if (g_paramSuite->paramGetHandle(paramSet, kParamProfileQuality, &qP, nullptr) == kOfxStatOK) {
                std::string qStr = std::to_string(static_cast<int>(std::round(result.uniformity_score))) + "%";
                g_paramSuite->paramSetValue(qP, qStr.c_str());
            }
            if (g_paramSuite->paramGetHandle(paramSet, kParamProfileAssessment, &aP, nullptr) == kOfxStatOK) {
                g_paramSuite->paramSetValue(aP, result.assessment.c_str());
            }

            // Update frequency bands
            OfxParamHandle bVL = nullptr, bL = nullptr, bM = nullptr, bH = nullptr, bVH = nullptr;
            if (g_paramSuite->paramGetHandle(paramSet, kParamBandVeryLow, &bVL, nullptr) == kOfxStatOK)
                g_paramSuite->paramSetValue(bVL, static_cast<double>(result.measured_psd[0]));
            if (g_paramSuite->paramGetHandle(paramSet, kParamBandLow, &bL, nullptr) == kOfxStatOK)
                g_paramSuite->paramSetValue(bL, static_cast<double>(result.measured_psd[1]));
            if (g_paramSuite->paramGetHandle(paramSet, kParamBandMid, &bM, nullptr) == kOfxStatOK)
                g_paramSuite->paramSetValue(bM, static_cast<double>(result.measured_psd[2]));
            if (g_paramSuite->paramGetHandle(paramSet, kParamBandHigh, &bH, nullptr) == kOfxStatOK)
                g_paramSuite->paramSetValue(bH, static_cast<double>(result.measured_psd[3]));
            if (g_paramSuite->paramGetHandle(paramSet, kParamBandVeryHigh, &bVH, nullptr) == kOfxStatOK)
                g_paramSuite->paramSetValue(bVH, static_cast<double>(result.measured_psd[4]));

            auto set_secret_val = [&](const char* sname, double sval) {
                OfxParamHandle sh = nullptr;
                if (g_paramSuite->paramGetHandle(paramSet, sname, &sh, nullptr) == kOfxStatOK && sh) {
                    g_paramSuite->paramSetValue(sh, sval);
                }
            };
            set_secret_val(kParamCalibrated, 1.0);
            set_secret_val(kParamNlfSlopeY, static_cast<double>(result.nlf_y.a));
            set_secret_val(kParamNlfInterceptY, static_cast<double>(result.nlf_y.b));
            set_secret_val(kParamNlfSlopeCb, static_cast<double>(result.nlf_cb.a));
            set_secret_val(kParamNlfInterceptCb, static_cast<double>(result.nlf_cb.b));
            set_secret_val(kParamNlfSlopeCr, static_cast<double>(result.nlf_cr.a));
            set_secret_val(kParamNlfInterceptCr, static_cast<double>(result.nlf_cr.b));
        }
        return kOfxStatOK;
    }

    // Load Profile
    if (std::strcmp(paramName, kParamLoadProfile) == 0) {
        OfxParamHandle pPath = nullptr;
        char* pathStr = nullptr;
        if (g_paramSuite->paramGetHandle(paramSet, kParamProfilePath, &pPath, nullptr) == kOfxStatOK) {
            g_paramSuite->paramGetValue(pPath, &pathStr);
        }
        if (pathStr && strlen(pathStr) > 0) {
            librenoise::LibreNoiseDeviceProfile prof;
            if (librenoise::ProfileSerializer::load_from_file(pathStr, prof)) {
                OfxParamHandle p = nullptr;
                if (g_paramSuite->paramGetHandle(paramSet, kParamProfileDevice, &p, nullptr) == kOfxStatOK) g_paramSuite->paramSetValue(p, prof.device_name.c_str());
                if (g_paramSuite->paramGetHandle(paramSet, kParamProfileMode, &p, nullptr) == kOfxStatOK) g_paramSuite->paramSetValue(p, prof.mode_description.c_str());
                if (g_paramSuite->paramGetHandle(paramSet, kParamProfileQuality, &p, nullptr) == kOfxStatOK) g_paramSuite->paramSetValue(p, (std::to_string(static_cast<int>(prof.uniformity_score)) + "%").c_str());
                if (g_paramSuite->paramGetHandle(paramSet, kParamProfileAssessment, &p, nullptr) == kOfxStatOK) g_paramSuite->paramSetValue(p, prof.quality_assessment.c_str());
                OfxParamHandle boxP1 = nullptr, boxP2 = nullptr;
                if (g_paramSuite->paramGetHandle(paramSet, kParamAnalysisBoxCorner1, &boxP1, nullptr) == kOfxStatOK && boxP1) {
                    g_paramSuite->paramSetValue(boxP1, static_cast<double>(prof.analysis_roi.x), static_cast<double>(prof.analysis_roi.y));
                }
                if (g_paramSuite->paramGetHandle(paramSet, kParamAnalysisBoxCorner2, &boxP2, nullptr) == kOfxStatOK && boxP2) {
                    g_paramSuite->paramSetValue(boxP2, static_cast<double>(prof.analysis_roi.x + prof.analysis_roi.width), static_cast<double>(prof.analysis_roi.y + prof.analysis_roi.height));
                }
                if (g_paramSuite->paramGetHandle(paramSet, kParamBandVeryLow, &p, nullptr) == kOfxStatOK) g_paramSuite->paramSetValue(p, static_cast<double>(prof.psd_5band[0]));
                if (g_paramSuite->paramGetHandle(paramSet, kParamBandLow, &p, nullptr) == kOfxStatOK) g_paramSuite->paramSetValue(p, static_cast<double>(prof.psd_5band[1]));
                if (g_paramSuite->paramGetHandle(paramSet, kParamBandMid, &p, nullptr) == kOfxStatOK) g_paramSuite->paramSetValue(p, static_cast<double>(prof.psd_5band[2]));
                if (g_paramSuite->paramGetHandle(paramSet, kParamBandHigh, &p, nullptr) == kOfxStatOK) g_paramSuite->paramSetValue(p, static_cast<double>(prof.psd_5band[3]));
                if (g_paramSuite->paramGetHandle(paramSet, kParamBandVeryHigh, &p, nullptr) == kOfxStatOK) g_paramSuite->paramSetValue(p, static_cast<double>(prof.psd_5band[4]));

                const auto& pts = prof.tonal_curve.points();
                if (pts.size() >= 5) {
                    if (g_paramSuite->paramGetHandle(paramSet, kParamTonalShadows, &p, nullptr) == kOfxStatOK) g_paramSuite->paramSetValue(p, static_cast<double>(pts[0].y));
                    if (g_paramSuite->paramGetHandle(paramSet, kParamTonalDarkMid, &p, nullptr) == kOfxStatOK) g_paramSuite->paramSetValue(p, static_cast<double>(pts[1].y));
                    if (g_paramSuite->paramGetHandle(paramSet, kParamTonalMidtones, &p, nullptr) == kOfxStatOK) g_paramSuite->paramSetValue(p, static_cast<double>(pts[2].y));
                    if (g_paramSuite->paramGetHandle(paramSet, kParamTonalLightMid, &p, nullptr) == kOfxStatOK) g_paramSuite->paramSetValue(p, static_cast<double>(pts[3].y));
                    if (g_paramSuite->paramGetHandle(paramSet, kParamTonalHighlights, &p, nullptr) == kOfxStatOK) g_paramSuite->paramSetValue(p, static_cast<double>(pts[4].y));
                }

                auto set_secret_val = [&](const char* sname, double sval) {
                    OfxParamHandle sh = nullptr;
                    if (g_paramSuite->paramGetHandle(paramSet, sname, &sh, nullptr) == kOfxStatOK && sh) {
                        g_paramSuite->paramSetValue(sh, sval);
                    }
                };
                set_secret_val(kParamCalibrated, 1.0);
                set_secret_val(kParamNlfSlopeY, static_cast<double>(prof.nlf_y.a));
                set_secret_val(kParamNlfInterceptY, static_cast<double>(prof.nlf_y.b));
                set_secret_val(kParamNlfSlopeCb, static_cast<double>(prof.nlf_cb.a));
                set_secret_val(kParamNlfInterceptCb, static_cast<double>(prof.nlf_cb.b));
                set_secret_val(kParamNlfSlopeCr, static_cast<double>(prof.nlf_cr.a));
                set_secret_val(kParamNlfInterceptCr, static_cast<double>(prof.nlf_cr.b));
            }
        }
        return kOfxStatOK;
    }

    // Save Profile
    if (std::strcmp(paramName, kParamSaveProfile) == 0) {
        OfxParamHandle pPath = nullptr;
        char* pathStr = nullptr;
        if (g_paramSuite->paramGetHandle(paramSet, kParamProfilePath, &pPath, nullptr) == kOfxStatOK) {
            g_paramSuite->paramGetValue(pPath, &pathStr);
        }
        if (pathStr && strlen(pathStr) > 0) {
            librenoise::LibreNoiseDeviceProfile prof;
            OfxParamHandle p = nullptr;
            char* devStr = nullptr;
            char* modeStr = nullptr;
            if (g_paramSuite->paramGetHandle(paramSet, kParamProfileDevice, &p, nullptr) == kOfxStatOK && g_paramSuite->paramGetValue(p, &devStr) == kOfxStatOK && devStr) prof.device_name = devStr;
            if (g_paramSuite->paramGetHandle(paramSet, kParamProfileMode, &p, nullptr) == kOfxStatOK && g_paramSuite->paramGetValue(p, &modeStr) == kOfxStatOK && modeStr) prof.mode_description = modeStr;

            double coords[4] = {0, 0, 128, 128};
            OfxParamHandle boxP1 = nullptr, boxP2 = nullptr;
            if (g_paramSuite->paramGetHandle(paramSet, kParamAnalysisBoxCorner1, &boxP1, nullptr) == kOfxStatOK && boxP1) {
                g_paramSuite->paramGetValue(boxP1, &coords[0], &coords[1]);
            }
            if (g_paramSuite->paramGetHandle(paramSet, kParamAnalysisBoxCorner2, &boxP2, nullptr) == kOfxStatOK && boxP2) {
                g_paramSuite->paramGetValue(boxP2, &coords[2], &coords[3]);
            }
            int32_t x1 = static_cast<int32_t>(std::round(std::min(coords[0], coords[2])));
            int32_t y1 = static_cast<int32_t>(std::round(std::min(coords[1], coords[3])));
            int32_t x2 = static_cast<int32_t>(std::round(std::max(coords[0], coords[2])));
            int32_t y2 = static_cast<int32_t>(std::round(std::max(coords[1], coords[3])));
            prof.analysis_roi = {x1, y1, std::max(16, x2 - x1), std::max(16, y2 - y1)};

            double b0 = 1.0, b1 = 1.0, b2 = 1.0, b3 = 1.0, b4 = 1.0;
            if (g_paramSuite->paramGetHandle(paramSet, kParamBandVeryLow, &p, nullptr) == kOfxStatOK) g_paramSuite->paramGetValue(p, &b0);
            if (g_paramSuite->paramGetHandle(paramSet, kParamBandLow, &p, nullptr) == kOfxStatOK) g_paramSuite->paramGetValue(p, &b1);
            if (g_paramSuite->paramGetHandle(paramSet, kParamBandMid, &p, nullptr) == kOfxStatOK) g_paramSuite->paramGetValue(p, &b2);
            if (g_paramSuite->paramGetHandle(paramSet, kParamBandHigh, &p, nullptr) == kOfxStatOK) g_paramSuite->paramGetValue(p, &b3);
            if (g_paramSuite->paramGetHandle(paramSet, kParamBandVeryHigh, &p, nullptr) == kOfxStatOK) g_paramSuite->paramGetValue(p, &b4);
            prof.psd_5band = {static_cast<float>(b0), static_cast<float>(b1), static_cast<float>(b2), static_cast<float>(b3), static_cast<float>(b4)};

            double t0 = 1.3, t1 = 1.15, t2 = 1.0, t3 = 0.9, t4 = 0.8;
            if (g_paramSuite->paramGetHandle(paramSet, kParamTonalShadows, &p, nullptr) == kOfxStatOK) g_paramSuite->paramGetValue(p, &t0);
            if (g_paramSuite->paramGetHandle(paramSet, kParamTonalDarkMid, &p, nullptr) == kOfxStatOK) g_paramSuite->paramGetValue(p, &t1);
            if (g_paramSuite->paramGetHandle(paramSet, kParamTonalMidtones, &p, nullptr) == kOfxStatOK) g_paramSuite->paramGetValue(p, &t2);
            if (g_paramSuite->paramGetHandle(paramSet, kParamTonalLightMid, &p, nullptr) == kOfxStatOK) g_paramSuite->paramGetValue(p, &t3);
            if (g_paramSuite->paramGetHandle(paramSet, kParamTonalHighlights, &p, nullptr) == kOfxStatOK) g_paramSuite->paramGetValue(p, &t4);
            prof.tonal_curve = librenoise::NoiseProfileCurve({
                {0.00f, static_cast<float>(t0)},
                {0.25f, static_cast<float>(t1)},
                {0.50f, static_cast<float>(t2)},
                {0.75f, static_cast<float>(t3)},
                {1.00f, static_cast<float>(t4)}
            });

            double sY = 0.00035, iY = 0.00003, sCb = 0.00030, iCb = 0.000025, sCr = 0.00030, iCr = 0.000025;
            if (g_paramSuite->paramGetHandle(paramSet, kParamNlfSlopeY, &p, nullptr) == kOfxStatOK && p) g_paramSuite->paramGetValue(p, &sY);
            if (g_paramSuite->paramGetHandle(paramSet, kParamNlfInterceptY, &p, nullptr) == kOfxStatOK && p) g_paramSuite->paramGetValue(p, &iY);
            if (g_paramSuite->paramGetHandle(paramSet, kParamNlfSlopeCb, &p, nullptr) == kOfxStatOK && p) g_paramSuite->paramGetValue(p, &sCb);
            if (g_paramSuite->paramGetHandle(paramSet, kParamNlfInterceptCb, &p, nullptr) == kOfxStatOK && p) g_paramSuite->paramGetValue(p, &iCb);
            if (g_paramSuite->paramGetHandle(paramSet, kParamNlfSlopeCr, &p, nullptr) == kOfxStatOK && p) g_paramSuite->paramGetValue(p, &sCr);
            if (g_paramSuite->paramGetHandle(paramSet, kParamNlfInterceptCr, &p, nullptr) == kOfxStatOK && p) g_paramSuite->paramGetValue(p, &iCr);
            prof.nlf_y = {static_cast<float>(sY), static_cast<float>(iY)};
            prof.nlf_cb = {static_cast<float>(sCb), static_cast<float>(iCb)};
            prof.nlf_cr = {static_cast<float>(sCr), static_cast<float>(iCr)};

            librenoise::ProfileSerializer::save_to_file(pathStr, prof);
        }
        return kOfxStatOK;
    }

    return kOfxStatReplyDefault;
}

// Action: Render
OfxStatus action_render(OfxImageEffectHandle effect, OfxPropertySetHandle inArgs) {
    double renderTime = 0.0;
    if (inArgs) {
        g_propSuite->propGetDouble(inArgs, kOfxPropTime, 0, &renderTime);
    }

    // Read Parameters
    OfxParamSetHandle paramSet = nullptr;
    g_effectSuite->getParamSet(effect, &paramSet);

    int engineChoice = 0;
    int accelMode = 0; // Auto
    int qualityChoice = 1; // Balanced
    int fastChroma = 1;
    int fastDct = 1;
    int earlyExitMotion = 1;
    int radius = 2;
    double tempStrength = 1.0;
    double spatStrength = 1.0;
    double lumaScale = 1.0;
    double chromaCr = 1.0;
    double chromaCb = 1.0;
    double mixWithOriginal = 0.0;
    double detailRec = 0.15;
    double grainAmt = 0.0;
    int previewMode = 0;

    int cudaEnabled = 0;
    if (inArgs) {
        g_propSuite->propGetInt(inArgs, kOfxImageEffectPropCudaEnabled, 0, &cudaEnabled);
    }

    double bandVL = 1.0, bandL = 1.0, bandM = 1.0, bandH = 1.0, bandVH = 1.0;
    double tonalSh = 1.3, tonalDM = 1.15, tonalMid = 1.0, tonalLM = 0.9, tonalHi = 0.8;
    double calibrated = 0.0;
    double nlfSlopeY = 0.00035, nlfInterceptY = 0.00003;
    double nlfSlopeCb = 0.00030, nlfInterceptCb = 0.000025;
    double nlfSlopeCr = 0.00030, nlfInterceptCr = 0.000025;

    if (paramSet) {
        OfxParamHandle p = nullptr;
        if (g_paramSuite->paramGetHandle(paramSet, kParamDenoiseEngine, &p, nullptr) == kOfxStatOK) g_paramSuite->paramGetValueAtTime(p, renderTime, &engineChoice);
        if (g_paramSuite->paramGetHandle(paramSet, kParamAccelerationMode, &p, nullptr) == kOfxStatOK) g_paramSuite->paramGetValueAtTime(p, renderTime, &accelMode);
        if (g_paramSuite->paramGetHandle(paramSet, kParamQualityPreset, &p, nullptr) == kOfxStatOK) g_paramSuite->paramGetValueAtTime(p, renderTime, &qualityChoice);
        if (g_paramSuite->paramGetHandle(paramSet, kParamFastChroma, &p, nullptr) == kOfxStatOK) g_paramSuite->paramGetValueAtTime(p, renderTime, &fastChroma);
        if (g_paramSuite->paramGetHandle(paramSet, kParamFastDct, &p, nullptr) == kOfxStatOK) g_paramSuite->paramGetValueAtTime(p, renderTime, &fastDct);
        if (g_paramSuite->paramGetHandle(paramSet, kParamEarlyExitMotion, &p, nullptr) == kOfxStatOK) g_paramSuite->paramGetValueAtTime(p, renderTime, &earlyExitMotion);
        if (g_paramSuite->paramGetHandle(paramSet, kParamTemporalRadius, &p, nullptr) == kOfxStatOK) g_paramSuite->paramGetValueAtTime(p, renderTime, &radius);
        if (g_paramSuite->paramGetHandle(paramSet, kParamTemporalStrength, &p, nullptr) == kOfxStatOK) g_paramSuite->paramGetValueAtTime(p, renderTime, &tempStrength);
        if (g_paramSuite->paramGetHandle(paramSet, kParamSpatialStrength, &p, nullptr) == kOfxStatOK) g_paramSuite->paramGetValueAtTime(p, renderTime, &spatStrength);
        if (g_paramSuite->paramGetHandle(paramSet, kParamLumaScale, &p, nullptr) == kOfxStatOK) g_paramSuite->paramGetValueAtTime(p, renderTime, &lumaScale);
        if (g_paramSuite->paramGetHandle(paramSet, kParamChromaCr, &p, nullptr) == kOfxStatOK) g_paramSuite->paramGetValueAtTime(p, renderTime, &chromaCr);
        if (g_paramSuite->paramGetHandle(paramSet, kParamChromaCb, &p, nullptr) == kOfxStatOK) g_paramSuite->paramGetValueAtTime(p, renderTime, &chromaCb);
        if (g_paramSuite->paramGetHandle(paramSet, kParamMixWithOriginal, &p, nullptr) == kOfxStatOK) g_paramSuite->paramGetValueAtTime(p, renderTime, &mixWithOriginal);
        if (g_paramSuite->paramGetHandle(paramSet, kParamBandVeryLow, &p, nullptr) == kOfxStatOK) g_paramSuite->paramGetValueAtTime(p, renderTime, &bandVL);
        if (g_paramSuite->paramGetHandle(paramSet, kParamBandLow, &p, nullptr) == kOfxStatOK) g_paramSuite->paramGetValueAtTime(p, renderTime, &bandL);
        if (g_paramSuite->paramGetHandle(paramSet, kParamBandMid, &p, nullptr) == kOfxStatOK) g_paramSuite->paramGetValueAtTime(p, renderTime, &bandM);
        if (g_paramSuite->paramGetHandle(paramSet, kParamBandHigh, &p, nullptr) == kOfxStatOK) g_paramSuite->paramGetValueAtTime(p, renderTime, &bandH);
        if (g_paramSuite->paramGetHandle(paramSet, kParamBandVeryHigh, &p, nullptr) == kOfxStatOK) g_paramSuite->paramGetValueAtTime(p, renderTime, &bandVH);
        if (g_paramSuite->paramGetHandle(paramSet, kParamTonalShadows, &p, nullptr) == kOfxStatOK) g_paramSuite->paramGetValueAtTime(p, renderTime, &tonalSh);
        if (g_paramSuite->paramGetHandle(paramSet, kParamTonalDarkMid, &p, nullptr) == kOfxStatOK) g_paramSuite->paramGetValueAtTime(p, renderTime, &tonalDM);
        if (g_paramSuite->paramGetHandle(paramSet, kParamTonalMidtones, &p, nullptr) == kOfxStatOK) g_paramSuite->paramGetValueAtTime(p, renderTime, &tonalMid);
        if (g_paramSuite->paramGetHandle(paramSet, kParamTonalLightMid, &p, nullptr) == kOfxStatOK) g_paramSuite->paramGetValueAtTime(p, renderTime, &tonalLM);
        if (g_paramSuite->paramGetHandle(paramSet, kParamTonalHighlights, &p, nullptr) == kOfxStatOK) g_paramSuite->paramGetValueAtTime(p, renderTime, &tonalHi);
        if (g_paramSuite->paramGetHandle(paramSet, kParamDetailRecovery, &p, nullptr) == kOfxStatOK) g_paramSuite->paramGetValueAtTime(p, renderTime, &detailRec);
        if (g_paramSuite->paramGetHandle(paramSet, kParamGrainAmount, &p, nullptr) == kOfxStatOK) g_paramSuite->paramGetValueAtTime(p, renderTime, &grainAmt);
        if (g_paramSuite->paramGetHandle(paramSet, kParamPreviewMode, &p, nullptr) == kOfxStatOK) g_paramSuite->paramGetValueAtTime(p, renderTime, &previewMode);

        OfxParamHandle hp = nullptr;
        if (g_paramSuite->paramGetHandle(paramSet, kParamCalibrated, &hp, nullptr) == kOfxStatOK && hp) g_paramSuite->paramGetValue(hp, &calibrated);
        if (g_paramSuite->paramGetHandle(paramSet, kParamNlfSlopeY, &hp, nullptr) == kOfxStatOK && hp) g_paramSuite->paramGetValue(hp, &nlfSlopeY);
        if (g_paramSuite->paramGetHandle(paramSet, kParamNlfInterceptY, &hp, nullptr) == kOfxStatOK && hp) g_paramSuite->paramGetValue(hp, &nlfInterceptY);
        if (g_paramSuite->paramGetHandle(paramSet, kParamNlfSlopeCb, &hp, nullptr) == kOfxStatOK && hp) g_paramSuite->paramGetValue(hp, &nlfSlopeCb);
        if (g_paramSuite->paramGetHandle(paramSet, kParamNlfInterceptCb, &hp, nullptr) == kOfxStatOK && hp) g_paramSuite->paramGetValue(hp, &nlfInterceptCb);
        if (g_paramSuite->paramGetHandle(paramSet, kParamNlfSlopeCr, &hp, nullptr) == kOfxStatOK && hp) g_paramSuite->paramGetValue(hp, &nlfSlopeCr);
        if (g_paramSuite->paramGetHandle(paramSet, kParamNlfInterceptCr, &hp, nullptr) == kOfxStatOK && hp) g_paramSuite->paramGetValue(hp, &nlfInterceptCr);
    }

    // Get Output Image
    OfxImageClipHandle outputClip = nullptr;
    g_effectSuite->clipGetHandle(effect, kOfxImageEffectOutputClipName, &outputClip, nullptr);
    if (!outputClip) return kOfxStatFailed;

    OfxPropertySetHandle outImage = nullptr;
    g_effectSuite->clipGetImage(outputClip, renderTime, nullptr, &outImage);
    if (!outImage) return kOfxStatFailed;

    OfxImgInfo outInfo = query_ofx_image(outImage);
    int width = outInfo.width();
    int height = outInfo.height();
    if (!outInfo.valid()) {
        g_effectSuite->clipReleaseImage(outImage);
        return kOfxStatOK;
    }

    // Get Source Clip and Temporal Frames
    OfxImageClipHandle sourceClip = nullptr;
    g_effectSuite->clipGetHandle(effect, kOfxImageEffectSimpleSourceClipName, &sourceClip, nullptr);
    if (!sourceClip) {
        g_effectSuite->clipReleaseImage(outImage);
        return kOfxStatFailed;
    }

    std::vector<OfxPropertySetHandle> sourceImages;
    std::vector<OfxImgInfo> srcInfos;

    for (int t = -radius; t <= radius; ++t) {
        double frameTime = renderTime + t;
        OfxPropertySetHandle srcImg = nullptr;
        g_effectSuite->clipGetImage(sourceClip, frameTime, nullptr, &srcImg);
        if (!srcImg) {
            g_effectSuite->clipGetImage(sourceClip, renderTime, nullptr, &srcImg);
        }
        if (srcImg) {
            OfxImgInfo si = query_ofx_image(srcImg);
            if (si.valid()) {
                sourceImages.push_back(srcImg);
                srcInfos.push_back(si);
            } else {
                g_effectSuite->clipReleaseImage(srcImg);
            }
        }
    }

    auto release_all_clips = [&]() {
        for (auto img : sourceImages) g_effectSuite->clipReleaseImage(img);
        g_effectSuite->clipReleaseImage(outImage);
    };

    if (sourceImages.empty()) {
        release_all_clips();
        return kOfxStatOK;
    }

    const int cpuThreads = compute_polite_cpu_threads();
    const size_t refIdx = srcInfos.size() / 2;
    const OfxImgInfo& refSrcInfo = srcInfos[refIdx];

    // Detect if source has active alpha or is an RGB / DPX plate without alpha (all 0.0)
    bool sourceHasAlpha = false;
    if (refSrcInfo.channels >= 4) {
        int stepY = std::max(1, height / 8);
        int stepX = std::max(1, width / 8);
        for (int py = 0; py < height; py += stepY) {
            for (int px = 0; px < width; px += stepX) {
                if (sample_ofx_alpha(refSrcInfo, outInfo.bounds[0] + px, outInfo.bounds[1] + py) > 0.001f) {
                    sourceHasAlpha = true;
                    break;
                }
            }
            if (sourceHasAlpha) break;
        }
    }

    bool canUseCuda = (accelMode != 2 && librenoise::is_cuda_available());

    // =========================================================================
    // Track B: GPU Acceleration Pipeline (CUDA sm_86)
    // =========================================================================
    if (canUseCuda) {
        librenoise::NoiseProfile activeProfile;
        if (calibrated != 0.0) {
            activeProfile.profile_name = "CalibratedSensor";
            activeProfile.nlf_per_channel[0] = {static_cast<float>(nlfSlopeY), static_cast<float>(nlfInterceptY)};
            activeProfile.nlf_per_channel[1] = {static_cast<float>(nlfSlopeCb), static_cast<float>(nlfInterceptCb)};
            activeProfile.nlf_per_channel[2] = {static_cast<float>(nlfSlopeCr), static_cast<float>(nlfInterceptCr)};
        } else {
            size_t auto_ref_idx = srcInfos.size() / 2;
            const OfxImgInfo& refInfo = srcInfos[auto_ref_idx];
            int rw = refInfo.width();
            int rh = refInfo.height();
            librenoise::VideoFrame32F refF(rw, rh);

            if (cudaEnabled != 0) {
#if defined(LIBRENOISE_HAS_CUDA) && (LIBRENOISE_HAS_CUDA != 0)
                size_t totalBytes = static_cast<size_t>(std::abs(refInfo.rowBytes)) * static_cast<size_t>(rh);
                std::vector<uint8_t> h_copy(totalBytes);
                if (cudaMemcpy(h_copy.data(), refInfo.data, totalBytes, cudaMemcpyDeviceToHost) == cudaSuccess) {
                    OfxImgInfo hostRef = refInfo;
                    hostRef.data = h_copy.data();
                    hostRef.rowBytes = std::abs(refInfo.rowBytes);
                    #pragma omp parallel for schedule(static) num_threads(cpuThreads)
                    for (int y = 0; y < rh; ++y) {
                        for (int x = 0; x < rw; ++x) {
                            float rgba[4];
                            sample_ofx_pixel_rgba(hostRef, hostRef.bounds[0] + x, hostRef.bounds[1] + y, rgba);
                            refF.channels[0].at(x, y) = rgba[0];
                            refF.channels[1].at(x, y) = rgba[1];
                            refF.channels[2].at(x, y) = rgba[2];
                        }
                    }
                    activeProfile = librenoise::NoiseProfiler::auto_estimate(refF);
                }
#endif
            } else {
                #pragma omp parallel for schedule(static) num_threads(cpuThreads)
                for (int y = 0; y < rh; ++y) {
                    for (int x = 0; x < rw; ++x) {
                        float rgba[4];
                        sample_ofx_pixel_rgba(refInfo, refInfo.bounds[0] + x, refInfo.bounds[1] + y, rgba);
                        refF.channels[0].at(x, y) = rgba[0];
                        refF.channels[1].at(x, y) = rgba[1];
                        refF.channels[2].at(x, y) = rgba[2];
                    }
                }
                activeProfile = librenoise::NoiseProfiler::auto_estimate(refF);
            }
        }

        librenoise::CudaDenoiseParams cp;
        cp.width = width;
        cp.height = height;
        cp.step_size = (qualityChoice == 0) ? 6 : (qualityChoice == 1) ? 4 : 3;
        cp.engine_choice = engineChoice;
        cp.early_exit_motion = earlyExitMotion;
        cp.fast_chroma = fastChroma;
        cp.temporal_weight = static_cast<float>(tempStrength);
        cp.spatial_weight = static_cast<float>(spatStrength);
        cp.band_weights[0] = static_cast<float>(bandVL);
        cp.band_weights[1] = static_cast<float>(bandL);
        cp.band_weights[2] = static_cast<float>(bandM);
        cp.band_weights[3] = static_cast<float>(bandH);
        cp.band_weights[4] = static_cast<float>(bandVH);
        cp.tonal_lut[0] = static_cast<float>(tonalSh);
        cp.tonal_lut[1] = static_cast<float>(tonalDM);
        cp.tonal_lut[2] = static_cast<float>(tonalMid);
        cp.tonal_lut[3] = static_cast<float>(tonalLM);
        cp.tonal_lut[4] = static_cast<float>(tonalHi);
        for (int c = 0; c < 3; ++c) {
            cp.nlf_a[c] = activeProfile.nlf_per_channel[c].a;
            cp.nlf_b[c] = activeProfile.nlf_per_channel[c].b;
            for (int b = 0; b < 5; ++b) {
                cp.psd_weights[c][b] = activeProfile.psd.octave_weights[c][b];
            }
        }
        cp.detail_recovery = static_cast<float>(detailRec);
        cp.grain_amount = static_cast<float>(grainAmt);
        cp.mix_original = static_cast<float>(mixWithOriginal);
        cp.luma_scale = static_cast<float>(lumaScale);
        cp.chroma_scale_cr = static_cast<float>(chromaCr);
        cp.chroma_scale_cb = static_cast<float>(chromaCb);
        cp.preview_mode = previewMode;
        cp.force_opaque_alpha = sourceHasAlpha ? 0 : 1;

#if defined(LIBRENOISE_HAS_CUDA) && (LIBRENOISE_HAS_CUDA != 0)
        // Path B1: Host passed native CUDA device pointers (Zero-copy GPU render)
        if (cudaEnabled != 0) {
            bool directCompatible = (outInfo.bitDepth && std::strcmp(outInfo.bitDepth, kOfxBitDepthFloat) == 0 &&
                                     outInfo.rowBytes > 0 && outInfo.channels >= 3);
            for (const auto& si : srcInfos) {
                if (si.rowBytes != outInfo.rowBytes || si.channels != outInfo.channels ||
                    si.bounds[0] != outInfo.bounds[0] || si.bounds[1] != outInfo.bounds[1] ||
                    si.bounds[2] != outInfo.bounds[2] || si.bounds[3] != outInfo.bounds[3]) {
                    directCompatible = false;
                    break;
                }
            }

            if (directCompatible) {
                cp.stride_floats = outInfo.rowBytes / static_cast<int32_t>(sizeof(float));
                cp.channels = outInfo.channels;

                void* cudaStream = nullptr;
                g_propSuite->propGetPointer(inArgs, kOfxImageEffectPropCudaStream, 0, &cudaStream);
                cudaStream_t stream = reinterpret_cast<cudaStream_t>(cudaStream);

                std::vector<const float*> d_ptrs;
                d_ptrs.reserve(srcInfos.size());
                for (const auto& si : srcInfos) {
                    d_ptrs.push_back(reinterpret_cast<const float*>(si.data));
                }

                const float** d_frame_arr = nullptr;
                if (cudaMalloc(&d_frame_arr, d_ptrs.size() * sizeof(float*)) == cudaSuccess) {
                    cudaMemcpyAsync(d_frame_arr, d_ptrs.data(), d_ptrs.size() * sizeof(float*), cudaMemcpyHostToDevice, stream);

                    int32_t ref_idx = static_cast<int32_t>(d_ptrs.size() / 2);
                    bool ok = librenoise::cuda_denoise_device(
                        d_frame_arr,
                        static_cast<int32_t>(d_ptrs.size()),
                        ref_idx,
                        reinterpret_cast<float*>(outInfo.data),
                        cp,
                        cudaStream
                    );
                    cudaFree(d_frame_arr);

                    if (ok) {
                        for (auto img : sourceImages) g_effectSuite->clipReleaseImage(img);
                        g_effectSuite->clipReleaseImage(outImage);
                        return kOfxStatOK;
                    }
                }
            }
        } else {
            // Path B2: Host passed CPU system memory (Float/Short/Byte, RGB/RGBA, +/- RowBytes, any Bounds)
            cp.stride_floats = width * 4;
            cp.channels = 4;
            size_t packedFloats = static_cast<size_t>(width) * static_cast<size_t>(height) * 4;
            size_t packedBytes = packedFloats * sizeof(float);

            std::vector<float*> d_alloc_ptrs(srcInfos.size(), nullptr);
            std::vector<float> h_stage(packedFloats);
            bool allocOk = true;

            for (size_t i = 0; i < srcInfos.size(); ++i) {
                const OfxImgInfo& si = srcInfos[i];
                // Fast direct copy if already contiguous positive-rowBytes RGBA32F with identical bounds and active alpha
                bool isExactRgba32F = (sourceHasAlpha &&
                                       si.bitDepth && std::strcmp(si.bitDepth, kOfxBitDepthFloat) == 0 &&
                                       si.channels == 4 && si.rowBytes == width * 4 * static_cast<int>(sizeof(float)) &&
                                       si.bounds[0] == outInfo.bounds[0] && si.bounds[1] == outInfo.bounds[1] &&
                                       si.bounds[2] == outInfo.bounds[2] && si.bounds[3] == outInfo.bounds[3]);

                if (cudaMalloc(&d_alloc_ptrs[i], packedBytes) != cudaSuccess) {
                    allocOk = false;
                    break;
                }

                if (isExactRgba32F) {
                    cudaMemcpy(d_alloc_ptrs[i], si.data, packedBytes, cudaMemcpyHostToDevice);
                } else {
                    #pragma omp parallel for schedule(static) num_threads(cpuThreads)
                    for (int y = 0; y < height; ++y) {
                        int wy = outInfo.bounds[1] + y;
                        float* dst_row = h_stage.data() + static_cast<size_t>(y) * static_cast<size_t>(width) * 4;
                        for (int x = 0; x < width; ++x) {
                            sample_ofx_pixel_rgba(si, outInfo.bounds[0] + x, wy, dst_row + x * 4);
                            if (!sourceHasAlpha) dst_row[x * 4 + 3] = 1.0f;
                        }
                    }
                    cudaMemcpy(d_alloc_ptrs[i], h_stage.data(), packedBytes, cudaMemcpyHostToDevice);
                }
            }

            float* d_out_buf = nullptr;
            if (allocOk && cudaMalloc(&d_out_buf, packedBytes) == cudaSuccess) {
                const float** d_frame_arr = nullptr;
                bool ok = false;
                if (cudaMalloc(&d_frame_arr, d_alloc_ptrs.size() * sizeof(float*)) == cudaSuccess) {
                    cudaMemcpy(d_frame_arr, d_alloc_ptrs.data(), d_alloc_ptrs.size() * sizeof(float*), cudaMemcpyHostToDevice);

                    int32_t ref_idx = static_cast<int32_t>(d_alloc_ptrs.size() / 2);
                    ok = librenoise::cuda_denoise_device(
                        d_frame_arr,
                        static_cast<int32_t>(d_alloc_ptrs.size()),
                        ref_idx,
                        d_out_buf,
                        cp,
                        nullptr
                    );
                    cudaFree(d_frame_arr);
                }

                if (ok) {
                    bool outExactRgba32F = (sourceHasAlpha &&
                                            outInfo.bitDepth && std::strcmp(outInfo.bitDepth, kOfxBitDepthFloat) == 0 &&
                                            outInfo.channels == 4 &&
                                            outInfo.bounds[0] == 0 && outInfo.bounds[1] == 0 &&
                                            outInfo.rowBytes == width * 4 * static_cast<int>(sizeof(float)));
                    if (outExactRgba32F) {
                        cudaMemcpy(outInfo.data, d_out_buf, packedBytes, cudaMemcpyDeviceToHost);
                    } else {
                        cudaMemcpy(h_stage.data(), d_out_buf, packedBytes, cudaMemcpyDeviceToHost);
                        const int outCh = outInfo.channels;
                        if (outInfo.bitDepth && std::strcmp(outInfo.bitDepth, kOfxBitDepthFloat) == 0) {
                            #pragma omp parallel for schedule(static) num_threads(cpuThreads)
                            for (int y = 0; y < height; ++y) {
                                auto* dst_row = reinterpret_cast<float*>(
                                    reinterpret_cast<uint8_t*>(outInfo.data) +
                                    static_cast<ptrdiff_t>(y) * static_cast<ptrdiff_t>(outInfo.rowBytes));
                                const float* src_row = h_stage.data() + static_cast<size_t>(y) * static_cast<size_t>(width) * 4;
                                for (int x = 0; x < width; ++x) {
                                    if (outCh >= 3) {
                                        dst_row[x * outCh + 0] = src_row[x * 4 + 0];
                                        dst_row[x * outCh + 1] = src_row[x * 4 + 1];
                                        dst_row[x * outCh + 2] = src_row[x * 4 + 2];
                                        if (outCh >= 4) dst_row[x * outCh + 3] = sourceHasAlpha ? src_row[x * 4 + 3] : 1.0f;
                                    } else if (outCh == 1) {
                                        dst_row[x] = src_row[x * 4 + 0];
                                    }
                                }
                            }
                        } else if (outInfo.bitDepth && std::strcmp(outInfo.bitDepth, kOfxBitDepthShort) == 0) {
                            #pragma omp parallel for schedule(static) num_threads(cpuThreads)
                            for (int y = 0; y < height; ++y) {
                                auto* dst_row = reinterpret_cast<uint16_t*>(
                                    reinterpret_cast<uint8_t*>(outInfo.data) +
                                    static_cast<ptrdiff_t>(y) * static_cast<ptrdiff_t>(outInfo.rowBytes));
                                const float* src_row = h_stage.data() + static_cast<size_t>(y) * static_cast<size_t>(width) * 4;
                                for (int x = 0; x < width; ++x) {
                                    for (int c = 0; c < std::min(outCh, 4); ++c) {
                                        dst_row[x * outCh + c] = static_cast<uint16_t>(std::clamp(src_row[x * 4 + c] * 65535.0f, 0.0f, 65535.0f));
                                    }
                                }
                            }
                        } else if (outInfo.bitDepth && std::strcmp(outInfo.bitDepth, kOfxBitDepthByte) == 0) {
                            #pragma omp parallel for schedule(static) num_threads(cpuThreads)
                            for (int y = 0; y < height; ++y) {
                                auto* dst_row = reinterpret_cast<uint8_t*>(outInfo.data) +
                                                static_cast<ptrdiff_t>(y) * static_cast<ptrdiff_t>(outInfo.rowBytes);
                                const float* src_row = h_stage.data() + static_cast<size_t>(y) * static_cast<size_t>(width) * 4;
                                for (int x = 0; x < width; ++x) {
                                    for (int c = 0; c < std::min(outCh, 4); ++c) {
                                        dst_row[x * outCh + c] = static_cast<uint8_t>(std::clamp(src_row[x * 4 + c] * 255.0f, 0.0f, 255.0f));
                                    }
                                }
                            }
                        }
                    }
                }

                cudaFree(d_out_buf);
                for (auto ptr : d_alloc_ptrs) if (ptr) cudaFree(ptr);

                if (ok) {
                    release_all_clips();
                    return kOfxStatOK;
                }
            } else {
                for (auto ptr : d_alloc_ptrs) if (ptr) cudaFree(ptr);
            }
        }
#endif
    }

    // =========================================================================
    // Track A: Multi-threaded AVX2 CPU Pipeline (Cooperative Abort & Memory-Bounded)
    // =========================================================================
    ScopedCpuRenderLock cpuGate(g_cpuRenderLock);
    if (is_ofx_aborted(effect)) {
        release_all_clips();
        return kOfxStatFailed;
    }

    std::vector<librenoise::VideoFrame32F> frameWindow;
    frameWindow.reserve(srcInfos.size());

    for (size_t i = 0; i < srcInfos.size(); ++i) {
        if (is_ofx_aborted(effect)) {
            release_all_clips();
            return kOfxStatFailed;
        }
        const OfxImgInfo& si = srcInfos[i];
        librenoise::VideoFrame32F f(width, height);

        #pragma omp parallel for schedule(static) num_threads(cpuThreads)
        for (int y = 0; y < height; ++y) {
            int wy = outInfo.bounds[1] + y;
            for (int x = 0; x < width; ++x) {
                float rgba[4];
                sample_ofx_pixel_rgba(si, outInfo.bounds[0] + x, wy, rgba);
                f.channels[0].at(x, y) = rgba[0];
                f.channels[1].at(x, y) = rgba[1];
                f.channels[2].at(x, y) = rgba[2];
            }
        }
        frameWindow.push_back(std::move(f));
    }

    librenoise::NoiseProfile activeProfile;
    if (calibrated != 0.0) {
        activeProfile.profile_name = "CalibratedSensor";
        activeProfile.nlf_per_channel[0] = {static_cast<float>(nlfSlopeY), static_cast<float>(nlfInterceptY)};
        activeProfile.nlf_per_channel[1] = {static_cast<float>(nlfSlopeCb), static_cast<float>(nlfInterceptCb)};
        activeProfile.nlf_per_channel[2] = {static_cast<float>(nlfSlopeCr), static_cast<float>(nlfInterceptCr)};
    } else {
        activeProfile = librenoise::NoiseProfiler::auto_estimate(frameWindow[refIdx]);
    }

    if (is_ofx_aborted(effect)) {
        release_all_clips();
        return kOfxStatFailed;
    }

    librenoise::VideoDenoiser::Config config;
    config.temporal_radius = radius;
    config.filter_params.engine = static_cast<librenoise::DenoiseEngine>(engineChoice);
    config.filter_params.quality = static_cast<librenoise::QualityPreset>(qualityChoice);
    config.filter_params.fast_chroma = (fastChroma != 0);
    config.filter_params.fast_dct = (fastDct != 0);
    config.filter_params.early_exit_motion = (earlyExitMotion != 0);
    config.filter_params.temporal_weight = static_cast<float>(tempStrength);
    config.filter_params.spatial_weight = static_cast<float>(spatStrength);
    config.filter_params.luma_scale = static_cast<float>(lumaScale);
    config.filter_params.chroma_scale_cr = static_cast<float>(chromaCr);
    config.filter_params.chroma_scale_cb = static_cast<float>(chromaCb);
    config.filter_params.abort_fn = &ofx_abort_trampoline;
    config.filter_params.abort_user_data = effect;
    config.filter_params.max_threads = cpuThreads;

    config.filter_params.band_weights[0] = static_cast<float>(bandVL);
    config.filter_params.band_weights[1] = static_cast<float>(bandL);
    config.filter_params.band_weights[2] = static_cast<float>(bandM);
    config.filter_params.band_weights[3] = static_cast<float>(bandH);
    config.filter_params.band_weights[4] = static_cast<float>(bandVH);

    config.filter_params.tonal_lut[0] = static_cast<float>(tonalSh);
    config.filter_params.tonal_lut[1] = static_cast<float>(tonalDM);
    config.filter_params.tonal_lut[2] = static_cast<float>(tonalMid);
    config.filter_params.tonal_lut[3] = static_cast<float>(tonalLM);
    config.filter_params.tonal_lut[4] = static_cast<float>(tonalHi);

    config.detail_params.detail_recovery = static_cast<float>(detailRec);
    config.detail_params.grain_amount = static_cast<float>(grainAmt);

    librenoise::VideoDenoiser denoiser(config);
    librenoise::VideoFrame32F denoised = denoiser.process_window(frameWindow, &activeProfile);

    if (is_ofx_aborted(effect)) {
        release_all_clips();
        return kOfxStatFailed;
    }

    const auto& refFrame = frameWindow[refIdx];
    float mixFactor = static_cast<float>(mixWithOriginal);
    const int outCh = outInfo.channels;

    if (outInfo.bitDepth && std::strcmp(outInfo.bitDepth, kOfxBitDepthFloat) == 0) {
        #pragma omp parallel for schedule(static) num_threads(cpuThreads)
        for (int y = 0; y < height; ++y) {
            const int wy = outInfo.bounds[1] + y;
            auto* dstRow = reinterpret_cast<float*>(
                reinterpret_cast<uint8_t*>(outInfo.data) +
                static_cast<ptrdiff_t>(y) * static_cast<ptrdiff_t>(outInfo.rowBytes));
            for (int x = 0; x < width; ++x) {
                float r = denoised.channels[0].at(x, y);
                float g = denoised.channels[1].at(x, y);
                float b = denoised.channels[2].at(x, y);

                float rawR = refFrame.channels[0].at(x, y);
                float rawG = refFrame.channels[1].at(x, y);
                float rawB = refFrame.channels[2].at(x, y);

                if (previewMode == 1) {
                    r = (rawR - r) * 2.0f + 0.5f;
                    g = (rawG - g) * 2.0f + 0.5f;
                    b = (rawB - b) * 2.0f + 0.5f;
                } else if (mixFactor > 0.0f) {
                    r = r * (1.0f - mixFactor) + rawR * mixFactor;
                    g = g * (1.0f - mixFactor) + rawG * mixFactor;
                    b = b * (1.0f - mixFactor) + rawB * mixFactor;
                }

                // Preserve linear-light HDR values > 1.0f and original source alpha (or 1.0 for RGB/DPX)
                if (outCh >= 3) {
                    dstRow[x * outCh + 0] = std::max(0.0f, r);
                    dstRow[x * outCh + 1] = std::max(0.0f, g);
                    dstRow[x * outCh + 2] = std::max(0.0f, b);
                    if (outCh >= 4) dstRow[x * outCh + 3] = sourceHasAlpha ? sample_ofx_alpha(refSrcInfo, outInfo.bounds[0] + x, wy) : 1.0f;
                } else if (outCh == 1) {
                    dstRow[x] = std::max(0.0f, r);
                }
            }
        }
    } else if (outInfo.bitDepth && std::strcmp(outInfo.bitDepth, kOfxBitDepthShort) == 0) {
        #pragma omp parallel for schedule(static) num_threads(cpuThreads)
        for (int y = 0; y < height; ++y) {
            const int wy = outInfo.bounds[1] + y;
            auto* dstRow = reinterpret_cast<uint16_t*>(
                reinterpret_cast<uint8_t*>(outInfo.data) +
                static_cast<ptrdiff_t>(y) * static_cast<ptrdiff_t>(outInfo.rowBytes));
            for (int x = 0; x < width; ++x) {
                float rgb[4] = {
                    denoised.channels[0].at(x, y),
                    denoised.channels[1].at(x, y),
                    denoised.channels[2].at(x, y),
                    (outCh >= 4) ? (sourceHasAlpha ? sample_ofx_alpha(refSrcInfo, outInfo.bounds[0] + x, wy) : 1.0f) : 1.0f
                };
                float raw[3] = {refFrame.channels[0].at(x, y), refFrame.channels[1].at(x, y), refFrame.channels[2].at(x, y)};
                for (int c = 0; c < 3; ++c) {
                    if (previewMode == 1) rgb[c] = (raw[c] - rgb[c]) * 2.0f + 0.5f;
                    else if (mixFactor > 0.0f) rgb[c] = rgb[c] * (1.0f - mixFactor) + raw[c] * mixFactor;
                }
                for (int c = 0; c < std::min(outCh, 4); ++c) {
                    dstRow[x * outCh + c] = static_cast<uint16_t>(std::clamp(rgb[c] * 65535.0f, 0.0f, 65535.0f));
                }
            }
        }
    } else if (outInfo.bitDepth && std::strcmp(outInfo.bitDepth, kOfxBitDepthByte) == 0) {
        #pragma omp parallel for schedule(static) num_threads(cpuThreads)
        for (int y = 0; y < height; ++y) {
            const int wy = outInfo.bounds[1] + y;
            auto* dstRow = reinterpret_cast<uint8_t*>(outInfo.data) +
                           static_cast<ptrdiff_t>(y) * static_cast<ptrdiff_t>(outInfo.rowBytes);
            for (int x = 0; x < width; ++x) {
                float rgb[4] = {
                    denoised.channels[0].at(x, y),
                    denoised.channels[1].at(x, y),
                    denoised.channels[2].at(x, y),
                    (outCh >= 4) ? (sourceHasAlpha ? sample_ofx_alpha(refSrcInfo, outInfo.bounds[0] + x, wy) : 1.0f) : 1.0f
                };
                float raw[3] = {refFrame.channels[0].at(x, y), refFrame.channels[1].at(x, y), refFrame.channels[2].at(x, y)};
                for (int c = 0; c < 3; ++c) {
                    if (previewMode == 1) rgb[c] = (raw[c] - rgb[c]) * 2.0f + 0.5f;
                    else if (mixFactor > 0.0f) rgb[c] = rgb[c] * (1.0f - mixFactor) + raw[c] * mixFactor;
                }
                for (int c = 0; c < std::min(outCh, 4); ++c) {
                    dstRow[x * outCh + c] = static_cast<uint8_t>(std::clamp(rgb[c] * 255.0f, 0.0f, 255.0f));
                }
            }
        }
    }

    release_all_clips();

    return kOfxStatOK;
}

// Primary Plugin Entry Point
OfxStatus plugin_main_entry(
    const char* action,
    const void* handle,
    OfxPropertySetHandle inArgs,
    OfxPropertySetHandle outArgs
) {
    if (std::strcmp(action, kOfxActionLoad) == 0) {
        return action_load();
    }
    if (std::strcmp(action, kOfxActionUnload) == 0) {
        librenoise::release_cuda_resources();
        return kOfxStatOK;
    }
    if (std::strcmp(action, kOfxActionDescribe) == 0) {
        auto effect = reinterpret_cast<OfxImageEffectHandle>(const_cast<void*>(handle));
        return action_describe(effect);
    }
    if (std::strcmp(action, kOfxImageEffectActionDescribeInContext) == 0) {
        auto effect = reinterpret_cast<OfxImageEffectHandle>(const_cast<void*>(handle));
        return action_describe_in_context(effect, inArgs);
    }
    if (std::strcmp(action, kOfxActionCreateInstance) == 0 ||
        std::strcmp(action, kOfxActionDestroyInstance) == 0) {
        return kOfxStatOK;
    }
    if (std::strcmp(action, kOfxImageEffectActionGetFramesNeeded) == 0) {
        auto effect = reinterpret_cast<OfxImageEffectHandle>(const_cast<void*>(handle));
        return action_get_frames_needed(effect, inArgs, outArgs);
    }
    if (std::strcmp(action, kOfxActionInstanceChanged) == 0) {
        auto effect = reinterpret_cast<OfxImageEffectHandle>(const_cast<void*>(handle));
        return action_instance_changed(effect, inArgs, outArgs);
    }
    if (std::strcmp(action, kOfxImageEffectActionRender) == 0) {
        auto effect = reinterpret_cast<OfxImageEffectHandle>(const_cast<void*>(handle));
        return action_render(effect, inArgs);
    }
    if (std::strcmp(action, kOfxImageEffectActionIsIdentity) == 0 ||
        std::strcmp(action, kOfxImageEffectActionGetRegionOfDefinition) == 0 ||
        std::strcmp(action, kOfxImageEffectActionGetRegionsOfInterest) == 0) {
        return kOfxStatReplyDefault;
    }

    return kOfxStatReplyDefault;
}

void set_host(OfxHost* host) {
    g_host = host;
}

static OfxPlugin g_plugin = {
    kOfxImageEffectPluginApi,
    kOfxImageEffectPluginApiVersion,
    "org.librenoise.denoiser",
    LIBRENOISE_VERSION_MAJOR,
    LIBRENOISE_VERSION_MINOR,
    set_host,
    plugin_main_entry
};

} // anonymous namespace

#ifdef __cplusplus
extern "C" {
#endif

OFX_EXPORT int OfxGetNumberOfPlugins(void) {
    return 1;
}

OFX_EXPORT OfxPlugin* OfxGetPlugin(int nth) {
    if (nth == 0) {
        return &g_plugin;
    }
    return nullptr;
}

#ifdef __cplusplus
}
#endif
