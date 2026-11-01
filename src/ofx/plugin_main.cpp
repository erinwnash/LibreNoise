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

#if defined(_WIN32) || defined(__WIN32__)
#define OFX_EXPORT __declspec(dllexport)
#else
#define OFX_EXPORT __attribute__((visibility("default")))
#endif

// Host Suites with external linkage so ofx_interact can access them
OfxHost* g_host = nullptr;
const OfxPropertySuiteV1* g_propSuite = nullptr;
const OfxParameterSuiteV1* g_paramSuite = nullptr;
const OfxImageEffectSuiteV1* g_effectSuite = nullptr;

namespace {

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

// Action: Load
OfxStatus action_load() {
    if (!g_host || !g_host->host || !g_host->fetchSuite) return kOfxStatErrFatal;
    g_propSuite   = static_cast<const OfxPropertySuiteV1*>(g_host->fetchSuite(g_host->host, kOfxPropertySuite, 1));
    g_paramSuite  = static_cast<const OfxParameterSuiteV1*>(g_host->fetchSuite(g_host->host, kOfxParameterSuite, 1));
    g_effectSuite = static_cast<const OfxImageEffectSuiteV1*>(g_host->fetchSuite(g_host->host, kOfxImageEffectSuite, 1));

    if (!g_propSuite || !g_paramSuite || !g_effectSuite) {
        return kOfxStatErrMissingHostFeature;
    }
    return kOfxStatOK;
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

    // Request temporal frame access
    g_propSuite->propSetInt(effectProps, kOfxImageEffectPropTemporalClipAccess, 0, 1);

    // OpenFX 1.5 GPU Render Support (Track B)
    g_propSuite->propSetString(effectProps, kOfxImageEffectPropCudaRenderSupported, 0, "true");
    g_propSuite->propSetString(effectProps, kOfxImageEffectPropCudaStreamSupported, 0, "true");
    g_propSuite->propSetString(effectProps, kOfxImageEffectPropOpenCLRenderSupported, 0, "true");

    // Viewer Overlay Interact Entry (Track B / Interactive Parity)
    g_propSuite->propSetPointer(effectProps, kOfxImageEffectPluginPropOverlayInteractV1, 0,
        reinterpret_cast<void*>(libre_noise_overlay_interact_entry));

    return kOfxStatOK;
}

// Action: DescribeInContext
OfxStatus action_describe_in_context(OfxImageEffectHandle effect, OfxPropertySetHandle inArgs) {
    (void)inArgs;
    // Clips: Define Output and Source
    OfxPropertySetHandle outputProps = nullptr;
    g_effectSuite->clipDefine(effect, kOfxImageEffectOutputClipName, &outputProps);
    g_propSuite->propSetString(outputProps, kOfxImageEffectPropSupportedComponents, 0, kOfxImageComponentRGBA);
    g_propSuite->propSetString(outputProps, kOfxImageEffectPropSupportedComponents, 1, kOfxImageComponentRGB);

    OfxPropertySetHandle sourceProps = nullptr;
    g_effectSuite->clipDefine(effect, kOfxImageEffectSimpleSourceClipName, &sourceProps);
    g_propSuite->propSetString(sourceProps, kOfxImageEffectPropSupportedComponents, 0, kOfxImageComponentRGBA);
    g_propSuite->propSetString(sourceProps, kOfxImageEffectPropSupportedComponents, 1, kOfxImageComponentRGB);
    g_propSuite->propSetInt(sourceProps, kOfxImageEffectPropTemporalClipAccess, 0, 1);

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
    g_propSuite->propSetString(pProps, kOfxPropLabel, 0, "Noise Profile & Analysis");

    // Analysis Box: Double2D Corner 1 and Corner 2
    g_paramSuite->paramDefine(paramSet, kOfxParamTypeDouble2D, kParamAnalysisBoxCorner1, &pProps);
    g_propSuite->propSetString(pProps, kOfxPropLabel, 0, "Region Top-Left");
    g_propSuite->propSetString(pProps, kOfxParamPropDoubleType, 0, kOfxParamDoubleTypeXYAbsolute);
    g_propSuite->propSetString(pProps, kOfxParamPropParent, 0, kGroupProfile);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDefault, 0, 100.0);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDefault, 1, 100.0);

    g_paramSuite->paramDefine(paramSet, kOfxParamTypeDouble2D, kParamAnalysisBoxCorner2, &pProps);
    g_propSuite->propSetString(pProps, kOfxPropLabel, 0, "Region Bottom-Right");
    g_propSuite->propSetString(pProps, kOfxParamPropDoubleType, 0, kOfxParamDoubleTypeXYAbsolute);
    g_propSuite->propSetString(pProps, kOfxParamPropParent, 0, kGroupProfile);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDefault, 0, 260.0);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDefault, 1, 260.0);

    // Auto Profile Button
    g_paramSuite->paramDefine(paramSet, kOfxParamTypePushButton, kParamAutoProfile, &pProps);
    g_propSuite->propSetString(pProps, kOfxPropLabel, 0, "Auto Profile (Search & Analyze)");
    g_propSuite->propSetString(pProps, kOfxParamPropParent, 0, kGroupProfile);

    // Auto Detect Area Button
    g_paramSuite->paramDefine(paramSet, kOfxParamTypePushButton, kParamAutoDetectArea, &pProps);
    g_propSuite->propSetString(pProps, kOfxPropLabel, 0, "Auto Detect Area");
    g_propSuite->propSetString(pProps, kOfxParamPropParent, 0, kGroupProfile);

    // Build Profile from Selection
    g_paramSuite->paramDefine(paramSet, kOfxParamTypePushButton, kParamBuildProfile, &pProps);
    g_propSuite->propSetString(pProps, kOfxPropLabel, 0, "Build Profile from Selection");
    g_propSuite->propSetString(pProps, kOfxParamPropParent, 0, kGroupProfile);

    // Quality Meter (Read-only string readout)
    g_paramSuite->paramDefine(paramSet, kOfxParamTypeString, kParamProfileQuality, &pProps);
    g_propSuite->propSetString(pProps, kOfxPropLabel, 0, "Profile Quality");
    g_propSuite->propSetString(pProps, kOfxParamPropParent, 0, kGroupProfile);
    g_propSuite->propSetString(pProps, kOfxParamPropDefault, 0, "Uncalibrated");

    // Profile Assessment
    g_paramSuite->paramDefine(paramSet, kOfxParamTypeString, kParamProfileAssessment, &pProps);
    g_propSuite->propSetString(pProps, kOfxPropLabel, 0, "Assessment");
    g_propSuite->propSetString(pProps, kOfxParamPropParent, 0, kGroupProfile);
    g_propSuite->propSetString(pProps, kOfxParamPropDefault, 0, "Click Auto Profile or drag box on viewer");

    // Device & Mode Metadata
    g_paramSuite->paramDefine(paramSet, kOfxParamTypeString, kParamProfileDevice, &pProps);
    g_propSuite->propSetString(pProps, kOfxPropLabel, 0, "Camera / Device");
    g_propSuite->propSetString(pProps, kOfxParamPropParent, 0, kGroupProfile);
    g_propSuite->propSetString(pProps, kOfxParamPropDefault, 0, "Generic Digital Camera");

    g_paramSuite->paramDefine(paramSet, kOfxParamTypeString, kParamProfileMode, &pProps);
    g_propSuite->propSetString(pProps, kOfxPropLabel, 0, "Mode Description");
    g_propSuite->propSetString(pProps, kOfxParamPropParent, 0, kGroupProfile);
    g_propSuite->propSetString(pProps, kOfxParamPropDefault, 0, "Standard ISO Profile");

    // Profile File Path & Load / Save
    std::string defaultProfilePath = librenoise::ProfileSerializer::get_default_profiles_directory() + "/profile.lnp";
    g_paramSuite->paramDefine(paramSet, kOfxParamTypeString, kParamProfilePath, &pProps);
    g_propSuite->propSetString(pProps, kOfxPropLabel, 0, "Profile File (.lnp)");
    g_propSuite->propSetString(pProps, kOfxParamPropParent, 0, kGroupProfile);
    g_propSuite->propSetString(pProps, kOfxParamPropDefault, 0, defaultProfilePath.c_str());

    g_paramSuite->paramDefine(paramSet, kOfxParamTypePushButton, kParamLoadProfile, &pProps);
    g_propSuite->propSetString(pProps, kOfxPropLabel, 0, "Load Profile (.lnp)...");
    g_propSuite->propSetString(pProps, kOfxParamPropParent, 0, kGroupProfile);

    g_paramSuite->paramDefine(paramSet, kOfxParamTypePushButton, kParamSaveProfile, &pProps);
    g_propSuite->propSetString(pProps, kOfxPropLabel, 0, "Save Profile (.lnp)...");
    g_propSuite->propSetString(pProps, kOfxParamPropParent, 0, kGroupProfile);

    // =========================================================================
    // Group 2: Denoise Controls
    // =========================================================================
    g_paramSuite->paramDefine(paramSet, kOfxParamTypeGroup, kGroupDenoise, &pProps);
    g_propSuite->propSetString(pProps, kOfxPropLabel, 0, "Denoise Controls");

    // Engine Choice
    g_paramSuite->paramDefine(paramSet, kOfxParamTypeChoice, kParamDenoiseEngine, &pProps);
    g_propSuite->propSetString(pProps, kOfxPropLabel, 0, "Denoise Engine");
    g_propSuite->propSetString(pProps, kOfxParamPropParent, 0, kGroupDenoise);
    g_propSuite->propSetString(pProps, kOfxParamPropChoiceOption, 0, "3D Collaborative Transform (Highest Fidelity)");
    g_propSuite->propSetString(pProps, kOfxParamPropChoiceOption, 1, "5-Band Wavelet Decomposition (Fastest / Real-time)");
    g_propSuite->propSetString(pProps, kOfxParamPropChoiceOption, 2, "Non-Local Means (Best for Fabrics & Pores)");
    g_propSuite->propSetString(pProps, kOfxParamPropChoiceOption, 3, "Frequency-Domain 3D (Best for Sensor Banding)");
    g_propSuite->propSetInt(pProps, kOfxParamPropDefault, 0, 0);

    // Hardware Acceleration Mode (Track B)
    g_paramSuite->paramDefine(paramSet, kOfxParamTypeChoice, kParamAccelerationMode, &pProps);
    g_propSuite->propSetString(pProps, kOfxPropLabel, 0, "Hardware Acceleration");
    g_propSuite->propSetString(pProps, kOfxParamPropParent, 0, kGroupDenoise);
    g_propSuite->propSetString(pProps, kOfxParamPropChoiceOption, 0, "Auto (GPU if available, CPU fallback)");
    g_propSuite->propSetString(pProps, kOfxParamPropChoiceOption, 1, "Force GPU (CUDA sm_86+)");
    g_propSuite->propSetString(pProps, kOfxParamPropChoiceOption, 2, "Force CPU (Multi-core AVX2)");
    g_propSuite->propSetInt(pProps, kOfxParamPropDefault, 0, 0);

    // Quality Preset (Track A)
    g_paramSuite->paramDefine(paramSet, kOfxParamTypeChoice, kParamQualityPreset, &pProps);
    g_propSuite->propSetString(pProps, kOfxPropLabel, 0, "Quality Preset");
    g_propSuite->propSetString(pProps, kOfxParamPropParent, 0, kGroupDenoise);
    g_propSuite->propSetString(pProps, kOfxParamPropChoiceOption, 0, "Fast (Scrubbing / Realtime)");
    g_propSuite->propSetString(pProps, kOfxParamPropChoiceOption, 1, "Balanced (Production Grade)");
    g_propSuite->propSetString(pProps, kOfxParamPropChoiceOption, 2, "High Quality (Master Export)");
    g_propSuite->propSetInt(pProps, kOfxParamPropDefault, 0, 1);

    // Temporal Radius (Integer 0 to 4)
    g_paramSuite->paramDefine(paramSet, kOfxParamTypeInteger, kParamTemporalRadius, &pProps);
    g_propSuite->propSetString(pProps, kOfxPropLabel, 0, "Temporal Radius");
    g_propSuite->propSetString(pProps, kOfxParamPropParent, 0, kGroupDenoise);
    g_propSuite->propSetInt(pProps, kOfxParamPropDefault, 0, 2);
    g_propSuite->propSetInt(pProps, kOfxParamPropMin, 0, 0);
    g_propSuite->propSetInt(pProps, kOfxParamPropMax, 0, 4);

    // Temporal Strength
    g_paramSuite->paramDefine(paramSet, kOfxParamTypeDouble, kParamTemporalStrength, &pProps);
    g_propSuite->propSetString(pProps, kOfxPropLabel, 0, "Temporal Strength");
    g_propSuite->propSetString(pProps, kOfxParamPropParent, 0, kGroupDenoise);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDefault, 0, 1.0);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDisplayMin, 0, 0.0);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDisplayMax, 0, 2.0);

    // Spatial Strength
    g_paramSuite->paramDefine(paramSet, kOfxParamTypeDouble, kParamSpatialStrength, &pProps);
    g_propSuite->propSetString(pProps, kOfxPropLabel, 0, "Spatial Strength");
    g_propSuite->propSetString(pProps, kOfxParamPropParent, 0, kGroupDenoise);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDefault, 0, 1.0);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDisplayMin, 0, 0.0);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDisplayMax, 0, 2.0);

    // Chrominance Scaling
    g_paramSuite->paramDefine(paramSet, kOfxParamTypeDouble, kParamChromaCr, &pProps);
    g_propSuite->propSetString(pProps, kOfxPropLabel, 0, "Chroma Cr Scale");
    g_propSuite->propSetString(pProps, kOfxParamPropParent, 0, kGroupDenoise);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDefault, 0, 1.0);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDisplayMin, 0, 0.0);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDisplayMax, 0, 3.0);

    g_paramSuite->paramDefine(paramSet, kOfxParamTypeDouble, kParamChromaCb, &pProps);
    g_propSuite->propSetString(pProps, kOfxPropLabel, 0, "Chroma Cb Scale");
    g_propSuite->propSetString(pProps, kOfxParamPropParent, 0, kGroupDenoise);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDefault, 0, 1.0);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDisplayMin, 0, 0.0);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDisplayMax, 0, 3.0);

    // Mix With Original
    g_paramSuite->paramDefine(paramSet, kOfxParamTypeDouble, kParamMixWithOriginal, &pProps);
    g_propSuite->propSetString(pProps, kOfxPropLabel, 0, "Mix with Original");
    g_propSuite->propSetString(pProps, kOfxParamPropParent, 0, kGroupDenoise);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDefault, 0, 0.0);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDisplayMin, 0, 0.0);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDisplayMax, 0, 1.0);

    // =========================================================================
    // Group 3: Frequency Equalizer
    // =========================================================================
    g_paramSuite->paramDefine(paramSet, kOfxParamTypeGroup, kGroupFrequency, &pProps);
    g_propSuite->propSetString(pProps, kOfxPropLabel, 0, "Frequency Equalizer");

    g_paramSuite->paramDefine(paramSet, kOfxParamTypeDouble, kParamBandVeryLow, &pProps);
    g_propSuite->propSetString(pProps, kOfxPropLabel, 0, "Band: Very Low");
    g_propSuite->propSetString(pProps, kOfxParamPropParent, 0, kGroupFrequency);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDefault, 0, 1.0);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDisplayMin, 0, 0.0);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDisplayMax, 0, 3.0);

    g_paramSuite->paramDefine(paramSet, kOfxParamTypeDouble, kParamBandLow, &pProps);
    g_propSuite->propSetString(pProps, kOfxPropLabel, 0, "Band: Low");
    g_propSuite->propSetString(pProps, kOfxParamPropParent, 0, kGroupFrequency);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDefault, 0, 1.0);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDisplayMin, 0, 0.0);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDisplayMax, 0, 3.0);

    g_paramSuite->paramDefine(paramSet, kOfxParamTypeDouble, kParamBandMid, &pProps);
    g_propSuite->propSetString(pProps, kOfxPropLabel, 0, "Band: Mid");
    g_propSuite->propSetString(pProps, kOfxParamPropParent, 0, kGroupFrequency);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDefault, 0, 1.0);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDisplayMin, 0, 0.0);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDisplayMax, 0, 3.0);

    g_paramSuite->paramDefine(paramSet, kOfxParamTypeDouble, kParamBandHigh, &pProps);
    g_propSuite->propSetString(pProps, kOfxPropLabel, 0, "Band: High");
    g_propSuite->propSetString(pProps, kOfxParamPropParent, 0, kGroupFrequency);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDefault, 0, 1.0);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDisplayMin, 0, 0.0);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDisplayMax, 0, 3.0);

    g_paramSuite->paramDefine(paramSet, kOfxParamTypeDouble, kParamBandVeryHigh, &pProps);
    g_propSuite->propSetString(pProps, kOfxPropLabel, 0, "Band: Very High");
    g_propSuite->propSetString(pProps, kOfxParamPropParent, 0, kGroupFrequency);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDefault, 0, 1.0);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDisplayMin, 0, 0.0);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDisplayMax, 0, 3.0);

    // =========================================================================
    // Group 4: Tonal Equalizer
    // =========================================================================
    g_paramSuite->paramDefine(paramSet, kOfxParamTypeGroup, kGroupTonal, &pProps);
    g_propSuite->propSetString(pProps, kOfxPropLabel, 0, "Tonal Equalizer");

    g_paramSuite->paramDefine(paramSet, kOfxParamTypeDouble, kParamTonalShadows, &pProps);
    g_propSuite->propSetString(pProps, kOfxPropLabel, 0, "Tonal: Shadows (0-20%)");
    g_propSuite->propSetString(pProps, kOfxParamPropParent, 0, kGroupTonal);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDefault, 0, 1.30);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDisplayMin, 0, 0.0);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDisplayMax, 0, 3.0);

    g_paramSuite->paramDefine(paramSet, kOfxParamTypeDouble, kParamTonalDarkMid, &pProps);
    g_propSuite->propSetString(pProps, kOfxPropLabel, 0, "Tonal: Dark Midtones (20-40%)");
    g_propSuite->propSetString(pProps, kOfxParamPropParent, 0, kGroupTonal);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDefault, 0, 1.15);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDisplayMin, 0, 0.0);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDisplayMax, 0, 3.0);

    g_paramSuite->paramDefine(paramSet, kOfxParamTypeDouble, kParamTonalMidtones, &pProps);
    g_propSuite->propSetString(pProps, kOfxPropLabel, 0, "Tonal: Midtones (40-60%)");
    g_propSuite->propSetString(pProps, kOfxParamPropParent, 0, kGroupTonal);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDefault, 0, 1.00);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDisplayMin, 0, 0.0);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDisplayMax, 0, 3.0);

    g_paramSuite->paramDefine(paramSet, kOfxParamTypeDouble, kParamTonalLightMid, &pProps);
    g_propSuite->propSetString(pProps, kOfxPropLabel, 0, "Tonal: Light Midtones (60-80%)");
    g_propSuite->propSetString(pProps, kOfxParamPropParent, 0, kGroupTonal);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDefault, 0, 0.90);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDisplayMin, 0, 0.0);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDisplayMax, 0, 3.0);

    g_paramSuite->paramDefine(paramSet, kOfxParamTypeDouble, kParamTonalHighlights, &pProps);
    g_propSuite->propSetString(pProps, kOfxPropLabel, 0, "Tonal: Highlights (80-100%)");
    g_propSuite->propSetString(pProps, kOfxParamPropParent, 0, kGroupTonal);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDefault, 0, 0.80);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDisplayMin, 0, 0.0);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDisplayMax, 0, 3.0);

    // =========================================================================
    // Group 5: Detail & Grain Recovery
    // =========================================================================
    g_paramSuite->paramDefine(paramSet, kOfxParamTypeGroup, kGroupDetail, &pProps);
    g_propSuite->propSetString(pProps, kOfxPropLabel, 0, "Detail & Grain Recovery");

    g_paramSuite->paramDefine(paramSet, kOfxParamTypeDouble, kParamDetailRecovery, &pProps);
    g_propSuite->propSetString(pProps, kOfxPropLabel, 0, "Detail Recovery");
    g_propSuite->propSetString(pProps, kOfxParamPropParent, 0, kGroupDetail);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDefault, 0, 0.15);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDisplayMin, 0, 0.0);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDisplayMax, 0, 1.0);

    g_paramSuite->paramDefine(paramSet, kOfxParamTypeDouble, kParamGrainAmount, &pProps);
    g_propSuite->propSetString(pProps, kOfxPropLabel, 0, "Grain Amount");
    g_propSuite->propSetString(pProps, kOfxParamPropParent, 0, kGroupDetail);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDefault, 0, 0.0);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDisplayMin, 0, 0.0);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDisplayMax, 0, 0.5);

    g_paramSuite->paramDefine(paramSet, kOfxParamTypeChoice, kParamPreviewMode, &pProps);
    g_propSuite->propSetString(pProps, kOfxPropLabel, 0, "Preview Mode");
    g_propSuite->propSetString(pProps, kOfxParamPropParent, 0, kGroupDetail);
    g_propSuite->propSetString(pProps, kOfxParamPropChoiceOption, 0, "Denoised Output");
    g_propSuite->propSetString(pProps, kOfxParamPropChoiceOption, 1, "Residual (Noise Only)");
    g_propSuite->propSetInt(pProps, kOfxParamPropDefault, 0, 0);

    // =========================================================================
    // Group 6: Performance & Optimizations
    // =========================================================================
    g_paramSuite->paramDefine(paramSet, kOfxParamTypeGroup, kGroupPerformance, &pProps);
    g_propSuite->propSetString(pProps, kOfxPropLabel, 0, "Performance & Optimizations");

    g_paramSuite->paramDefine(paramSet, kOfxParamTypeBoolean, kParamFastChroma, &pProps);
    g_propSuite->propSetString(pProps, kOfxPropLabel, 0, "Fast Chroma Subsampling");
    g_propSuite->propSetString(pProps, kOfxParamPropParent, 0, kGroupPerformance);
    g_propSuite->propSetInt(pProps, kOfxParamPropDefault, 0, 1);

    g_paramSuite->paramDefine(paramSet, kOfxParamTypeBoolean, kParamFastDct, &pProps);
    g_propSuite->propSetString(pProps, kOfxPropLabel, 0, "Fast Butterfly DCT");
    g_propSuite->propSetString(pProps, kOfxParamPropParent, 0, kGroupPerformance);
    g_propSuite->propSetInt(pProps, kOfxParamPropDefault, 0, 1);

    g_paramSuite->paramDefine(paramSet, kOfxParamTypeBoolean, kParamEarlyExitMotion, &pProps);
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

    // Helper lambda to fetch the current source frame
    auto fetch_current_frame = [&]() -> std::optional<librenoise::VideoFrame32F> {
        OfxImageClipHandle srcClip = nullptr;
        if (g_effectSuite->clipGetHandle(effect, kOfxImageEffectSimpleSourceClipName, &srcClip, nullptr) != kOfxStatOK || !srcClip) {
            return std::nullopt;
        }
        OfxPropertySetHandle srcImg = nullptr;
        if (g_effectSuite->clipGetImage(srcClip, changeTime, nullptr, &srcImg) != kOfxStatOK || !srcImg) {
            return std::nullopt;
        }

        void* srcData = nullptr;
        g_propSuite->propGetPointer(srcImg, kOfxImagePropData, 0, &srcData);
        int srcBounds[4] = {0, 0, 0, 0};
        g_propSuite->propGetIntN(srcImg, kOfxImagePropBounds, 4, srcBounds);
        int srcRowBytes = 0;
        g_propSuite->propGetInt(srcImg, kOfxImagePropRowBytes, 0, &srcRowBytes);
        char* bitDepth = nullptr;
        g_propSuite->propGetString(srcImg, kOfxImageEffectPropPixelDepth, 0, &bitDepth);

        int width = srcBounds[2] - srcBounds[0];
        int height = srcBounds[3] - srcBounds[1];
        if (width <= 0 || height <= 0 || !srcData) {
            g_effectSuite->clipReleaseImage(srcImg);
            return std::nullopt;
        }

        librenoise::VideoFrame32F frame(width, height);
        if (bitDepth && std::strcmp(bitDepth, kOfxBitDepthFloat) == 0) {
            const auto* s = reinterpret_cast<const float*>(srcData);
            int stride = srcRowBytes / sizeof(float);
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    frame.channels[0].at(x, y) = s[y * stride + x * 4 + 0];
                    frame.channels[1].at(x, y) = s[y * stride + x * 4 + 1];
                    frame.channels[2].at(x, y) = s[y * stride + x * 4 + 2];
                }
            }
        } else if (bitDepth && std::strcmp(bitDepth, kOfxBitDepthByte) == 0) {
            const auto* s = reinterpret_cast<const uint8_t*>(srcData);
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    frame.channels[0].at(x, y) = s[y * srcRowBytes + x * 4 + 0] / 255.0f;
                    frame.channels[1].at(x, y) = s[y * srcRowBytes + x * 4 + 1] / 255.0f;
                    frame.channels[2].at(x, y) = s[y * srcRowBytes + x * 4 + 2] / 255.0f;
                }
            }
        }
        g_effectSuite->clipReleaseImage(srcImg);
        return frame;
    };

    // Auto Profile or Auto Detect Area
    if (std::strcmp(paramName, kParamAutoProfile) == 0 || std::strcmp(paramName, kParamAutoDetectArea) == 0) {
        auto optFrame = fetch_current_frame();
        if (optFrame.has_value()) {
            const auto& frame = *optFrame;
            auto roi = librenoise::NoiseRoiAnalyzer::find_optimal_roi(frame);

            OfxParamHandle boxP1 = nullptr, boxP2 = nullptr;
            if (g_paramSuite->paramGetHandle(paramSet, kParamAnalysisBoxCorner1, &boxP1, nullptr) == kOfxStatOK && boxP1) {
                g_paramSuite->paramSetValue(boxP1, static_cast<double>(roi.x), static_cast<double>(roi.y));
            }
            if (g_paramSuite->paramGetHandle(paramSet, kParamAnalysisBoxCorner2, &boxP2, nullptr) == kOfxStatOK && boxP2) {
                g_paramSuite->paramSetValue(boxP2, static_cast<double>(roi.x + roi.width), static_cast<double>(roi.y + roi.height));
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
            const auto& frame = *optFrame;
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
            librenoise::AnalysisRoi roi{x1, y1, std::max(16, x2 - x1), std::max(16, y2 - y1)};

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
    }

    // Get Output Image
    OfxImageClipHandle outputClip = nullptr;
    g_effectSuite->clipGetHandle(effect, kOfxImageEffectOutputClipName, &outputClip, nullptr);
    if (!outputClip) return kOfxStatFailed;

    OfxPropertySetHandle outImage = nullptr;
    g_effectSuite->clipGetImage(outputClip, renderTime, nullptr, &outImage);
    if (!outImage) return kOfxStatFailed;

    void* outData = nullptr;
    g_propSuite->propGetPointer(outImage, kOfxImagePropData, 0, &outData);
    int outRowBytes = 0;
    g_propSuite->propGetInt(outImage, kOfxImagePropRowBytes, 0, &outRowBytes);
    int bounds[4] = {0, 0, 0, 0};
    g_propSuite->propGetIntN(outImage, kOfxImagePropBounds, 4, bounds);
    char* bitDepth = nullptr;
    g_propSuite->propGetString(outImage, kOfxImageEffectPropPixelDepth, 0, &bitDepth);

    int width = bounds[2] - bounds[0];
    int height = bounds[3] - bounds[1];
    if (width <= 0 || height <= 0 || !outData) {
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

    for (int t = -radius; t <= radius; ++t) {
        double frameTime = renderTime + t;
        OfxPropertySetHandle srcImg = nullptr;
        g_effectSuite->clipGetImage(sourceClip, frameTime, nullptr, &srcImg);
        if (!srcImg) {
            g_effectSuite->clipGetImage(sourceClip, renderTime, nullptr, &srcImg);
        }
        if (srcImg) {
            sourceImages.push_back(srcImg);
        }
    }

    if (sourceImages.empty()) {
        g_effectSuite->clipReleaseImage(outImage);
        return kOfxStatOK;
    }

    // =========================================================================
    // Track B: GPU Acceleration Pipeline (CUDA sm_86)
    // =========================================================================
    bool canUseCuda = (bitDepth && std::strcmp(bitDepth, kOfxBitDepthFloat) == 0 &&
                       accelMode != 2 && librenoise::is_cuda_available());

    if (canUseCuda) {
        librenoise::CudaDenoiseParams cp;
        cp.width = width;
        cp.height = height;
        cp.stride_floats = outRowBytes / static_cast<int32_t>(sizeof(float));
        cp.channels = 4;
        cp.step_size = (qualityChoice == 0) ? 6 : (qualityChoice == 1) ? 4 : 3;
        cp.temporal_weight = static_cast<float>(tempStrength);
        cp.spatial_weight = static_cast<float>(spatStrength);
        cp.band_weights[0] = static_cast<float>(bandVL);
        cp.band_weights[1] = static_cast<float>(bandL);
        cp.band_weights[2] = static_cast<float>(bandM);
        cp.band_weights[3] = static_cast<float>(bandH);
        cp.band_weights[4] = static_cast<float>(bandVH);
        double calibrated = 0.0;
        double nlfSlopeY = 0.00035, nlfInterceptY = 0.00003;
        double nlfSlopeCb = 0.00030, nlfInterceptCb = 0.000025;
        double nlfSlopeCr = 0.00030, nlfInterceptCr = 0.000025;
        if (paramSet) {
            OfxParamHandle hp = nullptr;
            if (g_paramSuite->paramGetHandle(paramSet, kParamCalibrated, &hp, nullptr) == kOfxStatOK && hp) g_paramSuite->paramGetValue(hp, &calibrated);
            if (g_paramSuite->paramGetHandle(paramSet, kParamNlfSlopeY, &hp, nullptr) == kOfxStatOK && hp) g_paramSuite->paramGetValue(hp, &nlfSlopeY);
            if (g_paramSuite->paramGetHandle(paramSet, kParamNlfInterceptY, &hp, nullptr) == kOfxStatOK && hp) g_paramSuite->paramGetValue(hp, &nlfInterceptY);
            if (g_paramSuite->paramGetHandle(paramSet, kParamNlfSlopeCb, &hp, nullptr) == kOfxStatOK && hp) g_paramSuite->paramGetValue(hp, &nlfSlopeCb);
            if (g_paramSuite->paramGetHandle(paramSet, kParamNlfInterceptCb, &hp, nullptr) == kOfxStatOK && hp) g_paramSuite->paramGetValue(hp, &nlfInterceptCb);
            if (g_paramSuite->paramGetHandle(paramSet, kParamNlfSlopeCr, &hp, nullptr) == kOfxStatOK && hp) g_paramSuite->paramGetValue(hp, &nlfSlopeCr);
            if (g_paramSuite->paramGetHandle(paramSet, kParamNlfInterceptCr, &hp, nullptr) == kOfxStatOK && hp) g_paramSuite->paramGetValue(hp, &nlfInterceptCr);
        }

        float spatScale = static_cast<float>(spatStrength * spatStrength);
        cp.nlf_a[0] = static_cast<float>(nlfSlopeY) * spatScale;
        cp.nlf_b[0] = static_cast<float>(nlfInterceptY) * spatScale;
        cp.nlf_a[1] = static_cast<float>(nlfSlopeCb) * spatScale * static_cast<float>(chromaCb * chromaCb);
        cp.nlf_b[1] = static_cast<float>(nlfInterceptCb) * spatScale * static_cast<float>(chromaCb * chromaCb);
        cp.nlf_a[2] = static_cast<float>(nlfSlopeCr) * spatScale * static_cast<float>(chromaCr * chromaCr);
        cp.nlf_b[2] = static_cast<float>(nlfInterceptCr) * spatScale * static_cast<float>(chromaCr * chromaCr);
        cp.detail_recovery = static_cast<float>(detailRec);
        cp.grain_amount = static_cast<float>(grainAmt);
        cp.mix_original = static_cast<float>(mixWithOriginal);
        cp.chroma_scale_cr = static_cast<float>(chromaCr);
        cp.chroma_scale_cb = static_cast<float>(chromaCb);
        cp.preview_mode = previewMode;

        // Path B1: Host passed native CUDA device pointers (Zero-copy GPU render)
        if (cudaEnabled != 0) {
            void* cudaStream = nullptr;
            g_propSuite->propGetPointer(inArgs, kOfxImageEffectPropCudaStream, 0, &cudaStream);

            std::vector<const float*> d_ptrs;
            for (auto img : sourceImages) {
                void* d_src = nullptr;
                g_propSuite->propGetPointer(img, kOfxImagePropData, 0, &d_src);
                d_ptrs.push_back(reinterpret_cast<const float*>(d_src));
            }

#if defined(LIBRENOISE_HAS_CUDA) && (LIBRENOISE_HAS_CUDA != 0)
            const float** d_frame_arr = nullptr;
            cudaStream_t stream = reinterpret_cast<cudaStream_t>(cudaStream);
            cudaMallocAsync(&d_frame_arr, d_ptrs.size() * sizeof(float*), stream);
            cudaMemcpyAsync(d_frame_arr, d_ptrs.data(), d_ptrs.size() * sizeof(float*), cudaMemcpyHostToDevice, stream);

            int32_t ref_idx = static_cast<int32_t>(d_ptrs.size() / 2);
            bool ok = librenoise::cuda_denoise_device(
                d_frame_arr,
                static_cast<int32_t>(d_ptrs.size()),
                ref_idx,
                reinterpret_cast<float*>(outData),
                cp,
                cudaStream
            );
            cudaFree(d_frame_arr);

            if (ok) {
                for (auto img : sourceImages) g_effectSuite->clipReleaseImage(img);
                g_effectSuite->clipReleaseImage(outImage);
                return kOfxStatOK;
            }
#endif
        } else {
            // Path B2: Host passed CPU system memory, but GPU acceleration is available
#if defined(LIBRENOISE_HAS_CUDA) && (LIBRENOISE_HAS_CUDA != 0)
            size_t frameBytes = static_cast<size_t>(outRowBytes * height);
            std::vector<float*> d_alloc_ptrs(sourceImages.size(), nullptr);
            bool allocOk = true;

            for (size_t i = 0; i < sourceImages.size(); ++i) {
                void* hostSrc = nullptr;
                g_propSuite->propGetPointer(sourceImages[i], kOfxImagePropData, 0, &hostSrc);
                if (!hostSrc || cudaMalloc(&d_alloc_ptrs[i], frameBytes) != cudaSuccess) {
                    allocOk = false;
                    break;
                }
                cudaMemcpy(d_alloc_ptrs[i], hostSrc, frameBytes, cudaMemcpyHostToDevice);
            }

            float* d_out_buf = nullptr;
            if (allocOk && cudaMalloc(&d_out_buf, frameBytes) == cudaSuccess) {
                const float** d_frame_arr = nullptr;
                cudaMalloc(&d_frame_arr, d_alloc_ptrs.size() * sizeof(float*));
                cudaMemcpy(d_frame_arr, d_alloc_ptrs.data(), d_alloc_ptrs.size() * sizeof(float*), cudaMemcpyHostToDevice);

                int32_t ref_idx = static_cast<int32_t>(d_alloc_ptrs.size() / 2);
                bool ok = librenoise::cuda_denoise_device(
                    d_frame_arr,
                    static_cast<int32_t>(d_alloc_ptrs.size()),
                    ref_idx,
                    d_out_buf,
                    cp,
                    nullptr
                );

                if (ok) {
                    cudaMemcpy(outData, d_out_buf, frameBytes, cudaMemcpyDeviceToHost);
                }

                cudaFree(d_frame_arr);
                cudaFree(d_out_buf);
                for (auto ptr : d_alloc_ptrs) if (ptr) cudaFree(ptr);

                if (ok) {
                    for (auto img : sourceImages) g_effectSuite->clipReleaseImage(img);
                    g_effectSuite->clipReleaseImage(outImage);
                    return kOfxStatOK;
                }
            } else {
                for (auto ptr : d_alloc_ptrs) if (ptr) cudaFree(ptr);
            }
#endif
        }
    }

    // =========================================================================
    // Track A: Multi-threaded AVX2 CPU Fallback
    // =========================================================================
    std::vector<librenoise::VideoFrame32F> frameWindow;
    frameWindow.reserve(sourceImages.size());

    for (auto img : sourceImages) {
        void* srcData = nullptr;
        g_propSuite->propGetPointer(img, kOfxImagePropData, 0, &srcData);
        int srcRowBytes = 0;
        g_propSuite->propGetInt(img, kOfxImagePropRowBytes, 0, &srcRowBytes);

        librenoise::VideoFrame32F f(width, height);

        if (bitDepth && std::strcmp(bitDepth, kOfxBitDepthFloat) == 0) {
            auto* s = reinterpret_cast<float*>(srcData);
            int floatStride = srcRowBytes / sizeof(float);
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    f.channels[0].at(x, y) = s[y * floatStride + x * 4 + 0];
                    f.channels[1].at(x, y) = s[y * floatStride + x * 4 + 1];
                    f.channels[2].at(x, y) = s[y * floatStride + x * 4 + 2];
                }
            }
        } else if (bitDepth && std::strcmp(bitDepth, kOfxBitDepthByte) == 0) {
            auto* s = reinterpret_cast<uint8_t*>(srcData);
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    f.channels[0].at(x, y) = s[y * srcRowBytes + x * 4 + 0] / 255.0f;
                    f.channels[1].at(x, y) = s[y * srcRowBytes + x * 4 + 1] / 255.0f;
                    f.channels[2].at(x, y) = s[y * srcRowBytes + x * 4 + 2] / 255.0f;
                }
            }
        }
        frameWindow.push_back(std::move(f));
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

    config.filter_params.band_weights[0] = static_cast<float>(bandVL);
    config.filter_params.band_weights[1] = static_cast<float>(bandL);
    config.filter_params.band_weights[2] = static_cast<float>(bandM);
    config.filter_params.band_weights[3] = static_cast<float>(bandH);
    config.filter_params.band_weights[4] = static_cast<float>(bandVH);

    config.detail_params.detail_recovery = static_cast<float>(detailRec);
    config.detail_params.grain_amount = static_cast<float>(grainAmt);

    librenoise::VideoDenoiser denoiser(config);
    librenoise::VideoFrame32F denoised = denoiser.process_window(frameWindow);

    size_t refIdx = frameWindow.size() / 2;
    const auto& refFrame = frameWindow[refIdx];

    // Write back to host buffer with tonal/chroma and mix blending
    float mixFactor = static_cast<float>(mixWithOriginal);

    if (bitDepth && std::strcmp(bitDepth, kOfxBitDepthFloat) == 0) {
        auto* dstPtr = reinterpret_cast<float*>(outData);
        int floatStride = outRowBytes / sizeof(float);

        for (int y = 0; y < height; ++y) {
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

                dstPtr[y * floatStride + x * 4 + 0] = std::clamp(r, 0.0f, 1.0f);
                dstPtr[y * floatStride + x * 4 + 1] = std::clamp(g, 0.0f, 1.0f);
                dstPtr[y * floatStride + x * 4 + 2] = std::clamp(b, 0.0f, 1.0f);
                dstPtr[y * floatStride + x * 4 + 3] = 1.0f;
            }
        }
    } else if (bitDepth && std::strcmp(bitDepth, kOfxBitDepthByte) == 0) {
        auto* dstPtr = reinterpret_cast<uint8_t*>(outData);

        for (int y = 0; y < height; ++y) {
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

                dstPtr[y * outRowBytes + x * 4 + 0] = static_cast<uint8_t>(std::clamp(r * 255.0f, 0.0f, 255.0f));
                dstPtr[y * outRowBytes + x * 4 + 1] = static_cast<uint8_t>(std::clamp(g * 255.0f, 0.0f, 255.0f));
                dstPtr[y * outRowBytes + x * 4 + 2] = static_cast<uint8_t>(std::clamp(b * 255.0f, 0.0f, 255.0f));
                dstPtr[y * outRowBytes + x * 4 + 3] = 255;
            }
        }
    }

    for (auto img : sourceImages) g_effectSuite->clipReleaseImage(img);
    g_effectSuite->clipReleaseImage(outImage);

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
