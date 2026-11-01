#include "openfx/ofxCore.h"
#include "openfx/ofxProperty.h"
#include "openfx/ofxParam.h"
#include "openfx/ofxImageEffect.h"
#include "librenoise/librenoise.hpp"

#include <cstring>
#include <algorithm>
#include <vector>
#include <cmath>
#include <iostream>
#include <string>

#if defined(_WIN32) || defined(__WIN32__)
#define OFX_EXPORT __declspec(dllexport)
#else
#define OFX_EXPORT __attribute__((visibility("default")))
#endif

namespace {

// Host Suites
OfxHost* g_host = nullptr;
const OfxPropertySuiteV1* g_propSuite = nullptr;
const OfxParameterSuiteV1* g_paramSuite = nullptr;
const OfxImageEffectSuiteV1* g_effectSuite = nullptr;

// Parameter Names
constexpr const char* kParamDenoiseEngine    = "denoiseEngine";
constexpr const char* kParamTemporalRadius   = "temporalRadius";
constexpr const char* kParamTemporalStrength = "temporalStrength";
constexpr const char* kParamSpatialStrength  = "spatialStrength";
constexpr const char* kParamDetailRecovery   = "detailRecovery";
constexpr const char* kParamGrainAmount      = "grainAmount";
constexpr const char* kParamPreviewMode      = "previewMode";

// 5-Band Strengths
constexpr const char* kParamBandVeryLow      = "bandVeryLow";
constexpr const char* kParamBandLow          = "bandLow";
constexpr const char* kParamBandMid          = "bandMid";
constexpr const char* kParamBandHigh         = "bandHigh";
constexpr const char* kParamBandVeryHigh     = "bandVeryHigh";

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
        "Production-grade open-source temporal and spatial video denoiser.");

    // Contexts supported: Filter & General
    g_propSuite->propSetString(effectProps, kOfxImageEffectPropSupportedContexts, 0, kOfxImageEffectContextFilter);
    g_propSuite->propSetString(effectProps, kOfxImageEffectPropSupportedContexts, 1, kOfxImageEffectContextGeneral);

    // Pixel depths
    g_propSuite->propSetString(effectProps, kOfxImageEffectPropSupportedPixelDepths, 0, kOfxBitDepthFloat);
    g_propSuite->propSetString(effectProps, kOfxImageEffectPropSupportedPixelDepths, 1, kOfxBitDepthShort);
    g_propSuite->propSetString(effectProps, kOfxImageEffectPropSupportedPixelDepths, 2, kOfxBitDepthByte);

    // Request temporal frame access
    g_propSuite->propSetInt(effectProps, kOfxImageEffectPropTemporalClipAccess, 0, 1);

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

    // Parameter Set
    OfxParamSetHandle paramSet = nullptr;
    g_effectSuite->getParamSet(effect, &paramSet);

    // 0. Processing Engine Choice
    OfxPropertySetHandle pProps = nullptr;
    g_paramSuite->paramDefine(paramSet, kOfxParamTypeChoice, kParamDenoiseEngine, &pProps);
    g_propSuite->propSetString(pProps, kOfxPropLabel, 0, "Denoise Engine");
    g_propSuite->propSetString(pProps, kOfxParamPropChoiceOption, 0, "3D Collaborative Transform (Highest Fidelity)");
    g_propSuite->propSetString(pProps, kOfxParamPropChoiceOption, 1, "5-Band Wavelet Decomposition (Fastest / Real-time)");
    g_propSuite->propSetString(pProps, kOfxParamPropChoiceOption, 2, "Non-Local Means (Best for Fabrics & Pores)");
    g_propSuite->propSetString(pProps, kOfxParamPropChoiceOption, 3, "Frequency-Domain 3D (Best for Sensor Banding)");
    g_propSuite->propSetInt(pProps, kOfxParamPropDefault, 0, 0);

    // 1. Temporal Radius (Integer 0 to 4)
    g_paramSuite->paramDefine(paramSet, kOfxParamTypeInteger, kParamTemporalRadius, &pProps);
    g_propSuite->propSetString(pProps, kOfxPropLabel, 0, "Temporal Radius");
    g_propSuite->propSetInt(pProps, kOfxParamPropDefault, 0, 2);
    g_propSuite->propSetInt(pProps, kOfxParamPropMin, 0, 0);
    g_propSuite->propSetInt(pProps, kOfxParamPropMax, 0, 4);

    // 2. Temporal Strength
    g_paramSuite->paramDefine(paramSet, kOfxParamTypeDouble, kParamTemporalStrength, &pProps);
    g_propSuite->propSetString(pProps, kOfxPropLabel, 0, "Temporal Strength");
    g_propSuite->propSetDouble(pProps, kOfxParamPropDefault, 0, 1.0);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDisplayMin, 0, 0.0);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDisplayMax, 0, 2.0);

    // 3. Spatial Strength
    g_paramSuite->paramDefine(paramSet, kOfxParamTypeDouble, kParamSpatialStrength, &pProps);
    g_propSuite->propSetString(pProps, kOfxPropLabel, 0, "Spatial Strength");
    g_propSuite->propSetDouble(pProps, kOfxParamPropDefault, 0, 1.0);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDisplayMin, 0, 0.0);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDisplayMax, 0, 2.0);

    // 5-Band Fine Controls
    g_paramSuite->paramDefine(paramSet, kOfxParamTypeDouble, kParamBandVeryLow, &pProps);
    g_propSuite->propSetString(pProps, kOfxPropLabel, 0, "Band: Very Low");
    g_propSuite->propSetDouble(pProps, kOfxParamPropDefault, 0, 1.0);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDisplayMin, 0, 0.0);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDisplayMax, 0, 3.0);

    g_paramSuite->paramDefine(paramSet, kOfxParamTypeDouble, kParamBandLow, &pProps);
    g_propSuite->propSetString(pProps, kOfxPropLabel, 0, "Band: Low");
    g_propSuite->propSetDouble(pProps, kOfxParamPropDefault, 0, 1.0);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDisplayMin, 0, 0.0);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDisplayMax, 0, 3.0);

    g_paramSuite->paramDefine(paramSet, kOfxParamTypeDouble, kParamBandMid, &pProps);
    g_propSuite->propSetString(pProps, kOfxPropLabel, 0, "Band: Mid");
    g_propSuite->propSetDouble(pProps, kOfxParamPropDefault, 0, 1.0);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDisplayMin, 0, 0.0);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDisplayMax, 0, 3.0);

    g_paramSuite->paramDefine(paramSet, kOfxParamTypeDouble, kParamBandHigh, &pProps);
    g_propSuite->propSetString(pProps, kOfxPropLabel, 0, "Band: High");
    g_propSuite->propSetDouble(pProps, kOfxParamPropDefault, 0, 1.0);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDisplayMin, 0, 0.0);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDisplayMax, 0, 3.0);

    g_paramSuite->paramDefine(paramSet, kOfxParamTypeDouble, kParamBandVeryHigh, &pProps);
    g_propSuite->propSetString(pProps, kOfxPropLabel, 0, "Band: Very High");
    g_propSuite->propSetDouble(pProps, kOfxParamPropDefault, 0, 1.0);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDisplayMin, 0, 0.0);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDisplayMax, 0, 3.0);

    // 4. Detail Recovery
    g_paramSuite->paramDefine(paramSet, kOfxParamTypeDouble, kParamDetailRecovery, &pProps);
    g_propSuite->propSetString(pProps, kOfxPropLabel, 0, "Detail Recovery");
    g_propSuite->propSetDouble(pProps, kOfxParamPropDefault, 0, 0.15);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDisplayMin, 0, 0.0);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDisplayMax, 0, 1.0);

    // 5. Grain Re-injection
    g_paramSuite->paramDefine(paramSet, kOfxParamTypeDouble, kParamGrainAmount, &pProps);
    g_propSuite->propSetString(pProps, kOfxPropLabel, 0, "Grain Amount");
    g_propSuite->propSetDouble(pProps, kOfxParamPropDefault, 0, 0.0);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDisplayMin, 0, 0.0);
    g_propSuite->propSetDouble(pProps, kOfxParamPropDisplayMax, 0, 0.5);

    // 6. Preview Mode
    g_paramSuite->paramDefine(paramSet, kOfxParamTypeChoice, kParamPreviewMode, &pProps);
    g_propSuite->propSetString(pProps, kOfxPropLabel, 0, "Preview Mode");
    g_propSuite->propSetString(pProps, kOfxParamPropChoiceOption, 0, "Denoised Output");
    g_propSuite->propSetString(pProps, kOfxParamPropChoiceOption, 1, "Residual (Noise Only)");
    g_propSuite->propSetInt(pProps, kOfxParamPropDefault, 0, 0);

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

    // Per OpenFX standard: "OfxImageClipPropFrameRange_" post-pended with clip name
    std::string propName = std::string("OfxImageClipPropFrameRange_") + kOfxImageEffectSimpleSourceClipName;
    g_propSuite->propSetDoubleN(outArgs, propName.c_str(), 2, range);

    return kOfxStatOK;
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
    int radius = 2;
    double tempStrength = 1.0;
    double spatStrength = 1.0;
    double detailRec = 0.15;
    double grainAmt = 0.0;
    int previewMode = 0;

    double bandVL = 1.0, bandL = 1.0, bandM = 1.0, bandH = 1.0, bandVH = 1.0;

    if (paramSet) {
        OfxParamHandle p = nullptr;
        if (g_paramSuite->paramGetHandle(paramSet, kParamDenoiseEngine, &p, nullptr) == kOfxStatOK) g_paramSuite->paramGetValueAtTime(p, renderTime, &engineChoice);
        if (g_paramSuite->paramGetHandle(paramSet, kParamTemporalRadius, &p, nullptr) == kOfxStatOK) g_paramSuite->paramGetValueAtTime(p, renderTime, &radius);
        if (g_paramSuite->paramGetHandle(paramSet, kParamTemporalStrength, &p, nullptr) == kOfxStatOK) g_paramSuite->paramGetValueAtTime(p, renderTime, &tempStrength);
        if (g_paramSuite->paramGetHandle(paramSet, kParamSpatialStrength, &p, nullptr) == kOfxStatOK) g_paramSuite->paramGetValueAtTime(p, renderTime, &spatStrength);
        if (g_paramSuite->paramGetHandle(paramSet, kParamBandVeryLow, &p, nullptr) == kOfxStatOK) g_paramSuite->paramGetValueAtTime(p, renderTime, &bandVL);
        if (g_paramSuite->paramGetHandle(paramSet, kParamBandLow, &p, nullptr) == kOfxStatOK) g_paramSuite->paramGetValueAtTime(p, renderTime, &bandL);
        if (g_paramSuite->paramGetHandle(paramSet, kParamBandMid, &p, nullptr) == kOfxStatOK) g_paramSuite->paramGetValueAtTime(p, renderTime, &bandM);
        if (g_paramSuite->paramGetHandle(paramSet, kParamBandHigh, &p, nullptr) == kOfxStatOK) g_paramSuite->paramGetValueAtTime(p, renderTime, &bandH);
        if (g_paramSuite->paramGetHandle(paramSet, kParamBandVeryHigh, &p, nullptr) == kOfxStatOK) g_paramSuite->paramGetValueAtTime(p, renderTime, &bandVH);
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

    std::vector<librenoise::VideoFrame32F> window;
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
            void* srcData = nullptr;
            g_propSuite->propGetPointer(srcImg, kOfxImagePropData, 0, &srcData);
            int srcRowBytes = 0;
            g_propSuite->propGetInt(srcImg, kOfxImagePropRowBytes, 0, &srcRowBytes);

            librenoise::VideoFrame32F frame(width, height, 3);
            if (bitDepth && std::strcmp(bitDepth, kOfxBitDepthFloat) == 0 && srcData) {
                const auto* srcPtr = reinterpret_cast<const float*>(srcData);
                int floatStride = srcRowBytes / sizeof(float);
                for (int y = 0; y < height; ++y) {
                    for (int x = 0; x < width; ++x) {
                        frame.channels[0].at(x, y) = srcPtr[y * floatStride + x * 4 + 0];
                        frame.channels[1].at(x, y) = srcPtr[y * floatStride + x * 4 + 1];
                        frame.channels[2].at(x, y) = srcPtr[y * floatStride + x * 4 + 2];
                    }
                }
            } else if (bitDepth && std::strcmp(bitDepth, kOfxBitDepthByte) == 0 && srcData) {
                const auto* srcPtr = reinterpret_cast<const uint8_t*>(srcData);
                for (int y = 0; y < height; ++y) {
                    for (int x = 0; x < width; ++x) {
                        frame.channels[0].at(x, y) = srcPtr[y * srcRowBytes + x * 4 + 0] / 255.0f;
                        frame.channels[1].at(x, y) = srcPtr[y * srcRowBytes + x * 4 + 1] / 255.0f;
                        frame.channels[2].at(x, y) = srcPtr[y * srcRowBytes + x * 4 + 2] / 255.0f;
                    }
                }
            }
            window.push_back(frame);
        }
    }

    if (window.empty()) {
        for (auto img : sourceImages) g_effectSuite->clipReleaseImage(img);
        g_effectSuite->clipReleaseImage(outImage);
        return kOfxStatOK;
    }

    // Configure Denoiser
    librenoise::VideoDenoiser::Config config;
    config.temporal_radius = radius;
    config.filter_params.engine = static_cast<librenoise::DenoiseEngine>(std::clamp(engineChoice, 0, 3));
    config.filter_params.band_weights[0] = static_cast<float>(bandVL);
    config.filter_params.band_weights[1] = static_cast<float>(bandL);
    config.filter_params.band_weights[2] = static_cast<float>(bandM);
    config.filter_params.band_weights[3] = static_cast<float>(bandH);
    config.filter_params.band_weights[4] = static_cast<float>(bandVH);
    config.filter_params.patch_size = 8;
    config.filter_params.step_size = 3;
    config.filter_params.temporal_weight = static_cast<float>(tempStrength);
    config.filter_params.spatial_weight = static_cast<float>(spatStrength);
    config.detail_params.detail_recovery = static_cast<float>(detailRec);
    config.detail_params.grain_amount = static_cast<float>(grainAmt);

    librenoise::VideoDenoiser denoiser(config);
    librenoise::VideoFrame32F denoised = denoiser.process_window(window);

    const size_t centerIdx = window.size() / 2;
    const auto& refFrame = window[centerIdx];

    if (bitDepth && std::strcmp(bitDepth, kOfxBitDepthFloat) == 0) {
        auto* dstPtr = reinterpret_cast<float*>(outData);
        int floatStride = outRowBytes / sizeof(float);

        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                float r = denoised.channels[0].at(x, y);
                float g = denoised.channels[1].at(x, y);
                float b = denoised.channels[2].at(x, y);

                if (previewMode == 1) {
                    r = (refFrame.channels[0].at(x, y) - r) * 2.0f + 0.5f;
                    g = (refFrame.channels[1].at(x, y) - g) * 2.0f + 0.5f;
                    b = (refFrame.channels[2].at(x, y) - b) * 2.0f + 0.5f;
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

                if (previewMode == 1) {
                    r = (refFrame.channels[0].at(x, y) - r) * 2.0f + 0.5f;
                    g = (refFrame.channels[1].at(x, y) - g) * 2.0f + 0.5f;
                    b = (refFrame.channels[2].at(x, y) - b) * 2.0f + 0.5f;
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
    if (std::strcmp(action, kOfxImageEffectActionRender) == 0) {
        auto effect = reinterpret_cast<OfxImageEffectHandle>(const_cast<void*>(handle));
        return action_render(effect, inArgs);
    }
    if (std::strcmp(action, kOfxImageEffectActionIsIdentity) == 0 ||
        std::strcmp(action, kOfxImageEffectActionGetRegionOfDefinition) == 0 ||
        std::strcmp(action, kOfxImageEffectActionGetRegionsOfInterest) == 0) {
        return kOfxStatReplyDefault;
    }

    // Standard OpenFX contract: ignored actions MUST return kOfxStatReplyDefault
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
