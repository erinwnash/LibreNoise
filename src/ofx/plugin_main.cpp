// LibreNoise v0.2 OpenFX plug-in.
//
// Identity is new for v0.2 (org.librenoise.denoiser_v02, bundle LibreNoise_v02.ofx.bundle) so it installs next to
// v0.1 and both can be compared in the same host.

#include "openfx/ofxCore.h"
#include "openfx/ofxProperty.h"
#include "openfx/ofxParam.h"
#include "openfx/ofxImageEffect.h"
#include "openfx/ofxGPURender.h"
#include "librenoise/librenoise.hpp"
#include "ofx_interact.hpp"

#if defined(LIBRENOISE_HAS_CUDA) && (LIBRENOISE_HAS_CUDA != 0)
#include <cuda_runtime.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

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

// Host suites with external linkage (ofx_interact.cpp uses them)
OfxHost* g_host = nullptr;
const OfxPropertySuiteV1* g_propSuite = nullptr;
const OfxParameterSuiteV1* g_paramSuite = nullptr;
const OfxImageEffectSuiteV1* g_effectSuite = nullptr;
const OfxInteractSuiteV1* g_interactSuite = nullptr;

namespace {

namespace v2 = librenoise::v2;

// ------------------------------------------------------------------------------------------------------------
// Parameter names
// ------------------------------------------------------------------------------------------------------------
constexpr const char* kGroupProfile = "groupProfile";
constexpr const char* kGroupTemporal = "groupTemporal";
constexpr const char* kGroupSpatial = "groupSpatial";
constexpr const char* kGroupFrequency = "groupFrequency";
constexpr const char* kGroupTonal = "groupTonal";
constexpr const char* kGroupFinishing = "groupFinishing";
constexpr const char* kGroupProcessing = "groupProcessing";

constexpr const char* kParamAnalysisBoxCorner1 = "analysisBoxCorner1";
constexpr const char* kParamAnalysisBoxCorner2 = "analysisBoxCorner2";
constexpr const char* kParamAutoProfile = "autoProfile";
constexpr const char* kParamBuildProfile = "buildProfile";
constexpr const char* kParamClearProfile = "clearProfile";
constexpr const char* kParamSourceType = "sourceType";
constexpr const char* kParamProfileStatus = "profileStatus";
constexpr const char* kParamProfileAssessment = "profileAssessment";
constexpr const char* kParamProfileDevice = "profileDevice";
constexpr const char* kParamProfileMode = "profileMode";
constexpr const char* kParamProfilePath = "profilePath";
constexpr const char* kParamLoadProfile = "loadProfile";
constexpr const char* kParamSaveProfile = "saveProfile";
constexpr const char* kParamNoiseModel = "noiseModel";   // secret: serialised measured model

constexpr const char* kParamTemporalRadius = "temporalRadius";
constexpr const char* kParamTemporalStrength = "temporalStrength";
constexpr const char* kParamFlicker = "flickerReduction";
constexpr const char* kParamRepeatedFrames = "skipRepeatedFrames";

constexpr const char* kParamAmountY = "amountLuma";
constexpr const char* kParamAmountCr = "amountCr";
constexpr const char* kParamAmountCb = "amountCb";
constexpr const char* kParamNoiseLevel = "noiseLevel";
constexpr const char* kParamNoiseLevelY = "noiseLevelLuma";
constexpr const char* kParamNoiseLevelC = "noiseLevelChroma";

const char* const kBandNoiseParams[5] = {"noiseLevelHigh", "noiseLevelMid", "noiseLevelLow", "noiseLevelVeryLow", "noiseLevelUltraLow"};
const char* const kBandAmountParams[5] = {"amountHigh", "amountMid", "amountLow", "amountVeryLow", "amountUltraLow"};
const char* const kBandLabels[5] = {"High", "Mid", "Low", "Very Low", "Ultra Low"};
const char* const kBandSizes[5] = {"finest grain, 1-2 pixel speckle", "main grain, about 2-4 pixels", "mottling, about 4-8 pixels",
                                   "blotches, about 8-16 pixels", "large soft patches, 16+ pixels"};
const double kBandNoiseDefaults[5] = {0.8, 0.9, 1.0, 1.0, 1.0};
const char* const kTonalParams[5] = {"tonalShadows", "tonalDarkMid", "tonalMidtones", "tonalLightMid", "tonalHighlights"};
const char* const kTonalLabels[5] = {"Shadows (-5 stops)", "Dark Midtones (-2.5 stops)", "Midtones (18% grey)", "Light Midtones (+1.5 stops)", "Highlights (+3 stops)"};

constexpr const char* kParamArtifacts = "artifactRemoval";
constexpr const char* kParamSharpen = "sharpening";
constexpr const char* kParamGrain = "grainAmount";
constexpr const char* kParamMix = "mixWithOriginal";
constexpr const char* kParamPreview = "previewMode";
constexpr const char* kParamAccel = "accelerationMode";
constexpr const char* kParamQuality = "qualityPreset";

// ------------------------------------------------------------------------------------------------------------
// Host helpers
// ------------------------------------------------------------------------------------------------------------
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
CpuRenderLock g_cpuRenderLock;
struct ScopedCpuRenderLock {
    CpuRenderLock& m;
    explicit ScopedCpuRenderLock(CpuRenderLock& l) noexcept : m(l) { m.acquire(); }
    ~ScopedCpuRenderLock() noexcept { m.release(); }
};

int compute_polite_cpu_threads() noexcept {
    int hw = static_cast<int>(std::thread::hardware_concurrency());
    if (hw <= 0) hw = 8;
    if (g_isBackgroundHost != 0) return std::clamp(hw / 4, 2, 6);
    return std::clamp((hw * 3) / 4, 2, 32);
}

bool is_ofx_aborted(OfxImageEffectHandle effect) noexcept {
    return effect != nullptr && g_effectSuite != nullptr && g_effectSuite->abort != nullptr && g_effectSuite->abort(effect) == 1;
}
bool ofx_abort_trampoline(void* user) { return is_ofx_aborted(reinterpret_cast<OfxImageEffectHandle>(user)); }

void set_hint(OfxPropertySetHandle props, const char* text) { g_propSuite->propSetString(props, kOfxParamPropHint, 0, text); }

// Per-instance cache of the automatic noise model, so an un-profiled clip is measured once (not per frame).
struct InstanceData {
    std::mutex m;
    bool has_auto = false;
    int w = 0, h = 0;
    v2::NoiseModel auto_model;
    std::string cached_text;
    v2::NoiseModel cached_model;
};

InstanceData* instance_data(OfxImageEffectHandle effect) {
    OfxPropertySetHandle props = nullptr;
    g_effectSuite->getPropertySet(effect, &props);
    void* p = nullptr;
    if (props) g_propSuite->propGetPointer(props, kOfxPropInstanceData, 0, &p);
    return reinterpret_cast<InstanceData*>(p);
}

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
    bool is_float() const { return bitDepth && std::strcmp(bitDepth, kOfxBitDepthFloat) == 0; }
};

OfxImgInfo query_ofx_image(OfxPropertySetHandle img) {
    OfxImgInfo info;
    if (!img) return info;
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

// Reads any host pixel (float / 16-bit / 8-bit, RGB / RGBA / A) at world coordinates, clamped to bounds.
inline void sample_rgba(const OfxImgInfo& info, int wx, int wy, float out[4]) {
    const int sx = std::clamp(wx, info.bounds[0], info.bounds[2] - 1) - info.bounds[0];
    const int sy = std::clamp(wy, info.bounds[1], info.bounds[3] - 1) - info.bounds[1];
    const auto* row = reinterpret_cast<const uint8_t*>(info.data) + static_cast<ptrdiff_t>(sy) * info.rowBytes;
    const int ch = info.channels;
    float v[4] = {0, 0, 0, 1};
    if (info.is_float()) {
        const float* p = reinterpret_cast<const float*>(row) + sx * ch;
        for (int c = 0; c < std::min(ch, 4); ++c) v[c] = p[c];
    } else if (info.bitDepth && std::strcmp(info.bitDepth, kOfxBitDepthShort) == 0) {
        const uint16_t* p = reinterpret_cast<const uint16_t*>(row) + sx * ch;
        for (int c = 0; c < std::min(ch, 4); ++c) v[c] = p[c] * (1.0f / 65535.0f);
    } else {
        const uint8_t* p = row + sx * ch;
        for (int c = 0; c < std::min(ch, 4); ++c) v[c] = p[c] * (1.0f / 255.0f);
    }
    if (ch >= 3) { out[0] = v[0]; out[1] = v[1]; out[2] = v[2]; out[3] = ch >= 4 ? v[3] : 1.0f; }
    else { out[0] = out[1] = out[2] = v[0]; out[3] = 1.0f; }
}

// Packs a host image into a tightly packed RGBA float buffer covering `bounds`.
void pack_rgba(const OfxImgInfo& src, const int* bounds, std::vector<float>& dst, int threads) {
    const int w = bounds[2] - bounds[0], h = bounds[3] - bounds[1];
    dst.resize(static_cast<size_t>(w) * h * 4);
    #pragma omp parallel for schedule(static) num_threads(threads)
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) sample_rgba(src, bounds[0] + x, bounds[1] + y, &dst[(static_cast<size_t>(y) * w + x) * 4]);
}

// Writes a packed RGBA float buffer into the host output image (any supported depth).
void unpack_rgba(const std::vector<float>& src, const OfxImgInfo& out, int threads) {
    const int w = out.width(), h = out.height(), ch = out.channels;
    const bool is_short = out.bitDepth && std::strcmp(out.bitDepth, kOfxBitDepthShort) == 0;
    #pragma omp parallel for schedule(static) num_threads(threads)
    for (int y = 0; y < h; ++y) {
        uint8_t* row = reinterpret_cast<uint8_t*>(out.data) + static_cast<ptrdiff_t>(y) * out.rowBytes;
        for (int x = 0; x < w; ++x) {
            const float* s = &src[(static_cast<size_t>(y) * w + x) * 4];
            float v[4] = {s[0], s[1], s[2], s[3]};
            if (ch == 1) v[0] = s[3];
            for (int c = 0; c < std::min(ch, 4); ++c) {
                if (out.is_float()) reinterpret_cast<float*>(row)[x * ch + c] = v[c];
                else if (is_short) reinterpret_cast<uint16_t*>(row)[x * ch + c] = static_cast<uint16_t>(std::clamp(v[c] * 65535.0f + 0.5f, 0.0f, 65535.0f));
                else row[x * ch + c] = static_cast<uint8_t>(std::clamp(v[c] * 255.0f + 0.5f, 0.0f, 255.0f));
            }
        }
    }
}

// Parameter access
OfxParamHandle handle(OfxParamSetHandle ps, const char* name) {
    OfxParamHandle h = nullptr;
    if (g_paramSuite->paramGetHandle(ps, name, &h, nullptr) != kOfxStatOK) return nullptr;
    return h;
}
double get_double(OfxParamSetHandle ps, const char* name, double t, double def) {
    double v = def;
    if (auto h = handle(ps, name)) g_paramSuite->paramGetValueAtTime(h, t, &v);
    return v;
}
int get_int(OfxParamSetHandle ps, const char* name, double t, int def) {
    int v = def;
    if (auto h = handle(ps, name)) g_paramSuite->paramGetValueAtTime(h, t, &v);
    return v;
}
std::string get_string(OfxParamSetHandle ps, const char* name) {
    char* v = nullptr;
    if (auto h = handle(ps, name)) g_paramSuite->paramGetValue(h, &v);
    return v ? std::string(v) : std::string();
}
void set_string(OfxParamSetHandle ps, const char* name, const std::string& v) {
    if (auto h = handle(ps, name)) g_paramSuite->paramSetValue(h, v.c_str());
}

// ------------------------------------------------------------------------------------------------------------
// Load / describe
// ------------------------------------------------------------------------------------------------------------
OfxStatus action_load() {
    if (!g_host || !g_host->fetchSuite) return kOfxStatErrFatal;
    g_propSuite = static_cast<const OfxPropertySuiteV1*>(g_host->fetchSuite(g_host->host, kOfxPropertySuite, 1));
    g_paramSuite = static_cast<const OfxParameterSuiteV1*>(g_host->fetchSuite(g_host->host, kOfxParameterSuite, 1));
    g_effectSuite = static_cast<const OfxImageEffectSuiteV1*>(g_host->fetchSuite(g_host->host, kOfxImageEffectSuite, 1));
    g_interactSuite = static_cast<const OfxInteractSuiteV1*>(g_host->fetchSuite(g_host->host, kOfxInteractSuite, 1));
    if (!g_propSuite || !g_paramSuite || !g_effectSuite) return kOfxStatErrMissingHostFeature;
    g_propSuite->propGetInt(g_host->host, kOfxImageEffectHostPropIsBackground, 0, &g_isBackgroundHost);
    return kOfxStatOK;
}

OfxStatus action_describe(OfxImageEffectHandle effect) {
    OfxPropertySetHandle p = nullptr;
    g_effectSuite->getPropertySet(effect, &p);
    g_propSuite->propSetString(p, kOfxPropLabel, 0, "LibreNoise v0.2");
    g_propSuite->propSetString(p, kOfxPropShortLabel, 0, "LibreNoise 0.2");
    g_propSuite->propSetString(p, kOfxPropLongLabel, 0, "LibreNoise v0.2 Video Denoiser");
    g_propSuite->propSetString(p, kOfxImageEffectPluginPropGrouping, 0, "LibreNoise");
    g_propSuite->propSetString(p, kOfxPropPluginDescription, 0,
        "Open-source temporal + multi-scale spatial denoiser. Measures the noise curve, spectrum and source type "
        "(film, digital sensor, compressed video) from the footage itself.");
    g_propSuite->propSetString(p, kOfxImageEffectPropSupportedContexts, 0, kOfxImageEffectContextFilter);
    g_propSuite->propSetString(p, kOfxImageEffectPropSupportedContexts, 1, kOfxImageEffectContextGeneral);
    g_propSuite->propSetString(p, kOfxImageEffectPropSupportedPixelDepths, 0, kOfxBitDepthFloat);
    g_propSuite->propSetString(p, kOfxImageEffectPropSupportedPixelDepths, 1, kOfxBitDepthShort);
    g_propSuite->propSetString(p, kOfxImageEffectPropSupportedPixelDepths, 2, kOfxBitDepthByte);
    g_propSuite->propSetInt(p, kOfxImageEffectPropSupportsTiles, 0, 0);
    g_propSuite->propSetInt(p, kOfxImageEffectPluginPropHostFrameThreading, 0, 0);
    g_propSuite->propSetString(p, kOfxImageEffectPluginRenderThreadSafety, 0, kOfxImageEffectRenderInstanceSafe);
    g_propSuite->propSetInt(p, kOfxImageEffectPropTemporalClipAccess, 0, 1);
    g_propSuite->propSetString(p, kOfxImageEffectPropCudaRenderSupported, 0, "true");
    g_propSuite->propSetString(p, kOfxImageEffectPropCudaStreamSupported, 0, "true");
    g_propSuite->propSetString(p, kOfxImageEffectPropOpenCLRenderSupported, 0, "false");
    g_propSuite->propSetPointer(p, kOfxImageEffectPluginPropOverlayInteractV1, 0, reinterpret_cast<void*>(libre_noise_overlay_interact_entry));
    return kOfxStatOK;
}

struct ParamDefiner {
    OfxParamSetHandle ps;
    void group(const char* name, const char* label, const char* hint, bool open) {
        OfxPropertySetHandle p = nullptr;
        g_paramSuite->paramDefine(ps, kOfxParamTypeGroup, name, &p);
        g_propSuite->propSetString(p, kOfxPropLabel, 0, label);
        set_hint(p, hint);
        g_propSuite->propSetInt(p, kOfxParamPropGroupOpen, 0, open ? 1 : 0);
    }
    OfxPropertySetHandle base(const char* type, const char* name, const char* label, const char* parent, const char* hint) {
        OfxPropertySetHandle p = nullptr;
        g_paramSuite->paramDefine(ps, type, name, &p);
        g_propSuite->propSetString(p, kOfxPropLabel, 0, label);
        if (parent) g_propSuite->propSetString(p, kOfxParamPropParent, 0, parent);
        set_hint(p, hint);
        return p;
    }
    void dbl(const char* name, const char* label, const char* parent, double def, double dmin, double dmax, double hmin, double hmax, const char* hint) {
        auto p = base(kOfxParamTypeDouble, name, label, parent, hint);
        g_propSuite->propSetDouble(p, kOfxParamPropDefault, 0, def);
        g_propSuite->propSetDouble(p, kOfxParamPropDisplayMin, 0, dmin);
        g_propSuite->propSetDouble(p, kOfxParamPropDisplayMax, 0, dmax);
        g_propSuite->propSetDouble(p, kOfxParamPropMin, 0, hmin);
        g_propSuite->propSetDouble(p, kOfxParamPropMax, 0, hmax);
    }
    void integer(const char* name, const char* label, const char* parent, int def, int mn, int mx, const char* hint) {
        auto p = base(kOfxParamTypeInteger, name, label, parent, hint);
        g_propSuite->propSetInt(p, kOfxParamPropDefault, 0, def);
        g_propSuite->propSetInt(p, kOfxParamPropMin, 0, mn);
        g_propSuite->propSetInt(p, kOfxParamPropMax, 0, mx);
        g_propSuite->propSetInt(p, kOfxParamPropDisplayMin, 0, mn);
        g_propSuite->propSetInt(p, kOfxParamPropDisplayMax, 0, mx);
    }
    void boolean(const char* name, const char* label, const char* parent, bool def, const char* hint) {
        auto p = base(kOfxParamTypeBoolean, name, label, parent, hint);
        g_propSuite->propSetInt(p, kOfxParamPropDefault, 0, def ? 1 : 0);
    }
    void choice(const char* name, const char* label, const char* parent, std::initializer_list<const char*> opts, int def, const char* hint) {
        auto p = base(kOfxParamTypeChoice, name, label, parent, hint);
        int i = 0;
        for (const char* o : opts) g_propSuite->propSetString(p, kOfxParamPropChoiceOption, i++, o);
        g_propSuite->propSetInt(p, kOfxParamPropDefault, 0, def);
    }
    void str(const char* name, const char* label, const char* parent, const char* def, const char* hint, bool secret = false) {
        auto p = base(kOfxParamTypeString, name, label, parent, hint);
        g_propSuite->propSetString(p, kOfxParamPropDefault, 0, def);
        if (secret) g_propSuite->propSetInt(p, kOfxParamPropSecret, 0, 1);
    }
    void button(const char* name, const char* label, const char* parent, const char* hint) {
        base(kOfxParamTypePushButton, name, label, parent, hint);
    }
    void xy(const char* name, const char* label, const char* parent, double x, double y, const char* hint) {
        auto p = base(kOfxParamTypeDouble2D, name, label, parent, hint);
        g_propSuite->propSetString(p, kOfxParamPropDoubleType, 0, kOfxParamDoubleTypeXYAbsolute);
        g_propSuite->propSetDouble(p, kOfxParamPropDefault, 0, x);
        g_propSuite->propSetDouble(p, kOfxParamPropDefault, 1, y);
    }
};

OfxStatus action_describe_in_context(OfxImageEffectHandle effect, OfxPropertySetHandle) {
    OfxPropertySetHandle cp = nullptr;
    g_effectSuite->clipDefine(effect, kOfxImageEffectOutputClipName, &cp);
    g_propSuite->propSetString(cp, kOfxImageEffectPropSupportedComponents, 0, kOfxImageComponentRGBA);
    g_propSuite->propSetString(cp, kOfxImageEffectPropSupportedComponents, 1, kOfxImageComponentRGB);
    g_propSuite->propSetInt(cp, kOfxImageEffectPropSupportsTiles, 0, 0);
    g_effectSuite->clipDefine(effect, kOfxImageEffectSimpleSourceClipName, &cp);
    g_propSuite->propSetString(cp, kOfxImageEffectPropSupportedComponents, 0, kOfxImageComponentRGBA);
    g_propSuite->propSetString(cp, kOfxImageEffectPropSupportedComponents, 1, kOfxImageComponentRGB);
    g_propSuite->propSetInt(cp, kOfxImageEffectPropTemporalClipAccess, 0, 1);
    g_propSuite->propSetInt(cp, kOfxImageEffectPropSupportsTiles, 0, 0);

    OfxPropertySetHandle ep = nullptr;
    g_effectSuite->getPropertySet(effect, &ep);
    if (ep) g_propSuite->propSetPointer(ep, kOfxImageEffectPluginPropOverlayInteractV1, 0, reinterpret_cast<void*>(libre_noise_overlay_interact_entry));

    OfxParamSetHandle ps = nullptr;
    g_effectSuite->getParamSet(effect, &ps);
    ParamDefiner d{ps};

    // ---------------------------------------------------------------- Noise profile
    d.group(kGroupProfile, "Noise Profile",
            "How LibreNoise learns this clip's noise. Without a profile the clip is measured automatically once and the "
            "result is reused for every frame. Auto Profile or a box on a flat area stores it with the project.", true);
    d.choice(kParamSourceType, "Source Type", kGroupProfile, {"Auto Detect", "Film (scanned grain)", "Digital Sensor", "Compressed Video"}, 0,
             "What the footage is. Auto Detect decides from the noise itself: film grain is independent per colour layer, "
             "sensor noise is correlated between channels, compressed video shows an 8x8 block grid. Override if the "
             "status line shows the wrong type. Compressed Video also forces deblocking on.");
    d.button(kParamAutoProfile, "Auto Profile", kGroupProfile,
             "Measure the noise on the current frame: brightness curve over the whole frame, noise spectrum from the "
             "flattest areas, source type and compression grid. Moves the box to the flattest area found.");
    d.xy(kParamAnalysisBoxCorner1, "Sample Box Corner 1", kGroupProfile, 100.0, 100.0,
         "One corner of the noise sample box. Put the box on a flat, featureless area (sky, wall, out-of-focus background).");
    d.xy(kParamAnalysisBoxCorner2, "Sample Box Corner 2", kGroupProfile, 260.0, 260.0,
         "Opposite corner of the noise sample box. Bigger is better: at least 64 x 64 pixels with no edges or texture.");
    d.button(kParamBuildProfile, "Build Profile from Box", kGroupProfile,
             "Measure the noise spectrum from inside the box (the brightness curve is still measured over the whole frame). "
             "Use this when Auto Profile picked an area that contains detail.");
    d.button(kParamClearProfile, "Clear Profile", kGroupProfile,
             "Forget the stored profile and go back to automatic measurement.");
    d.str(kParamProfileStatus, "Status", kGroupProfile, "Automatic (measured on first render)",
          "Detected source type, how noise grows with brightness (exponent: ~1 film, ~0.5 raw sensor), channel correlation, "
          "compression strength and profile quality.");
    d.str(kParamProfileAssessment, "Assessment", kGroupProfile, "",
          "Plain-language verdict on the profile and what to do if it is weak.");
    d.str(kParamProfileDevice, "Camera / Device", kGroupProfile, "Unspecified device",
          "Optional label stored in the .lnp profile file.");
    d.str(kParamProfileMode, "Mode / Stock", kGroupProfile, "",
          "Optional label (ISO, film stock, codec) stored in the .lnp profile file.");
    const std::string def_path = v2::default_profiles_directory() + "\\profile.lnp";
    d.str(kParamProfilePath, "Profile File (.lnp)", kGroupProfile, def_path.c_str(), "Path used by Load Profile and Save Profile.");
    d.button(kParamLoadProfile, "Load Profile", kGroupProfile, "Load a saved .lnp v0.2 profile and use it for this clip.");
    d.button(kParamSaveProfile, "Save Profile", kGroupProfile, "Save the current profile to the .lnp path for reuse on matching footage.");
    d.str(kParamNoiseModel, "Noise Model", kGroupProfile, "", "Stored measured noise model.", true);

    // ---------------------------------------------------------------- Temporal
    d.group(kGroupTemporal, "Temporal Filter",
            "Averages each pixel with its motion-compensated match in neighbouring frames. This removes noise without "
            "softening detail and keeps the result stable from frame to frame.", true);
    d.integer(kParamTemporalRadius, "Radius (frames each side)", kGroupTemporal, 2, 0, 5,
              "Frames used on each side of the current one. 2 = 5 frames total (default). Higher cleans static areas more "
              "and reduces boiling; costs render time and memory. 0 = spatial only.");
    d.dbl(kParamTemporalStrength, "Temporal Strength", kGroupTemporal, 1.0, 0.0, 2.0, 0.0, 4.0,
          "How different a matched pixel may be and still be averaged. Lower if moving objects smear or ghost; raise if "
          "static areas still flicker.");
    d.dbl(kParamFlicker, "Flicker Reduction", kGroupTemporal, 0.0, 0.0, 1.0, 0.0, 1.0,
          "Ignore slow brightness changes between frames when matching, so frame-to-frame flicker is averaged out "
          "together with the noise. 0 = off.");
    d.boolean(kParamRepeatedFrames, "Skip Repeated Frames", kGroupTemporal, true,
              "Exact duplicate frames (frame-rate conversion, clip ends) carry the same noise, so they are not counted "
              "as new samples.");

    // ---------------------------------------------------------------- Spatial
    d.group(kGroupSpatial, "Spatial Filter",
            "Within-frame noise reduction, done in Luma (Y) and two colour-difference channels (Cr, Cb). Noise Level sets "
            "what counts as noise; Amount sets how much of it is removed.", true);
    d.dbl(kParamAmountY, "Luma Amount", kGroupSpatial, 1.0, 0.0, 1.0, 0.0, 1.5,
          "How much of the detected brightness noise the spatial filter removes (1 = all, 0 = none). Lower it (0.6-0.8) "
          "to keep a film-like grain texture.");
    d.dbl(kParamAmountCr, "Chroma Cr (Red-Green) Amount", kGroupSpatial, 1.0, 0.0, 1.0, 0.0, 1.5,
          "How much red / green colour noise is removed. Colour noise can usually be removed completely.");
    d.dbl(kParamAmountCb, "Chroma Cb (Blue-Yellow) Amount", kGroupSpatial, 1.0, 0.0, 1.0, 0.0, 1.5,
          "How much blue / yellow colour noise is removed.");
    d.dbl(kParamNoiseLevel, "Noise Level", kGroupSpatial, 1.0, 0.5, 2.0, 0.1, 4.0,
          "Master threshold relative to the measured noise. Raise if noise remains, lower if fine detail is lost. "
          "Multiplies the per-frequency Noise Levels.");
    d.dbl(kParamNoiseLevelY, "Noise Level: Luma", kGroupSpatial, 1.0, 0.5, 2.0, 0.1, 4.0,
          "Threshold multiplier for the Luma (Y) channel only.");
    d.dbl(kParamNoiseLevelC, "Noise Level: Chroma", kGroupSpatial, 1.0, 0.5, 2.0, 0.1, 4.0,
          "Threshold multiplier for both colour channels (Cr, Cb).");

    // ---------------------------------------------------------------- Frequency
    d.group(kGroupFrequency, "Frequencies (noise size)",
            "Each band is one scale of a 5-level pyramid. Noise Level: what counts as noise at that size (1 = as measured). "
            "Amount: how much of it is removed.", false);
    for (int b = 0; b < 5; ++b) {
        const std::string ln = std::string("Noise Level: ") + kBandLabels[b];
        const std::string hn = std::string("Threshold for ") + kBandSizes[b] + ". 1 = the measured noise. Raise if this size of "
                               "noise remains, lower if detail of this size is lost.";
        d.dbl(kBandNoiseParams[b], ln.c_str(), kGroupFrequency, kBandNoiseDefaults[b], 0.0, 2.0, 0.0, 4.0, hn.c_str());
    }
    for (int b = 0; b < 5; ++b) {
        const std::string la = std::string("Amount: ") + kBandLabels[b];
        const std::string ha = std::string("How much of the noise at this size (") + kBandSizes[b] + ") is removed. 1 = all, 0 = none.";
        d.dbl(kBandAmountParams[b], la.c_str(), kGroupFrequency, 1.0, 0.0, 1.0, 0.0, 1.5, ha.c_str());
    }

    // ---------------------------------------------------------------- Tonal
    d.group(kGroupTonal, "Tonal Equalizer (brightness)",
            "Threshold multiplier by brightness, in stops around 18% grey (scene-linear). The measured noise curve already "
            "adapts to brightness, so 1.0 everywhere is neutral.", false);
    for (int t = 0; t < 5; ++t) {
        const std::string ht = std::string("Noise Level multiplier at ") + kTonalLabels[t] + ". 1 = neutral; raise to clean "
                               "this brightness range harder, lower to keep more texture there.";
        d.dbl(kTonalParams[t], kTonalLabels[t], kGroupTonal, 1.0, 0.0, 2.0, 0.0, 4.0, ht.c_str());
    }

    // ---------------------------------------------------------------- Finishing
    d.group(kGroupFinishing, "Artifacts, Sharpening & Output", "Compression artifact removal, sharpening, grain and output options.", true);
    d.dbl(kParamArtifacts, "Artifact Removal", kGroupFinishing, 1.0, 0.0, 2.0, 0.0, 2.0,
          "Strength of deblocking on the detected 8x8 compression grid (only acts on compressed footage). 0 = off.");
    d.dbl(kParamSharpen, "Sharpening", kGroupFinishing, 0.0, 0.0, 1.0, 0.0, 2.0,
          "Boost fine and medium detail of the denoised image. Noise is not amplified because only the cleaned detail is boosted.");
    d.dbl(kParamGrain, "Grain Amount", kGroupFinishing, 0.0, 0.0, 0.5, 0.0, 1.0,
          "Add back fresh, even grain at the measured noise strength so heavy cleaning does not look plastic. 0 = none.");
    d.dbl(kParamMix, "Mix with Original", kGroupFinishing, 0.0, 0.0, 1.0, 0.0, 1.0,
          "Blend the original back in. 0 = fully denoised, 1 = original.");
    d.choice(kParamPreview, "View", kGroupFinishing, {"Denoised Output", "Removed Noise (x2)"}, 0,
             "Removed Noise shows only what was taken out (x2 around grey). It should look like pure noise; visible edges "
             "or texture mean detail is being removed: lower the Noise Level.");

    // ---------------------------------------------------------------- Processing
    d.group(kGroupProcessing, "Processing", "Hardware and speed.", false);
    d.choice(kParamAccel, "Hardware", kGroupProcessing, {"Auto (GPU if available)", "Force GPU (CUDA)", "Force CPU"}, 0,
             "Auto uses the CUDA GPU (compute 8.6+) and falls back to the CPU. CPU and GPU give the same result.");
    d.choice(kParamQuality, "Quality", kGroupProcessing, {"Fast", "Balanced", "Best"}, 1,
             "Overlap of the spatial filter blocks. Fast for scrubbing, Balanced for most work, Best for final renders "
             "(about 3x the spatial cost of Balanced).");
    return kOfxStatOK;
}

// ------------------------------------------------------------------------------------------------------------
// Instance / frames
// ------------------------------------------------------------------------------------------------------------
OfxStatus action_create_instance(OfxImageEffectHandle effect) {
    OfxPropertySetHandle props = nullptr;
    g_effectSuite->getPropertySet(effect, &props);
    if (props) g_propSuite->propSetPointer(props, kOfxPropInstanceData, 0, new InstanceData());
    return kOfxStatOK;
}

OfxStatus action_destroy_instance(OfxImageEffectHandle effect) {
    delete instance_data(effect);
    OfxPropertySetHandle props = nullptr;
    g_effectSuite->getPropertySet(effect, &props);
    if (props) g_propSuite->propSetPointer(props, kOfxPropInstanceData, 0, nullptr);
    return kOfxStatOK;
}

OfxStatus action_get_frames_needed(OfxImageEffectHandle effect, OfxPropertySetHandle inArgs, OfxPropertySetHandle outArgs) {
    double time = 0.0;
    if (inArgs) g_propSuite->propGetDouble(inArgs, kOfxPropTime, 0, &time);
    OfxParamSetHandle ps = nullptr;
    g_effectSuite->getParamSet(effect, &ps);
    const int radius = std::clamp(get_int(ps, kParamTemporalRadius, time, 2), 0, 5);
    double range[2] = {time - radius, time + radius};
    const std::string prop = std::string("OfxImageClipPropFrameRange_") + kOfxImageEffectSimpleSourceClipName;
    g_propSuite->propSetDoubleN(outArgs, prop.c_str(), 2, range);
    return kOfxStatOK;
}

v2::SourceType source_type_param(OfxParamSetHandle ps, double t) {
    switch (get_int(ps, kParamSourceType, t, 0)) {
        case 1: return v2::SourceType::Film;
        case 2: return v2::SourceType::DigitalSensor;
        case 3: return v2::SourceType::CompressedVideo;
        default: return v2::SourceType::Auto;
    }
}

void publish_model(OfxParamSetHandle ps, const v2::NoiseModel& m) {
    set_string(ps, kParamNoiseModel, m.serialize());
    set_string(ps, kParamProfileStatus, m.summary());
    set_string(ps, kParamProfileAssessment, m.assessment);
}

// ------------------------------------------------------------------------------------------------------------
// Instance changed (buttons)
// ------------------------------------------------------------------------------------------------------------
OfxStatus action_instance_changed(OfxImageEffectHandle effect, OfxPropertySetHandle inArgs) {
    if (!inArgs) return kOfxStatReplyDefault;
    char* type = nullptr;
    g_propSuite->propGetString(inArgs, kOfxPropType, 0, &type);
    if (!type || std::strcmp(type, kOfxTypeParameter) != 0) return kOfxStatReplyDefault;
    char* name = nullptr;
    g_propSuite->propGetString(inArgs, kOfxPropName, 0, &name);
    if (!name) return kOfxStatReplyDefault;
    double t = 0.0;
    g_propSuite->propGetDouble(inArgs, kOfxPropTime, 0, &t);
    OfxParamSetHandle ps = nullptr;
    g_effectSuite->getParamSet(effect, &ps);
    if (!ps) return kOfxStatReplyDefault;

    const bool is_auto = std::strcmp(name, kParamAutoProfile) == 0;
    const bool is_box = std::strcmp(name, kParamBuildProfile) == 0;
    if (is_auto || is_box) {
        OfxImageClipHandle src = nullptr;
        if (g_effectSuite->clipGetHandle(effect, kOfxImageEffectSimpleSourceClipName, &src, nullptr) != kOfxStatOK || !src) return kOfxStatOK;
        OfxPropertySetHandle img = nullptr;
        if (g_effectSuite->clipGetImage(src, t, nullptr, &img) != kOfxStatOK || !img) return kOfxStatOK;
        const OfxImgInfo info = query_ofx_image(img);
        if (!info.valid()) { g_effectSuite->clipReleaseImage(img); return kOfxStatOK; }
        std::vector<float> packed;
        pack_rgba(info, info.bounds, packed, compute_polite_cpu_threads());
        const int w = info.width(), h = info.height(), ox = info.bounds[0], oy = info.bounds[1];
        g_effectSuite->clipReleaseImage(img);

        v2::ProfileOptions opt;
        opt.force_type = source_type_param(ps, t);
        if (is_box) {
            double c[4] = {0, 0, 0, 0};
            if (auto h1 = handle(ps, kParamAnalysisBoxCorner1)) g_paramSuite->paramGetValueAtTime(h1, t, &c[0], &c[1]);
            if (auto h2 = handle(ps, kParamAnalysisBoxCorner2)) g_paramSuite->paramGetValueAtTime(h2, t, &c[2], &c[3]);
            opt.roi_x = static_cast<int>(std::lround(std::min(c[0], c[2]))) - ox;
            opt.roi_y = static_cast<int>(std::lround(std::min(c[1], c[3]))) - oy;
            opt.roi_w = static_cast<int>(std::lround(std::abs(c[2] - c[0])));
            opt.roi_h = static_cast<int>(std::lround(std::abs(c[3] - c[1])));
            if (opt.roi_w < 32 || opt.roi_h < 32) {
                set_string(ps, kParamProfileAssessment, "Sample box too small: make it at least 64 x 64 pixels.");
                return kOfxStatOK;
            }
        }
        const v2::NoiseModel m = v2::estimate_noise_model(v2::FrameView{packed.data(), w * 4, 4}, w, h, opt);
        if (!m.valid) {
            set_string(ps, kParamProfileAssessment, m.assessment);
            return kOfxStatOK;
        }
        publish_model(ps, m);
        if (is_auto && m.sample_w > 0) {
            const int size = std::max(m.sample_w, 128);
            const double x1 = ox + std::clamp(m.sample_x + m.sample_w / 2 - size / 2, 0, std::max(0, w - size));
            const double y1 = oy + std::clamp(m.sample_y + m.sample_h / 2 - size / 2, 0, std::max(0, h - size));
            if (auto h1 = handle(ps, kParamAnalysisBoxCorner1)) g_paramSuite->paramSetValue(h1, x1, y1);
            if (auto h2 = handle(ps, kParamAnalysisBoxCorner2)) g_paramSuite->paramSetValue(h2, x1 + size, y1 + size);
        }
        return kOfxStatOK;
    }
    if (std::strcmp(name, kParamClearProfile) == 0) {
        set_string(ps, kParamNoiseModel, "");
        set_string(ps, kParamProfileStatus, "Automatic (measured on first render)");
        set_string(ps, kParamProfileAssessment, "");
        if (auto* inst = instance_data(effect)) { std::lock_guard<std::mutex> lk(inst->m); inst->has_auto = false; }
        return kOfxStatOK;
    }
    if (std::strcmp(name, kParamSaveProfile) == 0) {
        v2::ProfileFile f;
        f.device_name = get_string(ps, kParamProfileDevice);
        f.mode_description = get_string(ps, kParamProfileMode);
        if (!v2::NoiseModel::deserialize(get_string(ps, kParamNoiseModel), f.model) || !f.model.valid) {
            set_string(ps, kParamProfileAssessment, "Nothing to save: run Auto Profile or Build Profile from Box first.");
            return kOfxStatOK;
        }
        const std::string path = get_string(ps, kParamProfilePath);
        set_string(ps, kParamProfileAssessment, v2::save_profile(path, f) ? ("Saved " + path) : ("Could not write " + path));
        return kOfxStatOK;
    }
    if (std::strcmp(name, kParamLoadProfile) == 0) {
        const std::string path = get_string(ps, kParamProfilePath);
        v2::ProfileFile f;
        if (!v2::load_profile(path, f) || !f.model.valid) {
            set_string(ps, kParamProfileAssessment, "Could not read a v0.2 profile from " + path);
            return kOfxStatOK;
        }
        set_string(ps, kParamProfileDevice, f.device_name);
        set_string(ps, kParamProfileMode, f.mode_description);
        publish_model(ps, f.model);
        return kOfxStatOK;
    }
    return kOfxStatReplyDefault;
}

// ------------------------------------------------------------------------------------------------------------
// Render
// ------------------------------------------------------------------------------------------------------------
v2::Params read_params(OfxParamSetHandle ps, double t) {
    v2::Params P;
    P.temporal_radius = std::clamp(get_int(ps, kParamTemporalRadius, t, 2), 0, 5);
    P.temporal_strength = static_cast<float>(get_double(ps, kParamTemporalStrength, t, 1.0));
    P.flicker_reduction = static_cast<float>(get_double(ps, kParamFlicker, t, 0.0));
    P.skip_repeated_frames = get_int(ps, kParamRepeatedFrames, t, 1) != 0;
    P.amount_channel[0] = static_cast<float>(get_double(ps, kParamAmountY, t, 1.0));
    P.amount_channel[1] = static_cast<float>(get_double(ps, kParamAmountCb, t, 1.0));
    P.amount_channel[2] = static_cast<float>(get_double(ps, kParamAmountCr, t, 1.0));
    const float master = static_cast<float>(get_double(ps, kParamNoiseLevel, t, 1.0));
    P.channel_noise_level[0] = static_cast<float>(get_double(ps, kParamNoiseLevelY, t, 1.0));
    P.channel_noise_level[1] = P.channel_noise_level[2] = static_cast<float>(get_double(ps, kParamNoiseLevelC, t, 1.0));
    for (int b = 0; b < 5; ++b) {
        P.noise_level[b] = master * static_cast<float>(get_double(ps, kBandNoiseParams[b], t, kBandNoiseDefaults[b]));
        P.amount_band[b] = static_cast<float>(get_double(ps, kBandAmountParams[b], t, 1.0));
        P.tonal[b] = static_cast<float>(get_double(ps, kTonalParams[b], t, 1.0));
    }
    P.artifact_removal = static_cast<float>(get_double(ps, kParamArtifacts, t, 1.0));
    P.sharpening = static_cast<float>(get_double(ps, kParamSharpen, t, 0.0));
    P.grain = static_cast<float>(get_double(ps, kParamGrain, t, 0.0));
    P.grain_seed = 1337u + static_cast<uint32_t>(std::lround(t)) * 7919u;
    P.mix_original = static_cast<float>(get_double(ps, kParamMix, t, 0.0));
    P.preview = get_int(ps, kParamPreview, t, 0);
    const int q = get_int(ps, kParamQuality, t, 1);
    P.step = (q == 0) ? 6 : (q == 2) ? 2 : 4;
    P.source_type = source_type_param(ps, t);
    return P;
}

// Resolves the noise model: the stored profile, else the per-instance automatic one (measured once via fetch_ref).
template <class FetchRef>
bool resolve_model(OfxImageEffectHandle effect, OfxParamSetHandle ps, int w, int h, v2::SourceType forced,
                   FetchRef&& fetch_ref, v2::NoiseModel& out) {
    InstanceData* inst = instance_data(effect);
    const std::string text = get_string(ps, kParamNoiseModel);
    if (!text.empty()) {
        if (inst) {
            std::lock_guard<std::mutex> lk(inst->m);
            if (inst->cached_text != text) {
                v2::NoiseModel m;
                if (!v2::NoiseModel::deserialize(text, m)) return false;
                inst->cached_model = m;
                inst->cached_text = text;
            }
            out = inst->cached_model;
            return out.valid;
        }
        return v2::NoiseModel::deserialize(text, out) && out.valid;
    }
    if (inst) {
        std::lock_guard<std::mutex> lk(inst->m);
        if (inst->has_auto && inst->w == w && inst->h == h) { out = inst->auto_model; return true; }
    }
    std::vector<float> ref;
    if (!fetch_ref(ref)) return false;
    v2::ProfileOptions opt;
    opt.force_type = forced;
    out = v2::estimate_noise_model(v2::FrameView{ref.data(), w * 4, 4}, w, h, opt);
    if (!out.valid) return false;
    if (inst) {
        std::lock_guard<std::mutex> lk(inst->m);
        inst->auto_model = out;
        inst->has_auto = true;
        inst->w = w;
        inst->h = h;
    }
    return true;
}

OfxStatus action_render(OfxImageEffectHandle effect, OfxPropertySetHandle inArgs) {
    double t = 0.0;
    if (inArgs) g_propSuite->propGetDouble(inArgs, kOfxPropTime, 0, &t);
    OfxParamSetHandle ps = nullptr;
    g_effectSuite->getParamSet(effect, &ps);
    if (!ps) return kOfxStatFailed;
    v2::Params P = read_params(ps, t);
    const int accel = get_int(ps, kParamAccel, t, 0);
    int cudaEnabled = 0;
    if (inArgs) g_propSuite->propGetInt(inArgs, kOfxImageEffectPropCudaEnabled, 0, &cudaEnabled);

    OfxImageClipHandle outClip = nullptr, srcClip = nullptr;
    g_effectSuite->clipGetHandle(effect, kOfxImageEffectOutputClipName, &outClip, nullptr);
    g_effectSuite->clipGetHandle(effect, kOfxImageEffectSimpleSourceClipName, &srcClip, nullptr);
    if (!outClip || !srcClip) return kOfxStatFailed;
    OfxPropertySetHandle outImg = nullptr;
    g_effectSuite->clipGetImage(outClip, t, nullptr, &outImg);
    if (!outImg) return kOfxStatFailed;
    const OfxImgInfo outInfo = query_ofx_image(outImg);
    if (!outInfo.valid()) { g_effectSuite->clipReleaseImage(outImg); return kOfxStatOK; }
    const int w = outInfo.width(), h = outInfo.height();

    // Temporal window (missing frames fall back to the reference and are then skipped as duplicates)
    std::vector<OfxPropertySetHandle> imgs;
    std::vector<OfxImgInfo> infos;
    const int R = P.temporal_radius;
    int ref = 0;
    for (int d = -R; d <= R; ++d) {
        OfxPropertySetHandle img = nullptr;
        g_effectSuite->clipGetImage(srcClip, t + d, nullptr, &img);
        if (!img) g_effectSuite->clipGetImage(srcClip, t, nullptr, &img);
        if (!img) continue;
        const OfxImgInfo si = query_ofx_image(img);
        if (!si.valid()) { g_effectSuite->clipReleaseImage(img); continue; }
        if (d == 0) ref = static_cast<int>(imgs.size());
        imgs.push_back(img);
        infos.push_back(si);
    }
    auto release_all = [&]() {
        for (auto i : imgs) g_effectSuite->clipReleaseImage(i);
        g_effectSuite->clipReleaseImage(outImg);
    };
    if (imgs.empty()) { release_all(); return kOfxStatFailed; }
    const int n = static_cast<int>(imgs.size());
    const int threads = compute_polite_cpu_threads();
    P.abort_fn = &ofx_abort_trampoline;
    P.abort_user = effect;
    P.max_threads = threads;

    const bool want_gpu = accel != 2 && v2::cuda_available();

#if defined(LIBRENOISE_HAS_CUDA) && (LIBRENOISE_HAS_CUDA != 0)
    // ---------------- Host passed CUDA device memory
    if (cudaEnabled != 0) {
        bool direct = outInfo.is_float() && outInfo.channels >= 3 && outInfo.rowBytes > 0;
        for (const auto& si : infos)
            if (!si.is_float() || si.channels != outInfo.channels || si.rowBytes != outInfo.rowBytes ||
                std::memcmp(si.bounds, outInfo.bounds, sizeof(si.bounds)) != 0) direct = false;
        void* stream = nullptr;
        g_propSuite->propGetPointer(inArgs, kOfxImageEffectPropCudaStream, 0, &stream);
        auto fetch_ref = [&](std::vector<float>& ref_host) {
            const OfxImgInfo& ri = infos[static_cast<size_t>(ref)];
            std::vector<uint8_t> raw(static_cast<size_t>(std::abs(ri.rowBytes)) * ri.height());
            if (cudaMemcpy(raw.data(), ri.data, raw.size(), cudaMemcpyDeviceToHost) != cudaSuccess) return false;
            OfxImgInfo hi = ri;
            hi.data = raw.data();
            hi.rowBytes = std::abs(ri.rowBytes);
            pack_rgba(hi, outInfo.bounds, ref_host, threads);
            return true;
        };
        v2::NoiseModel model;
        bool ok = false;
        if (direct && resolve_model(effect, ps, w, h, P.source_type, fetch_ref, model)) {
            std::vector<v2::FrameView> views;
            for (const auto& si : infos) views.push_back(v2::FrameView{reinterpret_cast<const float*>(si.data), si.rowBytes / 4, si.channels});
            ok = v2::denoise_cuda(views.data(), n, ref, w, h, model, P,
                                  v2::OutputView{reinterpret_cast<float*>(outInfo.data), outInfo.rowBytes / 4, outInfo.channels}, stream);
        }
        release_all();
        return ok ? kOfxStatOK : kOfxStatFailed;
    }
#endif

    // ---------------- Host memory: pack every frame to RGBA float
    std::vector<std::vector<float>> packed(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        if (is_ofx_aborted(effect)) { release_all(); return kOfxStatFailed; }
        pack_rgba(infos[static_cast<size_t>(i)], outInfo.bounds, packed[static_cast<size_t>(i)], threads);
    }
    auto fetch_ref = [&](std::vector<float>& ref_host) { ref_host = packed[static_cast<size_t>(ref)]; return true; };
    v2::NoiseModel model;
    if (!resolve_model(effect, ps, w, h, P.source_type, fetch_ref, model)) { release_all(); return kOfxStatFailed; }

    std::vector<float> result(static_cast<size_t>(w) * h * 4, 0.0f);
    bool ok = false;
#if defined(LIBRENOISE_HAS_CUDA) && (LIBRENOISE_HAS_CUDA != 0)
    if (want_gpu) {
        const size_t bytes = result.size() * sizeof(float);
        std::vector<float*> dptr(static_cast<size_t>(n), nullptr);
        float* dout = nullptr;
        bool alloc_ok = cudaMalloc(&dout, bytes) == cudaSuccess;
        for (int i = 0; i < n && alloc_ok; ++i) {
            alloc_ok = cudaMalloc(&dptr[static_cast<size_t>(i)], bytes) == cudaSuccess &&
                       cudaMemcpy(dptr[static_cast<size_t>(i)], packed[static_cast<size_t>(i)].data(), bytes, cudaMemcpyHostToDevice) == cudaSuccess;
        }
        if (alloc_ok) {
            std::vector<v2::FrameView> views;
            for (int i = 0; i < n; ++i) views.push_back(v2::FrameView{dptr[static_cast<size_t>(i)], w * 4, 4});
            ok = v2::denoise_cuda(views.data(), n, ref, w, h, model, P, v2::OutputView{dout, w * 4, 4}, nullptr);
            if (ok) ok = cudaMemcpy(result.data(), dout, bytes, cudaMemcpyDeviceToHost) == cudaSuccess;
        }
        cudaGetLastError();
        if (dout) cudaFree(dout);
        for (float* p : dptr) if (p) cudaFree(p);
    }
#endif
    if (!ok) {
        if (accel == 1 && want_gpu) { release_all(); return kOfxStatFailed; }   // Force GPU: no silent CPU fallback
        ScopedCpuRenderLock gate(g_cpuRenderLock);
        std::vector<v2::FrameView> views;
        for (int i = 0; i < n; ++i) views.push_back(v2::FrameView{packed[static_cast<size_t>(i)].data(), w * 4, 4});
        ok = v2::denoise_cpu(views.data(), n, ref, w, h, model, P, v2::OutputView{result.data(), w * 4, 4});
    }
    if (ok) unpack_rgba(result, outInfo, threads);
    release_all();
    return ok ? kOfxStatOK : kOfxStatFailed;
}

// ------------------------------------------------------------------------------------------------------------
// Entry
// ------------------------------------------------------------------------------------------------------------
OfxStatus plugin_main_entry(const char* action, const void* h, OfxPropertySetHandle inArgs, OfxPropertySetHandle outArgs) {
    auto effect = reinterpret_cast<OfxImageEffectHandle>(const_cast<void*>(h));
    if (std::strcmp(action, kOfxActionLoad) == 0) return action_load();
    if (std::strcmp(action, kOfxActionUnload) == 0) { v2::cuda_release(); v2::release_cpu_buffers(); return kOfxStatOK; }
    if (std::strcmp(action, kOfxActionDescribe) == 0) return action_describe(effect);
    if (std::strcmp(action, kOfxImageEffectActionDescribeInContext) == 0) return action_describe_in_context(effect, inArgs);
    if (std::strcmp(action, kOfxActionCreateInstance) == 0) return action_create_instance(effect);
    if (std::strcmp(action, kOfxActionDestroyInstance) == 0) return action_destroy_instance(effect);
    if (std::strcmp(action, kOfxImageEffectActionGetFramesNeeded) == 0) return action_get_frames_needed(effect, inArgs, outArgs);
    if (std::strcmp(action, kOfxActionInstanceChanged) == 0) return action_instance_changed(effect, inArgs);
    if (std::strcmp(action, kOfxImageEffectActionRender) == 0) return action_render(effect, inArgs);
    return kOfxStatReplyDefault;
}

void set_host(OfxHost* host) { g_host = host; }

OfxPlugin g_plugin = {
    kOfxImageEffectPluginApi,
    kOfxImageEffectPluginApiVersion,
    "org.librenoise.denoiser_v02",
    LIBRENOISE_VERSION_MAJOR,
    LIBRENOISE_VERSION_MINOR,
    set_host,
    plugin_main_entry
};

} // namespace

extern "C" {
OFX_EXPORT int OfxGetNumberOfPlugins(void) { return 1; }
OFX_EXPORT OfxPlugin* OfxGetPlugin(int nth) { return nth == 0 ? &g_plugin : nullptr; }
}
