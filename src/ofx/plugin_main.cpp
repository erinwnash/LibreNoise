#include "openfx/ofxCore.h"
#include "openfx/ofxImageEffect.h"
#include "librenoise/librenoise.hpp"
#include <cstring>
#include <iostream>

#if defined(_WIN32) || defined(__WIN32__)
#define OFX_EXPORT __declspec(dllexport)
#else
#define OFX_EXPORT __attribute__((visibility("default")))
#endif

namespace {

OfxHost* g_host = nullptr;

OfxStatus plugin_main_entry(
    const char* action,
    const void* handle,
    OfxPropertySetHandle inArgs,
    OfxPropertySetHandle outArgs
) {
    if (std::strcmp(action, kOfxActionLoad) == 0) {
        return kOfxStatOK;
    }
    
    if (std::strcmp(action, kOfxActionUnload) == 0) {
        return kOfxStatOK;
    }

    if (std::strcmp(action, kOfxActionDescribe) == 0) {
        // Plugin descriptor logic
        return kOfxStatOK;
    }

    if (std::strcmp(action, kOfxImageEffectActionDescribeInContext) == 0) {
        // Declare RGBA & Float bitdepth support, temporal frame access
        return kOfxStatOK;
    }

    if (std::strcmp(action, kOfxImageEffectActionGetFramesNeeded) == 0) {
        // Temporal frame fetching: request t-2 to t+2 (radius = 2)
        return kOfxStatOK;
    }

    if (std::strcmp(action, kOfxImageEffectActionRender) == 0) {
        // Connect host buffer pointers to librenoise::VideoFrame32F and execute
        return kOfxStatOK;
    }

    return kOfxStatOK;
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
