#pragma once

/**
 * OpenFX Standard Core Headers (BSD 3-Clause)
 * Administered by the Academy Software Foundation
 */

#ifdef __cplusplus
extern "C" {
#endif

#define kOfxTypeImageEffectHost "OfxTypeImageEffectHost"
#define kOfxTypeImageEffect "OfxTypeImageEffect"

#define kOfxActionLoad "OfxActionLoad"
#define kOfxActionUnload "OfxActionUnload"
#define kOfxActionDescribe "OfxActionDescribe"
#define kOfxActionCreateInstance "OfxActionCreateInstance"
#define kOfxActionDestroyInstance "OfxActionDestroyInstance"
#define kOfxActionPurgeCaches "OfxActionPurgeCaches"
#define kOfxActionSyncPrivateData "OfxActionSyncPrivateData"

#define kOfxStatOK 0
#define kOfxStatFailed 1
#define kOfxStatErrFatal 2
#define kOfxStatErrMemory 3
#define kOfxStatErrUnsupported 4

typedef int OfxStatus;
typedef struct OfxPropertySetStruct* OfxPropertySetHandle;
typedef struct OfxImageEffectStruct* OfxImageEffectHandle;
typedef struct OfxClipStruct* OfxImageClipHandle;
typedef struct OfxParamStruct* OfxParamHandle;
typedef struct OfxParamSetStruct* OfxParamSetHandle;
typedef void* OfxHost;

typedef OfxStatus (*OfxPluginEntryPoint)(
    const char* action,
    const void* handle,
    OfxPropertySetHandle inArgs,
    OfxPropertySetHandle outArgs
);

typedef struct OfxPlugin {
    const char* pluginApi;
    int apiVersion;
    const char* pluginIdentifier;
    unsigned int pluginVersionMajor;
    unsigned int pluginVersionMinor;
    void (*setHost)(OfxHost* host);
    OfxPluginEntryPoint mainEntry;
} OfxPlugin;

typedef struct OfxHostStruct {
    OfxPropertySetHandle host;
    void* (*fetchSuite)(OfxPropertySetHandle host, const char* suiteName, int suiteVersion);
} OfxHostStruct;

#ifdef __cplusplus
}
#endif
