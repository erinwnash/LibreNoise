#pragma once

#include "ofxCore.h"
#include "ofxProperty.h"

#ifdef __cplusplus
extern "C" {
#endif

#define kOfxImageEffectPluginApi "OfxImageEffectPluginApi"
#define kOfxImageEffectPluginApiVersion 1

#define kOfxImageEffectSuite "OfxImageEffectSuite"

#define kOfxImageEffectActionDescribeInContext "OfxImageEffectActionDescribeInContext"
#define kOfxImageEffectActionGetRegionOfDefinition "OfxImageEffectActionGetRegionOfDefinition"
#define kOfxImageEffectActionGetRegionsOfInterest "OfxImageEffectActionGetRegionsOfInterest"
#define kOfxImageEffectActionGetFramesNeeded "OfxImageEffectActionGetFramesNeeded"
#define kOfxImageEffectActionIsIdentity "OfxImageEffectActionIsIdentity"
#define kOfxImageEffectActionRender "OfxImageEffectActionRender"
#define kOfxImageEffectActionBeginSequenceRender "OfxImageEffectActionBeginSequenceRender"
#define kOfxImageEffectActionEndSequenceRender "OfxImageEffectActionEndSequenceRender"

#define kOfxImageEffectContextFilter "OfxImageEffectContextFilter"
#define kOfxImageEffectContextGeneral "OfxImageEffectContextGeneral"

#define kOfxImageEffectPropSupportedContexts "OfxImageEffectPropSupportedContexts"
#define kOfxImageEffectPropSupportedPixelDepths "OfxImageEffectPropSupportedPixelDepths"
#define kOfxImageEffectPropSupportedComponents "OfxImageEffectPropSupportedComponents"
#define kOfxImageEffectPropContext "OfxImageEffectPropContext"
#define kOfxImageEffectPropPixelDepth "OfxImageEffectPropPixelDepth"
#define kOfxImageEffectPropComponents "OfxImageEffectPropComponents"
#define kOfxImageEffectPropTemporalClipAccess "OfxImageEffectPropTemporalClipAccess"

#define kOfxBitDepthNone "OfxBitDepthNone"
#define kOfxBitDepthByte "OfxBitDepthByte"
#define kOfxBitDepthShort "OfxBitDepthShort"
#define kOfxBitDepthFloat "OfxBitDepthFloat"

#define kOfxImageComponentNone "OfxImageComponentNone"
#define kOfxImageComponentRGBA "OfxImageComponentRGBA"
#define kOfxImageComponentRGB "OfxImageComponentRGB"
#define kOfxImageComponentAlpha "OfxImageComponentAlpha"

#define kOfxImageClipOutput "Output"
#define kOfxImageClipSource "Source"

#define kOfxImagePropData "OfxImagePropData"
#define kOfxImagePropRowBytes "OfxImagePropRowBytes"
#define kOfxImagePropBounds "OfxImagePropBounds"
#define kOfxImagePropRegionOfDefinition "OfxImagePropRegionOfDefinition"

#define kOfxImageEffectPropFrameRange "OfxImageEffectPropFrameRange"
#define kOfxImageEffectPropPreMultiplication "OfxImageEffectPropPreMultiplication"

typedef struct OfxTimeRange {
    double min;
    double max;
} OfxTimeRange;

typedef struct OfxRangeD {
    double min;
    double max;
} OfxRangeD;

typedef struct OfxRangeI {
    int min;
    int max;
} OfxRangeI;

typedef struct OfxPointD {
    double x;
    double y;
} OfxPointD;

typedef struct OfxRectD {
    double x1;
    double y1;
    double x2;
    double y2;
} OfxRectD;

typedef struct OfxRectI {
    int x1;
    int y1;
    int x2;
    int y2;
} OfxRectI;

typedef struct OfxImageEffectSuiteV1 {
    OfxStatus (*getPropertySet)(OfxImageEffectHandle imageEffect, OfxPropertySetHandle* propHandle);
    OfxStatus (*getParamSet)(OfxImageEffectHandle imageEffect, OfxParamSetHandle* paramSet);
    OfxStatus (*clipDefine)(OfxImageEffectHandle imageEffect, const char* name, OfxPropertySetHandle* propertySet);
    OfxStatus (*clipGetHandle)(OfxImageEffectHandle imageEffect, const char* name, OfxImageClipHandle* clip, OfxPropertySetHandle* propertySet);
    OfxStatus (*clipGetPropertySet)(OfxImageClipHandle clip, OfxPropertySetHandle* propHandle);
    OfxStatus (*clipGetImage)(OfxImageClipHandle clip, double time, const OfxRectD* region, OfxPropertySetHandle* image);
    OfxStatus (*clipReleaseImage)(OfxPropertySetHandle image);
    OfxStatus (*clipGetRegionOfDefinition)(OfxImageClipHandle clip, double time, OfxRectD* bounds);
    int (*abort)(OfxImageEffectHandle imageEffect);
    OfxStatus (*imageMemoryAlloc)(OfxImageEffectHandle instanceHandle, size_t nBytes, void** allocatedData);
    OfxStatus (*imageMemoryFree)(void* allocatedData);
} OfxImageEffectSuiteV1;

#ifdef __cplusplus
}
#endif
