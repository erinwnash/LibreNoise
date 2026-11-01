#pragma once

#include "ofxCore.h"

#ifdef __cplusplus
extern "C" {
#endif

#define kOfxImageEffectPluginApi "OfxImageEffectPluginApi"
#define kOfxImageEffectPluginApiVersion 1

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

#define kOfxBitDepthByte "OfxBitDepthByte"
#define kOfxBitDepthShort "OfxBitDepthShort"
#define kOfxBitDepthFloat "OfxBitDepthFloat"

#define kOfxImageComponentRGBA "OfxImageComponentRGBA"
#define kOfxImageComponentRGB "OfxImageComponentRGB"
#define kOfxImageComponentAlpha "OfxImageComponentAlpha"

#define kOfxImageClipOutput "Output"
#define kOfxImageClipSource "Source"

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

#ifdef __cplusplus
}
#endif
