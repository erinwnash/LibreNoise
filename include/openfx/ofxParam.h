#pragma once

#include "ofxCore.h"

#ifdef __cplusplus
extern "C" {
#endif

#define kOfxParameterSuite "OfxParameterSuite"

#define kOfxParamTypeInteger "OfxParamTypeInteger"
#define kOfxParamTypeDouble "OfxParamTypeDouble"
#define kOfxParamTypeBoolean "OfxParamTypeBoolean"
#define kOfxParamTypeChoice "OfxParamTypeChoice"
#define kOfxParamTypeRGBA "OfxParamTypeRGBA"
#define kOfxParamTypeRGB "OfxParamTypeRGB"
#define kOfxParamTypeDouble2D "OfxParamTypeDouble2D"
#define kOfxParamTypeInteger2D "OfxParamTypeInteger2D"
#define kOfxParamTypeDouble3D "OfxParamTypeDouble3D"
#define kOfxParamTypeInteger3D "OfxParamTypeInteger3D"
#define kOfxParamTypeString "OfxParamTypeString"
#define kOfxParamTypeCustom "OfxParamTypeCustom"
#define kOfxParamTypeGroup "OfxParamTypeGroup"
#define kOfxParamTypePage "OfxParamTypePage"
#define kOfxParamTypePushButton "OfxParamTypePushButton"

#define kOfxParamPropDefault "OfxParamPropDefault"
#define kOfxParamPropMin "OfxParamPropMin"
#define kOfxParamPropMax "OfxParamPropMax"
#define kOfxParamPropDisplayMin "OfxParamPropDisplayMin"
#define kOfxParamPropDisplayMax "OfxParamPropDisplayMax"
#define kOfxParamPropDigits "OfxParamPropDigits"
#define kOfxParamPropChoiceOption "OfxParamPropChoiceOption"
#define kOfxParamPropPageChild "OfxParamPropPageChild"
#define kOfxParamPropParent "OfxParamPropParent"

typedef struct OfxParameterSuiteV1 {
    OfxStatus (*paramDefine)(
        OfxParamSetHandle paramSet,
        const char* paramType,
        const char* name,
        OfxPropertySetHandle* propertySet
    );

    OfxStatus (*paramGetHandle)(
        OfxParamSetHandle paramSet,
        const char* name,
        OfxParamHandle* param,
        OfxPropertySetHandle* propertySet
    );

    OfxStatus (*paramSetGetPropertySet)(
        OfxParamSetHandle paramSet,
        OfxPropertySetHandle* propHandle
    );

    OfxStatus (*paramGetPropertySet)(
        OfxParamHandle param,
        OfxPropertySetHandle* propHandle
    );

    OfxStatus (*paramGetValue)(
        OfxParamHandle paramHandle,
        ...
    );

    OfxStatus (*paramGetValueAtTime)(
        OfxParamHandle paramHandle,
        double time,
        ...
    );

    OfxStatus (*paramGetDerivative)(
        OfxParamHandle paramHandle,
        double time,
        ...
    );

    OfxStatus (*paramGetIntegral)(
        OfxParamHandle paramHandle,
        double time1,
        double time2,
        ...
    );

    OfxStatus (*paramSetValue)(
        OfxParamHandle paramHandle,
        ...
    );

    OfxStatus (*paramSetValueAtTime)(
        OfxParamHandle paramHandle,
        double time,
        ...
    );

    OfxStatus (*paramGetNumKeys)(
        OfxParamHandle paramHandle,
        unsigned int* numberOfKeys
    );

    OfxStatus (*paramGetKeyTime)(
        OfxParamHandle paramHandle,
        unsigned int nthKey,
        double* time
    );

    OfxStatus (*paramGetKeyIndex)(
        OfxParamHandle paramHandle,
        double time,
        int direction,
        int* index
    );

    OfxStatus (*paramDeleteKey)(
        OfxParamHandle paramHandle,
        double time
    );

    OfxStatus (*paramDeleteAllKeys)(
        OfxParamHandle paramHandle
    );

    OfxStatus (*paramCopy)(
        OfxParamHandle paramTo,
        OfxParamHandle paramFrom,
        double dstOffset,
        const struct OfxRangeD* frameRange
    );

    OfxStatus (*paramEditBegin)(
        OfxParamSetHandle paramSet,
        const char* name
    );

    OfxStatus (*paramEditEnd)(
        OfxParamSetHandle paramSet
    );
} OfxParameterSuiteV1;

#ifdef __cplusplus
}
#endif
