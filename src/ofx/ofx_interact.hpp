#pragma once

#include "openfx/ofxCore.h"
#include "openfx/ofxProperty.h"
#include "openfx/ofxParam.h"
#include "openfx/ofxInteract.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief OpenFX Viewer Overlay Interact Entry Point for LibreNoise.
 * Handles interactive box drawing, corner dragging, and live ROI manipulation on the viewer.
 */
OfxStatus libre_noise_overlay_interact_entry(
    const char* action,
    const void* handle,
    OfxPropertySetHandle inArgs,
    OfxPropertySetHandle outArgs
);

#ifdef __cplusplus
}
#endif
