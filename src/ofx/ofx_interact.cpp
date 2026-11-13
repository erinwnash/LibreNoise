// SPDX-License-Identifier: Apache-2.0
#include "ofx_interact.hpp"
#include "openfx/ofxCore.h"
#include "openfx/ofxProperty.h"
#include "openfx/ofxParam.h"
#include "openfx/ofxImageEffect.h"
#include "openfx/ofxInteract.h"

#include <cmath>
#include <algorithm>
#include <cstring>
#include <string>
#include <unordered_map>

#if defined(_WIN32) || defined(__WIN32__)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <gl/gl.h>
#elif defined(__APPLE__)
#include <OpenGL/gl.h>
#else
#include <GL/gl.h>
#endif

// External suite pointers defined in plugin_main.cpp
extern const OfxPropertySuiteV1* g_propSuite;
extern const OfxParameterSuiteV1* g_paramSuite;
extern const OfxImageEffectSuiteV1* g_effectSuite;
extern const OfxInteractSuiteV1* g_interactSuite;

namespace {

// Parameter names matching plugin_main.cpp
constexpr const char* kParamAnalysisBoxCorner1 = "analysisBoxCorner1";
constexpr const char* kParamAnalysisBoxCorner2 = "analysisBoxCorner2";

enum class DragMode {
    None,
    Move,
    ResizeTL,
    ResizeTR,
    ResizeBL,
    ResizeBR,
    CreateNew
};

struct OverlayState {
    DragMode drag_mode = DragMode::None;
    double start_pen_x = 0.0;
    double start_pen_y = 0.0;
    double orig_box[4] = {100.0, 100.0, 260.0, 260.0}; // x1, y1, x2, y2
};

// Thread-safe fallback registry for hosts where interactGetPropertySet or kOfxPropInstanceData is unsupported
#if defined(_WIN32) || defined(__WIN32__)
static SRWLOCK s_stateLock = SRWLOCK_INIT;
static void acquire_state_lock() noexcept { AcquireSRWLockExclusive(&s_stateLock); }
static void release_state_lock() noexcept { ReleaseSRWLockExclusive(&s_stateLock); }
#else
static pthread_mutex_t s_stateLock = PTHREAD_MUTEX_INITIALIZER;
static void acquire_state_lock() noexcept { pthread_mutex_lock(&s_stateLock); }
static void release_state_lock() noexcept { pthread_mutex_unlock(&s_stateLock); }
#endif

static std::unordered_map<OfxInteractHandle, OverlayState*> s_interactStates;

static OfxPropertySetHandle get_interact_props(OfxInteractHandle interact) noexcept {
    if (!interact) return nullptr;
    OfxPropertySetHandle props = nullptr;
    if (g_interactSuite && g_interactSuite->interactGetPropertySet) {
        if (g_interactSuite->interactGetPropertySet(interact, &props) == kOfxStatOK && props) {
            return props;
        }
    }
    // Fallback: some hosts treat OfxInteractHandle directly as property set
    return reinterpret_cast<OfxPropertySetHandle>(interact);
}

static OverlayState* get_or_create_state(OfxInteractHandle interact) {
    if (!interact) return nullptr;

    // 1. Try host property set
    OfxPropertySetHandle props = get_interact_props(interact);
    if (props && g_propSuite) {
        void* data = nullptr;
        if (g_propSuite->propGetPointer(props, kOfxPropInstanceData, 0, &data) == kOfxStatOK && data) {
            return reinterpret_cast<OverlayState*>(data);
        }
    }

    // 2. Fallback to thread-safe internal registry
    acquire_state_lock();
    auto it = s_interactStates.find(interact);
    OverlayState* state = (it != s_interactStates.end()) ? it->second : nullptr;
    if (!state) {
        state = new OverlayState();
        s_interactStates[interact] = state;
        if (props && g_propSuite) {
            g_propSuite->propSetPointer(props, kOfxPropInstanceData, 0, state);
        }
    }
    release_state_lock();
    return state;
}

static OfxImageEffectHandle get_effect_handle(OfxInteractHandle interact, OfxPropertySetHandle inArgs) noexcept {
    void* effectPtr = nullptr;
    if (inArgs && g_propSuite) {
        if (g_propSuite->propGetPointer(inArgs, kOfxPropEffectInstance, 0, &effectPtr) == kOfxStatOK && effectPtr) {
            return reinterpret_cast<OfxImageEffectHandle>(effectPtr);
        }
    }
    OfxPropertySetHandle props = get_interact_props(interact);
    if (props && g_propSuite) {
        if (g_propSuite->propGetPointer(props, kOfxPropEffectInstance, 0, &effectPtr) == kOfxStatOK && effectPtr) {
            return reinterpret_cast<OfxImageEffectHandle>(effectPtr);
        }
    }
    return nullptr;
}

inline bool is_near(double x1, double y1, double x2, double y2, double radius) noexcept {
    double dx = x1 - x2;
    double dy = y1 - y2;
    return (dx * dx + dy * dy) <= (radius * radius);
}

} // anonymous namespace

extern "C" OfxStatus libre_noise_overlay_interact_entry(
    const char* action,
    const void* handle,
    OfxPropertySetHandle inArgs,
    OfxPropertySetHandle outArgs
) {
    (void)outArgs;
    auto interact = reinterpret_cast<OfxInteractHandle>(const_cast<void*>(handle));

    if (!g_propSuite || !g_paramSuite) {
        return kOfxStatReplyDefault;
    }

    // 1. Describe Interact
    if (std::strcmp(action, kOfxActionDescribe) == 0) {
        OfxPropertySetHandle descProps = get_interact_props(interact);
        if (descProps) {
            g_propSuite->propSetString(descProps, kOfxInteractPropSlaveToParam, 0, kParamAnalysisBoxCorner1);
            g_propSuite->propSetString(descProps, kOfxInteractPropSlaveToParam, 1, kParamAnalysisBoxCorner2);
        }
        return kOfxStatOK;
    }

    // 2. Create Instance
    if (std::strcmp(action, kOfxActionCreateInstance) == 0) {
        auto* state = new OverlayState();
        acquire_state_lock();
        s_interactStates[interact] = state;
        release_state_lock();

        OfxPropertySetHandle props = get_interact_props(interact);
        if (props) {
            g_propSuite->propSetPointer(props, kOfxPropInstanceData, 0, state);
        }
        return kOfxStatOK;
    }

    // 3. Destroy Instance
    if (std::strcmp(action, kOfxActionDestroyInstance) == 0) {
        OverlayState* state = nullptr;
        acquire_state_lock();
        auto it = s_interactStates.find(interact);
        if (it != s_interactStates.end()) {
            state = it->second;
            s_interactStates.erase(it);
        }
        release_state_lock();

        if (!state) {
            OfxPropertySetHandle props = get_interact_props(interact);
            void* data = nullptr;
            if (props && g_propSuite->propGetPointer(props, kOfxPropInstanceData, 0, &data) == kOfxStatOK) {
                state = reinterpret_cast<OverlayState*>(data);
            }
        }
        delete state;
        return kOfxStatOK;
    }

    // 4. Focus Events
    if (std::strcmp(action, kOfxInteractActionGainFocus) == 0 ||
        std::strcmp(action, kOfxInteractActionLoseFocus) == 0) {
        return kOfxStatOK;
    }

    // Retrieve state & effect instance
    OverlayState* state = get_or_create_state(interact);
    auto effect = get_effect_handle(interact, inArgs);

    OfxParamSetHandle paramSet = nullptr;
    if (effect && g_effectSuite) {
        g_effectSuite->getParamSet(effect, &paramSet);
    }

    double renderTime = 0.0;
    if (inArgs) {
        g_propSuite->propGetDouble(inArgs, kOfxPropTime, 0, &renderTime);
    }

    // Read current box coordinates from parameters
    double box[4] = {100.0, 100.0, 260.0, 260.0};
    OfxParamHandle boxParam1 = nullptr;
    OfxParamHandle boxParam2 = nullptr;
    if (paramSet) {
        if (g_paramSuite->paramGetHandle(paramSet, kParamAnalysisBoxCorner1, &boxParam1, nullptr) == kOfxStatOK && boxParam1) {
            g_paramSuite->paramGetValueAtTime(boxParam1, renderTime, &box[0], &box[1]);
        }
        if (g_paramSuite->paramGetHandle(paramSet, kParamAnalysisBoxCorner2, &boxParam2, nullptr) == kOfxStatOK && boxParam2) {
            g_paramSuite->paramGetValueAtTime(boxParam2, renderTime, &box[2], &box[3]);
        }
    }

    // Canonical ordering
    double min_x = std::min(box[0], box[2]);
    double max_x = std::max(box[0], box[2]);
    double min_y = std::min(box[1], box[3]);
    double max_y = std::max(box[1], box[3]);

    // Viewport pixel scale (canonical pixel size relative to screen pixels)
    double pixelScale[2] = {1.0, 1.0};
    if (inArgs) {
        g_propSuite->propGetDoubleN(inArgs, kOfxInteractPropPixelScale, 2, pixelScale);
    }
    if (pixelScale[0] <= 0.0) pixelScale[0] = 1.0;
    if (pixelScale[1] <= 0.0) pixelScale[1] = pixelScale[0];

    // =========================================================================
    // Action: Draw Overlay
    // =========================================================================
    if (std::strcmp(action, kOfxInteractActionDraw) == 0) {
        glPushAttrib(GL_ALL_ATTRIB_BITS);
        glDisable(GL_DEPTH_TEST);
        glDisable(GL_TEXTURE_2D);
#ifdef GL_TEXTURE_RECTANGLE_ARB
        glDisable(GL_TEXTURE_RECTANGLE_ARB);
#endif
        glDisable(GL_LIGHTING);
        glDisable(GL_CULL_FACE);

        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        // Semi-transparent neon green fill
        glColor4f(0.0f, 0.95f, 0.45f, 0.12f);
        glBegin(GL_QUADS);
        glVertex2d(min_x, min_y);
        glVertex2d(max_x, min_y);
        glVertex2d(max_x, max_y);
        glVertex2d(min_x, max_y);
        glEnd();

        // High-visibility green border
        glLineWidth(2.0f);
        glColor4f(0.0f, 1.0f, 0.5f, 0.95f);
        glBegin(GL_LINE_LOOP);
        glVertex2d(min_x, min_y);
        glVertex2d(max_x, min_y);
        glVertex2d(max_x, max_y);
        glVertex2d(min_x, max_y);
        glEnd();

        // Corner resize handles (scaled by pixelScale so always visible regardless of zoom)
        const double hs = 6.0 * pixelScale[0];
        auto draw_handle = [&](double cx, double cy) {
            // White interior
            glColor4f(1.0f, 1.0f, 1.0f, 0.95f);
            glBegin(GL_QUADS);
            glVertex2d(cx - hs, cy - hs);
            glVertex2d(cx + hs, cy - hs);
            glVertex2d(cx + hs, cy + hs);
            glVertex2d(cx - hs, cy + hs);
            glEnd();

            // Dark green border
            glColor4f(0.0f, 0.5f, 0.25f, 1.0f);
            glLineWidth(1.5f);
            glBegin(GL_LINE_LOOP);
            glVertex2d(cx - hs, cy - hs);
            glVertex2d(cx + hs, cy - hs);
            glVertex2d(cx + hs, cy + hs);
            glVertex2d(cx - hs, cy + hs);
            glEnd();
        };

        draw_handle(min_x, min_y); // Bottom-left
        draw_handle(max_x, min_y); // Bottom-right
        draw_handle(max_x, max_y); // Top-right
        draw_handle(min_x, max_y); // Top-left

        // Center crosshair move handle
        double mid_x = (min_x + max_x) * 0.5;
        double mid_y = (min_y + max_y) * 0.5;
        double ch_arm = 8.0 * pixelScale[0];
        glColor4f(0.0f, 1.0f, 0.5f, 0.85f);
        glLineWidth(1.5f);
        glBegin(GL_LINES);
        glVertex2d(mid_x - ch_arm, mid_y);
        glVertex2d(mid_x + ch_arm, mid_y);
        glVertex2d(mid_x, mid_y - ch_arm);
        glVertex2d(mid_x, mid_y + ch_arm);
        glEnd();

        // Top HUD Header bar for easy grabbing
        double header_h = 18.0 * pixelScale[1];
        glColor4f(0.0f, 0.65f, 0.35f, 0.85f);
        glBegin(GL_QUADS);
        glVertex2d(min_x, max_y);
        glVertex2d(max_x, max_y);
        glVertex2d(max_x, max_y + header_h);
        glVertex2d(min_x, max_y + header_h);
        glEnd();

        glPopAttrib();
        return kOfxStatOK;
    }

    // =========================================================================
    // Action: Pen Down (Click & Drag initiation)
    // =========================================================================
    if (std::strcmp(action, kOfxInteractActionPenDown) == 0 && state && inArgs) {
        double pen[2] = {0.0, 0.0};
        g_propSuite->propGetDoubleN(inArgs, kOfxInteractPropPenPosition, 2, pen);

        state->start_pen_x = pen[0];
        state->start_pen_y = pen[1];
        state->orig_box[0] = min_x;
        state->orig_box[1] = min_y;
        state->orig_box[2] = max_x;
        state->orig_box[3] = max_y;

        double handleRadius = 16.0 * pixelScale[0];
        double header_h = 18.0 * pixelScale[1];

        // Hit-test corner handles with zoom-independent screen pixel tolerance
        if (is_near(pen[0], pen[1], min_x, min_y, handleRadius)) {
            state->drag_mode = DragMode::ResizeBL;
        } else if (is_near(pen[0], pen[1], max_x, min_y, handleRadius)) {
            state->drag_mode = DragMode::ResizeBR;
        } else if (is_near(pen[0], pen[1], min_x, max_y, handleRadius)) {
            state->drag_mode = DragMode::ResizeTL;
        } else if (is_near(pen[0], pen[1], max_x, max_y, handleRadius)) {
            state->drag_mode = DragMode::ResizeTR;
        } else if (pen[0] >= min_x && pen[0] <= max_x && pen[1] >= min_y && pen[1] <= (max_y + header_h)) {
            state->drag_mode = DragMode::Move;
        } else {
            // Click outside starts drawing a new ROI box
            state->drag_mode = DragMode::CreateNew;
            state->orig_box[0] = pen[0];
            state->orig_box[1] = pen[1];
            state->orig_box[2] = pen[0];
            state->orig_box[3] = pen[1];
        }

        return kOfxStatOK;
    }

    // =========================================================================
    // Action: Pen Motion (Dragging & Live Parameter Updating)
    // =========================================================================
    if (std::strcmp(action, kOfxInteractActionPenMotion) == 0 && state && inArgs) {
        if (state->drag_mode == DragMode::None) {
            return kOfxStatReplyDefault;
        }

        double pen[2] = {0.0, 0.0};
        g_propSuite->propGetDoubleN(inArgs, kOfxInteractPropPenPosition, 2, pen);

        double dx = pen[0] - state->start_pen_x;
        double dy = pen[1] - state->start_pen_y;

        double nbox[4] = {state->orig_box[0], state->orig_box[1], state->orig_box[2], state->orig_box[3]};

        switch (state->drag_mode) {
            case DragMode::Move:
                nbox[0] += dx;
                nbox[1] += dy;
                nbox[2] += dx;
                nbox[3] += dy;
                break;
            case DragMode::ResizeTL:
                nbox[0] += dx;
                nbox[3] += dy;
                break;
            case DragMode::ResizeTR:
                nbox[2] += dx;
                nbox[3] += dy;
                break;
            case DragMode::ResizeBL:
                nbox[0] += dx;
                nbox[1] += dy;
                break;
            case DragMode::ResizeBR:
                nbox[2] += dx;
                nbox[1] += dy;
                break;
            case DragMode::CreateNew:
                nbox[0] = state->start_pen_x;
                nbox[1] = state->start_pen_y;
                nbox[2] = pen[0];
                nbox[3] = pen[1];
                break;
            default:
                break;
        }

        // Live update OpenFX parameters
        if (paramSet) {
            if (!boxParam1) g_paramSuite->paramGetHandle(paramSet, kParamAnalysisBoxCorner1, &boxParam1, nullptr);
            if (!boxParam2) g_paramSuite->paramGetHandle(paramSet, kParamAnalysisBoxCorner2, &boxParam2, nullptr);
            if (boxParam1) g_paramSuite->paramSetValue(boxParam1, nbox[0], nbox[1]);
            if (boxParam2) g_paramSuite->paramSetValue(boxParam2, nbox[2], nbox[3]);
        }

        // Immediately request redraw for 60fps viewer feedback
        if (g_interactSuite && g_interactSuite->interactRedraw) {
            g_interactSuite->interactRedraw(interact);
        }

        return kOfxStatOK;
    }

    // =========================================================================
    // Action: Pen Up (Drop / Finish Drag)
    // =========================================================================
    if (std::strcmp(action, kOfxInteractActionPenUp) == 0 && state) {
        state->drag_mode = DragMode::None;
        if (g_interactSuite && g_interactSuite->interactRedraw) {
            g_interactSuite->interactRedraw(interact);
        }
        return kOfxStatOK;
    }

    return kOfxStatReplyDefault;
}
