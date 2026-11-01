#include "ofx_interact.hpp"
#include "openfx/ofxImageEffect.h"

#include <cmath>
#include <algorithm>
#include <cstring>
#include <string>

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
    double orig_box[4] = {0.0, 0.0, 0.0, 0.0}; // x1, y1, x2, y2
};

// Hit-test distance threshold in screen pixels
constexpr double kHandleRadius = 10.0;

bool is_near(double x1, double y1, double x2, double y2, double radius) {
    double dx = x1 - x2;
    double dy = y1 - y2;
    return (dx * dx + dy * dy) <= (radius * radius);
}

} // anonymous namespace

OfxStatus libre_noise_overlay_interact_entry(
    const char* action,
    const void* handle,
    OfxPropertySetHandle inArgs,
    OfxPropertySetHandle outArgs
) {
    (void)outArgs;
    auto interact = reinterpret_cast<OfxInteractHandle>(const_cast<void*>(handle));

    extern const OfxPropertySuiteV1* g_propSuite;
    extern const OfxParameterSuiteV1* g_paramSuite;
    extern const OfxImageEffectSuiteV1* g_effectSuite;

    if (!g_propSuite || !g_paramSuite) {
        return kOfxStatReplyDefault;
    }

    if (std::strcmp(action, kOfxActionDescribe) == 0) {
        OfxPropertySetHandle interactProps = nullptr;
        // In describe, handle is interact descriptor property set
        interactProps = reinterpret_cast<OfxPropertySetHandle>(const_cast<void*>(handle));
        if (interactProps) {
            g_propSuite->propSetString(interactProps, kOfxInteractPropSlaveToParam, 0, kParamAnalysisBoxCorner1);
            g_propSuite->propSetString(interactProps, kOfxInteractPropSlaveToParam, 1, kParamAnalysisBoxCorner2);
        }
        return kOfxStatOK;
    }

    if (std::strcmp(action, kOfxActionCreateInstance) == 0) {
        OfxPropertySetHandle interactProps = nullptr;
        // Interact instance props
        interactProps = reinterpret_cast<OfxPropertySetHandle>(const_cast<void*>(handle));
        auto* state = new OverlayState();
        g_propSuite->propSetPointer(interactProps, kOfxPropInstanceData, 0, state);
        return kOfxStatOK;
    }

    if (std::strcmp(action, kOfxActionDestroyInstance) == 0) {
        OfxPropertySetHandle interactProps = reinterpret_cast<OfxPropertySetHandle>(const_cast<void*>(handle));
        void* data = nullptr;
        g_propSuite->propGetPointer(interactProps, kOfxPropInstanceData, 0, &data);
        delete reinterpret_cast<OverlayState*>(data);
        return kOfxStatOK;
    }

    // Retrieve instance data and effect handle
    OfxPropertySetHandle interactProps = reinterpret_cast<OfxPropertySetHandle>(const_cast<void*>(handle));
    void* data = nullptr;
    g_propSuite->propGetPointer(interactProps, kOfxPropInstanceData, 0, &data);
    auto* state = reinterpret_cast<OverlayState*>(data);

    void* effectPtr = nullptr;
    g_propSuite->propGetPointer(interactProps, kOfxPropEffectInstance, 0, &effectPtr);
    auto effect = reinterpret_cast<OfxImageEffectHandle>(effectPtr);

    OfxParamSetHandle paramSet = nullptr;
    if (effect && g_effectSuite) {
        g_effectSuite->getParamSet(effect, &paramSet);
    }

    double renderTime = 0.0;
    if (inArgs) {
        g_propSuite->propGetDouble(inArgs, kOfxPropTime, 0, &renderTime);
    }

    // Read current box coordinates
    double box[4] = {100.0, 100.0, 300.0, 300.0};
    OfxParamHandle boxParam1 = nullptr;
    OfxParamHandle boxParam2 = nullptr;
    if (paramSet) {
        g_paramSuite->paramGetHandle(paramSet, kParamAnalysisBoxCorner1, &boxParam1, nullptr);
        if (boxParam1) {
            g_paramSuite->paramGetValueAtTime(boxParam1, renderTime, &box[0], &box[1]);
        }
        g_paramSuite->paramGetHandle(paramSet, kParamAnalysisBoxCorner2, &boxParam2, nullptr);
        if (boxParam2) {
            g_paramSuite->paramGetValueAtTime(boxParam2, renderTime, &box[2], &box[3]);
        }
    }

    // Ensure canonical ordering (x1 < x2, y1 < y2)
    double min_x = std::min(box[0], box[2]);
    double max_x = std::max(box[0], box[2]);
    double min_y = std::min(box[1], box[3]);
    double max_y = std::max(box[1], box[3]);

    if (std::strcmp(action, kOfxInteractActionDraw) == 0) {
        // Draw the interactive noise profile analysis box
        glPushAttrib(GL_ALL_ATTRIB_BITS);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        // Semi-transparent interior fill
        glColor4f(0.0f, 0.9f, 0.4f, 0.12f);
        glBegin(GL_QUADS);
        glVertex2d(min_x, min_y);
        glVertex2d(max_x, min_y);
        glVertex2d(max_x, max_y);
        glVertex2d(min_x, max_y);
        glEnd();

        // Vibrant neon green border
        glLineWidth(2.0f);
        glColor4f(0.0f, 1.0f, 0.5f, 0.95f);
        glBegin(GL_LINE_LOOP);
        glVertex2d(min_x, min_y);
        glVertex2d(max_x, min_y);
        glVertex2d(max_x, max_y);
        glVertex2d(min_x, max_y);
        glEnd();

        // Corner resize handles
        const double hs = 6.0; // handle half-size
        auto draw_handle = [&](double cx, double cy) {
            // White handle with green outline
            glColor4f(1.0f, 1.0f, 1.0f, 0.95f);
            glBegin(GL_QUADS);
            glVertex2d(cx - hs, cy - hs);
            glVertex2d(cx + hs, cy - hs);
            glVertex2d(cx + hs, cy + hs);
            glVertex2d(cx - hs, cy + hs);
            glEnd();

            glColor4f(0.0f, 0.4f, 0.2f, 1.0f);
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
        glColor4f(0.0f, 1.0f, 0.5f, 0.8f);
        glBegin(GL_LINES);
        glVertex2d(mid_x - 8.0, mid_y);
        glVertex2d(mid_x + 8.0, mid_y);
        glVertex2d(mid_x, mid_y - 8.0);
        glVertex2d(mid_x, mid_y + 8.0);
        glEnd();

        // Top HUD Header bar
        glColor4f(0.0f, 0.6f, 0.3f, 0.85f);
        glBegin(GL_QUADS);
        glVertex2d(min_x, max_y);
        glVertex2d(max_x, max_y);
        glVertex2d(max_x, max_y + 16.0);
        glVertex2d(min_x, max_y + 16.0);
        glEnd();

        glPopAttrib();
        return kOfxStatOK;
    }

    if (std::strcmp(action, kOfxInteractActionPenDown) == 0 && state && inArgs) {
        double pen[2] = {0.0, 0.0};
        g_propSuite->propGetDoubleN(inArgs, kOfxInteractPropPenPosition, 2, pen);

        state->start_pen_x = pen[0];
        state->start_pen_y = pen[1];
        state->orig_box[0] = min_x;
        state->orig_box[1] = min_y;
        state->orig_box[2] = max_x;
        state->orig_box[3] = max_y;

        // Hit-test corner handles
        if (is_near(pen[0], pen[1], min_x, min_y, kHandleRadius)) {
            state->drag_mode = DragMode::ResizeBL;
        } else if (is_near(pen[0], pen[1], max_x, min_y, kHandleRadius)) {
            state->drag_mode = DragMode::ResizeBR;
        } else if (is_near(pen[0], pen[1], min_x, max_y, kHandleRadius)) {
            state->drag_mode = DragMode::ResizeTL;
        } else if (is_near(pen[0], pen[1], max_x, max_y, kHandleRadius)) {
            state->drag_mode = DragMode::ResizeTR;
        } else if (pen[0] >= min_x && pen[0] <= max_x && pen[1] >= min_y && pen[1] <= max_y) {
            state->drag_mode = DragMode::Move;
        } else {
            // Click outside starts drawing a new box
            state->drag_mode = DragMode::CreateNew;
            state->orig_box[0] = pen[0];
            state->orig_box[1] = pen[1];
            state->orig_box[2] = pen[0];
            state->orig_box[3] = pen[1];
        }

        return kOfxStatOK;
    }

    if (std::strcmp(action, kOfxInteractActionPenMotion) == 0 && state && inArgs) {
        if (state->drag_mode == DragMode::None) return kOfxStatReplyDefault;

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

        if (g_paramSuite) {
            if (boxParam1) g_paramSuite->paramSetValue(boxParam1, nbox[0], nbox[1]);
            if (boxParam2) g_paramSuite->paramSetValue(boxParam2, nbox[2], nbox[3]);
        }

        return kOfxStatOK;
    }

    if (std::strcmp(action, kOfxInteractActionPenUp) == 0 && state) {
        state->drag_mode = DragMode::None;
        return kOfxStatOK;
    }

    return kOfxStatReplyDefault;
}
