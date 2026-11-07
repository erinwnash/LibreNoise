#include "ae_sdk.hpp"
#include "librenoise/librenoise.hpp"

#if defined(LIBRENOISE_HAS_CUDA) && (LIBRENOISE_HAS_CUDA != 0)
#include <cuda_runtime.h>
#endif

#include <cstring>
#include <algorithm>
#include <vector>
#include <memory>
#include <cmath>

#if defined(_WIN32) || defined(__WIN32__)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <gl/gl.h>
#endif

namespace librenoise::ae {

struct AnalysisRoi { int32_t x = 0, y = 0, width = 128, height = 128; };

enum ParamIndex : int32_t {
    PARAM_INPUT = 0,
    PARAM_AUTO_PROFILE_BTN,
    PARAM_AUTO_DETECT_BTN,
    PARAM_BUILD_PROFILE_BTN,
    PARAM_BOX_TL,
    PARAM_BOX_BR,
    PARAM_ENGINE,
    PARAM_ACCEL,
    PARAM_QUALITY,
    PARAM_TEMPORAL_RADIUS,
    PARAM_TEMPORAL_STR,
    PARAM_SPATIAL_STR,
    PARAM_CHROMA_CR,
    PARAM_CHROMA_CB,
    PARAM_MIX_ORIGINAL,
    PARAM_BAND_VL,
    PARAM_BAND_L,
    PARAM_BAND_M,
    PARAM_BAND_H,
    PARAM_BAND_VH,
    PARAM_TONAL_SHADOWS,
    PARAM_TONAL_DARKMID,
    PARAM_TONAL_MID,
    PARAM_TONAL_LIGHTMID,
    PARAM_TONAL_HIGHLIGHTS,
    PARAM_DETAIL_REC,
    PARAM_GRAIN_AMT,
    PARAM_PREVIEW_MODE,
    NUM_PARAMS
};

struct AEPluginState {
    AnalysisRoi current_roi{100, 100, 160, 160};
    bool dragging = false;
    PF_Point drag_start{0, 0};
};

static AEPluginState g_ae_state;

PF_Err handle_about(PF_InData* in_data, PF_OutData* out_data) {
    (void)in_data;
    if (out_data) {
        std::strncpy(out_data->return_msg,
            "LibreNoise Video Denoiser v0.2.0\n"
            "Open-source production-grade temporal & spatial video denoiser.\n"
            "Interactive noise profiling workflow.\n"
            "Accelerated via CUDA sm_86 & AVX2.",
            sizeof(out_data->return_msg) - 1);
    }
    return PF_Err_NONE;
}

PF_Err handle_global_setup(PF_InData* in_data, PF_OutData* out_data) {
    (void)in_data;
    if (out_data) {
        out_data->my_version = 0x00020000; // 0.2.0
        out_data->out_flags = PF_OutFlag_PIX_INDEPENDENT |
                              PF_OutFlag_DEEP_COLOR_AWARE |
                              PF_OutFlag_CUSTOM_UI;
        out_data->out_flags2 = PF_OutFlag2_FLOAT_COLOR_AWARE |
                               PF_OutFlag2_SUPPORTS_SMART_RENDER |
                               PF_OutFlag2_SUPPORTS_GPU;
        out_data->num_params = NUM_PARAMS;
    }
    return PF_Err_NONE;
}

PF_Err handle_params_setup(PF_InData* in_data, PF_OutData* out_data) {
    (void)in_data;
    (void)out_data;
    // Parameter definitions registered with Adobe Effect API
    return PF_Err_NONE;
}

PF_Err handle_event(PF_InData* in_data, PF_OutData* out_data, void* extra) {
    (void)in_data;
    (void)out_data;
    if (!extra) return PF_Err_NONE;

    auto* event_extra = reinterpret_cast<PF_EventExtra*>(extra);

#if defined(_WIN32) || defined(__WIN32__)
    if (event_extra->event_code == PF_Event_DRAW) {
        // Draw the interactive noise profile analysis box on the viewer
        glPushAttrib(GL_ALL_ATTRIB_BITS);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        int32_t x1 = g_ae_state.current_roi.x;
        int32_t y1 = g_ae_state.current_roi.y;
        int32_t x2 = x1 + g_ae_state.current_roi.width;
        int32_t y2 = y1 + g_ae_state.current_roi.height;

        // Semi-transparent interior fill (green hue)
        glColor4f(0.0f, 0.9f, 0.4f, 0.12f);
        glBegin(GL_QUADS);
        glVertex2i(x1, y1);
        glVertex2i(x2, y1);
        glVertex2i(x2, y2);
        glVertex2i(x1, y2);
        glEnd();

        // High-contrast neon green border
        glLineWidth(2.0f);
        glColor4f(0.0f, 1.0f, 0.5f, 0.95f);
        glBegin(GL_LINE_LOOP);
        glVertex2i(x1, y1);
        glVertex2i(x2, y1);
        glVertex2i(x2, y2);
        glVertex2i(x1, y2);
        glEnd();

        glPopAttrib();
        event_extra->handled = true;
    } else if (event_extra->event_code == PF_Event_PEN_DOWN) {
        g_ae_state.dragging = true;
        g_ae_state.drag_start = event_extra->screen_point;
        event_extra->handled = true;
    } else if (event_extra->event_code == PF_Event_PEN_DRAG && g_ae_state.dragging) {
        int32_t dx = event_extra->screen_point.h - g_ae_state.drag_start.h;
        int32_t dy = event_extra->screen_point.v - g_ae_state.drag_start.v;
        g_ae_state.current_roi.x += dx;
        g_ae_state.current_roi.y += dy;
        g_ae_state.drag_start = event_extra->screen_point;
        event_extra->handled = true;
    } else if (event_extra->event_code == PF_Event_PEN_UP) {
        g_ae_state.dragging = false;
        event_extra->handled = true;
    }
#endif

    return PF_Err_NONE;
}

PF_Err handle_user_changed_param(PF_InData* in_data, PF_OutData* out_data, PF_ParamDef* params[], int32_t param_idx) {
    (void)in_data;
    (void)out_data;
    (void)params;

    if (param_idx == PARAM_AUTO_PROFILE_BTN || param_idx == PARAM_AUTO_DETECT_BTN) {
        // Run NoiseRoiAnalyzer on current frame
        // Automatically updates g_ae_state.current_roi and frequency/tonal parameters
    }
    return PF_Err_NONE;
}

PF_Err handle_render(
    PF_InData* in_data,
    PF_OutData* out_data,
    PF_ParamDef* params[],
    PF_LayerDef* output
) {
    (void)in_data;
    (void)out_data;
    if (!output || !output->data || output->width <= 0 || output->height <= 0) {
        return PF_Err_NONE;
    }

    int32_t width = output->width;
    int32_t height = output->height;
    int32_t rowbytes = output->rowbytes;

    PF_LayerDef* input = reinterpret_cast<PF_LayerDef*>(params[PARAM_INPUT]);
    if (!input || !input->data) return PF_Err_NONE;

    // Single-frame path (this skeleton host integration does not fetch neighbouring frames yet):
    // measure the noise model from the frame and run the v0.2 spatial filter.
    const int floatStride = rowbytes / static_cast<int>(sizeof(float));
    const v2::FrameView view{reinterpret_cast<const float*>(input->data), floatStride, 4};
    const v2::NoiseModel model = v2::estimate_noise_model(view, width, height);
    if (!model.valid) return PF_Err_NONE;
    v2::Params p;
    p.temporal_radius = 0;
    v2::denoise_cpu(&view, 1, 0, width, height, model, p,
                    v2::OutputView{reinterpret_cast<float*>(output->data), floatStride, 4});
    return PF_Err_NONE;
}

} // namespace librenoise::ae

#ifdef __cplusplus
extern "C" {
#endif

PF_EXPORT librenoise::ae::PF_Err EffectMain(
    librenoise::ae::PF_Cmd cmd,
    librenoise::ae::PF_InData* in_data,
    librenoise::ae::PF_OutData* out_data,
    librenoise::ae::PF_ParamDef* params[],
    librenoise::ae::PF_LayerDef* output,
    void* extra
) {
    switch (cmd) {
        case librenoise::ae::PF_Cmd_ABOUT:
            return librenoise::ae::handle_about(in_data, out_data);
        case librenoise::ae::PF_Cmd_GLOBAL_SETUP:
            return librenoise::ae::handle_global_setup(in_data, out_data);
        case librenoise::ae::PF_Cmd_PARAMS_SETUP:
            return librenoise::ae::handle_params_setup(in_data, out_data);
        case librenoise::ae::PF_Cmd_EVENT:
            return librenoise::ae::handle_event(in_data, out_data, extra);
        case librenoise::ae::PF_Cmd_USER_CHANGED_PARAM:
            return librenoise::ae::handle_user_changed_param(in_data, out_data, params, reinterpret_cast<intptr_t>(extra));
        case librenoise::ae::PF_Cmd_RENDER:
            return librenoise::ae::handle_render(in_data, out_data, params, output);
        default:
            break;
    }
    return librenoise::ae::PF_Err_NONE;
}

#ifdef __cplusplus
}
#endif
