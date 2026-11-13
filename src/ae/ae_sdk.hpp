// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <cstddef>

#if defined(_WIN32) || defined(__WIN32__)
#define PF_EXPORT __declspec(dllexport)
#else
#define PF_EXPORT __attribute__((visibility("default")))
#endif

namespace librenoise::ae {

using PF_Err = int32_t;
using PF_Boolean = uint8_t;
using PF_FpLong = double;
using PF_FpShort = float;
using PF_Fixed = int32_t;

struct PF_Point {
    int32_t h;
    int32_t v;
};

// Standard Error Codes
constexpr PF_Err PF_Err_NONE = 0;
constexpr PF_Err PF_Err_OUT_OF_MEMORY = 512;
constexpr PF_Err PF_Err_INTERNAL_STRUCT_DAMAGED = 513;
constexpr PF_Err PF_Err_INVALID_INDEX = 514;
constexpr PF_Err PF_Err_UNRECOGNIZED_PARAM_TYPE = 515;
constexpr PF_Err PF_Err_BAD_CALLBACK_PARAM = 517;

// Command Codes
enum PF_Cmd : int32_t {
    PF_Cmd_ABOUT = 0,
    PF_Cmd_GLOBAL_SETUP = 1,
    PF_Cmd_GLOBAL_SETDOWN = 2,
    PF_Cmd_PARAMS_SETUP = 3,
    PF_Cmd_SEQUENCE_SETUP = 4,
    PF_Cmd_SEQUENCE_SETDOWN = 5,
    PF_Cmd_SEQUENCE_RESETUP = 6,
    PF_Cmd_FRAME_SETUP = 7,
    PF_Cmd_RENDER = 8,
    PF_Cmd_FRAME_SETDOWN = 9,
    PF_Cmd_USER_CHANGED_PARAM = 10,
    PF_Cmd_UPDATE_PARAMS_UI = 11,
    PF_Cmd_EVENT = 14,
    PF_Cmd_DO_DIALOG = 15,
    PF_Cmd_SMART_PRE_RENDER = 44,
    PF_Cmd_SMART_RENDER = 45
};

// Parameter Types
enum PF_ParamType : int32_t {
    PF_Param_RESERVED = 0,
    PF_Param_CHECKBOX = 1,
    PF_Param_SLIDER = 2,
    PF_Param_ANGLE = 3,
    PF_Param_COLOR = 4,
    PF_Param_POINT = 5,
    PF_Param_POPUP = 6,
    PF_Param_GROUP_START = 8,
    PF_Param_GROUP_END = 9,
    PF_Param_BUTTON = 10,
    PF_Param_FLOAT_SLIDER = 11,
    PF_Param_POINT_3D = 12
};

// OutFlags
constexpr int32_t PF_OutFlag_NONE = 0;
constexpr int32_t PF_OutFlag_PIX_INDEPENDENT = (1 << 1);
constexpr int32_t PF_OutFlag_DEEP_COLOR_AWARE = (1 << 3);
constexpr int32_t PF_OutFlag_CUSTOM_UI = (1 << 6);
constexpr int32_t PF_OutFlag_SEND_DO_DIALOG = (1 << 15);
constexpr int32_t PF_OutFlag2_FLOAT_COLOR_AWARE = (1 << 0);
constexpr int32_t PF_OutFlag2_SUPPORTS_SMART_RENDER = (1 << 2);
constexpr int32_t PF_OutFlag2_SUPPORTS_GPU = (1 << 18);

struct PF_Pixel8 {
    uint8_t alpha;
    uint8_t red;
    uint8_t green;
    uint8_t blue;
};

struct PF_Pixel16 {
    uint16_t alpha;
    uint16_t red;
    uint16_t green;
    uint16_t blue;
};

struct PF_PixelFloat {
    float alpha;
    float red;
    float green;
    float blue;
};

struct PF_EffectWorld {
    int32_t width;
    int32_t height;
    int32_t rowbytes;
    void* data;
};

using PF_LayerDef = PF_EffectWorld;

struct PF_InData {
    int16_t appl_id;
    int16_t version_major;
    int16_t version_minor;
    int32_t current_time;
    int32_t time_scale;
    int32_t time_step;
    int32_t width;
    int32_t height;
    PF_Point origin;
    void* pica_basic_suite;
    void* sequence_data;
    void* global_data;
};

struct PF_OutData {
    int32_t my_version;
    int32_t out_flags;
    int32_t out_flags2;
    int32_t num_params;
    void* sequence_data;
    void* global_data;
    char return_msg[256];
};

struct PF_ParamDef {
    PF_ParamType param_type;
    char name[32];
    int32_t flags;
    union {
        struct {
            int32_t value;
            int32_t default_val;
        } button_d;
        struct {
            int32_t value;
            int32_t default_val;
            int32_t valid_min;
            int32_t valid_max;
            int32_t slider_min;
            int32_t slider_max;
        } sd;
        struct {
            double value;
            double default_val;
            double valid_min;
            double valid_max;
            double slider_min;
            double slider_max;
            int32_t precision;
        } fs_d;
        struct {
            int32_t value;
            int32_t default_val;
            int32_t num_choices;
            const char** names;
        } pd;
        struct {
            double x_val;
            double y_val;
            double default_x;
            double default_y;
        } td;
    } u;
};

// Event structures for Custom UI & Comp Viewer Overlay
enum PF_EventCode : int32_t {
    PF_Event_DRAW = 1,
    PF_Event_PEN_DOWN = 2,
    PF_Event_PEN_UP = 3,
    PF_Event_PEN_DRAG = 4,
    PF_Event_KEY_DOWN = 5
};

struct PF_EventExtra {
    PF_EventCode event_code;
    PF_Point screen_point;
    bool handled;
};

} // namespace librenoise::ae
