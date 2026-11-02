#pragma once

#include "types.hpp"
#include "noise_model.hpp"
#include <array>
#include <cmath>
#include <algorithm>

namespace librenoise {

/**
 * @brief Working colour space used by every denoise engine: Rec.709-weighted Y plus
 * scaled colour-difference planes Cb = (B - Y) / 1.8556 and Cr = (R - Y) / 1.5748.
 *
 * Denoising in an opponent space decorrelates luminance noise from chrominance noise so each
 * can be profiled and reduced independently (Luma / Cb / Cr controls). The transform is linear,
 * so it is valid for scene-linear and display-referred data alike. Chroma planes may be negative.
 */
namespace colorspace {

constexpr float kWr = 0.2126f;
constexpr float kWg = 0.7152f;
constexpr float kWb = 0.0722f;
constexpr float kCbScale = 1.8556f; // 2 * (1 - Wb)
constexpr float kCrScale = 1.5748f; // 2 * (1 - Wr)

inline void rgb_to_ycc(float r, float g, float b, float& y, float& cb, float& cr) noexcept {
    y = kWr * r + kWg * g + kWb * b;
    cb = (b - y) * (1.0f / kCbScale);
    cr = (r - y) * (1.0f / kCrScale);
}

inline void ycc_to_rgb(float y, float cb, float cr, float& r, float& g, float& b) noexcept {
    r = y + kCrScale * cr;
    b = y + kCbScale * cb;
    g = (y - kWr * r - kWb * b) * (1.0f / kWg);
}

/// Converts a 3-plane RGB frame to Y/Cb/Cr planes (other channels are copied untouched).
inline VideoFrame32F to_ycc(const VideoFrame32F& rgb) {
    VideoFrame32F out = rgb;
    if (rgb.channel_count() < 3) return out;
    const int32_t w = rgb.width, h = rgb.height;
    for (int32_t y = 0; y < h; ++y) {
        for (int32_t x = 0; x < w; ++x) {
            float Y, Cb, Cr;
            rgb_to_ycc(rgb.channels[0].at(x, y), rgb.channels[1].at(x, y), rgb.channels[2].at(x, y), Y, Cb, Cr);
            out.channels[0].at(x, y) = Y;
            out.channels[1].at(x, y) = Cb;
            out.channels[2].at(x, y) = Cr;
        }
    }
    return out;
}

/// Inverse of to_ycc().
inline VideoFrame32F to_rgb(const VideoFrame32F& ycc) {
    VideoFrame32F out = ycc;
    if (ycc.channel_count() < 3) return out;
    const int32_t w = ycc.width, h = ycc.height;
    for (int32_t y = 0; y < h; ++y) {
        for (int32_t x = 0; x < w; ++x) {
            float R, G, B;
            ycc_to_rgb(ycc.channels[0].at(x, y), ycc.channels[1].at(x, y), ycc.channels[2].at(x, y), R, G, B);
            out.channels[0].at(x, y) = R;
            out.channels[1].at(x, y) = G;
            out.channels[2].at(x, y) = B;
        }
    }
    return out;
}

/**
 * @brief Converts per-RGB-channel noise level functions (all evaluated at pixel intensity I)
 * into the equivalent Y / Cb / Cr functions expressed against *luma* intensity.
 * Assumes the three colour channels carry independent noise.
 */
inline std::array<NoiseLevelFunction, 3> rgb_nlf_to_ycc(const std::array<NoiseLevelFunction, 3>& rgb) {
    const float coef[3][3] = {
        {kWr, kWg, kWb},
        {-kWr / kCbScale, -kWg / kCbScale, (1.0f - kWb) / kCbScale},
        {(1.0f - kWr) / kCrScale, -kWg / kCrScale, -kWb / kCrScale},
    };
    std::array<NoiseLevelFunction, 3> out{};
    for (int c = 0; c < 3; ++c) {
        float a = 0.0f, b = 0.0f;
        for (int k = 0; k < 3; ++k) {
            a += coef[c][k] * coef[c][k] * rgb[static_cast<size_t>(k)].a;
            b += coef[c][k] * coef[c][k] * rgb[static_cast<size_t>(k)].b;
        }
        out[static_cast<size_t>(c)].a = a;
        out[static_cast<size_t>(c)].b = b;
    }
    return out;
}

/**
 * @brief Inverse of rgb_nlf_to_ycc(): recovers per-RGB-channel NLFs from a Y/Cb/Cr profile by solving the
 * 3x3 system of squared transform coefficients (independent channel noise assumed). Used by the CUDA path,
 * whose kernels operate directly on RGB. Negative solutions (inconsistent profiles) are clamped to a small
 * positive fraction of the mean so a bad chroma measurement can never zero a channel's threshold.
 */
inline std::array<NoiseLevelFunction, 3> ycc_nlf_to_rgb(const std::array<NoiseLevelFunction, 3>& ycc) {
    const double coef[3][3] = {
        {kWr, kWg, kWb},
        {-kWr / kCbScale, -kWg / kCbScale, (1.0 - kWb) / kCbScale},
        {(1.0 - kWr) / kCrScale, -kWg / kCrScale, -kWb / kCrScale},
    };
    double m[3][3];
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) m[i][j] = coef[i][j] * coef[i][j];
    auto det3 = [](const double a[3][3]) {
        return a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
             - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
             + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
    };
    const double d = det3(m);
    std::array<NoiseLevelFunction, 3> out{};
    for (int which = 0; which < 2; ++which) {
        const double rhs[3] = {which == 0 ? ycc[0].a : ycc[0].b,
                               which == 0 ? ycc[1].a : ycc[1].b,
                               which == 0 ? ycc[2].a : ycc[2].b};
        double sol[3];
        for (int k = 0; k < 3; ++k) {
            double t[3][3];
            for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) t[i][j] = (j == k) ? rhs[i] : m[i][j];
            sol[k] = (std::abs(d) > 1e-12) ? det3(t) / d : rhs[0] / 0.5619;
        }
        const double mean = (sol[0] + sol[1] + sol[2]) / 3.0;
        const double floor_v = std::max(1e-9, 0.1 * mean);
        for (int k = 0; k < 3; ++k) {
            const float v = static_cast<float>(std::max(sol[k], floor_v));
            if (which == 0) out[static_cast<size_t>(k)].a = v; else out[static_cast<size_t>(k)].b = v;
        }
    }
    return out;
}

/**
 * @brief Maps the user's Luma / Cb / Cr reduction amounts onto the R, G, B channels (for engines that
 * filter RGB directly): each channel receives the RMS of the component amounts weighted by how much of
 * that component it carries. Exact for white noise, a close approximation otherwise.
 */
inline std::array<float, 3> ycc_amounts_to_rgb(float luma, float cb, float cr) {
    const float load[3][3] = {          // rows: R, G, B ; cols: Y, Cb, Cr (ycc_to_rgb coefficients)
        {1.0f, 0.0f, kCrScale},
        {1.0f, -kWb * kCbScale / kWg, -kWr * kCrScale / kWg},
        {1.0f, kCbScale, 0.0f},
    };
    const float s[3] = {luma, cb, cr};
    std::array<float, 3> out{};
    for (int c = 0; c < 3; ++c) {
        float num = 0.0f, den = 0.0f;
        for (int k = 0; k < 3; ++k) {
            const float l2 = load[c][k] * load[c][k];
            num += l2 * s[k] * s[k];
            den += l2;
        }
        out[static_cast<size_t>(c)] = std::sqrt(num / den);
    }
    return out;
}

} // namespace colorspace
} // namespace librenoise
