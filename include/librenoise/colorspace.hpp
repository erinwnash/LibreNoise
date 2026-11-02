#pragma once

#include <array>
#include <cmath>
#include <algorithm>

namespace librenoise {

/**
 * @brief Rec.709 luma / colour-difference helpers. The engines filter RGB directly; these map the
 * user-facing Luma / Cb / Cr sliders onto the R, G, B planes.
 */
namespace colorspace {

constexpr float kWr = 0.2126f;
constexpr float kWg = 0.7152f;
constexpr float kWb = 0.0722f;
constexpr float kCbScale = 1.8556f; // 2 * (1 - Wb)
constexpr float kCrScale = 1.5748f; // 2 * (1 - Wr)

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
