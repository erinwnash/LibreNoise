#pragma once
// Per-pixel and per-block kernels shared by the CPU (MSVC/OpenMP) and CUDA (nvcc) back ends.
// Every function here is a pure function of its inputs so the two back ends produce the same result.

#include "librenoise/v2/model.hpp"
#include <cstdint>
#include <cmath>

#if defined(__CUDACC__)
#define LN_HD __host__ __device__ __forceinline__
#define LN_LAMBDA [=] __host__ __device__
#else
#define LN_HD inline
#define LN_LAMBDA [=]
#endif

namespace librenoise::v2::k {

// ------------------------------------------------------------------------------------------------
// Constant tables (orthonormal 8-point DCT-II, Kaiser beta=2 aggregation window)
// ------------------------------------------------------------------------------------------------
#define LN_DCT_TABLE { \
    0.353553391f, 0.353553391f, 0.353553391f, 0.353553391f, 0.353553391f, 0.353553391f, 0.353553391f, 0.353553391f, \
    0.490392640f, 0.415734806f, 0.277785117f, 0.097545161f, -0.097545161f, -0.277785117f, -0.415734806f, -0.490392640f, \
    0.461939766f, 0.191341716f, -0.191341716f, -0.461939766f, -0.461939766f, -0.191341716f, 0.191341716f, 0.461939766f, \
    0.415734806f, -0.097545161f, -0.490392640f, -0.277785117f, 0.277785117f, 0.490392640f, 0.097545161f, -0.415734806f, \
    0.353553391f, -0.353553391f, -0.353553391f, 0.353553391f, 0.353553391f, -0.353553391f, -0.353553391f, 0.353553391f, \
    0.277785117f, -0.490392640f, 0.097545161f, 0.415734806f, -0.415734806f, -0.097545161f, 0.490392640f, -0.277785117f, \
    0.191341716f, -0.461939766f, 0.461939766f, -0.191341716f, -0.191341716f, 0.461939766f, -0.461939766f, 0.191341716f, \
    0.097545161f, -0.277785117f, 0.415734806f, -0.490392640f, 0.490392640f, -0.415734806f, 0.277785117f, -0.097545161f }
#define LN_KAISER_TABLE { 0.438676280f, 0.681324263f, 0.876839905f, 0.985822506f, 0.985822506f, 0.876839905f, 0.681324263f, 0.438676280f }

#if defined(__CUDACC__)
__constant__ float c_dct[64] = LN_DCT_TABLE;
__constant__ float c_kaiser[8] = LN_KAISER_TABLE;
#endif
static const float h_dct[64] = LN_DCT_TABLE;
static const float h_kaiser[8] = LN_KAISER_TABLE;

LN_HD float dct_c(int k, int n) {
#if defined(__CUDA_ARCH__)
    return c_dct[k * 8 + n];
#else
    return h_dct[k * 8 + n];
#endif
}
LN_HD float kaiser(int i) {
#if defined(__CUDA_ARCH__)
    return c_kaiser[i];
#else
    return h_kaiser[i];
#endif
}

LN_HD int imin(int a, int b) { return a < b ? a : b; }
LN_HD int imax(int a, int b) { return a > b ? a : b; }
LN_HD int iclamp(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }
LN_HD float fclampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }
LN_HD int reflect(int i, int n) {
    if (n <= 1) return 0;
    while (i < 0 || i >= n) {
        if (i < 0) i = -i;
        if (i >= n) i = 2 * n - 2 - i;
    }
    return i;
}

// ------------------------------------------------------------------------------------------------
// Variance stabilisation
// ------------------------------------------------------------------------------------------------
LN_HD float vst_fwd(const VstLut& L, int c, float x) {
    if (x <= L.x0) return x / L.sigma0[c];
    if (x >= L.x1) return L.f1[c] + (x - L.x1) / L.sigma1[c];
    const float t = (log2f(x) - L.log2x0) / (L.log2x1 - L.log2x0) * static_cast<float>(kVstFwd - 1);
    const int i = iclamp(static_cast<int>(t), 0, kVstFwd - 2);
    const float fr = t - static_cast<float>(i);
    return L.fwd[c][i] + (L.fwd[c][i + 1] - L.fwd[c][i]) * fr;
}

LN_HD float vst_inv(const VstLut& L, int c, float y) {
    if (y <= L.f0[c]) return y * L.sigma0[c];
    if (y >= L.f1[c]) return L.x1 + (y - L.f1[c]) * L.sigma1[c];
    const float t = (y - L.f0[c]) / (L.f1[c] - L.f0[c]) * static_cast<float>(kVstInv - 1);
    const int i = iclamp(static_cast<int>(t), 0, kVstInv - 2);
    const float fr = t - static_cast<float>(i);
    return exp2f(L.inv[c][i] + (L.inv[c][i + 1] - L.inv[c][i]) * fr);
}

// Orthonormal opponent transform: Y = (R+G+B)/sqrt3, Cb = (2B-R-G)/sqrt6, Cr = (R-G)/sqrt2
constexpr float kS3 = 0.577350269f, kS6 = 0.408248290f, kS2 = 0.707106781f;

LN_HD void rgb_to_work(const VstLut& L, float r, float g, float b, float& Y, float& Cb, float& Cr) {
    const float fr = vst_fwd(L, 0, r), fg = vst_fwd(L, 1, g), fb = vst_fwd(L, 2, b);
    Y = (fr + fg + fb) * kS3;
    Cb = (2.0f * fb - fr - fg) * kS6;
    Cr = (fr - fg) * kS2;
}

LN_HD void work_to_rgb(const VstLut& L, float Y, float Cb, float Cr, float& r, float& g, float& b) {
    const float fr = Y * kS3 - Cb * kS6 + Cr * kS2;
    const float fg = Y * kS3 - Cb * kS6 - Cr * kS2;
    const float fb = Y * kS3 + 2.0f * Cb * kS6;
    r = vst_inv(L, 0, fr);
    g = vst_inv(L, 1, fg);
    b = vst_inv(L, 2, fb);
}

// ------------------------------------------------------------------------------------------------
// Pyramid (Burt-Adelson 5-tap binomial)
// ------------------------------------------------------------------------------------------------
// Output pixel (x, y) of the half-size image.
LN_HD float pyr_down(const float* src, int w, int h, int x, int y) {
    const float wt[5] = {1.0f / 16.0f, 4.0f / 16.0f, 6.0f / 16.0f, 4.0f / 16.0f, 1.0f / 16.0f};
    float acc = 0.0f;
    for (int j = 0; j < 5; ++j) {
        const int sy = reflect(2 * y + j - 2, h);
        float row = 0.0f;
        for (int i = 0; i < 5; ++i) {
            const int sx = reflect(2 * x + i - 2, w);
            row += wt[i] * src[static_cast<size_t>(sy) * static_cast<size_t>(w) + static_cast<size_t>(sx)];
        }
        acc += wt[j] * row;
    }
    return acc;
}

// Value of up(small) at full-resolution pixel (x, y); small is sw x sh.
LN_HD float pyr_up(const float* s, int sw, int sh, int x, int y) {
    int xi[3], yi[3];
    float xw[3], yw[3];
    int nx, ny;
    const int hx = x >> 1, hy = y >> 1;
    if ((x & 1) == 0) { xi[0] = hx - 1; xi[1] = hx; xi[2] = hx + 1; xw[0] = 0.125f; xw[1] = 0.75f; xw[2] = 0.125f; nx = 3; }
    else { xi[0] = hx; xi[1] = hx + 1; xi[2] = hx + 1; xw[0] = 0.5f; xw[1] = 0.5f; xw[2] = 0.0f; nx = 2; }
    if ((y & 1) == 0) { yi[0] = hy - 1; yi[1] = hy; yi[2] = hy + 1; yw[0] = 0.125f; yw[1] = 0.75f; yw[2] = 0.125f; ny = 3; }
    else { yi[0] = hy; yi[1] = hy + 1; yi[2] = hy + 1; yw[0] = 0.5f; yw[1] = 0.5f; yw[2] = 0.0f; ny = 2; }
    float acc = 0.0f;
    for (int j = 0; j < ny; ++j) {
        const int sy = reflect(yi[j], sh);
        float row = 0.0f;
        for (int i = 0; i < nx; ++i) {
            row += xw[i] * s[static_cast<size_t>(sy) * static_cast<size_t>(sw) + static_cast<size_t>(reflect(xi[i], sw))];
        }
        acc += yw[j] * row;
    }
    return acc;
}

// 2x2 box average (used for motion pyramids and factor maps)
LN_HD float box_down(const float* src, int w, int h, int x, int y) {
    const int x0 = imin(2 * x, w - 1), x1 = imin(2 * x + 1, w - 1);
    const int y0 = imin(2 * y, h - 1), y1 = imin(2 * y + 1, h - 1);
    const size_t W = static_cast<size_t>(w);
    return 0.25f * (src[y0 * W + x0] + src[y0 * W + x1] + src[y1 * W + x0] + src[y1 * W + x1]);
}

// ------------------------------------------------------------------------------------------------
// 8x8 DCT
// ------------------------------------------------------------------------------------------------
LN_HD void dct8x8(const float* in, float* out) {
    float tmp[64];
    for (int r = 0; r < 8; ++r) {
        for (int k = 0; k < 8; ++k) {
            float s = 0.0f;
            for (int n = 0; n < 8; ++n) s += dct_c(k, n) * in[r * 8 + n];
            tmp[r * 8 + k] = s;
        }
    }
    for (int c = 0; c < 8; ++c) {
        for (int k = 0; k < 8; ++k) {
            float s = 0.0f;
            for (int n = 0; n < 8; ++n) s += dct_c(k, n) * tmp[n * 8 + c];
            out[k * 8 + c] = s;
        }
    }
}

LN_HD void idct8x8(const float* in, float* out) {
    float tmp[64];
    for (int r = 0; r < 8; ++r) {
        for (int n = 0; n < 8; ++n) {
            float s = 0.0f;
            for (int k = 0; k < 8; ++k) s += dct_c(k, n) * in[r * 8 + k];
            tmp[r * 8 + n] = s;
        }
    }
    for (int c = 0; c < 8; ++c) {
        for (int n = 0; n < 8; ++n) {
            float s = 0.0f;
            for (int k = 0; k < 8; ++k) s += dct_c(k, n) * tmp[k * 8 + c];
            out[n * 8 + c] = s;
        }
    }
}

// Hard-threshold pilot for one block. img/num/den are pw-wide padded planes; (bx, by) = block origin.
LN_HD void block_ht(const float* img, int pw, int bx, int by, const float* sig, float factor, float lambda,
                    float* num, float* den) {
    float blk[64], co[64];
    for (int j = 0; j < 8; ++j)
        for (int i = 0; i < 8; ++i)
            blk[j * 8 + i] = img[static_cast<size_t>(by + j) * pw + (bx + i)];
    dct8x8(blk, co);
    int nnz = 0;
    for (int q = 0; q < 64; ++q) {
        const float t = lambda * sig[q] * factor;
        if (fabsf(co[q]) < t) co[q] = 0.0f; else ++nnz;
    }
    idct8x8(co, blk);
    const float wgt = 1.0f / static_cast<float>(imax(1, nnz));
    for (int j = 0; j < 8; ++j) {
        const float wj = kaiser(j);
        for (int i = 0; i < 8; ++i) {
            const float ww = wgt * wj * kaiser(i);
            const size_t o = static_cast<size_t>(by + j) * pw + (bx + i);
            num[o] += ww * blk[j * 8 + i];
            den[o] += ww;
        }
    }
}

// Empirical Wiener for one block using the pilot estimate.
LN_HD void block_wiener(const float* img, const float* pilot, int pw, int bx, int by, const float* sig, float factor,
                        float* num, float* den) {
    float blk[64], co[64], pc[64];
    for (int j = 0; j < 8; ++j)
        for (int i = 0; i < 8; ++i) {
            const size_t o = static_cast<size_t>(by + j) * pw + (bx + i);
            blk[j * 8 + i] = img[o];
            co[j * 8 + i] = pilot[o];
        }
    dct8x8(co, pc);
    dct8x8(blk, co);
    float wsum = 0.0f;
    for (int q = 0; q < 64; ++q) {
        const float s = sig[q] * factor;
        const float s2 = s * s;
        const float p2 = pc[q] * pc[q];
        const float g = p2 / (p2 + s2 + 1e-12f);
        co[q] *= g;
        wsum += g * g * s2;
    }
    idct8x8(co, blk);
    const float wgt = 1.0f / (wsum + 1e-6f);
    for (int j = 0; j < 8; ++j) {
        const float wj = kaiser(j);
        for (int i = 0; i < 8; ++i) {
            const float ww = wgt * wj * kaiser(i);
            const size_t o = static_cast<size_t>(by + j) * pw + (bx + i);
            num[o] += ww * blk[j * 8 + i];
            den[o] += ww;
        }
    }
}

// ------------------------------------------------------------------------------------------------
// Motion estimation (block SAD on the stabilised luma)
// ------------------------------------------------------------------------------------------------
struct Vec2 { float x, y; };

LN_HD float sample_clamped(const float* p, int w, int h, int x, int y) {
    return p[static_cast<size_t>(iclamp(y, 0, h - 1)) * w + iclamp(x, 0, w - 1)];
}

LN_HD float sample_bilinear(const float* p, int w, int h, float x, float y) {
    const float fx = floorf(x), fy = floorf(y);
    const int x0 = static_cast<int>(fx), y0 = static_cast<int>(fy);
    const float ax = x - fx, ay = y - fy;
    const float a = sample_clamped(p, w, h, x0, y0), b = sample_clamped(p, w, h, x0 + 1, y0);
    const float c = sample_clamped(p, w, h, x0, y0 + 1), d = sample_clamped(p, w, h, x0 + 1, y0 + 1);
    return (a + (b - a) * ax) * (1.0f - ay) + (c + (d - c) * ax) * ay;
}

// SAD of 8x8 block at (bx, by) in ref against cand displaced by integer (dx, dy). row_step 1 or 2.
LN_HD float block_sad(const float* ref, const float* cand, int w, int h, int bx, int by, int dx, int dy, int row_step) {
    float s = 0.0f;
    for (int j = 0; j < 8; j += row_step) {
        const int ry = imin(by + j, h - 1);
        for (int i = 0; i < 8; ++i) {
            const int rx = imin(bx + i, w - 1);
            s += fabsf(ref[static_cast<size_t>(ry) * w + rx] - sample_clamped(cand, w, h, rx + dx, ry + dy));
        }
    }
    return s;
}

LN_HD float block_sad_sub(const float* ref, const float* cand, int w, int h, int bx, int by, float dx, float dy) {
    float s = 0.0f;
    for (int j = 0; j < 8; j += 2) {
        const int ry = imin(by + j, h - 1);
        for (int i = 0; i < 8; ++i) {
            const int rx = imin(bx + i, w - 1);
            s += fabsf(ref[static_cast<size_t>(ry) * w + rx] - sample_bilinear(cand, w, h, rx + dx, ry + dy));
        }
    }
    return s;
}

// Full search at the coarsest level.
LN_HD Vec2 me_full(const float* ref, const float* cand, int w, int h, int bxi, int byi, int radius) {
    const int bx = bxi * 8, by = byi * 8;
    float best = block_sad(ref, cand, w, h, bx, by, 0, 0, 1) * 0.98f;
    int bdx = 0, bdy = 0;
    for (int dy = -radius; dy <= radius; ++dy)
        for (int dx = -radius; dx <= radius; ++dx) {
            if (dx == 0 && dy == 0) continue;
            const float s = block_sad(ref, cand, w, h, bx, by, dx, dy, 1);
            if (s < best) { best = s; bdx = dx; bdy = dy; }
        }
    Vec2 v; v.x = static_cast<float>(bdx); v.y = static_cast<float>(bdy);
    return v;
}

// Predictive refinement: evaluate candidate vectors, refine the best by +/-1, optional half-pel step.
LN_HD Vec2 me_refine(const float* ref, const float* cand, int w, int h, int bxi, int byi,
                     const Vec2* cands, int ncand, int row_step, bool half_pel) {
    const int bx = bxi * 8, by = byi * 8;
    float best = block_sad(ref, cand, w, h, bx, by, 0, 0, row_step) * 0.98f;
    int bdx = 0, bdy = 0;
    for (int c = 0; c < ncand; ++c) {
        const int dx = static_cast<int>(lrintf(cands[c].x)), dy = static_cast<int>(lrintf(cands[c].y));
        if (dx == bdx && dy == bdy) continue;
        const float s = block_sad(ref, cand, w, h, bx, by, dx, dy, row_step);
        if (s < best) { best = s; bdx = dx; bdy = dy; }
    }
    for (int it = 0; it < 2; ++it) {
        int cx = bdx, cy = bdy;
        bool moved = false;
        for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx) {
                if (dx == 0 && dy == 0) continue;
                const float s = block_sad(ref, cand, w, h, bx, by, cx + dx, cy + dy, row_step);
                if (s < best) { best = s; bdx = cx + dx; bdy = cy + dy; moved = true; }
            }
        if (!moved) break;
    }
    Vec2 v; v.x = static_cast<float>(bdx); v.y = static_cast<float>(bdy);
    if (half_pel) {
        float bs = block_sad_sub(ref, cand, w, h, bx, by, v.x, v.y);
        float hx = v.x, hy = v.y;
        for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx) {
                if (dx == 0 && dy == 0) continue;
                const float tx = v.x + 0.5f * dx, ty = v.y + 0.5f * dy;
                const float s = block_sad_sub(ref, cand, w, h, bx, by, tx, ty);
                if (s < bs) { bs = s; hx = tx; hy = ty; }
            }
        v.x = hx; v.y = hy;
    }
    return v;
}

// ------------------------------------------------------------------------------------------------
// Deblocking (applied in the stabilised domain when a compression grid was detected)
// ------------------------------------------------------------------------------------------------
// p1, p0 | q0, q1 straddle a block boundary; role 0..3 selects which of the four is being corrected.
LN_HD float deblock_value(float p1, float p0, float q0, float q1, int role, float thresh, float strength) {
    const float step = q0 - p0;
    const float natural = 0.5f * ((p0 - p1) + (q1 - q0));
    const float e = step - natural;
    const float a = fabsf(step);
    if (a >= thresh) {
        return role == 0 ? p1 : role == 1 ? p0 : role == 2 ? q0 : q1;
    }
    const float fade = 1.0f - a / thresh;  // fade the correction out as the step approaches an edge
    const float c = e * strength * fade;
    switch (role) {
        case 0: return p1 + c * 0.125f;
        case 1: return p0 + c * 0.375f;
        case 2: return q0 - c * 0.375f;
        default: return q1 - c * 0.125f;
    }
}

// Horizontal pass (vertical boundaries). Boundary lies between x and x+1 where x % 8 == phase.
LN_HD float deblock_h(const float* src, int w, int x, int y, int phase, float thresh, float strength) {
    const size_t row = static_cast<size_t>(y) * w;
    const int m = ((x - phase) % 8 + 8) % 8;   // 0 => x is p0, 1 => q0, 7 => p1, 2 => q1
    int bnd, role;
    if (m == 0) { bnd = x; role = 1; }
    else if (m == 1) { bnd = x - 1; role = 2; }
    else if (m == 7) { bnd = x + 1; role = 0; }
    else if (m == 2) { bnd = x - 2; role = 3; }
    else return src[row + x];
    if (bnd - 1 < 0 || bnd + 2 >= w) return src[row + x];
    return deblock_value(src[row + bnd - 1], src[row + bnd], src[row + bnd + 1], src[row + bnd + 2], role, thresh, strength);
}

LN_HD float deblock_v(const float* src, int w, int h, int x, int y, int phase, float thresh, float strength) {
    const int m = ((y - phase) % 8 + 8) % 8;
    int bnd, role;
    if (m == 0) { bnd = y; role = 1; }
    else if (m == 1) { bnd = y - 1; role = 2; }
    else if (m == 7) { bnd = y + 1; role = 0; }
    else if (m == 2) { bnd = y - 2; role = 3; }
    else return src[static_cast<size_t>(y) * w + x];
    if (bnd - 1 < 0 || bnd + 2 >= h) return src[static_cast<size_t>(y) * w + x];
    const size_t W = static_cast<size_t>(w);
    return deblock_value(src[(bnd - 1) * W + x], src[bnd * W + x], src[(bnd + 1) * W + x], src[(bnd + 2) * W + x], role, thresh, strength);
}

// ------------------------------------------------------------------------------------------------
// Misc
// ------------------------------------------------------------------------------------------------
LN_HD uint32_t hash32(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16;
    return x;
}

LN_HD float hash_gauss(int x, int y, uint32_t seed) {
    const uint32_t h1 = hash32(static_cast<uint32_t>(x) * 73856093U ^ static_cast<uint32_t>(y) * 19349663U ^ seed);
    const uint32_t h2 = hash32(h1 ^ 0x9e3779b9U);
    const float u1 = (static_cast<float>(h1 >> 8) + 0.5f) * (1.0f / 16777216.0f);
    const float u2 = static_cast<float>(h2 >> 8) * (1.0f / 16777216.0f);
    return sqrtf(-2.0f * logf(u1)) * cosf(6.283185307f * u2);
}

// Tonal multiplier from linear luma: knots at -5, -2.5, 0, +1.5, +3 stops around 18% grey.
LN_HD float tonal_factor(const float* t, float luma) {
    const float stops = log2f(fmaxf(luma, 1e-6f) / 0.18f);
    const float k0 = -5.0f, k1 = -2.5f, k2 = 0.0f, k3 = 1.5f, k4 = 3.0f;
    if (stops <= k0) return t[0];
    if (stops >= k4) return t[4];
    if (stops <= k1) return t[0] + (t[1] - t[0]) * (stops - k0) / (k1 - k0);
    if (stops <= k2) return t[1] + (t[2] - t[1]) * (stops - k1) / (k2 - k1);
    if (stops <= k3) return t[2] + (t[3] - t[2]) * (stops - k2) / (k3 - k2);
    return t[3] + (t[4] - t[3]) * (stops - k3) / (k4 - k3);
}

} // namespace librenoise::v2::k
