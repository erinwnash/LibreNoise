#pragma once
// Per-pixel and per-block kernels shared by the CPU (MSVC/OpenMP) and CUDA (nvcc) back ends.
// Every function here is a pure function of its inputs so the two back ends produce the same result.

#include "librenoise/v2/model.hpp"
#include <cstdint>
#include <cmath>
#include <cstring>

#if !defined(__CUDACC__) && defined(__AVX2__)
#include <immintrin.h>
#define LN_CPU_AVX2 1
#endif

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
#define LN_DCT_T_TABLE { \
    0.353553391f, 0.490392640f, 0.461939766f, 0.415734806f, 0.353553391f, 0.277785117f, 0.191341716f, 0.097545161f, \
    0.353553391f, 0.415734806f, 0.191341716f, -0.097545161f, -0.353553391f, -0.490392640f, -0.461939766f, -0.277785117f, \
    0.353553391f, 0.277785117f, -0.191341716f, -0.490392640f, -0.353553391f, 0.097545161f, 0.461939766f, 0.415734806f, \
    0.353553391f, 0.097545161f, -0.461939766f, -0.277785117f, 0.353553391f, 0.415734806f, -0.191341716f, -0.490392640f, \
    0.353553391f, -0.097545161f, -0.461939766f, 0.277785117f, 0.353553391f, -0.415734806f, -0.191341716f, 0.490392640f, \
    0.353553391f, -0.277785117f, -0.191341716f, 0.490392640f, -0.353553391f, -0.097545161f, 0.461939766f, -0.415734806f, \
    0.353553391f, -0.415734806f, 0.191341716f, 0.097545161f, -0.353553391f, 0.490392640f, -0.461939766f, 0.277785117f, \
    0.353553391f, -0.490392640f, 0.461939766f, -0.415734806f, 0.353553391f, -0.277785117f, 0.191341716f, -0.097545161f }
#define LN_KAISER_TABLE { 0.438676280f, 0.681324263f, 0.876839905f, 0.985822506f, 0.985822506f, 0.876839905f, 0.681324263f, 0.438676280f }

#if defined(__CUDACC__)
__constant__ float c_dct[64] = LN_DCT_TABLE;
__constant__ float c_kaiser[8] = LN_KAISER_TABLE;
__constant__ float c_dct_t[64] = LN_DCT_T_TABLE;
#endif
static const float h_dct[64] = LN_DCT_TABLE;
static const float h_kaiser[8] = LN_KAISER_TABLE;
static const float h_dct_t[64] = LN_DCT_T_TABLE;

LN_HD float dct_c(int k, int n) {
#if defined(__CUDA_ARCH__)
    return c_dct[k * 8 + n];
#else
    return h_dct[k * 8 + n];
#endif
}
// Transposed basis: dct_t(n, k) == dct_c(k, n), stored so the inner loops below run over contiguous memory.
LN_HD float dct_t(int n, int k) {
#if defined(__CUDA_ARCH__)
    return c_dct_t[n * 8 + k];
#else
    return h_dct_t[n * 8 + k];
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
// Fast log2 / exp2 (~1e-7 relative), identical on CPU and GPU. The CRT versions dominate per-pixel cost on MSVC.
// ------------------------------------------------------------------------------------------------
LN_HD int f2i_bits(float f) {
#if defined(__CUDA_ARCH__)
    return __float_as_int(f);
#else
    int i;
    std::memcpy(&i, &f, sizeof(i));
    return i;
#endif
}
LN_HD float i2f_bits(int i) {
#if defined(__CUDA_ARCH__)
    return __int_as_float(i);
#else
    float f;
    std::memcpy(&f, &i, sizeof(f));
    return f;
#endif
}

LN_HD float fast_log2(float x) {           // x > 0, finite
    int bits = f2i_bits(x);
    int e = ((bits >> 23) & 255) - 127;
    float m = i2f_bits((bits & 0x007fffff) | 0x3f800000);   // [1, 2)
    if (m > 1.41421356f) { m *= 0.5f; e += 1; }             // [0.707, 1.414)
    const float s = (m - 1.0f) / (m + 1.0f);
    const float s2 = s * s;
    const float ln = 2.0f * s * (1.0f + s2 * (0.333333333f + s2 * (0.2f + s2 * (0.142857143f + s2 * 0.111111111f))));
    return static_cast<float>(e) + ln * 1.442695041f;
}

LN_HD float fast_exp2(float y) {
    if (y < -126.0f) return 0.0f;
    if (y > 127.0f) y = 127.0f;
    const float fi = floorf(y);
    const float z = (y - fi - 0.5f) * 0.693147181f;           // [-0.347, 0.347)
    const float p = 1.0f + z * (1.0f + z * (0.5f + z * (0.166666667f + z * (0.041666667f + z * (0.008333333f + z * 0.001388889f)))));
    return p * 1.414213562f * i2f_bits((static_cast<int>(fi) + 127) << 23);
}

LN_HD float fast_exp(float x) { return fast_exp2(x * 1.442695041f); }

// ------------------------------------------------------------------------------------------------
// Variance stabilisation
// ------------------------------------------------------------------------------------------------
LN_HD float vst_fwd(const VstLut& L, int c, float x) {
    if (x <= L.x0) return x / L.sigma0[c];
    if (x >= L.x1) return L.f1[c] + (x - L.x1) / L.sigma1[c];
    const float t = (fast_log2(x) - L.log2x0) / (L.log2x1 - L.log2x0) * static_cast<float>(kVstFwd - 1);
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
    return fast_exp2(L.inv[c][i] + (L.inv[c][i + 1] - L.inv[c][i]) * fr);
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
#if defined(LN_CPU_AVX2)
// AVX2 / FMA versions of the separable transforms for the CPU build (same arithmetic order as the scalar form).
inline void dct8x8_avx(const float* in, float* out) {
    __m256 tmp[8];
    for (int r = 0; r < 8; ++r) {
        __m256 acc = _mm256_mul_ps(_mm256_set1_ps(in[r * 8]), _mm256_loadu_ps(h_dct_t));
        for (int n = 1; n < 8; ++n) acc = _mm256_fmadd_ps(_mm256_set1_ps(in[r * 8 + n]), _mm256_loadu_ps(h_dct_t + n * 8), acc);
        tmp[r] = acc;
    }
    for (int k = 0; k < 8; ++k) {
        __m256 acc = _mm256_mul_ps(_mm256_set1_ps(h_dct[k * 8]), tmp[0]);
        for (int n = 1; n < 8; ++n) acc = _mm256_fmadd_ps(_mm256_set1_ps(h_dct[k * 8 + n]), tmp[n], acc);
        _mm256_storeu_ps(out + k * 8, acc);
    }
}
inline void idct8x8_avx(const float* in, float* out) {
    __m256 tmp[8];
    for (int r = 0; r < 8; ++r) {
        __m256 acc = _mm256_mul_ps(_mm256_set1_ps(in[r * 8]), _mm256_loadu_ps(h_dct));
        for (int k = 1; k < 8; ++k) acc = _mm256_fmadd_ps(_mm256_set1_ps(in[r * 8 + k]), _mm256_loadu_ps(h_dct + k * 8), acc);
        tmp[r] = acc;
    }
    for (int n = 0; n < 8; ++n) {
        __m256 acc = _mm256_mul_ps(_mm256_set1_ps(h_dct_t[n * 8]), tmp[0]);
        for (int k = 1; k < 8; ++k) acc = _mm256_fmadd_ps(_mm256_set1_ps(h_dct_t[n * 8 + k]), tmp[k], acc);
        _mm256_storeu_ps(out + n * 8, acc);
    }
}
// num[row] += wgt * kaiser(j) * kaiser(i) * blk[row], den[row] += same weight, for the 8 rows of a block
inline void accumulate_block_avx(const float* blk, float wgt, float* num, float* den, size_t row0, int pw) {
    const __m256 kx = _mm256_loadu_ps(h_kaiser);
    for (int j = 0; j < 8; ++j) {
        const __m256 ww = _mm256_mul_ps(_mm256_set1_ps(wgt * h_kaiser[j]), kx);
        float* nr = num + row0 + static_cast<size_t>(j) * pw;
        float* dr = den + row0 + static_cast<size_t>(j) * pw;
        _mm256_storeu_ps(nr, _mm256_fmadd_ps(ww, _mm256_loadu_ps(blk + j * 8), _mm256_loadu_ps(nr)));
        _mm256_storeu_ps(dr, _mm256_add_ps(ww, _mm256_loadu_ps(dr)));
    }
}
#endif

// Separable 8x8 transforms. Every inner loop runs over 8 contiguous floats (AVX on the CPU, unrolled registers on CUDA).
LN_HD void dct8x8(const float* in, float* out) {
#if defined(LN_CPU_AVX2)
    dct8x8_avx(in, out);
#else
    float tmp[64];
    for (int r = 0; r < 8; ++r) {              // tmp[r][k] = sum_n in[r][n] C[k][n]
        float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (int n = 0; n < 8; ++n) {
            const float v = in[r * 8 + n];
            for (int k = 0; k < 8; ++k) acc[k] += v * dct_t(n, k);
        }
        for (int k = 0; k < 8; ++k) tmp[r * 8 + k] = acc[k];
    }
    for (int k = 0; k < 8; ++k) {              // out[k][c] = sum_n C[k][n] tmp[n][c]
        float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (int n = 0; n < 8; ++n) {
            const float cv = dct_c(k, n);
            for (int c = 0; c < 8; ++c) acc[c] += cv * tmp[n * 8 + c];
        }
        for (int c = 0; c < 8; ++c) out[k * 8 + c] = acc[c];
    }
#endif
}

LN_HD void idct8x8(const float* in, float* out) {
#if defined(LN_CPU_AVX2)
    idct8x8_avx(in, out);
#else
    float tmp[64];
    for (int r = 0; r < 8; ++r) {              // tmp[r][n] = sum_k in[r][k] C[k][n]
        float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (int k = 0; k < 8; ++k) {
            const float v = in[r * 8 + k];
            for (int n = 0; n < 8; ++n) acc[n] += v * dct_c(k, n);
        }
        for (int n = 0; n < 8; ++n) tmp[r * 8 + n] = acc[n];
    }
    for (int n = 0; n < 8; ++n) {              // out[n][c] = sum_k C[k][n] tmp[k][c]
        float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (int k = 0; k < 8; ++k) {
            const float cv = dct_t(n, k);
            for (int c = 0; c < 8; ++c) acc[c] += cv * tmp[k * 8 + c];
        }
        for (int c = 0; c < 8; ++c) out[n * 8 + c] = acc[c];
    }
#endif
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
#if defined(LN_CPU_AVX2)
    accumulate_block_avx(blk, wgt, num, den, static_cast<size_t>(by) * pw + bx, pw);
#else
    for (int j = 0; j < 8; ++j) {
        const float wj = kaiser(j);
        for (int i = 0; i < 8; ++i) {
            const float ww = wgt * wj * kaiser(i);
            const size_t o = static_cast<size_t>(by + j) * pw + (bx + i);
            num[o] += ww * blk[j * 8 + i];
            den[o] += ww;
        }
    }
#endif
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
#if defined(LN_CPU_AVX2)
    accumulate_block_avx(blk, wgt, num, den, static_cast<size_t>(by) * pw + bx, pw);
#else
    for (int j = 0; j < 8; ++j) {
        const float wj = kaiser(j);
        for (int i = 0; i < 8; ++i) {
            const float ww = wgt * wj * kaiser(i);
            const size_t o = static_cast<size_t>(by + j) * pw + (bx + i);
            num[o] += ww * blk[j * 8 + i];
            den[o] += ww;
        }
    }
#endif
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
    if (x0 >= 0 && y0 >= 0 && x0 + 1 < w && y0 + 1 < h) {
        const float* r0 = p + static_cast<size_t>(y0) * w + x0;
        const float* r1 = r0 + w;
        return (r0[0] + (r0[1] - r0[0]) * ax) * (1.0f - ay) + (r1[0] + (r1[1] - r1[0]) * ax) * ay;
    }
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
// `lam` charges each pixel of vector length, so in flat areas (where every displacement matches equally well) the
// search stays at zero instead of wandering to a random far-away match; real motion over texture gains far more
// SAD than it costs.
LN_HD Vec2 me_full(const float* ref, const float* cand, int w, int h, int bxi, int byi, int radius, float lam) {
    const int bx = bxi * 8, by = byi * 8;
    float best = block_sad(ref, cand, w, h, bx, by, 0, 0, 1) * 0.98f;
    int bdx = 0, bdy = 0;
    for (int dy = -radius; dy <= radius; ++dy)
        for (int dx = -radius; dx <= radius; ++dx) {
            if (dx == 0 && dy == 0) continue;
            const float s = block_sad(ref, cand, w, h, bx, by, dx, dy, 1) + lam * static_cast<float>(abs(dx) + abs(dy));
            if (s < best) { best = s; bdx = dx; bdy = dy; }
        }
    Vec2 v; v.x = static_cast<float>(bdx); v.y = static_cast<float>(bdy);
    return v;
}

// Predictive refinement: evaluate candidate vectors, refine the best by +/-1, optional half-pel step.
LN_HD Vec2 me_refine(const float* ref, const float* cand, int w, int h, int bxi, int byi,
                     const Vec2* cands, int ncand, int row_step, bool half_pel, float lam) {
    const int bx = bxi * 8, by = byi * 8;
    if (row_step == 2) lam *= 0.5f;            // the SAD only sums half of the rows
    float best = block_sad(ref, cand, w, h, bx, by, 0, 0, row_step) * 0.98f;
    int bdx = 0, bdy = 0;
    for (int c = 0; c < ncand; ++c) {
        const int dx = static_cast<int>(lrintf(cands[c].x)), dy = static_cast<int>(lrintf(cands[c].y));
        if (dx == bdx && dy == bdy) continue;
        const float s = block_sad(ref, cand, w, h, bx, by, dx, dy, row_step) + lam * static_cast<float>(abs(dx) + abs(dy));
        if (s < best) { best = s; bdx = dx; bdy = dy; }
    }
    for (int it = 0; it < 2; ++it) {
        int cx = bdx, cy = bdy;
        bool moved = false;
        for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx) {
                if (dx == 0 && dy == 0) continue;
                const float s = block_sad(ref, cand, w, h, bx, by, cx + dx, cy + dy, row_step) +
                                lam * static_cast<float>(abs(cx + dx) + abs(cy + dy));
                if (s < best) { best = s; bdx = cx + dx; bdy = cy + dy; moved = true; }
            }
        if (!moved) break;
    }
    Vec2 v; v.x = static_cast<float>(bdx); v.y = static_cast<float>(bdy);
    if (half_pel) {
        float bs = block_sad_sub(ref, cand, w, h, bx, by, v.x, v.y) + lam * (fabsf(v.x) + fabsf(v.y));
        float hx = v.x, hy = v.y;
        for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx) {
                if (dx == 0 && dy == 0) continue;
                const float tx = v.x + 0.5f * dx, ty = v.y + 0.5f * dy;
                const float s = block_sad_sub(ref, cand, w, h, bx, by, tx, ty) + lam * (fabsf(tx) + fabsf(ty));
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
    const float stops = fast_log2(fmaxf(luma, 1e-6f) / 0.18f);
    const float k0 = -5.0f, k1 = -2.5f, k2 = 0.0f, k3 = 1.5f, k4 = 3.0f;
    if (stops <= k0) return t[0];
    if (stops >= k4) return t[4];
    if (stops <= k1) return t[0] + (t[1] - t[0]) * (stops - k0) / (k1 - k0);
    if (stops <= k2) return t[1] + (t[2] - t[1]) * (stops - k1) / (k2 - k1);
    if (stops <= k3) return t[2] + (t[3] - t[2]) * (stops - k2) / (k3 - k2);
    return t[3] + (t[4] - t[3]) * (stops - k3) / (k4 - k3);
}

} // namespace librenoise::v2::k
