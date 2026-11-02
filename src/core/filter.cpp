#include "librenoise/filter.hpp"
#include <cmath>
#include <vector>
#include <algorithm>
#include <array>
#include <atomic>
#include <thread>
#include <cstdint>
#include <immintrin.h>

#ifdef _MSC_VER
#include <intrin.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

namespace librenoise {

namespace {

inline int resolve_filter_threads(int32_t requested) noexcept {
    if (requested > 0) return requested;
    int hw = static_cast<int>(std::thread::hardware_concurrency());
    if (hw <= 0) hw = 8;
    return std::clamp((hw * 3) / 4, 2, 16);
}

inline int popcount8(int mask8) noexcept {
#ifdef _MSC_VER
    return static_cast<int>(__popcnt(static_cast<unsigned int>(mask8)));
#else
    return __builtin_popcount(static_cast<unsigned int>(mask8));
#endif
}

inline float hsum256_ps(__m256 v) noexcept {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(s);
    __m128 sums = _mm_add_ps(s, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

// =========================================================================
// Precomputed Orthonormal Temporal DCT (K = 1..12) & Spatial Band/Feather LUTs
// Zero runtime trigonometric or square-root calls
// =========================================================================
struct TemporalDctTable {
    float mat[13][12][12]{};

    TemporalDctTable() {
        constexpr double pi = 3.14159265358979323846;
        mat[1][0][0] = 1.0f;
        for (int K = 2; K <= 12; ++K) {
            const double alpha0 = std::sqrt(1.0 / static_cast<double>(K));
            const double alpha = std::sqrt(2.0 / static_cast<double>(K));
            for (int k = 0; k < K; ++k) {
                const double factor = (k == 0) ? alpha0 : alpha;
                for (int n = 0; n < K; ++n) {
                    mat[K][k][n] = static_cast<float>(
                        factor * std::cos((pi / static_cast<double>(K)) * (static_cast<double>(n) + 0.5) * static_cast<double>(k))
                    );
                }
            }
        }
    }
};
static const TemporalDctTable g_tdct_table;

struct SpatialLutInit {
    alignas(32) float feather_2d[64]{};
    int band[64]{};

    SpatialLutInit() {
        constexpr float f1d[8] = {0.38f, 0.68f, 0.92f, 1.0f, 1.0f, 0.92f, 0.68f, 0.38f};
        for (int v = 0; v < 8; ++v) {
            for (int u = 0; u < 8; ++u) {
                feather_2d[v * 8 + u] = f1d[v] * f1d[u];
                float r = std::sqrt(static_cast<float>(u * u + v * v));
                band[v * 8 + u] = (r < 1.5f) ? 0 : (r < 3.0f) ? 1 : (r < 5.0f) ? 2 : (r < 7.5f) ? 3 : 4;
            }
        }
    }
};
static const SpatialLutInit g_spatial_lut;

// =========================================================================
// AVX2 Exact Orthonormal 8x8 2D Discrete Cosine Transform & Inverse
// =========================================================================
inline void transpose8x8_ps(
    __m256& r0, __m256& r1, __m256& r2, __m256& r3,
    __m256& r4, __m256& r5, __m256& r6, __m256& r7
) noexcept {
    __m256 t0 = _mm256_unpacklo_ps(r0, r1);
    __m256 t1 = _mm256_unpackhi_ps(r0, r1);
    __m256 t2 = _mm256_unpacklo_ps(r2, r3);
    __m256 t3 = _mm256_unpackhi_ps(r2, r3);
    __m256 t4 = _mm256_unpacklo_ps(r4, r5);
    __m256 t5 = _mm256_unpackhi_ps(r4, r5);
    __m256 t6 = _mm256_unpacklo_ps(r6, r7);
    __m256 t7 = _mm256_unpackhi_ps(r6, r7);

    __m256 s0 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(1, 0, 1, 0));
    __m256 s1 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(3, 2, 3, 2));
    __m256 s2 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(1, 0, 1, 0));
    __m256 s3 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(3, 2, 3, 2));
    __m256 s4 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(1, 0, 1, 0));
    __m256 s5 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(3, 2, 3, 2));
    __m256 s6 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(1, 0, 1, 0));
    __m256 s7 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(3, 2, 3, 2));

    r0 = _mm256_permute2f128_ps(s0, s4, 0x20);
    r1 = _mm256_permute2f128_ps(s1, s5, 0x20);
    r2 = _mm256_permute2f128_ps(s2, s6, 0x20);
    r3 = _mm256_permute2f128_ps(s3, s7, 0x20);
    r4 = _mm256_permute2f128_ps(s0, s4, 0x31);
    r5 = _mm256_permute2f128_ps(s1, s5, 0x31);
    r6 = _mm256_permute2f128_ps(s2, s6, 0x31);
    r7 = _mm256_permute2f128_ps(s3, s7, 0x31);
}

inline void dct8_cols_avx2(
    __m256& x0, __m256& x1, __m256& x2, __m256& x3,
    __m256& x4, __m256& x5, __m256& x6, __m256& x7
) noexcept {
    // 0.5 * cos(k * pi / 16) orthonormal DCT-II constants
    const __m256 kS0 = _mm256_set1_ps(0.35355339059327376f);
    const __m256 kC1 = _mm256_set1_ps(0.49039264020161522f);
    const __m256 kC2 = _mm256_set1_ps(0.46193976625564337f);
    const __m256 kC3 = _mm256_set1_ps(0.41573480615127262f);
    const __m256 kC5 = _mm256_set1_ps(0.27778511650980114f);
    const __m256 kC6 = _mm256_set1_ps(0.19134171618254492f);
    const __m256 kC7 = _mm256_set1_ps(0.09754516100806417f);

    __m256 s0 = _mm256_add_ps(x0, x7);
    __m256 d0 = _mm256_sub_ps(x0, x7);
    __m256 s1 = _mm256_add_ps(x1, x6);
    __m256 d1 = _mm256_sub_ps(x1, x6);
    __m256 s2 = _mm256_add_ps(x2, x5);
    __m256 d2 = _mm256_sub_ps(x2, x5);
    __m256 s3 = _mm256_add_ps(x3, x4);
    __m256 d3 = _mm256_sub_ps(x3, x4);

    __m256 e0 = _mm256_add_ps(s0, s3);
    __m256 e3 = _mm256_sub_ps(s0, s3);
    __m256 e1 = _mm256_add_ps(s1, s2);
    __m256 e2 = _mm256_sub_ps(s1, s2);

    x0 = _mm256_mul_ps(_mm256_add_ps(e0, e1), kS0);
    x4 = _mm256_mul_ps(_mm256_sub_ps(e0, e1), kS0);
    x2 = _mm256_fmadd_ps(e3, kC2, _mm256_mul_ps(e2, kC6));
    x6 = _mm256_fmsub_ps(e3, kC6, _mm256_mul_ps(e2, kC2));

    x1 = _mm256_fmadd_ps(kC1, d0, _mm256_fmadd_ps(kC3, d1, _mm256_fmadd_ps(kC5, d2, _mm256_mul_ps(kC7, d3))));
    x3 = _mm256_fmsub_ps(kC3, d0, _mm256_fmadd_ps(kC7, d1, _mm256_fmadd_ps(kC1, d2, _mm256_mul_ps(kC5, d3))));
    x5 = _mm256_fmadd_ps(kC5, d0, _mm256_fnmadd_ps(kC1, d1, _mm256_fmadd_ps(kC7, d2, _mm256_mul_ps(kC3, d3))));
    x7 = _mm256_fmadd_ps(kC7, d0, _mm256_fnmadd_ps(kC5, d1, _mm256_fmsub_ps(kC3, d2, _mm256_mul_ps(kC1, d3))));
}

inline void idct8_cols_avx2(
    __m256& y0, __m256& y1, __m256& y2, __m256& y3,
    __m256& y4, __m256& y5, __m256& y6, __m256& y7
) noexcept {
    const __m256 kS0 = _mm256_set1_ps(0.35355339059327376f);
    const __m256 kC1 = _mm256_set1_ps(0.49039264020161522f);
    const __m256 kC2 = _mm256_set1_ps(0.46193976625564337f);
    const __m256 kC3 = _mm256_set1_ps(0.41573480615127262f);
    const __m256 kC5 = _mm256_set1_ps(0.27778511650980114f);
    const __m256 kC6 = _mm256_set1_ps(0.19134171618254492f);
    const __m256 kC7 = _mm256_set1_ps(0.09754516100806417f);

    __m256 p0 = _mm256_mul_ps(_mm256_add_ps(y0, y4), kS0);
    __m256 p1 = _mm256_mul_ps(_mm256_sub_ps(y0, y4), kS0);
    __m256 q0 = _mm256_fmadd_ps(y2, kC2, _mm256_mul_ps(y6, kC6));
    __m256 q1 = _mm256_fmsub_ps(y2, kC6, _mm256_mul_ps(y6, kC2));

    __m256 ev0 = _mm256_add_ps(p0, q0);
    __m256 ev3 = _mm256_sub_ps(p0, q0);
    __m256 ev1 = _mm256_add_ps(p1, q1);
    __m256 ev2 = _mm256_sub_ps(p1, q1);

    __m256 od0 = _mm256_fmadd_ps(kC1, y1, _mm256_fmadd_ps(kC3, y3, _mm256_fmadd_ps(kC5, y5, _mm256_mul_ps(kC7, y7))));
    __m256 od1 = _mm256_fmsub_ps(kC3, y1, _mm256_fmadd_ps(kC7, y3, _mm256_fmadd_ps(kC1, y5, _mm256_mul_ps(kC5, y7))));
    __m256 od2 = _mm256_fmadd_ps(kC5, y1, _mm256_fnmadd_ps(kC1, y3, _mm256_fmadd_ps(kC7, y5, _mm256_mul_ps(kC3, y7))));
    __m256 od3 = _mm256_fmadd_ps(kC7, y1, _mm256_fnmadd_ps(kC5, y3, _mm256_fmsub_ps(kC3, y5, _mm256_mul_ps(kC1, y7))));

    y0 = _mm256_add_ps(ev0, od0);
    y7 = _mm256_sub_ps(ev0, od0);
    y1 = _mm256_add_ps(ev1, od1);
    y6 = _mm256_sub_ps(ev1, od1);
    y2 = _mm256_add_ps(ev2, od2);
    y5 = _mm256_sub_ps(ev2, od2);
    y3 = _mm256_add_ps(ev3, od3);
    y4 = _mm256_sub_ps(ev3, od3);
}

inline void dct8x8_2d(const float in[64], float out[64]) noexcept {
    __m256 r0 = _mm256_loadu_ps(in + 0);
    __m256 r1 = _mm256_loadu_ps(in + 8);
    __m256 r2 = _mm256_loadu_ps(in + 16);
    __m256 r3 = _mm256_loadu_ps(in + 24);
    __m256 r4 = _mm256_loadu_ps(in + 32);
    __m256 r5 = _mm256_loadu_ps(in + 40);
    __m256 r6 = _mm256_loadu_ps(in + 48);
    __m256 r7 = _mm256_loadu_ps(in + 56);

    dct8_cols_avx2(r0, r1, r2, r3, r4, r5, r6, r7);
    transpose8x8_ps(r0, r1, r2, r3, r4, r5, r6, r7);
    dct8_cols_avx2(r0, r1, r2, r3, r4, r5, r6, r7);
    transpose8x8_ps(r0, r1, r2, r3, r4, r5, r6, r7);

    _mm256_storeu_ps(out + 0, r0);
    _mm256_storeu_ps(out + 8, r1);
    _mm256_storeu_ps(out + 16, r2);
    _mm256_storeu_ps(out + 24, r3);
    _mm256_storeu_ps(out + 32, r4);
    _mm256_storeu_ps(out + 40, r5);
    _mm256_storeu_ps(out + 48, r6);
    _mm256_storeu_ps(out + 56, r7);
}

inline void idct8x8_2d(const float in[64], float out[64]) noexcept {
    __m256 r0 = _mm256_loadu_ps(in + 0);
    __m256 r1 = _mm256_loadu_ps(in + 8);
    __m256 r2 = _mm256_loadu_ps(in + 16);
    __m256 r3 = _mm256_loadu_ps(in + 24);
    __m256 r4 = _mm256_loadu_ps(in + 32);
    __m256 r5 = _mm256_loadu_ps(in + 40);
    __m256 r6 = _mm256_loadu_ps(in + 48);
    __m256 r7 = _mm256_loadu_ps(in + 56);

    idct8_cols_avx2(r0, r1, r2, r3, r4, r5, r6, r7);
    transpose8x8_ps(r0, r1, r2, r3, r4, r5, r6, r7);
    idct8_cols_avx2(r0, r1, r2, r3, r4, r5, r6, r7);
    transpose8x8_ps(r0, r1, r2, r3, r4, r5, r6, r7);

    _mm256_storeu_ps(out + 0, r0);
    _mm256_storeu_ps(out + 8, r1);
    _mm256_storeu_ps(out + 16, r2);
    _mm256_storeu_ps(out + 24, r3);
    _mm256_storeu_ps(out + 32, r4);
    _mm256_storeu_ps(out + 40, r5);
    _mm256_storeu_ps(out + 48, r6);
    _mm256_storeu_ps(out + 56, r7);
}

inline float load_patch8x8_mean_avx2(const float* src, int32_t stride, float dst[64]) noexcept {
    __m256 sum = _mm256_setzero_ps();
    for (int py = 0; py < 8; ++py) {
        __m256 row = _mm256_loadu_ps(src + static_cast<ptrdiff_t>(py) * stride);
        _mm256_store_ps(dst + py * 8, row);
        sum = _mm256_add_ps(sum, row);
    }
    return hsum256_ps(sum) * (1.0f / 64.0f);
}

inline void load_patch8x8_avx2(const float* src, int32_t stride, float dst[64]) noexcept {
    for (int py = 0; py < 8; ++py) {
        _mm256_store_ps(dst + py * 8, _mm256_loadu_ps(src + static_cast<ptrdiff_t>(py) * stride));
    }
}

inline float load_cand_mse8x8_avx2(const float* cand_src, int32_t stride, const float ref_aligned[64], float cand_aligned[64]) noexcept {
    __m256 ssd = _mm256_setzero_ps();
    for (int py = 0; py < 8; ++py) {
        __m256 c = _mm256_loadu_ps(cand_src + static_cast<ptrdiff_t>(py) * stride);
        _mm256_store_ps(cand_aligned + py * 8, c);
        __m256 d = _mm256_sub_ps(_mm256_load_ps(ref_aligned + py * 8), c);
        ssd = _mm256_fmadd_ps(d, d, ssd);
    }
    return hsum256_ps(ssd) * (1.0f / 64.0f);
}

inline void accum_patch8x8_feathered_avx2(
    float* num_base,
    float* den_base,
    int32_t stride,
    const float patch_aligned[64],
    float inv_weight
) noexcept {
    const __m256 scale_v = _mm256_set1_ps(inv_weight);
    for (int py = 0; py < 8; ++py) {
        __m256 w_agg = _mm256_mul_ps(_mm256_load_ps(g_spatial_lut.feather_2d + py * 8), scale_v);
        __m256 fv = _mm256_load_ps(patch_aligned + py * 8);
        float* np = num_base + static_cast<ptrdiff_t>(py) * stride;
        float* dp = den_base + static_cast<ptrdiff_t>(py) * stride;
        _mm256_storeu_ps(np, _mm256_fmadd_ps(fv, w_agg, _mm256_loadu_ps(np)));
        _mm256_storeu_ps(dp, _mm256_add_ps(w_agg, _mm256_loadu_ps(dp)));
    }
}

inline float eval_tonal_lut_cpu(const std::array<float, 5>& lut, float val) noexcept {
    float t = std::clamp(val, 0.0f, 1.0f) * 4.0f;
    int idx = std::min(3, std::max(0, static_cast<int>(t)));
    float frac = t - static_cast<float>(idx);
    return lut[idx] * (1.0f - frac) + lut[idx + 1] * frac;
}

inline float effective_sigma_cpu_raw(const NoiseLevelFunction& nlf, const FilterParams& params, float luma_intensity) noexcept {
    const float amount = (params.channel_index == 0) ? params.luma_scale
                       : (params.channel_index == 1) ? params.chroma_scale_cb : params.chroma_scale_cr;
    return nlf.sigma_at(luma_intensity) * params.spatial_weight * eval_tonal_lut_cpu(params.tonal_lut, luma_intensity) * amount;
}

// Noise level lookups always use LUMA intensity (shot noise follows brightness). Luma plane filters use
// their own value; chroma planes consult the reference-frame luma guide when one is available.
inline float effective_sigma_cpu(const NoiseLevelFunction& nlf, const FilterParams& params, float val) noexcept {
    return effective_sigma_cpu_raw(nlf, params, val);
}

inline float guide_at(const FilterParams& params, int32_t x, int32_t y, float fallback) noexcept {
    if (params.luma_guide == nullptr || params.channel_index == 0) return fallback;
    const ImagePlane32F& g = *params.luma_guide;
    return g.at(std::clamp(x, 0, g.width() - 1), std::clamp(y, 0, g.height() - 1));
}

inline float guide_patch_mean(const FilterParams& params, int32_t x, int32_t y, int32_t ps, float plane_mean) noexcept {
    if (params.luma_guide == nullptr || params.channel_index == 0) return plane_mean;
    const ImagePlane32F& g = *params.luma_guide;
    const int32_t x0 = std::clamp(x, 0, std::max(0, g.width() - ps));
    const int32_t y0 = std::clamp(y, 0, std::max(0, g.height() - ps));
    float sum = 0.0f;
    for (int32_t py = 0; py < ps; ++py) {
        const float* row = g.data() + static_cast<ptrdiff_t>(y0 + py) * g.stride() + x0;
        for (int32_t px = 0; px < ps; ++px) sum += row[px];
    }
    return sum / static_cast<float>(ps * ps);
}

// Motion lookup for patch-matching engines: an untrusted block falls back to zero motion and the
// per-patch distance test (against the noise floor) decides whether the candidate is usable. This
// avoids whole regions flipping between temporally-filtered and spatial-only from frame to frame.
inline bool fetch_patch_motion(const MotionField* mf, int32_t px, int32_t py, int32_t& dx, int32_t& dy) noexcept {
    dx = 0; dy = 0;
    if (mf == nullptr) return true;
    if (mf->is_valid()) {
        const auto mv = mf->get_vector_at_pixel(px, py);
        if (mv.trusted) { dx = mv.dx; dy = mv.dy; }
        return true;
    }
    return !mf->is_scene_cut;
}

inline float max_band_scale_cpu(const NoisePsd& psd, const FilterParams& params) noexcept {
    size_t ch = static_cast<size_t>(std::clamp(params.channel_index, 0, 2));
    float m = 1.0f;
    for (size_t b = 0; b < 5; ++b) {
        m = std::max(m, psd.octave_weights[ch][b] * params.band_weights[b]);
    }
    return m;
}

// Motion-compensated temporal accumulation helper
ImagePlane32F compute_temporal_average(
    const std::vector<const ImagePlane32F*>& temporal_planes,
    size_t ref_index,
    const std::vector<const MotionField*>& motion_fields,
    const NoiseLevelFunction& nlf,
    const NoisePsd& psd,
    const FilterParams& params
) {
    const int32_t w = temporal_planes[ref_index]->width();
    const int32_t h = temporal_planes[ref_index]->height();
    if (temporal_planes.size() <= 1 || params.temporal_weight <= 0.01f) {
        return *temporal_planes[ref_index];
    }

    ImagePlane32F avg(w, h);
    const float max_b = max_band_scale_cpu(psd, params);
    const int n_threads = resolve_filter_threads(params.max_threads);
    std::atomic<bool> aborted{false};

    #pragma omp parallel for schedule(dynamic, 16) num_threads(n_threads)
    for (int32_t y = 0; y < h; ++y) {
        if ((y & 15) == 0) {
#ifdef _OPENMP
            if (omp_get_thread_num() == 0 && params.is_aborted()) {
                aborted.store(true, std::memory_order_relaxed);
            }
#else
            if (params.is_aborted()) aborted.store(true, std::memory_order_relaxed);
#endif
        }
        if (aborted.load(std::memory_order_relaxed)) continue;

        for (int32_t x = 0; x < w; ++x) {
            float ref_val = temporal_planes[ref_index]->at(x, y);
            float sigma = effective_sigma_cpu(nlf, params, guide_at(params, x, y, ref_val)) * max_b;
            float trust_limit = 3.5f * sigma * params.temporal_weight;

            float sum = ref_val;
            float weight_sum = 1.0f;

            for (size_t f = 0; f < temporal_planes.size(); ++f) {
                if (f == ref_index) continue;
                int32_t dx = 0, dy = 0;
                if (!fetch_patch_motion(motion_fields[f], x, y, dx, dy)) continue;

                int32_t cx = std::clamp(x + dx, 0, w - 1);
                int32_t cy = std::clamp(y + dy, 0, h - 1);
                float cand_val = temporal_planes[f]->at(cx, cy);

                // Smooth (Tukey biweight) acceptance: no hard in/out switch, so no frame-to-frame flicker
                const float t = std::abs(ref_val - cand_val) / std::max(trust_limit, 1e-8f);
                if (t < 1.0f) {
                    const float u = 1.0f - t * t;
                    const float wgt = u * u;
                    sum += wgt * cand_val;
                    weight_sum += wgt;
                }
            }
            avg.at(x, y) = sum / weight_sum;
        }
    }
    return avg;
}

} // anonymous namespace

// =========================================================================
// Engine 1: 3D Collaborative Transform Filter (2-Stage V-BM3D + Non-Local Wiener)
// Zero-allocation 2-phase conflict-free strip parallelism + AVX2 3D DCT
// =========================================================================
void CollaborativePatchFilter3D::process_plane(
    const std::vector<const ImagePlane32F*>& temporal_planes,
    size_t ref_index,
    const std::vector<const MotionField*>& motion_fields,
    const NoiseLevelFunction& nlf,
    const NoisePsd& psd,
    const FilterParams& params,
    ImagePlane32F& output_plane
) {
    const int32_t w = output_plane.width();
    const int32_t h = output_plane.height();
    const int32_t ps = 8;
    const int32_t step = std::max(1, params.step_size);
    const size_t ch_idx = static_cast<size_t>(std::clamp(params.channel_index, 0, 2));
    const float max_b = max_band_scale_cpu(psd, params);
    const int n_threads = resolve_filter_threads(params.max_threads);

    if (w < ps || h < ps) {
        output_plane = *temporal_planes[ref_index];
        return;
    }

    // Precompute 64-coefficient radial frequency band scales once per channel plane
    alignas(32) float plane_bscale[64];
    alignas(32) float plane_bscale_sq[64];
    for (int i = 0; i < 64; ++i) {
        int band = g_spatial_lut.band[i];
        float bs = std::max(0.05f, psd.octave_weights[ch_idx][band] * params.band_weights[band]);
        plane_bscale[i] = bs;
        plane_bscale_sq[i] = bs * bs;
    }

    const int32_t npx = (w - ps + step - 1) / step + 1;
    const int32_t npy = (h - ps + step - 1) / step + 1;
    constexpr int32_t kStripSteps = 16; // >= 16..96 rows per strip (> ps=8), guaranteeing disjoint parity writes
    const int32_t num_strips = (npy + kStripSteps - 1) / kStripSteps;

    ImagePlane32F numerator(w, h);
    ImagePlane32F denominator(w, h);
    numerator.fill(0.0f);
    denominator.fill(0.0f);

    const ImagePlane32F* ref_plane = temporal_planes[ref_index];
    const float* ref_data = ref_plane->data();
    const int32_t ref_stride = ref_plane->stride();
    float* num_data = numerator.data();
    float* den_data = denominator.data();
    const int32_t acc_stride = numerator.stride();

    const float tm_mult = std::max(0.5f, params.temporal_weight);
    std::atomic<bool> aborted{false};

    // Pass 1: Motion-Compensated 3D Hard-Thresholding -> Pilot
    for (int parity = 0; parity < 2; ++parity) {
        if (aborted.load(std::memory_order_relaxed) || params.is_aborted()) {
            output_plane = *ref_plane;
            return;
        }

        #pragma omp parallel for schedule(dynamic, 1) num_threads(n_threads)
        for (int32_t strip = parity; strip < num_strips; strip += 2) {
#ifdef _OPENMP
            if (omp_get_thread_num() == 0 && params.is_aborted()) {
                aborted.store(true, std::memory_order_relaxed);
            }
#else
            if (params.is_aborted()) aborted.store(true, std::memory_order_relaxed);
#endif
            if (aborted.load(std::memory_order_relaxed)) continue;

            const int32_t iy_start = strip * kStripSteps;
            const int32_t iy_end = std::min(npy, iy_start + kStripSteps);

            alignas(32) float group[9][64];
            alignas(32) float spatial_coeffs[9][64];
            alignas(32) float cand_patch[64];
            alignas(32) float ref0_spec[64];
            alignas(32) float filtered_ref_patch[64];

            const __m256 sign_mask = _mm256_set1_ps(-0.0f);
            const __m256 zero_v = _mm256_setzero_ps();

            for (int32_t iy = iy_start; iy < iy_end; ++iy) {
                int32_t y = std::min(iy * step, h - ps);
                for (int32_t ix = 0; ix < npx; ++ix) {
                    int32_t x = std::min(ix * step, w - ps);

                    float mean_v = load_patch8x8_mean_avx2(
                        ref_data + static_cast<ptrdiff_t>(y) * ref_stride + x,
                        ref_stride,
                        group[0]
                    );

                    float sigma = effective_sigma_cpu(nlf, params, guide_patch_mean(params, x, y, ps, mean_v));
                    float eff_sig = sigma * max_b;
                    float trust_dist = 4.5f * std::max(0.25f, params.temporal_weight) * (eff_sig * eff_sig + 1e-6f);

                    int K = 1;
                    if (temporal_planes.size() > 1 && params.temporal_weight > 0.01f) {
                        for (size_t f = 0; f < temporal_planes.size(); ++f) {
                            if (f == ref_index || K >= 9) continue;

                            int32_t disp_x = 0, disp_y = 0;
                            if (!fetch_patch_motion(motion_fields[f], x + 4, y + 4, disp_x, disp_y)) continue;

                            int32_t cand_x = std::clamp(x + disp_x, 0, w - ps);
                            int32_t cand_y = std::clamp(y + disp_y, 0, h - ps);
                            const float* cp = temporal_planes[f]->data() +
                                              static_cast<ptrdiff_t>(cand_y) * temporal_planes[f]->stride() + cand_x;

                            float dist = load_cand_mse8x8_avx2(cp, temporal_planes[f]->stride(), group[0], cand_patch);
                            if (dist <= trust_dist) {
                                for (int v = 0; v < 8; ++v) {
                                    _mm256_store_ps(group[K] + v * 8, _mm256_load_ps(cand_patch + v * 8));
                                }
                                K++;
                            }
                        }
                    }

                    for (int k = 0; k < K; ++k) {
                        dct8x8_2d(group[k], spatial_coeffs[k]);
                    }

                    const __m256 sp_scale = _mm256_set1_ps(params.hard_thresh_lambda * sigma);
                    const __m256 tm_scale = _mm256_set1_ps(params.hard_thresh_lambda * sigma * tm_mult);

                    int non_zero_count = 0;
                    if (K == 1) {
                        for (int v = 0; v < 8; ++v) {
                            __m256 c = _mm256_load_ps(spatial_coeffs[0] + v * 8);
                            __m256 th = _mm256_mul_ps(sp_scale, _mm256_load_ps(plane_bscale + v * 8));
                            if (v == 0) th = _mm256_blend_ps(th, zero_v, 0x01); // Keep 3D DC (0,0,0)
                            __m256 mask = _mm256_cmp_ps(_mm256_andnot_ps(sign_mask, c), th, _CMP_GE_OQ);
                            non_zero_count += popcount8(_mm256_movemask_ps(mask));
                            _mm256_store_ps(ref0_spec + v * 8, _mm256_and_ps(c, mask));
                        }
                    } else {
                        __m256 acc[8];
                        for (int v = 0; v < 8; ++v) acc[v] = zero_v;

                        for (int k = 0; k < K; ++k) {
                            const float* tk_row = g_tdct_table.mat[K][k];
                            const __m256 inv0_v = _mm256_set1_ps(tk_row[0]);
                            const __m256 th_scale = (k == 0) ? sp_scale : tm_scale;

                            for (int v = 0; v < 8; ++v) {
                                __m256 tdct = _mm256_mul_ps(_mm256_set1_ps(tk_row[0]), _mm256_load_ps(spatial_coeffs[0] + v * 8));
                                for (int n = 1; n < K; ++n) {
                                    tdct = _mm256_fmadd_ps(_mm256_set1_ps(tk_row[n]), _mm256_load_ps(spatial_coeffs[n] + v * 8), tdct);
                                }
                                __m256 th = _mm256_mul_ps(th_scale, _mm256_load_ps(plane_bscale + v * 8));
                                if (k == 0 && v == 0) th = _mm256_blend_ps(th, zero_v, 0x01);
                                __m256 mask = _mm256_cmp_ps(_mm256_andnot_ps(sign_mask, tdct), th, _CMP_GE_OQ);
                                non_zero_count += popcount8(_mm256_movemask_ps(mask));
                                __m256 shrunk = _mm256_and_ps(tdct, mask);
                                acc[v] = _mm256_fmadd_ps(inv0_v, shrunk, acc[v]);
                            }
                        }
                        for (int v = 0; v < 8; ++v) {
                            _mm256_store_ps(ref0_spec + v * 8, acc[v]);
                        }
                    }

                    idct8x8_2d(ref0_spec, filtered_ref_patch);

                    float inv_nz = 1.0f / static_cast<float>(std::max(1, non_zero_count));
                    accum_patch8x8_feathered_avx2(
                        num_data + static_cast<ptrdiff_t>(y) * acc_stride + x,
                        den_data + static_cast<ptrdiff_t>(y) * acc_stride + x,
                        acc_stride,
                        filtered_ref_patch,
                        inv_nz
                    );
                }
            }
        }
    }

    if (aborted.load(std::memory_order_relaxed) || params.is_aborted()) {
        output_plane = *ref_plane;
        return;
    }

    ImagePlane32F pilot(w, h);
    #pragma omp parallel for schedule(static) num_threads(n_threads)
    for (int32_t y = 0; y < h; ++y) {
        for (int32_t x = 0; x < w; ++x) {
            float d = denominator.at(x, y);
            pilot.at(x, y) = (d > 1e-7f) ? (numerator.at(x, y) / d) : ref_plane->at(x, y);
        }
    }

    if (!params.enable_wiener_pass) {
        output_plane = pilot;
        return;
    }

    // Pass 2: Spatiotemporal Non-Local Collaborative 3D Empirical Wiener
    numerator.fill(0.0f);
    denominator.fill(0.0f);

    const float* pilot_data = pilot.data();
    const int32_t pilot_stride = pilot.stride();

    const int ox_list[12] = {-5,  5,  0,  0, -5,  5, -5,  5, -10, 10,   0,  0};
    const int oy_list[12] = { 0,  0, -5,  5, -5, -5,  5,  5,   0,  0, -10, 10};
    int max_spatial_cands = (params.quality == QualityPreset::HighQuality) ? 12
                          : (params.quality == QualityPreset::Fast)        ? 4 : 8;
    if (params.channel_index > 0 && params.fast_chroma) {
        max_spatial_cands = std::min(max_spatial_cands, 4);
    }

    for (int parity = 0; parity < 2; ++parity) {
        if (aborted.load(std::memory_order_relaxed) || params.is_aborted()) {
            output_plane = pilot;
            return;
        }

        #pragma omp parallel for schedule(dynamic, 1) num_threads(n_threads)
        for (int32_t strip = parity; strip < num_strips; strip += 2) {
#ifdef _OPENMP
            if (omp_get_thread_num() == 0 && params.is_aborted()) {
                aborted.store(true, std::memory_order_relaxed);
            }
#else
            if (params.is_aborted()) aborted.store(true, std::memory_order_relaxed);
#endif
            if (aborted.load(std::memory_order_relaxed)) continue;

            const int32_t iy_start = strip * kStripSteps;
            const int32_t iy_end = std::min(npy, iy_start + kStripSteps);

            alignas(32) float noisy_grp[12][64];
            alignas(32) float pilot_grp[12][64];
            alignas(32) float cand_buf[64];
            alignas(32) float n_spec[12][64];
            alignas(32) float p_spec[12][64];
            alignas(32) float ref0_spec[64];
            alignas(32) float wiener_patch[64];

            const __m256 zero_v = _mm256_setzero_ps();
            const __m256 one_v = _mm256_set1_ps(1.0f);
            const __m256 eps_v = _mm256_set1_ps(1e-10f);

            for (int32_t iy = iy_start; iy < iy_end; ++iy) {
                int32_t y = std::min(iy * step, h - ps);
                for (int32_t ix = 0; ix < npx; ++ix) {
                    int32_t x = std::min(ix * step, w - ps);

                    load_patch8x8_avx2(
                        ref_data + static_cast<ptrdiff_t>(y) * ref_stride + x,
                        ref_stride,
                        noisy_grp[0]
                    );
                    float mean_p = load_patch8x8_mean_avx2(
                        pilot_data + static_cast<ptrdiff_t>(y) * pilot_stride + x,
                        pilot_stride,
                        pilot_grp[0]
                    );

                    float sigma = effective_sigma_cpu(nlf, params, guide_patch_mean(params, x, y, ps, mean_p));
                    float eff_var = (sigma * max_b) * (sigma * max_b) + 1e-7f;
                    int K = 1;

                    // Temporal matches
                    if (temporal_planes.size() > 1 && params.temporal_weight > 0.01f) {
                        float temp_trust = 4.0f * std::max(0.25f, params.temporal_weight) * eff_var;
                        for (size_t f = 0; f < temporal_planes.size(); ++f) {
                            if (f == ref_index || K >= 12) continue;
                            int32_t disp_x = 0, disp_y = 0;
                            if (!fetch_patch_motion(motion_fields[f], x + 4, y + 4, disp_x, disp_y)) continue;

                            int32_t cx = std::clamp(x + disp_x, 0, w - ps);
                            int32_t cy = std::clamp(y + disp_y, 0, h - ps);
                            const float* cp = temporal_planes[f]->data() +
                                              static_cast<ptrdiff_t>(cy) * temporal_planes[f]->stride() + cx;

                            float dist = load_cand_mse8x8_avx2(cp, temporal_planes[f]->stride(), noisy_grp[0], cand_buf);
                            if (dist <= temp_trust) {
                                for (int v = 0; v < 8; ++v) {
                                    _mm256_store_ps(noisy_grp[K] + v * 8, _mm256_load_ps(cand_buf + v * 8));
                                    _mm256_store_ps(pilot_grp[K] + v * 8, _mm256_load_ps(pilot_grp[0] + v * 8));
                                }
                                K++;
                            }
                        }
                    }

                    // Spatial non-local matches on denoised pilot
                    if (params.spatial_weight > 0.01f && K < 12) {
                        float spat_trust = 1.6f * eff_var;
                        for (int s = 0; s < max_spatial_cands && K < 12; ++s) {
                            int32_t sx = x + ox_list[s];
                            int32_t sy = y + oy_list[s];
                            if (sx < 0 || sx + ps > w || sy < 0 || sy + ps > h) continue;

                            const float* pp = pilot_data + static_cast<ptrdiff_t>(sy) * pilot_stride + sx;
                            float dist_p = load_cand_mse8x8_avx2(pp, pilot_stride, pilot_grp[0], cand_buf);
                            if (dist_p <= spat_trust) {
                                load_patch8x8_avx2(
                                    ref_data + static_cast<ptrdiff_t>(sy) * ref_stride + sx,
                                    ref_stride,
                                    noisy_grp[K]
                                );
                                for (int v = 0; v < 8; ++v) {
                                    _mm256_store_ps(pilot_grp[K] + v * 8, _mm256_load_ps(cand_buf + v * 8));
                                }
                                K++;
                            }
                        }
                    }

                    for (int k = 0; k < K; ++k) {
                        dct8x8_2d(noisy_grp[k], n_spec[k]);
                        dct8x8_2d(pilot_grp[k], p_spec[k]);
                    }

                    const __m256 sig_sq_sp = _mm256_set1_ps(sigma * sigma);
                    const __m256 sig_sq_tm = _mm256_set1_ps(sigma * sigma * tm_mult);
                    __m256 w_sq_acc = zero_v;

                    if (K == 1) {
                        for (int v = 0; v < 8; ++v) {
                            __m256 ns = _mm256_load_ps(n_spec[0] + v * 8);
                            __m256 ps_v = _mm256_load_ps(p_spec[0] + v * 8);
                            __m256 p2 = _mm256_mul_ps(ps_v, ps_v);
                            __m256 nvar = _mm256_mul_ps(sig_sq_sp, _mm256_load_ps(plane_bscale_sq + v * 8));
                            __m256 w_gain = _mm256_div_ps(p2, _mm256_add_ps(p2, _mm256_add_ps(nvar, eps_v)));
                            if (v == 0) w_gain = _mm256_blend_ps(w_gain, one_v, 0x01);
                            w_sq_acc = _mm256_fmadd_ps(w_gain, w_gain, w_sq_acc);
                            _mm256_store_ps(ref0_spec + v * 8, _mm256_mul_ps(ns, w_gain));
                        }
                    } else {
                        __m256 acc[8];
                        for (int v = 0; v < 8; ++v) acc[v] = zero_v;

                        for (int k = 0; k < K; ++k) {
                            const float* tk_row = g_tdct_table.mat[K][k];
                            const __m256 inv0_v = _mm256_set1_ps(tk_row[0]);
                            const __m256 sig_sq = (k == 0) ? sig_sq_sp : sig_sq_tm;

                            for (int v = 0; v < 8; ++v) {
                                __m256 t0 = _mm256_set1_ps(tk_row[0]);
                                __m256 n_tdct = _mm256_mul_ps(t0, _mm256_load_ps(n_spec[0] + v * 8));
                                __m256 p_tdct = _mm256_mul_ps(t0, _mm256_load_ps(p_spec[0] + v * 8));
                                for (int n = 1; n < K; ++n) {
                                    __m256 tn = _mm256_set1_ps(tk_row[n]);
                                    n_tdct = _mm256_fmadd_ps(tn, _mm256_load_ps(n_spec[n] + v * 8), n_tdct);
                                    p_tdct = _mm256_fmadd_ps(tn, _mm256_load_ps(p_spec[n] + v * 8), p_tdct);
                                }
                                __m256 p2 = _mm256_mul_ps(p_tdct, p_tdct);
                                __m256 nvar = _mm256_mul_ps(sig_sq, _mm256_load_ps(plane_bscale_sq + v * 8));
                                __m256 w_gain = _mm256_div_ps(p2, _mm256_add_ps(p2, _mm256_add_ps(nvar, eps_v)));
                                if (k == 0 && v == 0) w_gain = _mm256_blend_ps(w_gain, one_v, 0x01);
                                w_sq_acc = _mm256_fmadd_ps(w_gain, w_gain, w_sq_acc);
                                __m256 shrunk = _mm256_mul_ps(n_tdct, w_gain);
                                acc[v] = _mm256_fmadd_ps(inv0_v, shrunk, acc[v]);
                            }
                        }
                        for (int v = 0; v < 8; ++v) {
                            _mm256_store_ps(ref0_spec + v * 8, acc[v]);
                        }
                    }

                    idct8x8_2d(ref0_spec, wiener_patch);

                    float w_sq_sum = hsum256_ps(w_sq_acc);
                    float inv_w = 1.0f / std::max(1e-2f, w_sq_sum);
                    accum_patch8x8_feathered_avx2(
                        num_data + static_cast<ptrdiff_t>(y) * acc_stride + x,
                        den_data + static_cast<ptrdiff_t>(y) * acc_stride + x,
                        acc_stride,
                        wiener_patch,
                        inv_w
                    );
                }
            }
        }
    }

    if (aborted.load(std::memory_order_relaxed) || params.is_aborted()) {
        output_plane = pilot;
        return;
    }

    #pragma omp parallel for schedule(static) num_threads(n_threads)
    for (int32_t y = 0; y < h; ++y) {
        for (int32_t x = 0; x < w; ++x) {
            float d = denominator.at(x, y);
            output_plane.at(x, y) = (d > 1e-7f) ? (numerator.at(x, y) / d) : pilot.at(x, y);
        }
    }
}

// =========================================================================
// Engine 2: 5-Band Multi-Scale Wavelet / Pyramid Filter
// =========================================================================
namespace {

ImagePlane32F downsample2x(const ImagePlane32F& src, int n_threads) {
    int32_t sw = src.width();
    int32_t sh = src.height();
    int32_t dw = std::max(1, sw / 2);
    int32_t dh = std::max(1, sh / 2);
    ImagePlane32F dst(dw, dh);

    #pragma omp parallel for schedule(static) num_threads(n_threads)
    for (int32_t y = 0; y < dh; ++y) {
        int32_t sy = y * 2;
        int32_t sy1 = std::min(sh - 1, sy + 1);
        for (int32_t x = 0; x < dw; ++x) {
            int32_t sx = x * 2;
            int32_t sx1 = std::min(sw - 1, sx + 1);
            dst.at(x, y) = 0.25f * (src.at(sx, sy) + src.at(sx1, sy) + src.at(sx, sy1) + src.at(sx1, sy1));
        }
    }
    return dst;
}

ImagePlane32F upsample2x(const ImagePlane32F& src, int32_t target_w, int32_t target_h, int n_threads) {
    ImagePlane32F dst(target_w, target_h);
    int32_t sw = src.width();
    int32_t sh = src.height();

    #pragma omp parallel for schedule(static) num_threads(n_threads)
    for (int32_t y = 0; y < target_h; ++y) {
        float v = (static_cast<float>(y) + 0.5f) * (static_cast<float>(sh) / static_cast<float>(target_h)) - 0.5f;
        int32_t y0 = std::clamp(static_cast<int32_t>(std::floor(v)), 0, sh - 1);
        int32_t y1 = std::clamp(y0 + 1, 0, sh - 1);
        float fy = v - static_cast<float>(y0);

        for (int32_t x = 0; x < target_w; ++x) {
            float u = (static_cast<float>(x) + 0.5f) * (static_cast<float>(sw) / static_cast<float>(target_w)) - 0.5f;
            int32_t x0 = std::clamp(static_cast<int32_t>(std::floor(u)), 0, sw - 1);
            int32_t x1 = std::clamp(x0 + 1, 0, sw - 1);
            float fx = u - static_cast<float>(x0);

            float p00 = src.at(x0, y0);
            float p10 = src.at(x1, y0);
            float p01 = src.at(x0, y1);
            float p11 = src.at(x1, y1);

            dst.at(x, y) = (1.0f - fx) * (1.0f - fy) * p00 +
                           fx * (1.0f - fy) * p10 +
                           (1.0f - fx) * fy * p01 +
                           fx * fy * p11;
        }
    }
    return dst;
}

} // anonymous namespace

void Wavelet5BandFilter::process_plane(
    const std::vector<const ImagePlane32F*>& temporal_planes,
    size_t ref_index,
    const std::vector<const MotionField*>& motion_fields,
    const NoiseLevelFunction& nlf,
    const NoisePsd& psd,
    const FilterParams& params,
    ImagePlane32F& output_plane
) {
    const int n_threads = resolve_filter_threads(params.max_threads);
    ImagePlane32F current = compute_temporal_average(temporal_planes, ref_index, motion_fields, nlf, psd, params);
    if (params.is_aborted()) {
        output_plane = current;
        return;
    }

    const size_t ch_idx = static_cast<size_t>(std::clamp(params.channel_index, 0, 2));
    const int32_t num_levels = 5;
    std::vector<ImagePlane32F> pyramid(num_levels + 1);
    std::vector<ImagePlane32F> details(num_levels);

    pyramid[0] = current;

    // Luma guide pyramid so chroma planes evaluate their noise level against local brightness
    const bool use_guide = (params.luma_guide != nullptr && params.channel_index > 0);
    std::vector<ImagePlane32F> guide_pyr;
    if (use_guide) {
        guide_pyr.resize(num_levels + 1);
        guide_pyr[0] = *params.luma_guide;
        for (int l = 0; l < num_levels; ++l) guide_pyr[l + 1] = downsample2x(guide_pyr[l], n_threads);
    }
    for (int l = 0; l < num_levels; ++l) {
        if (params.is_aborted()) {
            output_plane = current;
            return;
        }
        pyramid[l + 1] = downsample2x(pyramid[l], n_threads);
        ImagePlane32F up = upsample2x(pyramid[l + 1], pyramid[l].width(), pyramid[l].height(), n_threads);

        details[l] = ImagePlane32F(pyramid[l].width(), pyramid[l].height());
        int band = std::clamp(4 - l, 0, 4);
        float band_mult = std::max(0.05f, psd.octave_weights[ch_idx][band] * params.band_weights[band]);
        float level_scale = std::pow(0.55f, static_cast<float>(l));

        int32_t pw = pyramid[l].width();
        int32_t ph = pyramid[l].height();

        #pragma omp parallel for schedule(static) num_threads(n_threads)
        for (int32_t y = 0; y < ph; ++y) {
            for (int32_t x = 0; x < pw; ++x) {
                float raw_diff = pyramid[l].at(x, y) - up.at(x, y);
                float base_val = use_guide ? guide_pyr[l].at(x, y) : up.at(x, y);
                float sigma = effective_sigma_cpu(nlf, params, base_val);

                float thresh = 1.6f * sigma * band_mult * level_scale;
                float pwr = raw_diff * raw_diff;
                float gain = std::max(0.0f, (pwr - thresh * thresh) / (pwr + 1e-8f));
                details[l].at(x, y) = raw_diff * gain;
            }
        }
    }

    ImagePlane32F recon = pyramid[num_levels];
    for (int l = num_levels - 1; l >= 0; --l) {
        ImagePlane32F up = upsample2x(recon, details[l].width(), details[l].height(), n_threads);
        recon = ImagePlane32F(details[l].width(), details[l].height());

        int32_t rw = details[l].width();
        int32_t rh = details[l].height();

        #pragma omp parallel for schedule(static) num_threads(n_threads)
        for (int32_t y = 0; y < rh; ++y) {
            for (int32_t x = 0; x < rw; ++x) {
                recon.at(x, y) = up.at(x, y) + details[l].at(x, y);
            }
        }
    }

    output_plane = recon;
}

// =========================================================================
// Engine 3: Accelerated Non-Local Means (NLM)
// =========================================================================
void NonLocalMeansFilter::process_plane(
    const std::vector<const ImagePlane32F*>& temporal_planes,
    size_t ref_index,
    const std::vector<const MotionField*>& motion_fields,
    const NoiseLevelFunction& nlf,
    const NoisePsd& psd,
    const FilterParams& params,
    ImagePlane32F& output_plane
) {
    const int n_threads = resolve_filter_threads(params.max_threads);
    ImagePlane32F temp_accum = compute_temporal_average(temporal_planes, ref_index, motion_fields, nlf, psd, params);
    if (params.is_aborted()) {
        output_plane = temp_accum;
        return;
    }

    const float max_b = max_band_scale_cpu(psd, params);
    const int32_t w = temp_accum.width();
    const int32_t h = temp_accum.height();
    const int32_t r_search = (params.quality == QualityPreset::Fast) ? 4 : params.nlm_search_radius;
    const int32_t s_step = (params.quality == QualityPreset::HighQuality) ? 1 : 2;
    const int32_t r_patch = params.nlm_patch_radius;
    const float patch_norm = 1.0f / static_cast<float>((2 * r_patch + 1) * (2 * r_patch + 1));
    const float* src_data = temp_accum.data();
    const int32_t stride = temp_accum.stride();

    std::atomic<bool> aborted{false};

    #pragma omp parallel for schedule(dynamic, 8) num_threads(n_threads)
    for (int32_t y = 0; y < h; ++y) {
        if ((y & 7) == 0) {
#ifdef _OPENMP
            if (omp_get_thread_num() == 0 && params.is_aborted()) {
                aborted.store(true, std::memory_order_relaxed);
            }
#else
            if (params.is_aborted()) aborted.store(true, std::memory_order_relaxed);
#endif
        }
        if (aborted.load(std::memory_order_relaxed)) continue;

        for (int32_t x = 0; x < w; ++x) {
            float center_val = temp_accum.at(x, y);
            float sigma = effective_sigma_cpu(nlf, params, guide_at(params, x, y, center_val)) * max_b;
            float sig_sq2 = 2.0f * sigma * sigma;
            float inv_h_sq = 1.0f / (params.nlm_h_factor * sig_sq2 + 1e-6f);

            float weight_sum = 1.0f; // Center self-weight
            float val_sum = center_val;

            int32_t min_sy = std::max(r_patch, y - r_search);
            int32_t max_sy = std::min(h - 1 - r_patch, y + r_search);
            int32_t min_sx = std::max(r_patch, x - r_search);
            int32_t max_sx = std::min(w - 1 - r_patch, x + r_search);
            const bool interior_x = (x >= r_patch && x < w - r_patch && y >= r_patch && y < h - r_patch);

            for (int32_t sy = min_sy; sy <= max_sy; sy += s_step) {
                for (int32_t sx = min_sx; sx <= max_sx; sx += s_step) {
                    if (sx == x && sy == y) continue;

                    float dist = 0.0f;
                    if (interior_x) {
                        for (int32_t py = -r_patch; py <= r_patch; ++py) {
                            const float* rp = src_data + static_cast<ptrdiff_t>(y + py) * stride + x;
                            const float* sp = src_data + static_cast<ptrdiff_t>(sy + py) * stride + sx;
                            for (int32_t px = -r_patch; px <= r_patch; ++px) {
                                float diff = rp[px] - sp[px];
                                dist += diff * diff;
                            }
                        }
                    } else {
                        for (int32_t py = -r_patch; py <= r_patch; ++py) {
                            int32_t ry = std::clamp(y + py, 0, h - 1);
                            for (int32_t px = -r_patch; px <= r_patch; ++px) {
                                int32_t rx = std::clamp(x + px, 0, w - 1);
                                float diff = temp_accum.at(rx, ry) - temp_accum.at(sx + px, sy + py);
                                dist += diff * diff;
                            }
                        }
                    }
                    dist *= patch_norm;

                    float arg = std::max(0.0f, dist - sig_sq2) * inv_h_sq;
                    if (arg < 8.0f) {
                        // Fast Padé/rational exp(-arg) accurate to <1e-3 on [0, 8]
                        float t = 1.0f + arg * (1.0f / 8.0f);
                        float t2 = t * t;
                        float t4 = t2 * t2;
                        float w_val = 1.0f / (t4 * t4);
                        val_sum += src_data[static_cast<ptrdiff_t>(sy) * stride + sx] * w_val;
                        weight_sum += w_val;
                    }
                }
            }

            output_plane.at(x, y) = val_sum / weight_sum;
        }
    }
}

// =========================================================================
// Engine 4: 3D Frequency-Domain Spectral Thresholding (DFTTest family)
// Zero-allocation 2-phase conflict-free strip parallelism + AVX2 8x8 DCT
// =========================================================================
void FrequencyDomainFilter3D::process_plane(
    const std::vector<const ImagePlane32F*>& temporal_planes,
    size_t ref_index,
    const std::vector<const MotionField*>& motion_fields,
    const NoiseLevelFunction& nlf,
    const NoisePsd& psd,
    const FilterParams& params,
    ImagePlane32F& output_plane
) {
    const int n_threads = resolve_filter_threads(params.max_threads);
    ImagePlane32F temp_accum = compute_temporal_average(temporal_planes, ref_index, motion_fields, nlf, psd, params);
    if (params.is_aborted()) {
        output_plane = temp_accum;
        return;
    }

    const size_t ch_idx = static_cast<size_t>(std::clamp(params.channel_index, 0, 2));
    const int32_t w = temp_accum.width();
    const int32_t h = temp_accum.height();
    const int32_t bs = 8;
    const int32_t step = 4;

    if (w < bs || h < bs) {
        output_plane = temp_accum;
        return;
    }

    alignas(32) float plane_bscale_sq[64];
    for (int i = 0; i < 64; ++i) {
        int band = g_spatial_lut.band[i];
        float bscale = std::max(0.05f, psd.octave_weights[ch_idx][band] * params.band_weights[band]);
        plane_bscale_sq[i] = bscale * bscale;
    }

    const int32_t npx = (w - bs + step - 1) / step + 1;
    const int32_t npy = (h - bs + step - 1) / step + 1;
    constexpr int32_t kStripSteps = 16;
    const int32_t num_strips = (npy + kStripSteps - 1) / kStripSteps;

    ImagePlane32F numerator(w, h);
    ImagePlane32F denominator(w, h);
    numerator.fill(0.0f);
    denominator.fill(0.0f);

    const float* src_data = temp_accum.data();
    const int32_t src_stride = temp_accum.stride();
    float* num_data = numerator.data();
    float* den_data = denominator.data();
    const int32_t acc_stride = numerator.stride();

    std::atomic<bool> aborted{false};

    for (int parity = 0; parity < 2; ++parity) {
        if (aborted.load(std::memory_order_relaxed) || params.is_aborted()) {
            output_plane = temp_accum;
            return;
        }

        #pragma omp parallel for schedule(dynamic, 1) num_threads(n_threads)
        for (int32_t strip = parity; strip < num_strips; strip += 2) {
#ifdef _OPENMP
            if (omp_get_thread_num() == 0 && params.is_aborted()) {
                aborted.store(true, std::memory_order_relaxed);
            }
#else
            if (params.is_aborted()) aborted.store(true, std::memory_order_relaxed);
#endif
            if (aborted.load(std::memory_order_relaxed)) continue;

            const int32_t iy_start = strip * kStripSteps;
            const int32_t iy_end = std::min(npy, iy_start + kStripSteps);

            alignas(32) float block[64];
            alignas(32) float spectrum[64];
            alignas(32) float filtered_block[64];

            const __m256 zero_v = _mm256_setzero_ps();
            const __m256 one_v = _mm256_set1_ps(1.0f);
            const __m256 eps_v = _mm256_set1_ps(1e-7f);

            for (int32_t iy = iy_start; iy < iy_end; ++iy) {
                int32_t y = std::min(iy * step, h - bs);
                for (int32_t ix = 0; ix < npx; ++ix) {
                    int32_t x = std::min(ix * step, w - bs);

                    float mean_v = load_patch8x8_mean_avx2(
                        src_data + static_cast<ptrdiff_t>(y) * src_stride + x,
                        src_stride,
                        block
                    );
                    float sigma = effective_sigma_cpu(nlf, params, guide_patch_mean(params, x, y, bs, mean_v));
                    __m256 sig_sq_v = _mm256_set1_ps(sigma * sigma);

                    dct8x8_2d(block, spectrum);

                    for (int v = 0; v < 8; ++v) {
                        __m256 coeff = _mm256_load_ps(spectrum + v * 8);
                        __m256 pwr = _mm256_mul_ps(coeff, coeff);
                        __m256 npwr = _mm256_mul_ps(sig_sq_v, _mm256_load_ps(plane_bscale_sq + v * 8));
                        __m256 gain = _mm256_max_ps(zero_v, _mm256_div_ps(_mm256_sub_ps(pwr, npwr), _mm256_add_ps(pwr, eps_v)));
                        if (v == 0) gain = _mm256_blend_ps(gain, one_v, 0x01); // Keep DC
                        _mm256_store_ps(spectrum + v * 8, _mm256_mul_ps(coeff, gain));
                    }

                    idct8x8_2d(spectrum, filtered_block);

                    accum_patch8x8_feathered_avx2(
                        num_data + static_cast<ptrdiff_t>(y) * acc_stride + x,
                        den_data + static_cast<ptrdiff_t>(y) * acc_stride + x,
                        acc_stride,
                        filtered_block,
                        1.0f
                    );
                }
            }
        }
    }

    if (aborted.load(std::memory_order_relaxed) || params.is_aborted()) {
        output_plane = temp_accum;
        return;
    }

    #pragma omp parallel for schedule(static) num_threads(n_threads)
    for (int32_t y = 0; y < h; ++y) {
        for (int32_t x = 0; x < w; ++x) {
            float d = denominator.at(x, y);
            output_plane.at(x, y) = (d > 1e-7f) ? (numerator.at(x, y) / d) : temp_accum.at(x, y);
        }
    }
}

// =========================================================================
// Unified Filter Dispatcher
// =========================================================================
void FilterDispatcher::process(
    const std::vector<VideoFrame32F>& temporal_frames,
    size_t ref_index,
    const std::vector<MotionField>& motion_fields,
    const NoiseProfile& profile,
    const FilterParams& params,
    VideoFrame32F& output_frame
) {
    if (temporal_frames.empty() || ref_index >= temporal_frames.size()) return;

    output_frame = VideoFrame32F(
        temporal_frames[ref_index].width,
        temporal_frames[ref_index].height,
        temporal_frames[ref_index].channel_count(),
        temporal_frames[ref_index].transfer
    );

    FilterParams active_params = params;
    if (active_params.quality == QualityPreset::Fast) {
        active_params.step_size = 6;
    } else if (active_params.quality == QualityPreset::Balanced) {
        active_params.step_size = 4;
    } else if (active_params.quality == QualityPreset::HighQuality) {
        active_params.step_size = 3;
    }

    std::unique_ptr<ISpatialEngine> engine;
    switch (active_params.engine) {
        case DenoiseEngine::Wavelet5Band:
            engine = std::make_unique<Wavelet5BandFilter>();
            break;
        case DenoiseEngine::NonLocalMeans:
            engine = std::make_unique<NonLocalMeansFilter>();
            break;
        case DenoiseEngine::FrequencyDomain3D:
            engine = std::make_unique<FrequencyDomainFilter3D>();
            break;
        case DenoiseEngine::Collaborative3D:
        default:
            engine = std::make_unique<CollaborativePatchFilter3D>(active_params);
            break;
    }

    for (size_t c = 0; c < output_frame.channel_count(); ++c) {
        if (active_params.is_aborted()) {
            output_frame.channels[c] = temporal_frames[ref_index].channels[c];
            continue;
        }

        std::vector<const ImagePlane32F*> planes;
        planes.reserve(temporal_frames.size());
        for (const auto& tf : temporal_frames) planes.push_back(&tf.channels[c]);

        std::vector<const MotionField*> mf_ptrs;
        mf_ptrs.reserve(motion_fields.size());
        for (const auto& mf : motion_fields) mf_ptrs.push_back(&mf);

        FilterParams channel_params = active_params;
        channel_params.channel_index = static_cast<int32_t>(c);
        channel_params.luma_guide = (c > 0 && temporal_frames[ref_index].channel_count() >= 3)
                                        ? &temporal_frames[ref_index].channels[0] : nullptr;
        if (c > 0 && active_params.fast_chroma) {
            channel_params.step_size = std::min(active_params.step_size + 2, 6);
        }

        const auto& nlf = profile.nlf_per_channel[std::min(c, size_t{2})];
        engine->process_plane(planes, ref_index, mf_ptrs, nlf, profile.psd, channel_params, output_frame.channels[c]);
    }
}

} // namespace librenoise
