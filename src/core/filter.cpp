#include "librenoise/filter.hpp"
#include "librenoise/colorspace.hpp"
#include <cmath>
#include <vector>
#include <algorithm>
#include <array>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

namespace librenoise {

namespace {

// =========================================================================
// Precomputed Exact Orthonormal 8-point Discrete Cosine Transform (0 Trig Calls)
// =========================================================================
struct DctTableInit {
    float mat[8][8];
    float mat_t[8][8]; // transpose, for row-vector products
    DctTableInit() {
        constexpr float s0 = 0.35355339059327376f;
        constexpr float s = 0.5f;
        constexpr double pi = 3.14159265358979323846;
        for (int k = 0; k < 8; ++k) {
            float factor = (k == 0) ? s0 : s;
            for (int n = 0; n < 8; ++n) {
                mat[k][n] = factor * static_cast<float>(std::cos((pi / 8.0) * (n + 0.5) * k));
            }
        }
        for (int k = 0; k < 8; ++k)
            for (int n = 0; n < 8; ++n) mat_t[n][k] = mat[k][n];
    }
};
static const DctTableInit g_dct_table;

inline void dct8_1d(const float in[8], float out[8]) {
    const auto& m = g_dct_table.mat;
    for (int k = 0; k < 8; ++k) {
        out[k] = m[k][0] * in[0] + m[k][1] * in[1] + m[k][2] * in[2] + m[k][3] * in[3] +
                 m[k][4] * in[4] + m[k][5] * in[5] + m[k][6] * in[6] + m[k][7] * in[7];
    }
}

inline void idct8_1d(const float in[8], float out[8]) {
    const auto& m = g_dct_table.mat;
    for (int n = 0; n < 8; ++n) {
        out[n] = m[0][n] * in[0] + m[1][n] * in[1] + m[2][n] * in[2] + m[3][n] * in[3] +
                 m[4][n] * in[4] + m[5][n] * in[5] + m[6][n] * in[6] + m[7][n] * in[7];
    }
}

#if defined(__AVX2__)
// 8x8 separable DCT as two 8x8 matrix products held entirely in YMM registers (128 FMAs per transform).
void dct8x8_2d(const float in[64], float out[64]) {
    const auto& m = g_dct_table;
    __m256 mt[8], t[8];
    for (int i = 0; i < 8; ++i) mt[i] = _mm256_loadu_ps(m.mat_t[i]);
    // T[r][:] = sum_n X[r][n] * M[:][n]
    for (int r = 0; r < 8; ++r) {
        __m256 acc = _mm256_mul_ps(_mm256_set1_ps(in[r * 8 + 0]), mt[0]);
        for (int n = 1; n < 8; ++n) acc = _mm256_fmadd_ps(_mm256_set1_ps(in[r * 8 + n]), mt[n], acc);
        t[r] = acc;
    }
    // out[k][:] = sum_r M[k][r] * T[r][:]
    for (int k = 0; k < 8; ++k) {
        __m256 acc = _mm256_mul_ps(_mm256_set1_ps(m.mat[k][0]), t[0]);
        for (int r = 1; r < 8; ++r) acc = _mm256_fmadd_ps(_mm256_set1_ps(m.mat[k][r]), t[r], acc);
        _mm256_storeu_ps(out + k * 8, acc);
    }
}

void idct8x8_2d(const float in[64], float out[64]) {
    const auto& m = g_dct_table;
    __m256 mr[8], u[8];
    for (int i = 0; i < 8; ++i) mr[i] = _mm256_loadu_ps(m.mat[i]);
    // U[k][:] = sum_l Y[k][l] * M[l][:]
    for (int k = 0; k < 8; ++k) {
        __m256 acc = _mm256_mul_ps(_mm256_set1_ps(in[k * 8 + 0]), mr[0]);
        for (int l = 1; l < 8; ++l) acc = _mm256_fmadd_ps(_mm256_set1_ps(in[k * 8 + l]), mr[l], acc);
        u[k] = acc;
    }
    // out[n][:] = sum_k M[k][n] * U[k][:]
    for (int n = 0; n < 8; ++n) {
        __m256 acc = _mm256_mul_ps(_mm256_set1_ps(m.mat[0][n]), u[0]);
        for (int k = 1; k < 8; ++k) acc = _mm256_fmadd_ps(_mm256_set1_ps(m.mat[k][n]), u[k], acc);
        _mm256_storeu_ps(out + n * 8, acc);
    }
}
#else
void dct8x8_2d(const float in[64], float out[64]) {
    float temp[64];
    for (int r = 0; r < 8; ++r) dct8_1d(&in[r * 8], &temp[r * 8]);
    for (int c = 0; c < 8; ++c) {
        float col_in[8], col_out[8];
        for (int r = 0; r < 8; ++r) col_in[r] = temp[r * 8 + c];
        dct8_1d(col_in, col_out);
        for (int r = 0; r < 8; ++r) out[r * 8 + c] = col_out[r];
    }
}

void idct8x8_2d(const float in[64], float out[64]) {
    float temp[64];
    for (int c = 0; c < 8; ++c) {
        float col_in[8], col_out[8];
        for (int r = 0; r < 8; ++r) col_in[r] = in[r * 8 + c];
        idct8_1d(col_in, col_out);
        for (int r = 0; r < 8; ++r) temp[r * 8 + c] = col_out[r];
    }
    for (int r = 0; r < 8; ++r) idct8_1d(&temp[r * 8], &out[r * 8]);
}
#endif

// Orthonormal DCT-II basis for every possible temporal group size (1..12), built once.
// basis[K][k][n] already includes the alpha normalisation, so no trig runs in the hot loops.
struct TemporalDctTable {
    float basis[13][12][12];
    TemporalDctTable() {
        constexpr double pi = 3.14159265358979323846;
        for (int K = 0; K <= 12; ++K)
            for (int k = 0; k < 12; ++k)
                for (int n = 0; n < 12; ++n) basis[K][k][n] = 0.0f;
        for (int K = 1; K <= 12; ++K) {
            const double a0 = std::sqrt(1.0 / K), a = std::sqrt(2.0 / K);
            for (int k = 0; k < K; ++k)
                for (int n = 0; n < K; ++n)
                    basis[K][k][n] = static_cast<float>((k == 0 ? a0 : a) * std::cos(pi / K * (n + 0.5) * k));
        }
    }
};
static const TemporalDctTable g_tdct_table;

void temporal_dct_forward(const float* in, float* out, int K) {
    if (K <= 1) { out[0] = in[0]; return; }
    const auto& b = g_tdct_table.basis[K];
    for (int k = 0; k < K; ++k) {
        float sum = 0.0f;
        for (int n = 0; n < K; ++n) sum += b[k][n] * in[n];
        out[k] = sum;
    }
}

void temporal_dct_inverse(const float* in, float* out, int K) {
    if (K <= 1) { out[0] = in[0]; return; }
    const auto& b = g_tdct_table.basis[K];
    for (int n = 0; n < K; ++n) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) sum += b[k][n] * in[k];
        out[n] = sum;
    }
}

inline float eval_tonal_lut_cpu(const std::array<float, 5>& lut, float val) {
    float t = std::clamp(val, 0.0f, 1.0f) * 4.0f;
    int idx = std::min(3, std::max(0, static_cast<int>(t)));
    float frac = t - static_cast<float>(idx);
    return lut[idx] * (1.0f - frac) + lut[idx + 1] * frac;
}

inline float effective_sigma_cpu(const NoiseLevelFunction& nlf, const FilterParams& params, float val) {
    return nlf.sigma_at(val) * params.spatial_weight * eval_tonal_lut_cpu(params.tonal_lut, val) * params.channel_amount;
}

inline int band_for_uv_cpu(int u, int v) {
    float r = std::sqrt(static_cast<float>(u * u + v * v));
    return (r < 1.5f) ? 0 : (r < 3.0f) ? 1 : (r < 5.0f) ? 2 : (r < 7.5f) ? 3 : 4;
}

inline float max_band_scale_cpu(const NoisePsd& psd, const FilterParams& params) {
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

    #pragma omp parallel for schedule(dynamic)
    for (int32_t y = 0; y < h; ++y) {
        for (int32_t x = 0; x < w; ++x) {
            float ref_val = temporal_planes[ref_index]->at(x, y);
            float sigma = effective_sigma_cpu(nlf, params, ref_val) * max_b;
            float trust_limit = 3.5f * sigma * params.temporal_weight;

            float sum = ref_val;
            float weight_sum = 1.0f;

            for (size_t f = 0; f < temporal_planes.size(); ++f) {
                if (f == ref_index) continue;
                int32_t dx = 0, dy = 0;
                bool trusted = true;
                if (motion_fields[f]) {
                    if (motion_fields[f]->is_valid()) {
                        auto mv = motion_fields[f]->get_vector_at_pixel(x, y);
                        dx = mv.dx;
                        dy = mv.dy;
                        trusted = mv.trusted;
                    } else if (motion_fields[f]->is_scene_cut) {
                        trusted = false;
                    }
                }
                if (!trusted) continue;

                int32_t cx = std::clamp(x + dx, 0, w - 1);
                int32_t cy = std::clamp(y + dy, 0, h - 1);
                float cand_val = temporal_planes[f]->at(cx, cy);

                if (std::abs(ref_val - cand_val) <= trust_limit) {
                    sum += cand_val;
                    weight_sum += 1.0f;
                }
            }
            avg.at(x, y) = sum / weight_sum;
        }
    }
    return avg;
}

} // anonymous namespace

// Row-band accumulator: a thread only needs the rows its patch strip can touch, not a full frame.
struct BandPlane {
    int32_t w = 0, y0 = 0, rows = 0;
    std::vector<float> d;
    BandPlane(int32_t w_, int32_t y0_, int32_t rows_)
        : w(w_), y0(y0_), rows(rows_), d(static_cast<size_t>(w_) * static_cast<size_t>(rows_), 0.0f) {}
    inline float& at(int32_t x, int32_t y) { return d[static_cast<size_t>(y - y0) * static_cast<size_t>(w) + static_cast<size_t>(x)]; }
};

inline int32_t strip_chunk_count(int32_t patch_rows) {
    int32_t threads = 1;
#ifdef _OPENMP
    threads = std::max(1, omp_get_max_threads());
#endif
    return std::max<int32_t>(1, std::min<int32_t>(patch_rows, threads * 4));
}

// =========================================================================
// Engine 1: 3D Collaborative Transform Filter (2-Stage V-BM3D + Non-Local Wiener)
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
    const size_t total_pixels = static_cast<size_t>(w * h);
    const size_t ch_idx = static_cast<size_t>(std::clamp(params.channel_index, 0, 2));
    const float max_b = max_band_scale_cpu(psd, params);
    const float feather_1d[8] = {0.38f, 0.68f, 0.92f, 1.0f, 1.0f, 0.92f, 0.68f, 0.38f};

    if (w < ps || h < ps) {
        output_plane = *temporal_planes[ref_index];
        return;
    }

    const int32_t npx = (w - ps + step - 1) / step + 1;
    const int32_t npy = (h - ps + step - 1) / step + 1;

    ImagePlane32F numerator(w, h);
    ImagePlane32F denominator(w, h);
    numerator.fill(0.0f);
    denominator.fill(0.0f);

    const ImagePlane32F* ref_plane = temporal_planes[ref_index];

    // Per-coefficient band scale (profile PSD x user band weight), shared by every patch
    alignas(32) float bsc[64];
    for (int c = 0; c < 64; ++c) {
        const int band = band_for_uv_cpu(c & 7, c >> 3);
        bsc[c] = std::max(0.05f, psd.octave_weights[ch_idx][band] * params.band_weights[band]);
    }

    // Pass 1: Motion-Compensated 3D Hard-Thresholding -> Pilot
    const int32_t n_chunks = strip_chunk_count(npy);
    #pragma omp parallel for schedule(dynamic, 1)
    for (int32_t chunk = 0; chunk < n_chunks; ++chunk) {
        const int32_t iy_lo = static_cast<int32_t>(static_cast<int64_t>(npy) * chunk / n_chunks);
        const int32_t iy_hi = static_cast<int32_t>(static_cast<int64_t>(npy) * (chunk + 1) / n_chunks);
        const int32_t band_y0 = std::min(iy_lo * step, h - ps);
        const int32_t band_y1 = std::min((iy_hi - 1) * step, h - ps) + ps;
        BandPlane local_num(w, band_y0, band_y1 - band_y0);
        BandPlane local_den(w, band_y0, band_y1 - band_y0);

        for (int32_t iy = iy_lo; iy < iy_hi; ++iy) {
            int32_t y = std::min(iy * step, h - ps);
            for (int32_t ix = 0; ix < npx; ++ix) {
                int32_t x = std::min(ix * step, w - ps);

                float ref_patch[64];
                float mean_v = 0.0f;
                for (int32_t py = 0; py < ps; ++py) {
                    for (int32_t px = 0; px < ps; ++px) {
                        float v = ref_plane->at(x + px, y + py);
                        ref_patch[py * 8 + px] = v;
                        mean_v += v;
                    }
                }
                mean_v *= (1.0f / 64.0f);

                float sigma = effective_sigma_cpu(nlf, params, mean_v);
                float eff_sig = sigma * max_b;
                float trust_dist = 4.5f * std::max(0.25f, params.temporal_weight) * (eff_sig * eff_sig + 1e-6f);

                float group[9][64];
                std::copy(ref_patch, ref_patch + 64, group[0]);
                int K = 1;

                if (temporal_planes.size() > 1 && params.temporal_weight > 0.01f) {
                    for (size_t f = 0; f < temporal_planes.size(); ++f) {
                        if (f == ref_index || K >= 9) continue;

                        int32_t disp_x = 0, disp_y = 0;
                        bool trusted = true;
                        if (motion_fields[f]) {
                            if (motion_fields[f]->is_valid()) {
                                auto mv = motion_fields[f]->get_vector_at_pixel(x + 4, y + 4);
                                disp_x = mv.dx;
                                disp_y = mv.dy;
                                trusted = mv.trusted;
                            } else if (motion_fields[f]->is_scene_cut) {
                                trusted = false;
                            }
                        }
                        if (!trusted) continue;

                        int32_t cand_x = std::clamp(x + disp_x, 0, w - ps);
                        int32_t cand_y = std::clamp(y + disp_y, 0, h - ps);

                        float cand_patch[64];
                        float dist = 0.0f;
                        for (int32_t py = 0; py < ps; ++py) {
                            for (int32_t px = 0; px < ps; ++px) {
                                float cv = temporal_planes[f]->at(cand_x + px, cand_y + py);
                                cand_patch[py * 8 + px] = cv;
                                float diff = ref_patch[py * 8 + px] - cv;
                                dist += diff * diff;
                            }
                        }
                        dist *= (1.0f / 64.0f);

                        if (dist <= trust_dist) {
                            std::copy(cand_patch, cand_patch + 64, group[K]);
                            K++;
                        }
                    }
                }

                float spatial_coeffs[9][64];
                for (int k = 0; k < K; ++k) dct8x8_2d(group[k], spatial_coeffs[k]);

                // 3D transform: temporal DCT across all 64 spatial coefficients at once
                const auto& tb = g_tdct_table.basis[K];
                alignas(32) float transform_3d[9][64];
                for (int k = 0; k < K; ++k) {
                    float* __restrict t3 = transform_3d[k];
                    for (int c = 0; c < 64; ++c) t3[c] = 0.0f;
                    for (int n = 0; n < K; ++n) {
                        const float bkn = tb[k][n];
                        const float* __restrict sc = spatial_coeffs[n];
                        for (int c = 0; c < 64; ++c) t3[c] += bkn * sc[c];
                    }
                }

                // Hard threshold, then inverse temporal DCT of output 0 only (the reference slice)
                const float tw_scale = std::max(0.5f, params.temporal_weight);
                alignas(32) float out0[64];
                for (int c = 0; c < 64; ++c) out0[c] = 0.0f;
                int non_zero_count = 0;
                for (int k = 0; k < K; ++k) {
                    float* __restrict t3 = transform_3d[k];
                    const float kt = params.hard_thresh_lambda * sigma * ((k == 0) ? 1.0f : tw_scale);
                    const float dc = t3[0];
                    int nz = 0;
                    for (int c = 0; c < 64; ++c) {
                        const bool keep = std::abs(t3[c]) >= kt * bsc[c];
                        nz += keep ? 1 : 0;
                        t3[c] = keep ? t3[c] : 0.0f;
                    }
                    if (k == 0) {   // DC of the reference slice is never thresholded and always counted
                        if (t3[0] == 0.0f && dc != 0.0f) nz++;
                        t3[0] = dc;
                    }
                    non_zero_count += nz;
                    const float inv0 = tb[k][0];
                    for (int c = 0; c < 64; ++c) out0[c] += inv0 * t3[c];
                }
                for (int c = 0; c < 64; ++c) spatial_coeffs[0][c] = out0[c];

                float filtered_ref_patch[64];
                idct8x8_2d(spatial_coeffs[0], filtered_ref_patch);

                float inv_nz = 1.0f / static_cast<float>(std::max(1, non_zero_count));
                for (int32_t py = 0; py < ps; ++py) {
                    for (int32_t px = 0; px < ps; ++px) {
                        float w_agg = feather_1d[py] * feather_1d[px] * inv_nz;
                        float fv = std::max(0.0f, filtered_ref_patch[py * 8 + px]);
                        local_num.at(x + px, y + py) += fv * w_agg;
                        local_den.at(x + px, y + py) += w_agg;
                    }
                }
            }
        }

        #pragma omp critical
        {
            for (int32_t yy = band_y0; yy < band_y1; ++yy) {
                for (int32_t xx = 0; xx < w; ++xx) {
                    numerator.at(xx, yy) += local_num.at(xx, yy);
                    denominator.at(xx, yy) += local_den.at(xx, yy);
                }
            }
        }
    }

    ImagePlane32F pilot(w, h);
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

    // Per-coefficient squared band scale (profile PSD x user band weight), shared by every patch
    alignas(32) float bs2[64];
    for (int c = 0; c < 64; ++c) {
        const float bscale = std::max(0.05f, psd.octave_weights[ch_idx][band_for_uv_cpu(c & 7, c >> 3)] * params.band_weights[band_for_uv_cpu(c & 7, c >> 3)]);
        bs2[c] = bscale * bscale;
    }

    // Pass 2: Spatiotemporal Non-Local Collaborative 3D Empirical Wiener
    numerator.fill(0.0f);
    denominator.fill(0.0f);

    #pragma omp parallel for schedule(dynamic, 1)
    for (int32_t chunk = 0; chunk < n_chunks; ++chunk) {
        const int32_t iy_lo = static_cast<int32_t>(static_cast<int64_t>(npy) * chunk / n_chunks);
        const int32_t iy_hi = static_cast<int32_t>(static_cast<int64_t>(npy) * (chunk + 1) / n_chunks);
        const int32_t band_y0 = std::min(iy_lo * step, h - ps);
        const int32_t band_y1 = std::min((iy_hi - 1) * step, h - ps) + ps;
        BandPlane local_num(w, band_y0, band_y1 - band_y0);
        BandPlane local_den(w, band_y0, band_y1 - band_y0);

        const int ox_list[12] = {-5,  5,  0,  0, -5,  5, -5,  5, -10, 10,   0,  0};
        const int oy_list[12] = { 0,  0, -5,  5, -5, -5,  5,  5,   0,  0, -10, 10};

        for (int32_t iy = iy_lo; iy < iy_hi; ++iy) {
            int32_t y = std::min(iy * step, h - ps);
            for (int32_t ix = 0; ix < npx; ++ix) {
                int32_t x = std::min(ix * step, w - ps);

                float noisy_grp[12][64];
                float pilot_grp[12][64];
                float mean_p = 0.0f;

                for (int32_t py = 0; py < ps; ++py) {
                    for (int32_t px = 0; px < ps; ++px) {
                        int idx = py * 8 + px;
                        noisy_grp[0][idx] = ref_plane->at(x + px, y + py);
                        float pv = pilot.at(x + px, y + py);
                        pilot_grp[0][idx] = pv;
                        mean_p += pv;
                    }
                }
                mean_p *= (1.0f / 64.0f);

                float sigma = effective_sigma_cpu(nlf, params, mean_p);
                float eff_var = (sigma * max_b) * (sigma * max_b) + 1e-7f;
                int K = 1;

                // Temporal matches
                if (temporal_planes.size() > 1 && params.temporal_weight > 0.01f) {
                    float temp_trust = 4.0f * std::max(0.25f, params.temporal_weight) * eff_var;
                    for (size_t f = 0; f < temporal_planes.size(); ++f) {
                        if (f == ref_index || K >= 12) continue;
                        int32_t disp_x = 0, disp_y = 0;
                        bool trusted = true;
                        if (motion_fields[f]) {
                            if (motion_fields[f]->is_valid()) {
                                auto mv = motion_fields[f]->get_vector_at_pixel(x + 4, y + 4);
                                disp_x = mv.dx;
                                disp_y = mv.dy;
                                trusted = mv.trusted;
                            } else if (motion_fields[f]->is_scene_cut) {
                                trusted = false;
                            }
                        }
                        if (!trusted) continue;

                        int32_t cx = std::clamp(x + disp_x, 0, w - ps);
                        int32_t cy = std::clamp(y + disp_y, 0, h - ps);
                        float cand_n[64];
                        float dist = 0.0f;
                        for (int32_t py = 0; py < ps; ++py) {
                            for (int32_t px = 0; px < ps; ++px) {
                                int idx = py * 8 + px;
                                float cv = temporal_planes[f]->at(cx + px, cy + py);
                                cand_n[idx] = cv;
                                float diff = noisy_grp[0][idx] - cv;
                                dist += diff * diff;
                            }
                        }
                        if (dist * (1.0f / 64.0f) <= temp_trust) {
                            std::copy(cand_n, cand_n + 64, noisy_grp[K]);
                            std::copy(pilot_grp[0], pilot_grp[0] + 64, pilot_grp[K]);
                            K++;
                        }
                    }
                }

                // Spatial non-local matches on denoised pilot
                if (params.spatial_weight > 0.01f && K < 12) {
                    float spat_trust = 1.6f * eff_var;
                    for (int s = 0; s < 12 && K < 12; ++s) {
                        int32_t sx = x + ox_list[s];
                        int32_t sy = y + oy_list[s];
                        if (sx < 0 || sx + ps > w || sy < 0 || sy + ps > h) continue;

                        float cand_p[64];
                        float dist_p = 0.0f;
                        for (int32_t py = 0; py < ps; ++py) {
                            for (int32_t px = 0; px < ps; ++px) {
                                int idx = py * 8 + px;
                                float pv = pilot.at(sx + px, sy + py);
                                cand_p[idx] = pv;
                                float diff = pilot_grp[0][idx] - pv;
                                dist_p += diff * diff;
                            }
                        }
                        if (dist_p * (1.0f / 64.0f) <= spat_trust) {
                            for (int32_t py = 0; py < ps; ++py) {
                                for (int32_t px = 0; px < ps; ++px) {
                                    noisy_grp[K][py * 8 + px] = ref_plane->at(sx + px, sy + py);
                                }
                            }
                            std::copy(cand_p, cand_p + 64, pilot_grp[K]);
                            K++;
                        }
                    }
                }

                float n_spec[12][64], p_spec[12][64];
                for (int k = 0; k < K; ++k) {
                    dct8x8_2d(noisy_grp[k], n_spec[k]);
                    dct8x8_2d(pilot_grp[k], p_spec[k]);
                }

                // Temporal DCT across all 64 spatial coefficients at once (contiguous, vectorizable)
                const auto& tb = g_tdct_table.basis[K];
                alignas(32) float n_t[12][64], p_t[12][64];
                for (int k = 0; k < K; ++k) {
                    float* __restrict nt = n_t[k];
                    float* __restrict pt = p_t[k];
                    for (int c = 0; c < 64; ++c) { nt[c] = 0.0f; pt[c] = 0.0f; }
                    for (int n = 0; n < K; ++n) {
                        const float bkn = tb[k][n];
                        const float* __restrict ns = n_spec[n];
                        const float* __restrict ps_ = p_spec[n];
                        for (int c = 0; c < 64; ++c) { nt[c] += bkn * ns[c]; pt[c] += bkn * ps_[c]; }
                    }
                }

                // Wiener shrinkage per (temporal k, spatial c); only temporal output 0 of the inverse is used
                const float sigma2 = sigma * sigma;
                const float tw_scale = std::max(0.5f, params.temporal_weight);
                alignas(32) float out0[64];
                for (int c = 0; c < 64; ++c) out0[c] = 0.0f;
                float w_sq_sum = 0.0f;
                for (int k = 0; k < K; ++k) {
                    const float kscale = sigma2 * ((k == 0) ? 1.0f : tw_scale);
                    alignas(32) float gain[64];
                    const float* __restrict pt = p_t[k];
                    for (int c = 0; c < 64; ++c) {
                        const float p2 = pt[c] * pt[c];
                        gain[c] = p2 / (p2 + kscale * bs2[c] + 1e-10f);
                    }
                    if (k == 0) gain[0] = 1.0f; // DC of the reference stays untouched
                    const float inv0 = tb[k][0];
                    const float* __restrict nt = n_t[k];
                    float gsq = 0.0f;
                    for (int c = 0; c < 64; ++c) {
                        out0[c] += inv0 * nt[c] * gain[c];
                        gsq += gain[c] * gain[c];
                    }
                    w_sq_sum += gsq;
                }
                for (int c = 0; c < 64; ++c) n_spec[0][c] = out0[c];

                float wiener_patch[64];
                idct8x8_2d(n_spec[0], wiener_patch);

                float inv_w = 1.0f / std::max(1e-2f, w_sq_sum);
                for (int32_t py = 0; py < ps; ++py) {
                    for (int32_t px = 0; px < ps; ++px) {
                        float w_agg = feather_1d[py] * feather_1d[px] * inv_w;
                        float fv = std::max(0.0f, wiener_patch[py * 8 + px]);
                        local_num.at(x + px, y + py) += fv * w_agg;
                        local_den.at(x + px, y + py) += w_agg;
                    }
                }
            }
        }

        #pragma omp critical
        {
            for (int32_t yy = band_y0; yy < band_y1; ++yy) {
                for (int32_t xx = 0; xx < w; ++xx) {
                    numerator.at(xx, yy) += local_num.at(xx, yy);
                    denominator.at(xx, yy) += local_den.at(xx, yy);
                }
            }
        }
    }

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

ImagePlane32F downsample2x(const ImagePlane32F& src) {
    int32_t sw = src.width();
    int32_t sh = src.height();
    int32_t dw = std::max(1, sw / 2);
    int32_t dh = std::max(1, sh / 2);
    ImagePlane32F dst(dw, dh);

    #pragma omp parallel for schedule(dynamic)
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

ImagePlane32F upsample2x(const ImagePlane32F& src, int32_t target_w, int32_t target_h) {
    ImagePlane32F dst(target_w, target_h);
    int32_t sw = src.width();
    int32_t sh = src.height();

    #pragma omp parallel for schedule(dynamic)
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
    // Step 1: Temporal Motion-Compensated baseline
    ImagePlane32F current = compute_temporal_average(temporal_planes, ref_index, motion_fields, nlf, psd, params);
    const size_t ch_idx = static_cast<size_t>(std::clamp(params.channel_index, 0, 2));

    const int32_t num_levels = 5;
    std::vector<ImagePlane32F> pyramid(num_levels + 1);
    std::vector<ImagePlane32F> details(num_levels);

    pyramid[0] = current;

    // Decomposition into 5 frequency octaves
    for (int l = 0; l < num_levels; ++l) {
        pyramid[l + 1] = downsample2x(pyramid[l]);
        ImagePlane32F up = upsample2x(pyramid[l + 1], pyramid[l].width(), pyramid[l].height());

        details[l] = ImagePlane32F(pyramid[l].width(), pyramid[l].height());
        int band = std::clamp(4 - l, 0, 4); // 4 = Very High down to 0 = Very Low
        float band_mult = std::max(0.05f, psd.octave_weights[ch_idx][band] * params.band_weights[band]);
        // Each 2x2 pyramid average halves white noise std (0.5^l)
        float level_scale = std::pow(0.55f, static_cast<float>(l));

        int32_t pw = pyramid[l].width();
        int32_t ph = pyramid[l].height();

        #pragma omp parallel for schedule(dynamic)
        for (int32_t y = 0; y < ph; ++y) {
            for (int32_t x = 0; x < pw; ++x) {
                float raw_diff = pyramid[l].at(x, y) - up.at(x, y);
                float base_val = up.at(x, y);
                float sigma = effective_sigma_cpu(nlf, params, base_val);

                // Empirical Wiener shrinkage per octave band (preserves gradients & crisp edges)
                float thresh = 1.6f * sigma * band_mult * level_scale;
                float pwr = raw_diff * raw_diff;
                float gain = std::max(0.0f, (pwr - thresh * thresh) / (pwr + 1e-8f));
                details[l].at(x, y) = raw_diff * gain;
            }
        }
    }

    // Reconstruction
    ImagePlane32F recon = pyramid[num_levels];
    for (int l = num_levels - 1; l >= 0; --l) {
        ImagePlane32F up = upsample2x(recon, details[l].width(), details[l].height());
        recon = ImagePlane32F(details[l].width(), details[l].height());

        int32_t rw = details[l].width();
        int32_t rh = details[l].height();

        #pragma omp parallel for schedule(dynamic)
        for (int32_t y = 0; y < rh; ++y) {
            for (int32_t x = 0; x < rw; ++x) {
                recon.at(x, y) = std::max(0.0f, up.at(x, y) + details[l].at(x, y));
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
    // Step 1: Temporal Motion-Compensated baseline
    ImagePlane32F temp_accum = compute_temporal_average(temporal_planes, ref_index, motion_fields, nlf, psd, params);
    const float max_b = max_band_scale_cpu(psd, params);

    const int32_t w = temp_accum.width();
    const int32_t h = temp_accum.height();
    const int32_t r_search = params.nlm_search_radius; // 6
    const int32_t r_patch = params.nlm_patch_radius;   // 2 -> 5x5 patch
    const float patch_norm = 1.0f / static_cast<float>((2 * r_patch + 1) * (2 * r_patch + 1));

    // Exact accelerated NLM: for every search offset, one squared-difference image is box-summed
    // (separable, O(1) per pixel) to give all patch distances at once, instead of re-comparing a full
    // patch for every pixel/candidate pair. Same weights, same accumulation order as the direct form.
    const int32_t R = r_patch;
    const int32_t pw = w + 2 * R, ph = h + 2 * R;
    // Edge-replicated padded copy: reproduces the clamped centre-patch reads of the direct form
    std::vector<float> pad(static_cast<size_t>(pw) * static_cast<size_t>(ph));
    #pragma omp parallel for schedule(static)
    for (int32_t j = 0; j < ph; ++j) {
        const int32_t yy = std::clamp(j - R, 0, h - 1);
        for (int32_t i = 0; i < pw; ++i) pad[static_cast<size_t>(j) * pw + i] = temp_accum.at(std::clamp(i - R, 0, w - 1), yy);
    }

    std::vector<float> sig2(static_cast<size_t>(w) * h), hsq(static_cast<size_t>(w) * h);
    std::vector<float> val_sum(static_cast<size_t>(w) * h, 0.0f), weight_sum(static_cast<size_t>(w) * h, 0.0f);
    #pragma omp parallel for schedule(static)
    for (int32_t y = 0; y < h; ++y) {
        for (int32_t x = 0; x < w; ++x) {
            const float sigma = effective_sigma_cpu(nlf, params, temp_accum.at(x, y)) * max_b;
            sig2[static_cast<size_t>(y) * w + x] = sigma * sigma;
            hsq[static_cast<size_t>(y) * w + x] = params.nlm_h_factor * sigma * sigma * 2.0f + 1e-6f;
        }
    }

    // Each task owns a band of output rows and runs every search offset inside it: no barriers,
    // no shared writes, and the working set stays in cache.
    const int32_t n_bands = std::max<int32_t>(1, std::min<int32_t>(h, strip_chunk_count(h)));
    #pragma omp parallel for schedule(dynamic, 1)
    for (int32_t band = 0; band < n_bands; ++band) {
        const int32_t y0 = static_cast<int32_t>(static_cast<int64_t>(h) * band / n_bands);
        const int32_t y1 = static_cast<int32_t>(static_cast<int64_t>(h) * (band + 1) / n_bands);
        if (y1 <= y0) continue;
        const int32_t rows = y1 - y0 + 2 * R;
        std::vector<float> hbuf(static_cast<size_t>(rows) * w);
        std::vector<float> dist(static_cast<size_t>(w)), wv(static_cast<size_t>(w));

        for (int32_t oy = -r_search; oy <= r_search; ++oy) {
            // Output rows whose candidate patch row stays inside the image
            const int32_t ya = std::max(y0, R - oy);
            const int32_t yb = std::min(y1 - 1, h - 1 - R - oy);
            if (ya > yb) continue;
            for (int32_t ox = -r_search; ox <= r_search; ++ox) {
                const int32_t xa = std::max(0, R - ox);
                const int32_t xb = std::min(w - 1, w - 1 - R - ox);
                if (xa > xb) continue;
                const int32_t n = xb - xa + 1;

                // Horizontal box sums of squared differences for padded rows ya .. yb+2R
                for (int32_t j = ya; j <= yb + 2 * R; ++j) {
                    float* __restrict hrow = &hbuf[static_cast<size_t>(j - y0) * w];
                    const float* __restrict ar = &pad[static_cast<size_t>(j) * pw];
                    const float* __restrict cr = &pad[static_cast<size_t>(j + oy) * pw + ox];
                    for (int32_t x = 0; x < n; ++x) hrow[xa + x] = 0.0f;
                    for (int32_t k = 0; k <= 2 * R; ++k) {
                        for (int32_t x = 0; x < n; ++x) {
                            const float d = ar[xa + x + k] - cr[xa + x + k];
                            hrow[xa + x] += d * d;
                        }
                    }
                }

                for (int32_t y = ya; y <= yb; ++y) {
                    float* __restrict dst = dist.data();
                    for (int32_t x = 0; x < n; ++x) dst[x] = 0.0f;
                    for (int32_t k = 0; k <= 2 * R; ++k) {
                        const float* __restrict hrow = &hbuf[static_cast<size_t>(y + k - y0) * w + xa];
                        for (int32_t x = 0; x < n; ++x) dst[x] += hrow[x];
                    }
                    const size_t base = static_cast<size_t>(y) * w + xa;
                    for (int32_t x = 0; x < n; ++x) {
                        const float d = dst[x] * patch_norm;
                        wv[x] = std::exp(-std::max(0.0f, d - 2.0f * sig2[base + x]) / hsq[base + x]);
                    }
                    const float* cand = temp_accum.data() + static_cast<ptrdiff_t>(y + oy) * temp_accum.stride() + xa + ox;
                    for (int32_t x = 0; x < n; ++x) {
                        val_sum[base + x] += cand[x] * wv[x];
                        weight_sum[base + x] += wv[x];
                    }
                }
            }
        }
    }

    #pragma omp parallel for schedule(static)
    for (int32_t y = 0; y < h; ++y) {
        for (int32_t x = 0; x < w; ++x) {
            const size_t idx = static_cast<size_t>(y) * w + x;
            output_plane.at(x, y) = (weight_sum[idx] > 1e-6f) ? (val_sum[idx] / weight_sum[idx]) : temp_accum.at(x, y);
        }
    }
}

// =========================================================================
// Engine 4: 3D Frequency-Domain Spectral Thresholding (DFTTest family)
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
    // Step 1: Temporal motion-compensated pre-estimate
    ImagePlane32F temp_accum = compute_temporal_average(temporal_planes, ref_index, motion_fields, nlf, psd, params);
    const size_t ch_idx = static_cast<size_t>(std::clamp(params.channel_index, 0, 2));

    const int32_t w = temp_accum.width();
    const int32_t h = temp_accum.height();
    const int32_t bs = 8;
    const int32_t step = 4; // Overlapping sliding blocks

    if (w < bs || h < bs) {
        output_plane = temp_accum;
        return;
    }

    const int32_t npx = (w - bs + step - 1) / step + 1;
    const int32_t npy = (h - bs + step - 1) / step + 1;

    ImagePlane32F numerator(w, h);
    ImagePlane32F denominator(w, h);
    numerator.fill(0.0f);
    denominator.fill(0.0f);

    const int32_t n_chunks = strip_chunk_count(npy);
    #pragma omp parallel for schedule(dynamic, 1)
    for (int32_t chunk = 0; chunk < n_chunks; ++chunk) {
        const int32_t iy_lo = static_cast<int32_t>(static_cast<int64_t>(npy) * chunk / n_chunks);
        const int32_t iy_hi = static_cast<int32_t>(static_cast<int64_t>(npy) * (chunk + 1) / n_chunks);
        const int32_t band_y0 = std::min(iy_lo * step, h - bs);
        const int32_t band_y1 = std::min((iy_hi - 1) * step, h - bs) + bs;
        BandPlane local_num(w, band_y0, band_y1 - band_y0);
        BandPlane local_den(w, band_y0, band_y1 - band_y0);

        for (int32_t iy = iy_lo; iy < iy_hi; ++iy) {
            int32_t y = std::min(iy * step, h - bs);
            for (int32_t ix = 0; ix < npx; ++ix) {
                int32_t x = std::min(ix * step, w - bs);
                float block[64];
                float mean_v = 0.0f;
                for (int32_t py = 0; py < bs; ++py) {
                    for (int32_t px = 0; px < bs; ++px) {
                        float v = temp_accum.at(x + px, y + py);
                        block[py * 8 + px] = v;
                        mean_v += v;
                    }
                }
                mean_v *= (1.0f / 64.0f);

                float sigma = effective_sigma_cpu(nlf, params, mean_v);

                // 2D DCT Transform
                float spectrum[64];
                dct8x8_2d(block, spectrum);

                // 5-Band Radial Frequency Mapping and Spectral Shrinkage
                for (int v = 0; v < 8; ++v) {
                    for (int u = 0; u < 8; ++u) {
                        if (u == 0 && v == 0) continue; // Keep DC

                        int band = band_for_uv_cpu(u, v);
                        float bscale = std::max(0.05f, psd.octave_weights[ch_idx][band] * params.band_weights[band]);
                        float coeff = spectrum[v * 8 + u];
                        float pwr = coeff * coeff;
                        float noise_pwr = (sigma * bscale) * (sigma * bscale);

                        // Wiener empirical spectral shrinkage
                        float gain = std::max(0.0f, (pwr - noise_pwr) / (pwr + 1e-7f));
                        spectrum[v * 8 + u] = coeff * gain;
                    }
                }

                // Inverse 2D IDCT
                float filtered_block[64];
                idct8x8_2d(spectrum, filtered_block);

                for (int32_t py = 0; py < bs; ++py) {
                    for (int32_t px = 0; px < bs; ++px) {
                        local_num.at(x + px, y + py) += std::max(0.0f, filtered_block[py * 8 + px]);
                        local_den.at(x + px, y + py) += 1.0f;
                    }
                }
            }
        }

        #pragma omp critical
        {
            for (int32_t yy = band_y0; yy < band_y1; ++yy) {
                for (int32_t xx = 0; xx < w; ++xx) {
                    numerator.at(xx, yy) += local_num.at(xx, yy);
                    denominator.at(xx, yy) += local_den.at(xx, yy);
                }
            }
        }
    }

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
    // The engines filter R/G/B directly; map the Luma / Cb / Cr sliders onto each colour channel
    const auto amounts = colorspace::ycc_amounts_to_rgb(params.luma_scale, params.chroma_scale_cb, params.chroma_scale_cr);
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
        channel_params.channel_amount = amounts[std::min(c, size_t{2})];
        if (c > 0 && active_params.fast_chroma) {
            channel_params.step_size = std::min(active_params.step_size + 2, 6);
        }

        const auto& nlf = profile.nlf_per_channel[std::min(c, size_t{2})];
        engine->process_plane(planes, ref_index, mf_ptrs, nlf, profile.psd, channel_params, output_frame.channels[c]);
    }
}

} // namespace librenoise
