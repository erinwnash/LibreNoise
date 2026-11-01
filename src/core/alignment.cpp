#include "librenoise/alignment.hpp"
#include <cmath>
#include <algorithm>
#include <thread>
#include <immintrin.h>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace librenoise {

namespace {

inline int resolve_alignment_threads(int32_t requested) noexcept {
    if (requested > 0) return requested;
    int hw = static_cast<int>(std::thread::hardware_concurrency());
    if (hw <= 0) hw = 8;
    return std::clamp((hw * 3) / 4, 2, 16);
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

inline float block_sad_avx2(
    const float* ref_base,
    int32_t ref_stride,
    const float* tgt_base,
    int32_t tgt_stride,
    int32_t bs,
    int32_t row_step,
    float gain_correction
) noexcept {
    if (bs == 16) {
        const __m256 gain_v = _mm256_set1_ps(gain_correction);
        const __m256 sign_mask = _mm256_set1_ps(-0.0f);
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        for (int32_t py = 0; py < 16; py += row_step) {
            const float* rp = ref_base + static_cast<ptrdiff_t>(py) * ref_stride;
            const float* tp = tgt_base + static_cast<ptrdiff_t>(py) * tgt_stride;
            __m256 r0 = _mm256_loadu_ps(rp);
            __m256 r1 = _mm256_loadu_ps(rp + 8);
            __m256 t0 = _mm256_mul_ps(_mm256_loadu_ps(tp), gain_v);
            __m256 t1 = _mm256_mul_ps(_mm256_loadu_ps(tp + 8), gain_v);
            acc0 = _mm256_add_ps(acc0, _mm256_andnot_ps(sign_mask, _mm256_sub_ps(r0, t0)));
            acc1 = _mm256_add_ps(acc1, _mm256_andnot_ps(sign_mask, _mm256_sub_ps(r1, t1)));
        }
        return hsum256_ps(_mm256_add_ps(acc0, acc1));
    }

    float sad = 0.0f;
    for (int32_t py = 0; py < bs; py += row_step) {
        const float* rp = ref_base + static_cast<ptrdiff_t>(py) * ref_stride;
        const float* tp = tgt_base + static_cast<ptrdiff_t>(py) * tgt_stride;
        for (int32_t px = 0; px < bs; ++px) {
            sad += std::abs(rp[px] - tp[px] * gain_correction);
        }
    }
    return sad;
}

} // anonymous namespace

bool MotionEstimator::is_scene_cut(
    const ImagePlane32F& ref_luma,
    const ImagePlane32F& target_luma,
    float threshold
) {
    if (ref_luma.width() != target_luma.width() || ref_luma.height() != target_luma.height()) {
        return true;
    }

    const int32_t step = 8;
    double sad = 0.0;
    double sum_ref = 0.0;
    int64_t count = 0;

    for (int32_t y = 0; y < ref_luma.height(); y += step) {
        for (int32_t x = 0; x < ref_luma.width(); x += step) {
            float r = ref_luma.at(x, y);
            float t = target_luma.at(x, y);
            sad += std::abs(r - t);
            sum_ref += std::abs(r);
            count++;
        }
    }

    if (count == 0 || sum_ref < 1e-4) return false;
    float relative_diff = static_cast<float>(sad / sum_ref);
    return relative_diff > threshold;
}

MotionField MotionEstimator::estimate(
    const ImagePlane32F& ref_luma,
    const ImagePlane32F& target_luma,
    const NoiseLevelFunction& nlf,
    float exposure_drift_ratio,
    const MatchParams& params
) {
    const int32_t bs = params.block_size;
    const int32_t w = ref_luma.width();
    const int32_t h = ref_luma.height();
    const int32_t num_bx = w / bs;
    const int32_t num_by = h / bs;

    MotionField raw_field(num_bx, num_by, bs);
    if (num_bx <= 0 || num_by <= 0) return raw_field;

    // Multiplier for exposure drift compensation
    const float gain_correction = (exposure_drift_ratio > 0.01f) ? (1.0f / exposure_drift_ratio) : 1.0f;
    const int32_t ref_stride = ref_luma.stride();
    const int32_t tgt_stride = target_luma.stride();
    const float* ref_data = ref_luma.data();
    const float* tgt_data = target_luma.data();
    const int n_threads = resolve_alignment_threads(params.max_threads);

    #pragma omp parallel for schedule(dynamic, 4) num_threads(n_threads)
    for (int32_t by = 0; by < num_by; ++by) {
        for (int32_t bx = 0; bx < num_bx; ++bx) {
            const int32_t ref_x = bx * bs;
            const int32_t ref_y = by * bs;
            const float* ref_block = ref_data + static_cast<ptrdiff_t>(ref_y) * ref_stride + ref_x;

            // Sample 4 interior points for stable intensity estimate under heavy grain
            float avg_intensity = 0.25f * (
                ref_luma.at(ref_x + bs / 4, ref_y + bs / 4) +
                ref_luma.at(ref_x + (3 * bs) / 4, ref_y + bs / 4) +
                ref_luma.at(ref_x + bs / 4, ref_y + (3 * bs) / 4) +
                ref_luma.at(ref_x + (3 * bs) / 4, ref_y + (3 * bs) / 4)
            );
            float expected_noise_sigma = nlf.sigma_at(avg_intensity);
            float trust_threshold = params.trust_sigma_multiplier * expected_noise_sigma;

            // Fast center check: if the block is stationary within grain floor, early-exit immediately
            const float* center_tgt = tgt_data + static_cast<ptrdiff_t>(ref_y) * tgt_stride + ref_x;
            float center_sad_sub = block_sad_avx2(ref_block, ref_stride, center_tgt, tgt_stride, bs, 2, gain_correction);
            float center_mae = center_sad_sub / static_cast<float>(bs * (bs / 2));

            if (params.early_exit && center_mae <= trust_threshold * 0.85f) {
                MotionVector mv;
                mv.dx = 0;
                mv.dy = 0;
                mv.residual = center_mae;
                mv.trusted = true;
                raw_field.at(bx, by) = mv;
                continue;
            }

            const int32_t rad = params.search_radius;
            const int32_t min_x = std::max(0, ref_x - rad);
            const int32_t max_x = std::min(w - bs, ref_x + rad);
            const int32_t min_y = std::max(0, ref_y - rad);
            const int32_t max_y = std::min(h - bs, ref_y + rad);

            int32_t best_dx = 0;
            int32_t best_dy = 0;
            float min_sad = center_sad_sub * 0.98f; // Slight zero-motion prior against grain drift

            // Stage 1: Coarse logarithmic search (stride 4)
            for (int32_t cy = min_y; cy <= max_y; cy += 4) {
                for (int32_t cx = min_x; cx <= max_x; cx += 4) {
                    const float* cand_ptr = tgt_data + static_cast<ptrdiff_t>(cy) * tgt_stride + cx;
                    float sad = block_sad_avx2(ref_block, ref_stride, cand_ptr, tgt_stride, bs, 2, gain_correction);
                    if (sad < min_sad) {
                        min_sad = sad;
                        best_dx = cx - ref_x;
                        best_dy = cy - ref_y;
                    }
                }
            }

            // Stage 2: Medium refinement (stride 2 around Stage 1 winner)
            const int32_t med_min_x = std::max(min_x, ref_x + best_dx - 2);
            const int32_t med_max_x = std::min(max_x, ref_x + best_dx + 2);
            const int32_t med_min_y = std::max(min_y, ref_y + best_dy - 2);
            const int32_t med_max_y = std::min(max_y, ref_y + best_dy + 2);

            for (int32_t cy = med_min_y; cy <= med_max_y; cy += 2) {
                for (int32_t cx = med_min_x; cx <= med_max_x; cx += 2) {
                    const float* cand_ptr = tgt_data + static_cast<ptrdiff_t>(cy) * tgt_stride + cx;
                    float sad = block_sad_avx2(ref_block, ref_stride, cand_ptr, tgt_stride, bs, 2, gain_correction);
                    if (sad < min_sad) {
                        min_sad = sad;
                        best_dx = cx - ref_x;
                        best_dy = cy - ref_y;
                    }
                }
            }

            // Stage 3: Full-resolution 1-pixel refinement around Stage 2 winner
            const int32_t fine_min_x = std::max(min_x, ref_x + best_dx - 1);
            const int32_t fine_max_x = std::min(max_x, ref_x + best_dx + 1);
            const int32_t fine_min_y = std::max(min_y, ref_y + best_dy - 1);
            const int32_t fine_max_y = std::min(max_y, ref_y + best_dy + 1);
            float fine_min_sad = 1e9f;

            for (int32_t cy = fine_min_y; cy <= fine_max_y; ++cy) {
                for (int32_t cx = fine_min_x; cx <= fine_max_x; ++cx) {
                    const float* cand_ptr = tgt_data + static_cast<ptrdiff_t>(cy) * tgt_stride + cx;
                    float sad = block_sad_avx2(ref_block, ref_stride, cand_ptr, tgt_stride, bs, 1, gain_correction);
                    if (sad < fine_min_sad) {
                        fine_min_sad = sad;
                        best_dx = cx - ref_x;
                        best_dy = cy - ref_y;
                    }
                }
            }

            float mae = fine_min_sad / static_cast<float>(bs * bs);

            MotionVector mv;
            mv.dx = static_cast<int16_t>(best_dx);
            mv.dy = static_cast<int16_t>(best_dy);
            mv.residual = mae;
            mv.trusted = (mae <= trust_threshold * 2.2f);

            raw_field.at(bx, by) = mv;
        }
    }

    if (num_bx < 3 || num_by < 3) {
        return raw_field;
    }

    // 3x3 Vector Median Outlier Rejection (matches GPU k_smooth_motion_vectors)
    MotionField smoothed(num_bx, num_by, bs);
    #pragma omp parallel for schedule(static) num_threads(n_threads)
    for (int32_t by = 0; by < num_by; ++by) {
        for (int32_t bx = 0; bx < num_bx; ++bx) {
            int16_t vx[9], vy[9];
            int k = 0;
            for (int dy = -1; dy <= 1; ++dy) {
                int ny = std::clamp(by + dy, 0, num_by - 1);
                for (int dx = -1; dx <= 1; ++dx) {
                    int nx = std::clamp(bx + dx, 0, num_bx - 1);
                    const auto& m = raw_field.at(nx, ny);
                    vx[k] = m.dx;
                    vy[k] = m.dy;
                    ++k;
                }
            }
            std::nth_element(vx, vx + 4, vx + 9);
            std::nth_element(vy, vy + 4, vy + 9);

            MotionVector out_mv = raw_field.at(bx, by);
            out_mv.dx = vx[4];
            out_mv.dy = vy[4];
            smoothed.at(bx, by) = out_mv;
        }
    }

    return smoothed;
}

} // namespace librenoise
