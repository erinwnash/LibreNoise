#include "librenoise/alignment.hpp"
#include <cmath>
#include <algorithm>

namespace librenoise {

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

    MotionField field(num_bx, num_by, bs);

    // Multiplier for exposure drift compensation
    const float gain_correction = (exposure_drift_ratio > 0.01f) ? (1.0f / exposure_drift_ratio) : 1.0f;

    #pragma omp parallel for schedule(dynamic)
    for (int32_t by = 0; by < num_by; ++by) {
        for (int32_t bx = 0; bx < num_bx; ++bx) {
            const int32_t ref_x = bx * bs;
            const int32_t ref_y = by * bs;

            float avg_intensity = ref_luma.at(ref_x + bs / 2, ref_y + bs / 2);
            float expected_noise_variance = nlf.variance_at(avg_intensity);
            float trust_threshold = params.trust_sigma_multiplier * std::sqrt(expected_noise_variance);

            // Fast center check: if the block is stationary, early-exit immediately
            if (params.early_exit) {
                float center_sad = 0.0f;
                for (int32_t py = 0; py < bs; py += 2) {
                    for (int32_t px = 0; px < bs; px += 2) {
                        float r = ref_luma.at(ref_x + px, ref_y + py);
                        float t = target_luma.at(ref_x + px, ref_y + py) * gain_correction;
                        center_sad += std::abs(r - t);
                    }
                }
                float center_mse = center_sad / static_cast<float>((bs / 2) * (bs / 2));
                if (center_mse <= trust_threshold * 0.7f) {
                    MotionVector mv;
                    mv.dx = 0;
                    mv.dy = 0;
                    mv.residual = center_mse;
                    mv.trusted = true;
                    field.at(bx, by) = mv;
                    continue;
                }
            }

            int32_t best_dx = 0;
            int32_t best_dy = 0;
            float min_sad = 1e9f;

            // Coarse search within search_radius
            const int32_t rad = params.search_radius;
            const int32_t min_x = std::max(0, ref_x - rad);
            const int32_t max_x = std::min(w - bs, ref_x + rad);
            const int32_t min_y = std::max(0, ref_y - rad);
            const int32_t max_y = std::min(h - bs, ref_y + rad);

            for (int32_t cy = min_y; cy <= max_y; cy += 2) {
                for (int32_t cx = min_x; cx <= max_x; cx += 2) {
                    float current_sad = 0.0f;
                    for (int32_t py = 0; py < bs; py += 2) {
                        for (int32_t px = 0; px < bs; px += 2) {
                            float r = ref_luma.at(ref_x + px, ref_y + py);
                            float t = target_luma.at(cx + px, cy + py) * gain_correction;
                            current_sad += std::abs(r - t);
                        }
                    }
                    if (current_sad < min_sad) {
                        min_sad = current_sad;
                        best_dx = cx - ref_x;
                        best_dy = cy - ref_y;
                    }
                }
            }

            // Fine search around best point
            const int32_t fine_min_x = std::max(0, ref_x + best_dx - 1);
            const int32_t fine_max_x = std::min(w - bs, ref_x + best_dx + 1);
            const int32_t fine_min_y = std::max(0, ref_y + best_dy - 1);
            const int32_t fine_max_y = std::min(h - bs, ref_y + best_dy + 1);

            for (int32_t cy = fine_min_y; cy <= fine_max_y; ++cy) {
                for (int32_t cx = fine_min_x; cx <= fine_max_x; ++cx) {
                    float current_sad = 0.0f;
                    for (int32_t py = 0; py < bs; ++py) {
                        for (int32_t px = 0; px < bs; ++px) {
                            float r = ref_luma.at(ref_x + px, ref_y + py);
                            float t = target_luma.at(cx + px, cy + py) * gain_correction;
                            current_sad += std::abs(r - t);
                        }
                    }
                    if (current_sad < min_sad) {
                        min_sad = current_sad;
                        best_dx = cx - ref_x;
                        best_dy = cy - ref_y;
                    }
                }
            }

            // Mean squared error per pixel
            float mse = (min_sad / static_cast<float>(bs * bs));

            MotionVector mv;
            mv.dx = static_cast<int16_t>(best_dx);
            mv.dy = static_cast<int16_t>(best_dy);
            mv.residual = mse;
            mv.trusted = (mse <= trust_threshold * 1.5f);

            field.at(bx, by) = mv;
        }
    }

    return field;
}

} // namespace librenoise
