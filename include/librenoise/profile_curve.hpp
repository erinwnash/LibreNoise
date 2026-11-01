#pragma once

#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdint>

namespace librenoise {

struct CurvePoint {
    float x = 0.0f; // Input intensity [0.0, 1.0]
    float y = 1.0f; // Noise scale factor / multiplier [0.0, 3.0]
};

/**
 * @brief Monotonic cubic Hermite spline curve for tone-dependent noise level tuning.
 * Guarantees smooth, non-oscillating curves for tone mapping noise thresholds.
 */
class NoiseProfileCurve {
public:
    NoiseProfileCurve() {
        // Default standard response
        points_ = {
            {0.00f, 1.30f}, // Shadows (sensor read noise dominant)
            {0.25f, 1.15f}, // Dark midtones
            {0.50f, 1.00f}, // Midtones (calibrated unity)
            {0.75f, 0.90f}, // Light midtones
            {1.00f, 0.80f}  // Highlights (shot noise saturation)
        };
        update_spline();
    }

    explicit NoiseProfileCurve(const std::vector<CurvePoint>& points) : points_(points) {
        if (points_.size() < 2) {
            points_ = {{0.0f, 1.0f}, {1.0f, 1.0f}};
        }
        std::sort(points_.begin(), points_.end(), [](const CurvePoint& a, const CurvePoint& b) {
            return a.x < b.x;
        });
        update_spline();
    }

    void set_point(size_t index, float x, float y) {
        if (index < points_.size()) {
            points_[index].x = std::clamp(x, 0.0f, 1.0f);
            points_[index].y = std::clamp(y, 0.0f, 5.0f);
            std::sort(points_.begin(), points_.end(), [](const CurvePoint& a, const CurvePoint& b) {
                return a.x < b.x;
            });
            update_spline();
        }
    }

    void add_point(float x, float y) {
        points_.push_back({std::clamp(x, 0.0f, 1.0f), std::clamp(y, 0.0f, 5.0f)});
        std::sort(points_.begin(), points_.end(), [](const CurvePoint& a, const CurvePoint& b) {
            return a.x < b.x;
        });
        update_spline();
    }

    const std::vector<CurvePoint>& points() const noexcept { return points_; }

    /**
     * @brief Sample the curve at a given image intensity using monotonic cubic Hermite interpolation.
     */
    float evaluate(float x) const noexcept {
        if (points_.empty()) return 1.0f;
        if (x <= points_.front().x) return points_.front().y;
        if (x >= points_.back().x) return points_.back().y;

        // Binary search segment
        size_t low = 0;
        size_t high = points_.size() - 1;
        while (high - low > 1) {
            size_t mid = (low + high) / 2;
            if (points_[mid].x > x) {
                high = mid;
            } else {
                low = mid;
            }
        }

        const float x0 = points_[low].x;
        const float x1 = points_[high].x;
        const float y0 = points_[low].y;
        const float y1 = points_[high].y;
        const float h = x1 - x0;
        if (h <= 1e-7f) return y0;

        const float t = (x - x0) / h;
        const float t2 = t * t;
        const float t3 = t2 * t;

        // Hermite basis functions
        const float h00 = 2.0f * t3 - 3.0f * t2 + 1.0f;
        const float h10 = t3 - 2.0f * t2 + t;
        const float h01 = -2.0f * t3 + 3.0f * t2;
        const float h11 = t3 - t2;

        const float m0 = tangents_[low];
        const float m1 = tangents_[high];

        float result = h00 * y0 + h10 * h * m0 + h01 * y1 + h11 * h * m1;
        return std::clamp(result, 0.0f, 5.0f);
    }

private:
    void update_spline() {
        const size_t n = points_.size();
        tangents_.assign(n, 0.0f);
        if (n < 2) return;

        std::vector<float> deltas(n - 1);
        for (size_t i = 0; i < n - 1; ++i) {
            float dx = points_[i + 1].x - points_[i].x;
            deltas[i] = (dx > 1e-7f) ? ((points_[i + 1].y - points_[i].y) / dx) : 0.0f;
        }

        // Fritsch-Carlson monotonic tangent initialization
        tangents_[0] = deltas[0];
        tangents_[n - 1] = deltas[n - 2];
        for (size_t i = 1; i < n - 1; ++i) {
            tangents_[i] = (deltas[i - 1] + deltas[i]) * 0.5f;
        }

        for (size_t i = 0; i < n - 1; ++i) {
            if (std::abs(deltas[i]) < 1e-7f) {
                tangents_[i] = 0.0f;
                tangents_[i + 1] = 0.0f;
            } else {
                float alpha = tangents_[i] / deltas[i];
                float beta = tangents_[i + 1] / deltas[i];
                float s = alpha * alpha + beta * beta;
                if (s > 9.0f) {
                    float tau = 3.0f / std::sqrt(s);
                    tangents_[i] = tau * alpha * deltas[i];
                    tangents_[i + 1] = tau * beta * deltas[i];
                }
            }
        }
    }

    std::vector<CurvePoint> points_;
    std::vector<float> tangents_;
};

} // namespace librenoise
