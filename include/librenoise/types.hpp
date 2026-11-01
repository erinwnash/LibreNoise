#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <memory>
#include <string>
#include <algorithm>
#include <cmath>

namespace librenoise {

enum class PixelFormat {
    Rgba8,
    Rgba16U,
    Rgba32F,
    PlanarRgb32F,
    PlanarYuv32F
};

enum class ColorTransfer {
    Linear,
    Rec709,
    SLog3,
    ArriLogC3,
    ArriLogC4,
    RedLog3G10,
    AcesCc
};

struct Rect {
    int32_t x = 0;
    int32_t y = 0;
    int32_t width = 0;
    int32_t height = 0;

    bool empty() const noexcept { return width <= 0 || height <= 0; }
    bool contains(int32_t px, int32_t py) const noexcept {
        return px >= x && px < (x + width) && py >= y && py < (y + height);
    }
};

/**
 * @brief High-performance contiguous float32 planar frame buffer.
 * Internally, LibreNoise processes image planes in linear float32 normalized [0.0, 1.0].
 */
class ImagePlane32F {
public:
    ImagePlane32F() = default;
    ImagePlane32F(int32_t width, int32_t height)
        : width_(width), height_(height), stride_(width), data_(static_cast<size_t>(width * height), 0.0f) {}
    
    ImagePlane32F(int32_t width, int32_t height, int32_t stride)
        : width_(width), height_(height), stride_(stride), data_(static_cast<size_t>(stride * height), 0.0f) {}

    int32_t width() const noexcept { return width_; }
    int32_t height() const noexcept { return height_; }
    int32_t stride() const noexcept { return stride_; }
    
    float* data() noexcept { return data_.data(); }
    const float* data() const noexcept { return data_.data(); }

    float& at(int32_t x, int32_t y) noexcept {
        return data_[static_cast<size_t>(y * stride_ + x)];
    }
    const float& at(int32_t x, int32_t y) const noexcept {
        return data_[static_cast<size_t>(y * stride_ + x)];
    }

    void copy_to(ImagePlane32F& dst) const {
        dst = *this;
    }

    void fill(float value) {
        std::fill(data_.begin(), data_.end(), value);
    }

private:
    int32_t width_ = 0;
    int32_t height_ = 0;
    int32_t stride_ = 0;
    std::vector<float> data_;
};

/**
 * @brief Multi-channel planar frame representation (3 or 4 channels).
 */
struct VideoFrame32F {
    int32_t width = 0;
    int32_t height = 0;
    ColorTransfer transfer = ColorTransfer::Linear;
    std::vector<ImagePlane32F> channels; // [0] = Y / R, [1] = U / G, [2] = V / B

    VideoFrame32F() = default;
    VideoFrame32F(int32_t w, int32_t h, size_t num_channels = 3, ColorTransfer trans = ColorTransfer::Linear)
        : width(w), height(h), transfer(trans) {
        channels.resize(num_channels);
        for (auto& ch : channels) {
            ch = ImagePlane32F(w, h);
        }
    }

    size_t channel_count() const noexcept { return channels.size(); }
    bool is_valid() const noexcept { return width > 0 && height > 0 && !channels.empty(); }
};

} // namespace librenoise
