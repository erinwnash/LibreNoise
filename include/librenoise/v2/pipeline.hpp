// SPDX-License-Identifier: Apache-2.0
#pragma once
// LibreNoise v0.2 public processing API.

#include "librenoise/v2/model.hpp"
#include <cstdint>
#include <string>

namespace librenoise::v2 {

// Interleaved float image (RGB or RGBA). For the CPU path the pointer is host memory, for the CUDA path device memory.
struct FrameView {
    const float* data = nullptr;
    int32_t stride_floats = 0;   // floats per row
    int32_t channels = 4;        // 3 or 4
};

struct OutputView {
    float* data = nullptr;
    int32_t stride_floats = 0;
    int32_t channels = 4;
};

struct ProfileOptions {
    // Optional user sample area (pixels, same orientation as the frame buffer). Width/height 0 = automatic.
    int32_t roi_x = 0, roi_y = 0, roi_w = 0, roi_h = 0;
    SourceType force_type = SourceType::Auto;
};

// Build a noise model from one host frame (interleaved float). Pure algorithm: nothing stored or looked up.
NoiseModel estimate_noise_model(const FrameView& frame, int32_t width, int32_t height, const ProfileOptions& opt = {});

// Denoise frames[ref] using frames[0..n). All views are width x height.
bool denoise_cpu(const FrameView* frames, int32_t n, int32_t ref, int32_t width, int32_t height,
                 const NoiseModel& model, const Params& params, const OutputView& out);

// Frees cached working buffers (both back ends keep them between frames for speed).
void release_cpu_buffers() noexcept;

// CUDA path: frame data, the frames array itself and the output live in device memory.
bool cuda_available() noexcept;
void cuda_release() noexcept;
bool denoise_cuda(const FrameView* host_array_of_device_frames, int32_t n, int32_t ref, int32_t width, int32_t height,
                  const NoiseModel& model, const Params& params, const OutputView& device_out, void* cuda_stream = nullptr);

// Last error message from either back end (empty if none).
std::string last_error();

} // namespace librenoise::v2
