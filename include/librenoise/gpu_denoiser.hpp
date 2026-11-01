#pragma once

#include <cstdint>

namespace librenoise {

struct CudaDenoiseParams {
    int32_t width = 0;
    int32_t height = 0;
    int32_t stride_floats = 0;
    int32_t channels = 4; // 4 for RGBA
    int32_t step_size = 4; // Stride of patches (4 = balanced, 6 = fast, 3 = high quality)
    float hard_thresh_lambda = 2.7f;
    float temporal_weight = 1.0f;
    float spatial_weight = 1.0f;
    float band_weights[5] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    float nlf_a[3] = {0.001f, 0.001f, 0.001f};
    float nlf_b[3] = {0.0001f, 0.0001f, 0.0001f};
    float detail_recovery = 0.15f;
    float grain_amount = 0.0f;
    float mix_original = 0.0f; // 0.0 = 100% denoised, 1.0 = 100% original
    float chroma_scale_cb = 1.0f;
    float chroma_scale_cr = 1.0f;
    int32_t preview_mode = 0; // 0 = denoised, 1 = residual
};

/**
 * @brief Checks if CUDA runtime and capable GPU are present.
 */
bool is_cuda_available() noexcept;

/**
 * @brief Launch GPU video denoiser on CUDA device pointers.
 *
 * @param d_input_frames Array of host pointers to CUDA device memory pointers (RGBA float32).
 * @param frame_count Number of temporal frames in the window.
 * @param ref_index Index of the reference frame.
 * @param d_output CUDA device pointer to RGBA float32 output buffer.
 * @param params Denoising and noise profile parameters.
 * @param cuda_stream Optional CUDA stream (cudaStream_t).
 * @return true on success, false on failure.
 */
bool cuda_denoise_device(
    const float* const* d_input_frames,
    int32_t frame_count,
    int32_t ref_index,
    float* d_output,
    const CudaDenoiseParams& params,
    void* cuda_stream = nullptr
);

} // namespace librenoise
