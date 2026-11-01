#include "librenoise/gpu_denoiser.hpp"

namespace librenoise {

#if !defined(LIBRENOISE_HAS_CUDA) || (LIBRENOISE_HAS_CUDA == 0)

bool is_cuda_available() noexcept {
    return false;
}

bool cuda_denoise_device(
    const float* const* /*d_input_frames*/,
    int32_t /*frame_count*/,
    int32_t /*ref_index*/,
    float* /*d_output*/,
    const CudaDenoiseParams& /*params*/,
    void* /*cuda_stream*/
) {
    return false;
}

#endif

} // namespace librenoise
