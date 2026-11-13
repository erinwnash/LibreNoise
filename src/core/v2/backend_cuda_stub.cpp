// SPDX-License-Identifier: Apache-2.0
// Used when the CUDA toolkit is not available at build time.
#include "librenoise/v2/pipeline.hpp"

namespace librenoise::v2 {
void set_last_error(const std::string& s);
bool cuda_available() noexcept { return false; }
void cuda_release() noexcept {}
bool denoise_cuda(const FrameView*, int32_t, int32_t, int32_t, int32_t, const NoiseModel&, const Params&, const OutputView&, void*) {
    set_last_error("built without CUDA");
    return false;
}
} // namespace librenoise::v2
