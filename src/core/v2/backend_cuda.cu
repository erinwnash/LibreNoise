// SPDX-License-Identifier: Apache-2.0
// CUDA back end for the v0.2 pipeline. Same templated pipeline as the CPU path (pipeline_impl.inl).
#include "librenoise/v2/pipeline.hpp"
#include "pipeline_impl.inl"

#include <cuda_runtime.h>

#include <map>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <vector>

namespace librenoise::v2 {

void set_last_error(const std::string& s);

namespace detail {

template <class F>
__global__ void k_for2d(int w, int h, F f) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < w && y < h) f(x, y);
}

template <class F>
__global__ void k_sum2d(int w, int h, F f, double* out) {
    __shared__ double sh[256];
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    sh[tid] = (x < w && y < h) ? f(x, y) : 0.0;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) sh[tid] += sh[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(out, sh[0]);
}

// Size-keyed device memory pool (frames of a clip have the same size, so buffers are reused across renders).
struct CudaPool {
    std::mutex m;
    std::multimap<size_t, float*> free_list;
    size_t held = 0;

    float* get(size_t n) {
        {
            std::lock_guard<std::mutex> lk(m);
            auto it = free_list.find(n);
            if (it != free_list.end()) {
                float* p = it->second;
                free_list.erase(it);
                return p;
            }
        }
        float* p = nullptr;
        if (cudaMalloc(&p, (n > 0 ? n : 1) * sizeof(float)) != cudaSuccess) {
            // free cached buffers and retry once
            release_all();
            cudaGetLastError();
            if (cudaMalloc(&p, (n > 0 ? n : 1) * sizeof(float)) != cudaSuccess) {
                cudaGetLastError();
                throw std::bad_alloc();
            }
        }
        return p;
    }
    void put(float* p, size_t n) {
        std::lock_guard<std::mutex> lk(m);
        free_list.emplace(n, p);
    }
    void release_all() {
        std::lock_guard<std::mutex> lk(m);
        for (auto& e : free_list) cudaFree(e.second);
        free_list.clear();
    }
};

CudaPool& pool() {
    static CudaPool* p = new CudaPool();   // intentionally leaked: released explicitly via cuda_release()
    return *p;
}

struct CudaBackend {
    struct Buf {
        float* p = nullptr;
        size_t n = 0;
        Buf() = default;
        explicit Buf(size_t count) : p(pool().get(count)), n(count) {}
        Buf(const Buf&) = delete;
        Buf& operator=(const Buf&) = delete;
        Buf(Buf&& o) noexcept : p(o.p), n(o.n) { o.p = nullptr; o.n = 0; }
        Buf& operator=(Buf&& o) noexcept {
            if (this != &o) { reset(); p = o.p; n = o.n; o.p = nullptr; o.n = 0; }
            return *this;
        }
        ~Buf() { reset(); }
        void reset() { if (p) pool().put(p, n); p = nullptr; n = 0; }
    };

    cudaStream_t stream = nullptr;
    const Params* params = nullptr;
    double* d_sum = nullptr;

    Buf alloc(size_t n) { return Buf(n); }
    Buf upload(const float* host, size_t n) {
        Buf b(n);
        cudaMemcpyAsync(b.p, host, n * sizeof(float), cudaMemcpyHostToDevice, stream);
        return b;
    }
    void zero(float* p, size_t n) { cudaMemsetAsync(p, 0, n * sizeof(float), stream); }
    void copy(float* dst, const float* src, size_t n) {
        if (dst != src) cudaMemcpyAsync(dst, src, n * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    }

    template <class F>
    void for2d(int w, int h, F f) {
        if (w <= 0 || h <= 0) return;
        const dim3 block(32, 8);
        const dim3 grid((w + 31) / 32, (h + 7) / 8);
        k_for2d<<<grid, block, 0, stream>>>(w, h, f);
    }

    template <class F>
    double sum2d(int w, int h, F f) {
        if (w <= 0 || h <= 0) return 0.0;
        cudaMemsetAsync(d_sum, 0, sizeof(double), stream);
        const dim3 block(32, 8);
        const dim3 grid((w + 31) / 32, (h + 7) / 8);
        k_sum2d<<<grid, block, 0, stream>>>(w, h, f, d_sum);
        double s = 0.0;
        cudaMemcpyAsync(&s, d_sum, sizeof(double), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        return s;
    }

    bool aborted() const { return params != nullptr && params->aborted(); }

    bool timing = std::getenv("LIBRENOISE_TIMING") != nullptr;
    std::chrono::steady_clock::time_point t_last = std::chrono::steady_clock::now();
    void mark(const char* stage) {
        if (!timing) return;
        cudaStreamSynchronize(stream);
        const auto now = std::chrono::steady_clock::now();
        std::fprintf(stderr, "  [%-18s] %8.1f ms\n", stage, std::chrono::duration<double, std::milli>(now - t_last).count());
        t_last = now;
    }
};

} // namespace detail

bool cuda_available() noexcept {
    static int cached = -1;
    if (cached < 0) {
        int n = 0;
        if (cudaGetDeviceCount(&n) != cudaSuccess || n <= 0) {
            cudaGetLastError();
            cached = 0;
        } else {
            cudaDeviceProp prop{};
            cached = (cudaGetDeviceProperties(&prop, 0) == cudaSuccess && prop.major * 10 + prop.minor >= 86) ? 1 : 0;
        }
    }
    return cached == 1;
}

void cuda_release() noexcept {
    detail::pool().release_all();
}

bool denoise_cuda(const FrameView* frames, int32_t n, int32_t ref, int32_t width, int32_t height,
                  const NoiseModel& model, const Params& params, const OutputView& out, void* cuda_stream) {
    if (frames == nullptr || n <= 0 || ref < 0 || ref >= n || width <= 0 || height <= 0 || out.data == nullptr) {
        set_last_error("denoise_cuda: invalid arguments");
        return false;
    }
    if (!model.valid) {
        set_last_error("denoise_cuda: noise model is not valid");
        return false;
    }
    if (!cuda_available()) {
        set_last_error("denoise_cuda: no CUDA device (compute 8.6+) available");
        return false;
    }
    detail::CudaBackend be;
    be.stream = reinterpret_cast<cudaStream_t>(cuda_stream);
    be.params = &params;
    bool ok = false;
    try {
        auto host_lut = std::make_unique<VstLut>();
        build_vst(model, *host_lut);
        const size_t lut_floats = (sizeof(VstLut) + sizeof(float) - 1) / sizeof(float);
        auto lut_buf = be.upload(reinterpret_cast<const float*>(host_lut.get()), lut_floats);
        auto sum_buf = be.alloc(2);
        be.d_sum = reinterpret_cast<double*>(sum_buf.p);
        ok = detail::run_pipeline(be, frames, n, ref, width, height, model, params, out,
                                  reinterpret_cast<const VstLut*>(lut_buf.p));
        const cudaError_t e = cudaStreamSynchronize(be.stream);
        if (e != cudaSuccess) {
            set_last_error(std::string("denoise_cuda: ") + cudaGetErrorString(e));
            ok = false;
        }
    } catch (const std::bad_alloc&) {
        set_last_error("denoise_cuda: out of GPU memory");
        cudaStreamSynchronize(be.stream);
        ok = false;
    }
    const cudaError_t le = cudaGetLastError();
    if (le != cudaSuccess) {
        set_last_error(std::string("denoise_cuda: ") + cudaGetErrorString(le));
        ok = false;
    }
    return ok;
}

} // namespace librenoise::v2
