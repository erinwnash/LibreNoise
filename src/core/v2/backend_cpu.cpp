// CPU (OpenMP) back end for the v0.2 pipeline.
#include "librenoise/v2/pipeline.hpp"
#include "pipeline_impl.inl"

#include <cstring>
#include <memory>
#include <mutex>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace librenoise::v2 {

namespace {
std::mutex g_err_mutex;
std::string g_last_error;
}

void set_last_error(const std::string& s) {
    std::lock_guard<std::mutex> lk(g_err_mutex);
    g_last_error = s;
}

std::string last_error() {
    std::lock_guard<std::mutex> lk(g_err_mutex);
    return g_last_error;
}

namespace detail {

struct CpuBackend {
    struct Buf {
        std::unique_ptr<float[]> mem;
        float* p = nullptr;
        size_t n = 0;
        Buf() = default;
        explicit Buf(size_t count) : mem(new float[count > 0 ? count : 1]), p(mem.get()), n(count) {}
        Buf(Buf&&) noexcept = default;
        Buf& operator=(Buf&&) noexcept = default;
    };

    const Params* params = nullptr;

    Buf alloc(size_t n) { return Buf(n); }
    Buf upload(const float* host, size_t n) {
        Buf b(n);
        std::memcpy(b.p, host, n * sizeof(float));
        return b;
    }
    void zero(float* p, size_t n) {
        const int64_t N = static_cast<int64_t>(n);
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < N; ++i) p[i] = 0.0f;
    }
    void copy(float* dst, const float* src, size_t n) {
        if (dst == src) return;
        const int64_t N = static_cast<int64_t>(n);
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < N; ++i) dst[i] = src[i];
    }

    template <class F>
    void for2d(int w, int h, F f) {
        if (w <= 0 || h <= 0) return;
        if (h >= 64) {
            #pragma omp parallel for schedule(static)
            for (int y = 0; y < h; ++y)
                for (int x = 0; x < w; ++x) f(x, y);
        } else {
            const int64_t total = static_cast<int64_t>(w) * h;
            #pragma omp parallel for schedule(static)
            for (int64_t i = 0; i < total; ++i) f(static_cast<int>(i % w), static_cast<int>(i / w));
        }
    }

    template <class F>
    double sum2d(int w, int h, F f) {
        double s = 0.0;
        if (w <= 0 || h <= 0) return s;
        #pragma omp parallel for schedule(static) reduction(+ : s)
        for (int y = 0; y < h; ++y) {
            double row = 0.0;
            for (int x = 0; x < w; ++x) row += f(x, y);
            s += row;
        }
        return s;
    }

    bool aborted() const { return params != nullptr && params->aborted(); }
};

} // namespace detail

bool denoise_cpu(const FrameView* frames, int32_t n, int32_t ref, int32_t width, int32_t height,
                 const NoiseModel& model, const Params& params, const OutputView& out) {
    if (frames == nullptr || n <= 0 || ref < 0 || ref >= n || width <= 0 || height <= 0 || out.data == nullptr) {
        set_last_error("denoise_cpu: invalid arguments");
        return false;
    }
    if (!model.valid) {
        set_last_error("denoise_cpu: noise model is not valid");
        return false;
    }
#ifdef _OPENMP
    if (params.max_threads > 0) omp_set_num_threads(params.max_threads);
#endif
    auto lut = std::make_unique<VstLut>();
    build_vst(model, *lut);
    detail::CpuBackend be;
    be.params = &params;
    try {
        return detail::run_pipeline(be, frames, n, ref, width, height, model, params, out, lut.get());
    } catch (const std::bad_alloc&) {
        set_last_error("denoise_cpu: out of memory");
        return false;
    }
}

} // namespace librenoise::v2
