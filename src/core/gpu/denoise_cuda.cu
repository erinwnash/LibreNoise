#include "librenoise/gpu_denoiser.hpp"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <algorithm>
#include <atomic>
#include <vector>
#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace librenoise {

// Constant memory for 8x8 Orthonormal DCT-II basis matrix and smooth Kaiser-like feather window
__constant__ float c_dct_mat[8][8];
__constant__ float c_feather_1d[8];

namespace {

constexpr int kMaxGroupSize = 12;

struct alignas(8) GpuMotionVec {
    int16_t dx;
    int16_t dy;
    int16_t trusted;
    int16_t pad;
};

struct GpuKernelConfig {
    int width;
    int height;
    int stride_floats;
    int channels;
    int step_size;
    int frame_count;
    int ref_index;
    int num_bx;
    int num_by;
    int early_exit_motion;
    float hard_thresh_lambda;
    float temporal_weight;
    float spatial_weight;
    float detail_recovery;
    float grain_amount;
    float mix_original;
    float chroma_scale_cb;
    float chroma_scale_cr;
    int preview_mode;
    int force_opaque_alpha;
    float band_weights[5];
    float psd_weights[3][5];
    float tonal_lut[5];
    float nlf_a[3];
    float nlf_b[3];
};

// Cached GPU scratch arena to avoid per-frame cudaMalloc/cudaFree stalls in interactive hosts
struct GpuScratchArena {
    size_t alloc_pixels = 0;
    int alloc_frames = 0;
    int alloc_mv_blocks = 0;

    float* d_luma = nullptr;        // frame_count * width * height
    float* d_pilot[3] = {};         // 3 * width * height
    float* d_num[3] = {};           // 3 * width * height
    float* d_den[3] = {};           // 3 * width * height
    float* d_frame_stats = nullptr; // frame_count * 4
    float* d_gain_corr = nullptr;   // frame_count
    int* d_scene_cut = nullptr;     // frame_count
    GpuMotionVec* d_mv_raw = nullptr;    // frame_count * num_bx * num_by
    GpuMotionVec* d_mv_smooth = nullptr; // frame_count * num_bx * num_by

    bool ensure_capacity(size_t pixels, int frames, int mv_blocks) {
        if (pixels <= alloc_pixels && frames <= alloc_frames && mv_blocks <= alloc_mv_blocks && d_luma != nullptr) {
            return true;
        }
        release();

        size_t p_cap = std::max<size_t>(pixels, 64 * 64);
        int f_cap = std::max(frames, 9);
        int mv_cap = std::max(mv_blocks, 64);

        auto ok = [&](cudaError_t err) { return err == cudaSuccess; };
        bool success = true;
        success = success && ok(cudaMalloc(&d_luma, static_cast<size_t>(f_cap) * p_cap * sizeof(float)));
        for (int c = 0; c < 3; ++c) {
            success = success && ok(cudaMalloc(&d_pilot[c], p_cap * sizeof(float)));
            success = success && ok(cudaMalloc(&d_num[c], p_cap * sizeof(float)));
            success = success && ok(cudaMalloc(&d_den[c], p_cap * sizeof(float)));
        }
        success = success && ok(cudaMalloc(&d_frame_stats, static_cast<size_t>(f_cap) * 4 * sizeof(float)));
        success = success && ok(cudaMalloc(&d_gain_corr, static_cast<size_t>(f_cap) * sizeof(float)));
        success = success && ok(cudaMalloc(&d_scene_cut, static_cast<size_t>(f_cap) * sizeof(int)));
        success = success && ok(cudaMalloc(&d_mv_raw, static_cast<size_t>(f_cap) * mv_cap * sizeof(GpuMotionVec)));
        success = success && ok(cudaMalloc(&d_mv_smooth, static_cast<size_t>(f_cap) * mv_cap * sizeof(GpuMotionVec)));

        if (!success) {
            release();
            return false;
        }

        alloc_pixels = p_cap;
        alloc_frames = f_cap;
        alloc_mv_blocks = mv_cap;
        return true;
    }

    void release() {
        // Fast no-op when no GPU scratch memory was ever allocated (e.g., during host startup plugin scan)
        if (d_luma == nullptr && alloc_pixels == 0) {
            return;
        }
        int dev = 0;
        cudaError_t state = cudaGetDevice(&dev);
        bool runtime_alive = (state == cudaSuccess);
        if (runtime_alive) {
            if (d_luma) cudaFree(d_luma);
            for (int c = 0; c < 3; ++c) {
                if (d_pilot[c]) cudaFree(d_pilot[c]);
                if (d_num[c]) cudaFree(d_num[c]);
                if (d_den[c]) cudaFree(d_den[c]);
            }
            if (d_frame_stats) cudaFree(d_frame_stats);
            if (d_gain_corr) cudaFree(d_gain_corr);
            if (d_scene_cut) cudaFree(d_scene_cut);
            if (d_mv_raw) cudaFree(d_mv_raw);
            if (d_mv_smooth) cudaFree(d_mv_smooth);
        }
        d_luma = nullptr;
        for (int c = 0; c < 3; ++c) {
            d_pilot[c] = nullptr;
            d_num[c] = nullptr;
            d_den[c] = nullptr;
        }
        d_frame_stats = nullptr;
        d_gain_corr = nullptr;
        d_scene_cut = nullptr;
        d_mv_raw = nullptr;
        d_mv_smooth = nullptr;
        alloc_pixels = 0;
        alloc_frames = 0;
        alloc_mv_blocks = 0;
    }

    ~GpuScratchArena() {
        // Do not invoke CUDA driver teardown during OS process exit / DllMain detach;
        // explicit cleanup is handled via release_cuda_resources() in kOfxActionUnload.
    }
};

// Zero-CRT-dependency lock (avoids MSVCP140.dll!_Mtx_lock ABI mismatch in hosts shipping older MSVC runtimes)
struct GpuArenaLock {
#if defined(_WIN32)
    SRWLOCK srw = SRWLOCK_INIT;
    void lock() noexcept { AcquireSRWLockExclusive(&srw); }
    void unlock() noexcept { ReleaseSRWLockExclusive(&srw); }
#else
    std::atomic_flag flag = ATOMIC_FLAG_INIT;
    void lock() noexcept { while (flag.test_and_set(std::memory_order_acquire)) {} }
    void unlock() noexcept { flag.clear(std::memory_order_release); }
#endif
};

struct GpuScopedLock {
    GpuArenaLock& m;
    explicit GpuScopedLock(GpuArenaLock& mtx) noexcept : m(mtx) { m.lock(); }
    ~GpuScopedLock() noexcept { m.unlock(); }
};

static GpuArenaLock g_cuda_lock;
static GpuScratchArena g_arena;

void init_dct_constants() {
    static bool initialized = false;
    if (initialized) return;

    float h_mat[8][8];
    constexpr float s0 = 0.35355339059327376f; // 1 / sqrt(8)
    constexpr float s = 0.5f;                  // sqrt(2/8)
    constexpr double pi = 3.14159265358979323846;

    for (int k = 0; k < 8; ++k) {
        float factor = (k == 0) ? s0 : s;
        for (int n = 0; n < 8; ++n) {
            h_mat[k][n] = factor * static_cast<float>(std::cos((pi / 8.0) * (n + 0.5) * k));
        }
    }

    const float h_feather[8] = {0.38f, 0.68f, 0.92f, 1.0f, 1.0f, 0.92f, 0.68f, 0.38f};

    cudaMemcpyToSymbol(c_dct_mat, h_mat, sizeof(h_mat));
    cudaMemcpyToSymbol(c_feather_1d, h_feather, sizeof(h_feather));
    initialized = true;
}

__device__ inline float hash_noise(int x, int y, uint32_t seed) {
    uint32_t n = static_cast<uint32_t>(x) + static_cast<uint32_t>(y) * 19849u + seed * 65537u;
    n = (n ^ (n >> 13)) * 1274126177u;
    n = n ^ (n >> 16);
    return (static_cast<float>(n & 0x00FFFFFF) / static_cast<float>(0x00FFFFFF)) * 2.0f - 1.0f;
}

__device__ inline float eval_tonal_curve(const GpuKernelConfig& cfg, float luma) {
    float t = fminf(1.0f, fmaxf(0.0f, luma)) * 4.0f;
    int idx = min(3, max(0, static_cast<int>(t)));
    float frac = t - static_cast<float>(idx);
    return cfg.tonal_lut[idx] * (1.0f - frac) + cfg.tonal_lut[idx + 1] * frac;
}

__device__ inline float eval_channel_sigma(const GpuKernelConfig& cfg, int c, float val, float luma) {
    float var = fmaxf(1e-8f, cfg.nlf_a[c] * fmaxf(0.0f, val) + cfg.nlf_b[c]);
    float chroma_mult = (c == 0) ? cfg.chroma_scale_cr : (c == 2) ? cfg.chroma_scale_cb : 1.0f;
    return sqrtf(var) * cfg.spatial_weight * eval_tonal_curve(cfg, luma) * chroma_mult;
}

__device__ inline int freq_band_for_uv(int u, int v) {
    float r = sqrtf(static_cast<float>(u * u + v * v));
    return (r < 1.5f) ? 0 : (r < 3.0f) ? 1 : (r < 5.0f) ? 2 : (r < 7.5f) ? 3 : 4;
}

__device__ inline float eval_band_scale(const GpuKernelConfig& cfg, int c, int band) {
    return fmaxf(0.05f, cfg.psd_weights[c][band] * cfg.band_weights[band]);
}

__device__ inline float max_band_scale_for_channel(const GpuKernelConfig& cfg, int c) {
    float m = 1.0f;
    #pragma unroll
    for (int b = 0; b < 5; ++b) {
        m = fmaxf(m, eval_band_scale(cfg, c, b));
    }
    return m;
}

// =========================================================================
// Kernel 0A: Extract 3x3 Gaussian-smoothed Luma + Subsampled Frame Stats
// =========================================================================
__global__ void k_extract_smoothed_luma(
    const float* const* __restrict__ d_frames,
    GpuKernelConfig cfg,
    float* __restrict__ d_luma_all,
    float* __restrict__ d_frame_stats
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int f = blockIdx.z;

    if (x >= cfg.width || y >= cfg.height || f >= cfg.frame_count) return;

    const float* frame = d_frames[f];
    const int ch = cfg.channels;
    const int stride = cfg.stride_floats;

    const float kw[3] = {0.25f, 0.50f, 0.25f};
    float luma_sum = 0.0f;

    #pragma unroll
    for (int dy = -1; dy <= 1; ++dy) {
        int py = min(cfg.height - 1, max(0, y + dy));
        float wy = kw[dy + 1];
        #pragma unroll
        for (int dx = -1; dx <= 1; ++dx) {
            int px = min(cfg.width - 1, max(0, x + dx));
            int idx = py * stride + px * ch;
            float r = frame[idx + 0];
            float g = frame[idx + 1];
            float b = frame[idx + 2];
            float y_val = 0.2126f * r + 0.7152f * g + 0.0722f * b;
            luma_sum += wy * kw[dx + 1] * y_val;
        }
    }

    size_t plane_offset = static_cast<size_t>(f) * static_cast<size_t>(cfg.width * cfg.height);
    d_luma_all[plane_offset + y * cfg.width + x] = luma_sum;

    // Subsampled scene-cut accumulation (every 4th pixel, clamped to [0, 2] so HDR fireflies cannot skew frame stats)
    if ((x & 3) == 0 && (y & 3) == 0) {
        const float* ref_frame = d_frames[cfg.ref_index];
        int cidx = y * stride + x * ch;
        float ref_y = fminf(2.0f, fmaxf(0.0f, 0.2126f * ref_frame[cidx + 0] + 0.7152f * ref_frame[cidx + 1] + 0.0722f * ref_frame[cidx + 2]));
        float cur_y = fminf(2.0f, fmaxf(0.0f, 0.2126f * frame[cidx + 0] + 0.7152f * frame[cidx + 1] + 0.0722f * frame[cidx + 2]));

        atomicAdd(&d_frame_stats[f * 4 + 0], fabsf(ref_y));
        atomicAdd(&d_frame_stats[f * 4 + 1], cur_y);
        atomicAdd(&d_frame_stats[f * 4 + 2], fabsf(ref_y - cur_y));
        atomicAdd(&d_frame_stats[f * 4 + 3], ref_y);
    }
}

__global__ void k_finalize_frame_stats(
    int frame_count,
    int ref_index,
    const float* __restrict__ d_frame_stats,
    float* __restrict__ d_gain_corr,
    int* __restrict__ d_scene_cut
) {
    int f = threadIdx.x;
    if (f >= frame_count) return;

    if (f == ref_index) {
        d_gain_corr[f] = 1.0f;
        d_scene_cut[f] = 0;
        return;
    }

    float abs_ref = d_frame_stats[f * 4 + 0];
    float sad = d_frame_stats[f * 4 + 2];

    float rel_diff = (abs_ref > 1e-4f) ? (sad / abs_ref) : 0.0f;
    d_scene_cut[f] = (rel_diff > 0.45f) ? 1 : 0;
    // Preserve exact linear radiometric values across frames (matches CPU filter pipeline)
    d_gain_corr[f] = 1.0f;
}

// =========================================================================
// Kernel 0B: Hierarchical 16x16 Luma Block Motion Estimation (1 Warp / Block)
// =========================================================================
__global__ void k_estimate_motion_luma(
    const float* __restrict__ d_luma_all,
    const float* __restrict__ d_gain_corr,
    const int* __restrict__ d_scene_cut,
    GpuKernelConfig cfg,
    GpuMotionVec* __restrict__ d_mv_raw
) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int f = blockIdx.z;
    int lane = threadIdx.x; // 0..31

    if (bx >= cfg.num_bx || by >= cfg.num_by || f >= cfg.frame_count) return;

    int mv_idx = (f * cfg.num_by + by) * cfg.num_bx + bx;

    if (f == cfg.ref_index) {
        if (lane == 0) d_mv_raw[mv_idx] = {0, 0, 1, 0};
        return;
    }

    if (d_scene_cut[f] != 0) {
        if (lane == 0) d_mv_raw[mv_idx] = {0, 0, 0, 0};
        return;
    }

    const int w = cfg.width;
    const int h = cfg.height;
    const size_t plane_sz = static_cast<size_t>(w * h);
    const float* ref_luma = d_luma_all + static_cast<size_t>(cfg.ref_index) * plane_sz;
    const float* cand_luma = d_luma_all + static_cast<size_t>(f) * plane_sz;
    const float gain = d_gain_corr[f];

    const int bs = min(16, min(w, h));
    const int ref_x = min(bx * 16, max(0, w - bs));
    const int ref_y = min(by * 16, max(0, h - bs));

    float center_luma = ref_luma[(ref_y + bs / 2) * w + (ref_x + bs / 2)];
    float var_0 = fmaxf(1e-8f, cfg.nlf_a[0] * fmaxf(0.0f, center_luma) + cfg.nlf_b[0]);
    float var_1 = fmaxf(1e-8f, cfg.nlf_a[1] * fmaxf(0.0f, center_luma) + cfg.nlf_b[1]);
    float sigma_luma = sqrtf(0.5f * (var_0 + var_1));

    // Subsampled 64-point MAE helper across the 32 warp lanes (2 samples per lane)
    auto eval_mae_64 = [&](int dx, int dy) -> float {
        int cx = min(max(0, ref_x + dx), max(0, w - bs));
        int cy = min(max(0, ref_y + dy), max(0, h - bs));
        float sad = 0.0f;
        #pragma unroll
        for (int s = 0; s < 2; ++s) {
            int idx = lane + s * 32; // 0..63
            int py = min(bs - 1, (idx >> 3) * 2);
            int px = min(bs - 1, (idx & 7) * 2);
            float r = ref_luma[(ref_y + py) * w + (ref_x + px)];
            float t = cand_luma[(cy + py) * w + (cx + px)] * gain;
            sad += fabsf(r - t);
        }
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            sad += __shfl_down_sync(0xffffffff, sad, offset);
        }
        return __shfl_sync(0xffffffff, sad, 0) * (1.0f / 64.0f);
    };

    float mae_zero = eval_mae_64(0, 0);
    if (cfg.early_exit_motion && mae_zero <= 1.25f * sigma_luma) {
        if (lane == 0) d_mv_raw[mv_idx] = {0, 0, 1, 0};
        return;
    }

    int best_dx = 0;
    int best_dy = 0;
    // Slight zero-motion preference prevents random drift on featureless skies
    float best_cost = mae_zero - 0.08f * sigma_luma;

    // Step 1: Coarse search [-16, +16] with stride 4
    for (int dy = -16; dy <= 16; dy += 4) {
        for (int dx = -16; dx <= 16; dx += 4) {
            if (dx == 0 && dy == 0) continue;
            if (ref_x + dx < 0 || ref_x + dx + bs > w || ref_y + dy < 0 || ref_y + dy + bs > h) continue;
            float cost = eval_mae_64(dx, dy);
            if (cost < best_cost) {
                best_cost = cost;
                best_dx = dx;
                best_dy = dy;
            }
        }
    }

    // Step 2: Medium search around coarse winner with stride 2
    int med_cx = best_dx;
    int med_cy = best_dy;
    for (int dy = med_cy - 2; dy <= med_cy + 2; dy += 2) {
        for (int dx = med_cx - 2; dx <= med_cx + 2; dx += 2) {
            if (dx == med_cx && dy == med_cy) continue;
            if (ref_x + dx < 0 || ref_x + dx + bs > w || ref_y + dy < 0 || ref_y + dy + bs > h) continue;
            float cost = eval_mae_64(dx, dy);
            if (cost < best_cost) {
                best_cost = cost;
                best_dx = dx;
                best_dy = dy;
            }
        }
    }

    // Step 3: Full 256-pixel 1px refinement around medium winner (8 samples per lane)
    int fine_cx = best_dx;
    int fine_cy = best_dy;
    float fine_best_mae = 1e9f;

    for (int dy = fine_cy - 1; dy <= fine_cy + 1; ++dy) {
        for (int dx = fine_cx - 1; dx <= fine_cx + 1; ++dx) {
            if (ref_x + dx < 0 || ref_x + dx + bs > w || ref_y + dy < 0 || ref_y + dy + bs > h) continue;
            int cx = ref_x + dx;
            int cy = ref_y + dy;
            float sad = 0.0f;
            int total_pts = bs * bs;
            for (int idx = lane; idx < total_pts; idx += 32) {
                int py = idx / bs;
                int px = idx % bs;
                float r = ref_luma[(ref_y + py) * w + (ref_x + px)];
                float t = cand_luma[(cy + py) * w + (cx + px)] * gain;
                sad += fabsf(r - t);
            }
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                sad += __shfl_down_sync(0xffffffff, sad, offset);
            }
            float mae = __shfl_sync(0xffffffff, sad, 0) / static_cast<float>(total_pts);
            float bias = (dx == 0 && dy == 0) ? (0.05f * sigma_luma) : 0.0f;
            if (mae - bias < fine_best_mae) {
                fine_best_mae = mae - bias;
                best_dx = dx;
                best_dy = dy;
            }
        }
    }

    if (lane == 0) {
        float max_b = max_band_scale_for_channel(cfg, 0);
        float trust_limit = fmaxf(0.015f, 3.5f * sigma_luma * max_b * fmaxf(0.5f, cfg.temporal_weight));
        int16_t is_trusted = (fine_best_mae <= trust_limit) ? 1 : 0;
        d_mv_raw[mv_idx] = {static_cast<int16_t>(best_dx), static_cast<int16_t>(best_dy), is_trusted, 0};
    }
}

// =========================================================================
// Kernel 0C: 3x3 Spatial Median Filter on Motion Vectors (Coherent R/G/B)
// =========================================================================
__global__ void k_smooth_motion_vectors(
    const GpuMotionVec* __restrict__ d_mv_raw,
    int num_bx,
    int num_by,
    int frame_count,
    int ref_index,
    GpuMotionVec* __restrict__ d_mv_smooth
) {
    int bx = blockIdx.x * blockDim.x + threadIdx.x;
    int by = blockIdx.y * blockDim.y + threadIdx.y;
    int f = blockIdx.z;

    if (bx >= num_bx || by >= num_by || f >= frame_count) return;

    int idx = (f * num_by + by) * num_bx + bx;
    if (f == ref_index) {
        d_mv_smooth[idx] = {0, 0, 1, 0};
        return;
    }

    GpuMotionVec center = d_mv_raw[idx];
    int16_t vx[9];
    int16_t vy[9];
    int count = 0;

    for (int dy = -1; dy <= 1; ++dy) {
        int ny = min(num_by - 1, max(0, by + dy));
        for (int dx = -1; dx <= 1; ++dx) {
            int nx = min(num_bx - 1, max(0, bx + dx));
            GpuMotionVec m = d_mv_raw[(f * num_by + ny) * num_bx + nx];
            if (m.trusted) {
                vx[count] = m.dx;
                vy[count] = m.dy;
                count++;
            }
        }
    }

    if (count == 0 || (!center.trusted && count < 5)) {
        d_mv_smooth[idx] = {0, 0, 0, 0};
        return;
    }

    // Insertion sort for exact median
    for (int i = 1; i < count; ++i) {
        int16_t kx = vx[i], ky = vy[i];
        int j = i - 1;
        while (j >= 0 && vx[j] > kx) { vx[j + 1] = vx[j]; --j; }
        vx[j + 1] = kx;
        j = i - 1;
        while (j >= 0 && vy[j] > ky) { vy[j + 1] = vy[j]; --j; }
        vy[j + 1] = ky;
    }

    d_mv_smooth[idx] = {vx[count / 2], vy[count / 2], 1, 0};
}

// =========================================================================
// Engine 0 - Pass 1: Motion-Compensated 3D Collaborative Hard-Thresholding
// =========================================================================
__global__ void k_bm3d_stage1_hard(
    const float* const* __restrict__ d_frames,
    const float* __restrict__ d_luma_all,
    const GpuMotionVec* __restrict__ d_mv_smooth,
    const float* __restrict__ d_gain_corr,
    GpuKernelConfig cfg,
    int channel,
    int step_override,
    float* __restrict__ d_numerator,
    float* __restrict__ d_denominator
) {
    __shared__ float s_group[9][8][8];
    __shared__ float s_temp[8][8];
    __shared__ float s_warp_a[2];
    __shared__ float s_warp_b[2];
    __shared__ float s_sigma;
    __shared__ int s_total_non_zero;
    __shared__ GpuMotionVec s_mv;

    const int tx = threadIdx.x; // 0..7
    const int ty = threadIdx.y; // 0..7
    const int tid = ty * 8 + tx;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;

    const int bx = min(static_cast<int>(blockIdx.x) * step_override, max(0, cfg.width - 8));
    const int by = min(static_cast<int>(blockIdx.y) * step_override, max(0, cfg.height - 8));

    const int px = min(bx + tx, cfg.width - 1);
    const int py = min(by + ty, cfg.height - 1);

    const float* ref_frame = d_frames[cfg.ref_index];
    const float* ref_luma = d_luma_all + static_cast<size_t>(cfg.ref_index) * static_cast<size_t>(cfg.width * cfg.height);

    float ref_val = ref_frame[py * cfg.stride_floats + px * cfg.channels + channel];
    float luma_val = ref_luma[py * cfg.width + px];
    s_group[0][ty][tx] = ref_val;

    // Warp-uniform reduction of 64-pixel patch mean and luma mean for rock-steady local sigma
    float sum_v = ref_val;
    float sum_l = luma_val;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_v += __shfl_down_sync(0xffffffff, sum_v, offset);
        sum_l += __shfl_down_sync(0xffffffff, sum_l, offset);
    }
    if (lane_id == 0) {
        s_warp_a[warp_id] = sum_v;
        s_warp_b[warp_id] = sum_l;
    }
    __syncthreads();

    if (tid == 0) {
        float mean_v = (s_warp_a[0] + s_warp_a[1]) * (1.0f / 64.0f);
        float mean_l = (s_warp_b[0] + s_warp_b[1]) * (1.0f / 64.0f);
        s_sigma = eval_channel_sigma(cfg, channel, mean_v, mean_l);
    }
    __syncthreads();

    float max_bscale = max_band_scale_for_channel(cfg, channel);
    float eff_sigma_sq = (s_sigma * max_bscale) * (s_sigma * max_bscale);
    float trust_dist = (4.5f * fmaxf(0.25f, cfg.temporal_weight)) * (eff_sigma_sq + 1e-6f);

    int mv_bx = min(cfg.num_bx - 1, (bx + 4) / 16);
    int mv_by = min(cfg.num_by - 1, (by + 4) / 16);

    int K = 1;
    if (cfg.frame_count > 1 && cfg.temporal_weight > 0.01f) {
        for (int f = 0; f < cfg.frame_count; ++f) {
            if (f == cfg.ref_index) continue;
            if (K >= 9) break;

            if (tid == 0) {
                s_mv = d_mv_smooth[(f * cfg.num_by + mv_by) * cfg.num_bx + mv_bx];
            }
            __syncthreads();

            if (s_mv.trusted) {
                int cx = min(max(0, bx + static_cast<int>(s_mv.dx)), max(0, cfg.width - 8));
                int cy = min(max(0, by + static_cast<int>(s_mv.dy)), max(0, cfg.height - 8));
                int cpx = min(cx + tx, cfg.width - 1);
                int cpy = min(cy + ty, cfg.height - 1);

                const float* cand_frame = d_frames[f];
                float cand_v = cand_frame[cpy * cfg.stride_floats + cpx * cfg.channels + channel] * d_gain_corr[f];
                float diff = ref_val - cand_v;
                float sq = diff * diff;

                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1) {
                    sq += __shfl_down_sync(0xffffffff, sq, offset);
                }
                if (lane_id == 0) s_warp_a[warp_id] = sq;
                __syncthreads();

                float dist = (s_warp_a[0] + s_warp_a[1]) * (1.0f / 64.0f);
                if (dist <= trust_dist) {
                    s_group[K][ty][tx] = cand_v;
                    K++;
                }
                __syncthreads();
            }
        }
    }

    // 2D Orthonormal Spatial DCT on each member k in [0, K-1]
    for (int k = 0; k < K; ++k) {
        float r_sum = 0.0f;
        #pragma unroll
        for (int n = 0; n < 8; ++n) {
            r_sum += c_dct_mat[tx][n] * s_group[k][ty][n];
        }
        s_temp[ty][tx] = r_sum;
        __syncthreads();

        float c_sum = 0.0f;
        #pragma unroll
        for (int m = 0; m < 8; ++m) {
            c_sum += c_dct_mat[ty][m] * s_temp[m][tx];
        }
        s_group[k][ty][tx] = c_sum;
        __syncthreads();
    }

    // 1D Temporal DCT across K members
    float temp_in[9];
    float temp_out[9];
    for (int k = 0; k < K; ++k) temp_in[k] = s_group[k][ty][tx];

    if (K == 1) {
        temp_out[0] = temp_in[0];
    } else {
        float inv_k = 1.0f / static_cast<float>(K);
        float alpha0 = sqrtf(inv_k);
        float alpha = sqrtf(2.0f * inv_k);
        constexpr float pi = 3.141592653589793f;
        for (int k = 0; k < K; ++k) {
            float factor = (k == 0) ? alpha0 : alpha;
            float sum = 0.0f;
            for (int n = 0; n < K; ++n) {
                sum += temp_in[n] * cosf((pi * inv_k) * (static_cast<float>(n) + 0.5f) * static_cast<float>(k));
            }
            temp_out[k] = factor * sum;
        }
    }

    int band = freq_band_for_uv(tx, ty);
    float band_scale = eval_band_scale(cfg, channel, band);
    float spatial_thresh = cfg.hard_thresh_lambda * s_sigma * band_scale;
    float temporal_thresh = spatial_thresh * fmaxf(0.5f, cfg.temporal_weight);

    int thread_nz = 0;
    for (int k = 0; k < K; ++k) {
        if (k == 0 && tx == 0 && ty == 0) {
            thread_nz++;
            continue;
        }
        float thresh = (k == 0) ? spatial_thresh : temporal_thresh;
        if (fabsf(temp_out[k]) < thresh) {
            temp_out[k] = 0.0f;
        } else {
            thread_nz++;
        }
    }

    int nz = thread_nz;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        nz += __shfl_down_sync(0xffffffff, nz, offset);
    }
    if (lane_id == 0) s_warp_a[warp_id] = static_cast<float>(nz);
    __syncthreads();
    if (tid == 0) {
        s_total_non_zero = max(1, static_cast<int>(s_warp_a[0] + s_warp_a[1]));
    }
    __syncthreads();

    // Inverse 1D Temporal DCT for reference member n = 0
    float inv_temp_0 = 0.0f;
    if (K == 1) {
        inv_temp_0 = temp_out[0];
    } else {
        float inv_k = 1.0f / static_cast<float>(K);
        float alpha0 = sqrtf(inv_k);
        float alpha = sqrtf(2.0f * inv_k);
        constexpr float pi = 3.141592653589793f;
        inv_temp_0 = alpha0 * temp_out[0];
        for (int k = 1; k < K; ++k) {
            inv_temp_0 += alpha * temp_out[k] * cosf((pi * inv_k) * 0.5f * static_cast<float>(k));
        }
    }
    s_group[0][ty][tx] = inv_temp_0;
    __syncthreads();

    // Inverse 2D Spatial IDCT
    float icol_sum = 0.0f;
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        icol_sum += c_dct_mat[k][ty] * s_group[0][k][tx];
    }
    s_temp[ty][tx] = icol_sum;
    __syncthreads();

    float irow_sum = 0.0f;
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        irow_sum += c_dct_mat[k][tx] * s_temp[ty][k];
    }
    irow_sum = fmaxf(0.0f, irow_sum);

    float feather = c_feather_1d[ty] * c_feather_1d[tx];
    float weight = feather / static_cast<float>(s_total_non_zero);
    int out_x = bx + tx;
    int out_y = by + ty;
    if (out_x < cfg.width && out_y < cfg.height) {
        int idx = out_y * cfg.width + out_x;
        atomicAdd(&d_numerator[idx], irow_sum * weight);
        atomicAdd(&d_denominator[idx], weight);
    }
}

__global__ void k_normalize_pilot(
    const float* const* __restrict__ d_frames,
    int ref_index,
    int width,
    int height,
    int stride_floats,
    int channels,
    const float* __restrict__ d_num_r, const float* __restrict__ d_den_r,
    const float* __restrict__ d_num_g, const float* __restrict__ d_den_g,
    const float* __restrict__ d_num_b, const float* __restrict__ d_den_b,
    float* __restrict__ d_pilot_r,
    float* __restrict__ d_pilot_g,
    float* __restrict__ d_pilot_b
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int flat = y * width + x;
    int pidx = y * stride_floats + x * channels;
    const float* ref = d_frames[ref_index];

    d_pilot_r[flat] = (d_den_r[flat] > 1e-7f) ? (d_num_r[flat] / d_den_r[flat]) : ref[pidx + 0];
    d_pilot_g[flat] = (d_den_g[flat] > 1e-7f) ? (d_num_g[flat] / d_den_g[flat]) : ref[pidx + 1];
    d_pilot_b[flat] = (d_den_b[flat] > 1e-7f) ? (d_num_b[flat] / d_den_b[flat]) : ref[pidx + 2];
}

// =========================================================================
// Engine 0 - Pass 2: Spatiotemporal Non-Local Collaborative 3D Wiener Filter
// Groups temporal motion-aligned patches AND clean 2D spatial non-local patches
// using 3-channel denoised pilot distances so R/G/B share identical patch groups
// (zero checkerboard phase mismatch) and coarse 8x8 DC grain blotches are wiped out!
// =========================================================================
struct PatchCoord {
    int16_t f;
    int16_t x;
    int16_t y;
    int16_t is_temporal;
};

__global__ void k_bm3d_stage2_wiener(
    const float* const* __restrict__ d_frames,
    const float* __restrict__ d_pilot_r,
    const float* __restrict__ d_pilot_g,
    const float* __restrict__ d_pilot_b,
    const GpuMotionVec* __restrict__ d_mv_smooth,
    const float* __restrict__ d_gain_corr,
    GpuKernelConfig cfg,
    float* __restrict__ d_num_r, float* __restrict__ d_den_r,
    float* __restrict__ d_num_g, float* __restrict__ d_den_g,
    float* __restrict__ d_num_b, float* __restrict__ d_den_b
) {
    __shared__ float s_noisy[kMaxGroupSize][8][8];
    __shared__ float s_pilot[kMaxGroupSize][8][8];
    __shared__ float s_temp[8][8];
    __shared__ float s_warp_a[2];
    __shared__ float s_sigma[3];
    __shared__ float s_wiener_w_sum;
    __shared__ PatchCoord s_coords[kMaxGroupSize];
    __shared__ GpuMotionVec s_mv;

    const int tx = threadIdx.x; // 0..7
    const int ty = threadIdx.y; // 0..7
    const int tid = ty * 8 + tx;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;

    const int bx = min(static_cast<int>(blockIdx.x) * cfg.step_size, max(0, cfg.width - 8));
    const int by = min(static_cast<int>(blockIdx.y) * cfg.step_size, max(0, cfg.height - 8));

    const int px = min(bx + tx, cfg.width - 1);
    const int py = min(by + ty, cfg.height - 1);
    const int ref_flat = py * cfg.width + px;

    float pr0 = d_pilot_r[ref_flat];
    float pg0 = d_pilot_g[ref_flat];
    float pb0 = d_pilot_b[ref_flat];

    // Compute clean 64-pixel pilot patch means for R, G, B
    float mr = pr0, mg = pg0, mb = pb0;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        mr += __shfl_down_sync(0xffffffff, mr, offset);
        mg += __shfl_down_sync(0xffffffff, mg, offset);
        mb += __shfl_down_sync(0xffffffff, mb, offset);
    }
    if (lane_id == 0) s_warp_a[warp_id] = mr;
    __syncthreads();
    float mean_r = (s_warp_a[0] + s_warp_a[1]) * (1.0f / 64.0f);
    __syncthreads();

    if (lane_id == 0) s_warp_a[warp_id] = mg;
    __syncthreads();
    float mean_g = (s_warp_a[0] + s_warp_a[1]) * (1.0f / 64.0f);
    __syncthreads();

    if (lane_id == 0) s_warp_a[warp_id] = mb;
    __syncthreads();
    float mean_b = (s_warp_a[0] + s_warp_a[1]) * (1.0f / 64.0f);
    __syncthreads();

    if (tid == 0) {
        float mean_luma = 0.2126f * mean_r + 0.7152f * mean_g + 0.0722f * mean_b;
        s_sigma[0] = eval_channel_sigma(cfg, 0, mean_r, mean_luma);
        s_sigma[1] = eval_channel_sigma(cfg, 1, mean_g, mean_luma);
        s_sigma[2] = eval_channel_sigma(cfg, 2, mean_b, mean_luma);
        s_coords[0] = {static_cast<int16_t>(cfg.ref_index), static_cast<int16_t>(bx), static_cast<int16_t>(by), 0};
    }
    __syncthreads();

    float avg_sig_sq = (s_sigma[0] * s_sigma[0] + s_sigma[1] * s_sigma[1] + s_sigma[2] * s_sigma[2]) * (1.0f / 3.0f);
    float max_b0 = fmaxf(max_band_scale_for_channel(cfg, 0), fmaxf(max_band_scale_for_channel(cfg, 1), max_band_scale_for_channel(cfg, 2)));
    float eff_var = avg_sig_sq * max_b0 * max_b0 + 1e-7f;

    int K = 1;
    int mv_bx = min(cfg.num_bx - 1, (bx + 4) / 16);
    int mv_by = min(cfg.num_by - 1, (by + 4) / 16);

    // 1. Gather temporal motion-compensated patches verified against pilot + raw consistency
    if (cfg.frame_count > 1 && cfg.temporal_weight > 0.01f) {
        const float* ref_raw = d_frames[cfg.ref_index];
        int rp = py * cfg.stride_floats + px * cfg.channels;
        float rr = ref_raw[rp + 0], rg = ref_raw[rp + 1], rb = ref_raw[rp + 2];
        float temp_trust = 4.0f * fmaxf(0.25f, cfg.temporal_weight) * eff_var;

        for (int f = 0; f < cfg.frame_count; ++f) {
            if (f == cfg.ref_index || K >= kMaxGroupSize) continue;

            if (tid == 0) {
                s_mv = d_mv_smooth[(f * cfg.num_by + mv_by) * cfg.num_bx + mv_bx];
            }
            __syncthreads();

            if (s_mv.trusted) {
                int cx = min(max(0, bx + static_cast<int>(s_mv.dx)), max(0, cfg.width - 8));
                int cy = min(max(0, by + static_cast<int>(s_mv.dy)), max(0, cfg.height - 8));
                int cpx = min(cx + tx, cfg.width - 1);
                int cpy = min(cy + ty, cfg.height - 1);

                const float* cand_raw = d_frames[f];
                float gcorr = d_gain_corr[f];
                int cp = cpy * cfg.stride_floats + cpx * cfg.channels;
                float dr = rr - cand_raw[cp + 0] * gcorr;
                float dg = rg - cand_raw[cp + 1] * gcorr;
                float db = rb - cand_raw[cp + 2] * gcorr;
                float sq = (dr * dr + dg * dg + db * db) * (1.0f / 3.0f);

                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1) {
                    sq += __shfl_down_sync(0xffffffff, sq, offset);
                }
                if (lane_id == 0) s_warp_a[warp_id] = sq;
                __syncthreads();

                float dist = (s_warp_a[0] + s_warp_a[1]) * (1.0f / 64.0f);
                if (dist <= temp_trust) {
                    if (tid == 0) {
                        s_coords[K] = {static_cast<int16_t>(f), static_cast<int16_t>(cx), static_cast<int16_t>(cy), 1};
                    }
                    K++;
                }
                __syncthreads();
            }
        }
    }

    // 2. Gather clean 2D spatial non-local patches in [-10, +10] using denoised 3-channel pilot!
    // Spans 28x28 pixels so correlated 3-8px camera grain blotches decorrelate completely while
    // real structural edges are strictly rejected by the noise-free pilot MSE.
    if (cfg.spatial_weight > 0.01f && K < kMaxGroupSize) {
        const int ox_list[12] = {-5,  5,  0,  0, -5,  5, -5,  5, -10, 10,   0,  0};
        const int oy_list[12] = { 0,  0, -5,  5, -5, -5,  5,  5,   0,  0, -10, 10};
        float spat_trust = 1.6f * eff_var;

        for (int s = 0; s < 12; ++s) {
            if (K >= kMaxGroupSize) break;
            int sx = bx + ox_list[s];
            int sy = by + oy_list[s];
            if (sx < 0 || sx + 8 > cfg.width || sy < 0 || sy + 8 > cfg.height) continue;

            int s_flat = (sy + ty) * cfg.width + (sx + tx);
            float dr = pr0 - d_pilot_r[s_flat];
            float dg = pg0 - d_pilot_g[s_flat];
            float db = pb0 - d_pilot_b[s_flat];
            float sq = (dr * dr + dg * dg + db * db) * (1.0f / 3.0f);

            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                sq += __shfl_down_sync(0xffffffff, sq, offset);
            }
            if (lane_id == 0) s_warp_a[warp_id] = sq;
            __syncthreads();

            float dist_pilot = (s_warp_a[0] + s_warp_a[1]) * (1.0f / 64.0f);
            if (dist_pilot <= spat_trust) {
                if (tid == 0) {
                    s_coords[K] = {static_cast<int16_t>(cfg.ref_index), static_cast<int16_t>(sx), static_cast<int16_t>(sy), 0};
                }
                K++;
            }
            __syncthreads();
        }
    }

    const float* pilots[3] = {d_pilot_r, d_pilot_g, d_pilot_b};
    float* out_nums[3] = {d_num_r, d_num_g, d_num_b};
    float* out_dens[3] = {d_den_r, d_den_g, d_den_b};

    int band = freq_band_for_uv(tx, ty);
    float feather = c_feather_1d[ty] * c_feather_1d[tx];

    // 3. Collaborative 3D Empirical Wiener Shrinkage per channel using unified group s_coords[0..K-1]
    for (int c = 0; c < 3; ++c) {
        const float* cur_pilot = pilots[c];

        for (int k = 0; k < K; ++k) {
            PatchCoord pc = s_coords[k];
            int cpx = min(static_cast<int>(pc.x) + tx, cfg.width - 1);
            int cpy = min(static_cast<int>(pc.y) + ty, cfg.height - 1);
            float gcorr = d_gain_corr[pc.f];

            s_noisy[k][ty][tx] = d_frames[pc.f][cpy * cfg.stride_floats + cpx * cfg.channels + c] * gcorr;
            // For temporal frames, pilot of ref frame at shifted or ref pos represents clean underlying structure:
            // Note: using cur_pilot at (pc.is_temporal ? (bx+tx, by+ty) : (cpx, cpy)) gives exact clean structure!
            int p_flat = pc.is_temporal ? ref_flat : (cpy * cfg.width + cpx);
            s_pilot[k][ty][tx] = cur_pilot[p_flat];
        }
        __syncthreads();

        // 2D Spatial DCT on both s_noisy and s_pilot
        for (int k = 0; k < K; ++k) {
            float rn = 0.0f, rp = 0.0f;
            #pragma unroll
            for (int n = 0; n < 8; ++n) {
                float dval = c_dct_mat[tx][n];
                rn += dval * s_noisy[k][ty][n];
                rp += dval * s_pilot[k][ty][n];
            }
            s_temp[ty][tx] = rn;
            __syncthreads();
            float cn = 0.0f;
            #pragma unroll
            for (int m = 0; m < 8; ++m) cn += c_dct_mat[ty][m] * s_temp[m][tx];
            s_noisy[k][ty][tx] = cn;
            __syncthreads();

            s_temp[ty][tx] = rp;
            __syncthreads();
            float cp = 0.0f;
            #pragma unroll
            for (int m = 0; m < 8; ++m) cp += c_dct_mat[ty][m] * s_temp[m][tx];
            s_pilot[k][ty][tx] = cp;
            __syncthreads();
        }

        // 1D Group DCT + Empirical Wiener Shrinkage
        float y_in[kMaxGroupSize], p_in[kMaxGroupSize], y_out[kMaxGroupSize];
        for (int k = 0; k < K; ++k) {
            y_in[k] = s_noisy[k][ty][tx];
            p_in[k] = s_pilot[k][ty][tx];
        }

        float b_scale = eval_band_scale(cfg, c, band);
        float sig_b = s_sigma[c] * b_scale;
        float noise_var = sig_b * sig_b;
        float w_sq_sum = 0.0f;

        if (K == 1) {
            if (tx == 0 && ty == 0) {
                y_out[0] = y_in[0];
                w_sq_sum = 1.0f;
            } else {
                float p2 = p_in[0] * p_in[0];
                float w = p2 / (p2 + noise_var + 1e-10f);
                y_out[0] = y_in[0] * w;
                w_sq_sum = w * w;
            }
        } else {
            float inv_k = 1.0f / static_cast<float>(K);
            float alpha0 = sqrtf(inv_k);
            float alpha = sqrtf(2.0f * inv_k);
            constexpr float pi = 3.141592653589793f;

            for (int k = 0; k < K; ++k) {
                float factor = (k == 0) ? alpha0 : alpha;
                float sy = 0.0f, sp = 0.0f;
                for (int n = 0; n < K; ++n) {
                    float basis = cosf((pi * inv_k) * (static_cast<float>(n) + 0.5f) * static_cast<float>(k));
                    sy += y_in[n] * basis;
                    sp += p_in[n] * basis;
                }
                sy *= factor;
                sp *= factor;

                if (k == 0 && tx == 0 && ty == 0) {
                    y_out[0] = sy;
                    w_sq_sum += 1.0f;
                } else {
                    float p2 = sp * sp;
                    float eff_nvar = (k == 0) ? noise_var : (noise_var * fmaxf(0.5f, cfg.temporal_weight));
                    float w = p2 / (p2 + eff_nvar + 1e-10f);
                    y_out[k] = sy * w;
                    w_sq_sum += w * w;
                }
            }
        }

        // Reduce sum of squared Wiener weights across 64 threads
        float red_w = w_sq_sum;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            red_w += __shfl_down_sync(0xffffffff, red_w, offset);
        }
        if (lane_id == 0) s_warp_a[warp_id] = red_w;
        __syncthreads();
        if (tid == 0) {
            s_wiener_w_sum = fmaxf(1e-2f, s_warp_a[0] + s_warp_a[1]);
        }
        __syncthreads();

        // Inverse 1D Group DCT for reference member n = 0
        float inv_0 = 0.0f;
        if (K == 1) {
            inv_0 = y_out[0];
        } else {
            float inv_k = 1.0f / static_cast<float>(K);
            float alpha0 = sqrtf(inv_k);
            float alpha = sqrtf(2.0f * inv_k);
            constexpr float pi = 3.141592653589793f;
            inv_0 = alpha0 * y_out[0];
            for (int k = 1; k < K; ++k) {
                inv_0 += alpha * y_out[k] * cosf((pi * inv_k) * 0.5f * static_cast<float>(k));
            }
        }
        s_noisy[0][ty][tx] = inv_0;
        __syncthreads();

        // Inverse 2D Spatial IDCT
        float icol = 0.0f;
        #pragma unroll
        for (int k = 0; k < 8; ++k) icol += c_dct_mat[k][ty] * s_noisy[0][k][tx];
        s_temp[ty][tx] = icol;
        __syncthreads();

        float irow = 0.0f;
        #pragma unroll
        for (int k = 0; k < 8; ++k) irow += c_dct_mat[k][tx] * s_temp[ty][k];
        irow = fmaxf(0.0f, irow);

        float agg_w = feather / s_wiener_w_sum;
        int out_x = bx + tx;
        int out_y = by + ty;
        if (out_x < cfg.width && out_y < cfg.height) {
            int out_idx = out_y * cfg.width + out_x;
            atomicAdd(&out_nums[c][out_idx], irow * agg_w);
            atomicAdd(&out_dens[c][out_idx], agg_w);
        }
        __syncthreads();
    }
}

// =========================================================================
// Shared Baseline for Engines 1, 2, 3: Motion-Compensated Temporal Average
// =========================================================================
__global__ void k_temporal_motion_average(
    const float* const* __restrict__ d_frames,
    const float* __restrict__ d_luma_all,
    const GpuMotionVec* __restrict__ d_mv_smooth,
    const float* __restrict__ d_gain_corr,
    GpuKernelConfig cfg,
    float* __restrict__ d_pilot_r,
    float* __restrict__ d_pilot_g,
    float* __restrict__ d_pilot_b
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= cfg.width || y >= cfg.height) return;

    int flat = y * cfg.width + x;
    int pidx = y * cfg.stride_floats + x * cfg.channels;
    const float* ref = d_frames[cfg.ref_index];
    float luma = d_luma_all[static_cast<size_t>(cfg.ref_index) * static_cast<size_t>(cfg.width * cfg.height) + flat];

    float rr = ref[pidx + 0], rg = ref[pidx + 1], rb = ref[pidx + 2];
    float sr = rr, sg = rg, sb = rb;
    float wr = 1.0f, wg = 1.0f, wb = 1.0f;

    if (cfg.frame_count > 1 && cfg.temporal_weight > 0.01f) {
        float sig_r = eval_channel_sigma(cfg, 0, rr, luma) * max_band_scale_for_channel(cfg, 0);
        float sig_g = eval_channel_sigma(cfg, 1, rg, luma) * max_band_scale_for_channel(cfg, 1);
        float sig_b = eval_channel_sigma(cfg, 2, rb, luma) * max_band_scale_for_channel(cfg, 2);
        float lim_r = 3.5f * sig_r * cfg.temporal_weight;
        float lim_g = 3.5f * sig_g * cfg.temporal_weight;
        float lim_b = 3.5f * sig_b * cfg.temporal_weight;

        int mv_bx = min(cfg.num_bx - 1, x / 16);
        int mv_by = min(cfg.num_by - 1, y / 16);

        for (int f = 0; f < cfg.frame_count; ++f) {
            if (f == cfg.ref_index) continue;
            GpuMotionVec mv = d_mv_smooth[(f * cfg.num_by + mv_by) * cfg.num_bx + mv_bx];
            if (!mv.trusted) continue;

            int cx = min(cfg.width - 1, max(0, x + static_cast<int>(mv.dx)));
            int cy = min(cfg.height - 1, max(0, y + static_cast<int>(mv.dy)));
            int cidx = cy * cfg.stride_floats + cx * cfg.channels;
            float gcorr = d_gain_corr[f];

            float cr = d_frames[f][cidx + 0] * gcorr;
            float cg = d_frames[f][cidx + 1] * gcorr;
            float cb = d_frames[f][cidx + 2] * gcorr;

            if (fabsf(rr - cr) <= lim_r) { sr += cr; wr += 1.0f; }
            if (fabsf(rg - cg) <= lim_g) { sg += cg; wg += 1.0f; }
            if (fabsf(rb - cb) <= lim_b) { sb += cb; wb += 1.0f; }
        }
    }

    d_pilot_r[flat] = sr / wr;
    d_pilot_g[flat] = sg / wg;
    d_pilot_b[flat] = sb / wb;
}

// =========================================================================
// Engine 1: GPU 5-Band Cascaded Edge-Preserving À-Trous B3-Spline Wavelet
// =========================================================================
__global__ void k_wavelet_atrous_level(
    const float* __restrict__ d_cur_r,
    const float* __restrict__ d_cur_g,
    const float* __restrict__ d_cur_b,
    GpuKernelConfig cfg,
    int level,
    float* __restrict__ d_next_r,
    float* __restrict__ d_next_g,
    float* __restrict__ d_next_b,
    float* __restrict__ d_accum_r,
    float* __restrict__ d_accum_g,
    float* __restrict__ d_accum_b
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= cfg.width || y >= cfg.height) return;

    int flat = y * cfg.width + x;
    const float* curs[3] = {d_cur_r, d_cur_g, d_cur_b};
    float* nexts[3] = {d_next_r, d_next_g, d_next_b};
    float* accums[3] = {d_accum_r, d_accum_g, d_accum_b};

    float luma = 0.2126f * d_cur_r[flat] + 0.7152f * d_cur_g[flat] + 0.0722f * d_cur_b[flat];
    const float k1d[5] = {0.0625f, 0.25f, 0.375f, 0.25f, 0.0625f};
    const float level_norms[5] = {0.89f, 0.42f, 0.21f, 0.105f, 0.052f};

    int step = 1 << level;
    int band = 4 - level;
    float lnorm = level_norms[level];

    for (int c = 0; c < 3; ++c) {
        const float* plane = curs[c];
        float center = plane[flat];
        float base_sig = eval_channel_sigma(cfg, c, center, luma);
        float bscale = eval_band_scale(cfg, c, band);
        float thresh = 1.6f * base_sig * bscale * lnorm;
        float edge_h2 = 16.0f * (base_sig * bscale) * (base_sig * bscale) + 1e-6f;

        float smooth = 0.0f;
        float wsum = 0.0f;
        #pragma unroll
        for (int ky = -2; ky <= 2; ++ky) {
            int py = min(cfg.height - 1, max(0, y + ky * step));
            float wy = k1d[ky + 2];
            #pragma unroll
            for (int kx = -2; kx <= 2; ++kx) {
                int px = min(cfg.width - 1, max(0, x + kx * step));
                float pv = plane[py * cfg.width + px];
                float diff = pv - center;
                // Edge-stopping weight prevents large coarse dilations (step=8,16) from haloing across HDR edges
                float we = (level >= 1) ? expf(-(diff * diff) / edge_h2) : 1.0f;
                float w = wy * k1d[kx + 2] * we;
                smooth += w * pv;
                wsum += w;
            }
        }
        smooth = (wsum > 1e-7f) ? (smooth / wsum) : center;

        float detail = center - smooth;
        float pwr = detail * detail;
        float t2 = thresh * thresh;
        float gain = fmaxf(0.0f, (pwr - t2) / (pwr + 1e-9f));

        nexts[c][flat] = smooth;
        accums[c][flat] += detail * gain;
    }
}

__global__ void k_wavelet_atrous_reconstruct(
    const float* __restrict__ d_coarse_r,
    const float* __restrict__ d_coarse_g,
    const float* __restrict__ d_coarse_b,
    int width,
    int height,
    float* __restrict__ d_num_r, float* __restrict__ d_den_r,
    float* __restrict__ d_num_g, float* __restrict__ d_den_g,
    float* __restrict__ d_num_b, float* __restrict__ d_den_b
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int flat = y * width + x;
    d_num_r[flat] = fmaxf(0.0f, d_coarse_r[flat] + d_num_r[flat]);
    d_den_r[flat] = 1.0f;
    d_num_g[flat] = fmaxf(0.0f, d_coarse_g[flat] + d_num_g[flat]);
    d_den_g[flat] = 1.0f;
    d_num_b[flat] = fmaxf(0.0f, d_coarse_b[flat] + d_num_b[flat]);
    d_den_b[flat] = 1.0f;
}

// =========================================================================
// Engine 3: GPU Overlapping 8x8 Frequency-Domain Spectral Wiener Filter
// =========================================================================
__global__ void k_freq3d_spectral_wiener(
    const float* __restrict__ d_in_r,
    const float* __restrict__ d_in_g,
    const float* __restrict__ d_in_b,
    GpuKernelConfig cfg,
    float* __restrict__ d_num_r, float* __restrict__ d_den_r,
    float* __restrict__ d_num_g, float* __restrict__ d_den_g,
    float* __restrict__ d_num_b, float* __restrict__ d_den_b
) {
    __shared__ float s_block[8][8];
    __shared__ float s_temp[8][8];
    __shared__ float s_warp_a[2];
    __shared__ float s_sigma[3];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * 8 + tx;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;

    const int bx = min(static_cast<int>(blockIdx.x) * cfg.step_size, max(0, cfg.width - 8));
    const int by = min(static_cast<int>(blockIdx.y) * cfg.step_size, max(0, cfg.height - 8));
    const int px = min(bx + tx, cfg.width - 1);
    const int py = min(by + ty, cfg.height - 1);
    const int flat = py * cfg.width + px;

    float vr = d_in_r[flat], vg = d_in_g[flat], vb = d_in_b[flat];
    float mr = vr, mg = vg, mb = vb;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        mr += __shfl_down_sync(0xffffffff, mr, offset);
        mg += __shfl_down_sync(0xffffffff, mg, offset);
        mb += __shfl_down_sync(0xffffffff, mb, offset);
    }
    if (lane_id == 0) s_warp_a[warp_id] = mr;
    __syncthreads();
    float mean_r = (s_warp_a[0] + s_warp_a[1]) * (1.0f / 64.0f);
    __syncthreads();
    if (lane_id == 0) s_warp_a[warp_id] = mg;
    __syncthreads();
    float mean_g = (s_warp_a[0] + s_warp_a[1]) * (1.0f / 64.0f);
    __syncthreads();
    if (lane_id == 0) s_warp_a[warp_id] = mb;
    __syncthreads();
    float mean_b = (s_warp_a[0] + s_warp_a[1]) * (1.0f / 64.0f);
    __syncthreads();

    if (tid == 0) {
        float mean_luma = 0.2126f * mean_r + 0.7152f * mean_g + 0.0722f * mean_b;
        s_sigma[0] = eval_channel_sigma(cfg, 0, mean_r, mean_luma);
        s_sigma[1] = eval_channel_sigma(cfg, 1, mean_g, mean_luma);
        s_sigma[2] = eval_channel_sigma(cfg, 2, mean_b, mean_luma);
    }
    __syncthreads();

    const float vals[3] = {vr, vg, vb};
    float* out_nums[3] = {d_num_r, d_num_g, d_num_b};
    float* out_dens[3] = {d_den_r, d_den_g, d_den_b};

    int band = freq_band_for_uv(tx, ty);
    float feather = c_feather_1d[ty] * c_feather_1d[tx];

    for (int c = 0; c < 3; ++c) {
        s_block[ty][tx] = vals[c];
        __syncthreads();

        float r_sum = 0.0f;
        #pragma unroll
        for (int n = 0; n < 8; ++n) r_sum += c_dct_mat[tx][n] * s_block[ty][n];
        s_temp[ty][tx] = r_sum;
        __syncthreads();

        float coeff = 0.0f;
        #pragma unroll
        for (int m = 0; m < 8; ++m) coeff += c_dct_mat[ty][m] * s_temp[m][tx];
        __syncthreads();

        if (!(tx == 0 && ty == 0)) {
            float bscale = eval_band_scale(cfg, c, band);
            float noise_pwr = (s_sigma[c] * bscale) * (s_sigma[c] * bscale);
            float pwr = coeff * coeff;
            float gain = fmaxf(0.0f, (pwr - noise_pwr) / (pwr + 1e-8f));
            coeff *= gain;
        }
        s_block[ty][tx] = coeff;
        __syncthreads();

        float icol = 0.0f;
        #pragma unroll
        for (int k = 0; k < 8; ++k) icol += c_dct_mat[k][ty] * s_block[k][tx];
        s_temp[ty][tx] = icol;
        __syncthreads();

        float irow = 0.0f;
        #pragma unroll
        for (int k = 0; k < 8; ++k) irow += c_dct_mat[k][tx] * s_temp[ty][k];
        irow = fmaxf(0.0f, irow);

        int out_x = bx + tx;
        int out_y = by + ty;
        if (out_x < cfg.width && out_y < cfg.height) {
            int out_idx = out_y * cfg.width + out_x;
            atomicAdd(&out_nums[c][out_idx], irow * feather);
            atomicAdd(&out_dens[c][out_idx], feather);
        }
        __syncthreads();
    }
}

// =========================================================================
// Engine 2: GPU Multi-Channel Accelerated Non-Local Means (NLM)
// =========================================================================
__global__ void k_nlm_spatial(
    const float* __restrict__ d_in_r,
    const float* __restrict__ d_in_g,
    const float* __restrict__ d_in_b,
    GpuKernelConfig cfg,
    float* __restrict__ d_num_r, float* __restrict__ d_den_r,
    float* __restrict__ d_num_g, float* __restrict__ d_den_g,
    float* __restrict__ d_num_b, float* __restrict__ d_den_b
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= cfg.width || y >= cfg.height) return;

    int flat = y * cfg.width + x;
    float cr = d_in_r[flat], cg = d_in_g[flat], cb = d_in_b[flat];
    float luma = 0.2126f * cr + 0.7152f * cg + 0.0722f * cb;

    float sr = eval_channel_sigma(cfg, 0, cr, luma) * max_band_scale_for_channel(cfg, 0);
    float sg = eval_channel_sigma(cfg, 1, cg, luma) * max_band_scale_for_channel(cfg, 1);
    float sb = eval_channel_sigma(cfg, 2, cb, luma) * max_band_scale_for_channel(cfg, 2);
    float sig_sq = (sr * sr + sg * sg + sb * sb) * (1.0f / 3.0f);
    float h_sq = 1.5f * sig_sq + 1e-7f;

    const int r_search = (cfg.step_size >= 6) ? 4 : 6;
    const int r_patch = 2; // 5x5 patch

    float sum_r = 0.0f, sum_g = 0.0f, sum_b = 0.0f, sum_w = 0.0f;

    for (int sy = max(0, y - r_search); sy <= min(cfg.height - 1, y + r_search); ++sy) {
        for (int sx = max(0, x - r_search); sx <= min(cfg.width - 1, x + r_search); ++sx) {
            float dist = 0.0f;
            #pragma unroll
            for (int py = -r_patch; py <= r_patch; ++py) {
                int y1 = min(cfg.height - 1, max(0, y + py));
                int y2 = min(cfg.height - 1, max(0, sy + py));
                #pragma unroll
                for (int px = -r_patch; px <= r_patch; ++px) {
                    int x1 = min(cfg.width - 1, max(0, x + px));
                    int x2 = min(cfg.width - 1, max(0, sx + px));
                    int i1 = y1 * cfg.width + x1;
                    int i2 = y2 * cfg.width + x2;
                    float dr = d_in_r[i1] - d_in_r[i2];
                    float dg = d_in_g[i1] - d_in_g[i2];
                    float db = d_in_b[i1] - d_in_b[i2];
                    dist += (dr * dr + dg * dg + db * db);
                }
            }
            dist *= (1.0f / 75.0f); // 25 pixels * 3 channels
            float w = expf(-fmaxf(0.0f, dist - 1.5f * sig_sq) / h_sq);
            int s_flat = sy * cfg.width + sx;
            sum_r += d_in_r[s_flat] * w;
            sum_g += d_in_g[s_flat] * w;
            sum_b += d_in_b[s_flat] * w;
            sum_w += w;
        }
    }

    d_num_r[flat] = (sum_w > 1e-7f) ? (sum_r / sum_w) : cr;
    d_den_r[flat] = 1.0f;
    d_num_g[flat] = (sum_w > 1e-7f) ? (sum_g / sum_w) : cg;
    d_den_g[flat] = 1.0f;
    d_num_b[flat] = (sum_w > 1e-7f) ? (sum_b / sum_w) : cb;
    d_den_b[flat] = 1.0f;
}

// =========================================================================
// Kernel Final: Post-processing Finalize (Edge Detail, Grain, Mix, HDR-safe)
// =========================================================================
__global__ void k_postprocess_finalize(
    const float* const* __restrict__ d_frames,
    const float* __restrict__ d_num_r, const float* __restrict__ d_den_r,
    const float* __restrict__ d_num_g, const float* __restrict__ d_den_g,
    const float* __restrict__ d_num_b, const float* __restrict__ d_den_b,
    GpuKernelConfig cfg,
    float* __restrict__ d_output
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= cfg.width || y >= cfg.height) return;

    int pixel_idx = y * cfg.stride_floats + x * cfg.channels;
    int flat_idx = y * cfg.width + x;

    const float* d_ref_frame = d_frames[cfg.ref_index];
    const float* nums[3] = {d_num_r, d_num_g, d_num_b};
    const float* dens[3] = {d_den_r, d_den_g, d_den_b};

    int left_x = max(0, x - 1);
    int right_x = min(cfg.width - 1, x + 1);
    int up_y = max(0, y - 1);
    int down_y = min(cfg.height - 1, y + 1);

    for (int c = 0; c < 3; ++c) {
        float ref_val = d_ref_frame[pixel_idx + c];
        float d = dens[c][flat_idx];
        float filtered = (d > 1e-6f) ? (nums[c][flat_idx] / d) : ref_val;

        float var = fmaxf(1e-8f, cfg.nlf_a[c] * fmaxf(0.0f, filtered) + cfg.nlf_b[c]);
        float sigma = sqrtf(var);
        float max_sig = sigma * max_band_scale_for_channel(cfg, c);

        // Edge-protected micro-contrast recovery (soft-clamped so grain spikes aren't re-injected)
        if (cfg.detail_recovery > 0.001f) {
            float d_l = dens[c][y * cfg.width + left_x];
            float d_r = dens[c][y * cfg.width + right_x];
            float d_u = dens[c][up_y * cfg.width + x];
            float d_d = dens[c][down_y * cfg.width + x];

            float fl = (d_l > 1e-6f) ? (nums[c][y * cfg.width + left_x] / d_l) : d_ref_frame[y * cfg.stride_floats + left_x * cfg.channels + c];
            float fr = (d_r > 1e-6f) ? (nums[c][y * cfg.width + right_x] / d_r) : d_ref_frame[y * cfg.stride_floats + right_x * cfg.channels + c];
            float fu = (d_u > 1e-6f) ? (nums[c][up_y * cfg.width + x] / d_u) : d_ref_frame[up_y * cfg.stride_floats + x * cfg.channels + c];
            float fd = (d_d > 1e-6f) ? (nums[c][down_y * cfg.width + x] / d_d) : d_ref_frame[down_y * cfg.stride_floats + x * cfg.channels + c];

            float gx = (fr - fl) * 0.5f;
            float gy = (fd - fu) * 0.5f;
            float grad_mag = sqrtf(gx * gx + gy * gy);

            if (grad_mag > 1.8f * max_sig) {
                float residual = fminf(1.5f * sigma, fmaxf(-1.5f * sigma, ref_val - filtered));
                float boost = fminf(0.5f, fmaxf(0.0f, cfg.detail_recovery * (grad_mag / (grad_mag + 2.0f * max_sig))));
                filtered = fmaxf(0.0f, filtered + boost * residual);
            }
        }

        // Calibrated synthetic film grain
        if (cfg.grain_amount > 0.001f) {
            float grain = hash_noise(x, y, static_cast<uint32_t>(c * 7919)) * sigma * cfg.grain_amount;
            filtered = fmaxf(0.0f, filtered + grain);
        }

        // Preview Mode: 1 = Residual visual check
        if (cfg.preview_mode == 1) {
            filtered = (ref_val - filtered) * 2.0f + 0.5f;
        } else if (cfg.mix_original > 0.0f) {
            filtered = filtered * (1.0f - cfg.mix_original) + ref_val * cfg.mix_original;
        }

        // Preserve linear-light HDR values > 1.0f (no upper clamp!)
        d_output[pixel_idx + c] = fmaxf(0.0f, filtered);
    }

    if (cfg.channels >= 4) {
        float alpha = (cfg.force_opaque_alpha != 0) ? 1.0f : d_ref_frame[pixel_idx + 3];
        d_output[pixel_idx + 3] = alpha;
    }
}

} // anonymous namespace

bool is_cuda_available() noexcept {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    return (err == cudaSuccess && device_count > 0);
}

void release_cuda_resources() noexcept {
    GpuScopedLock lock(g_cuda_lock);
    g_arena.release();
}

bool cuda_denoise_device(
    const float* const* d_input_frames,
    int32_t frame_count,
    int32_t ref_index,
    float* d_output,
    const CudaDenoiseParams& params,
    void* cuda_stream
) {
    if (!d_input_frames || !d_output || frame_count <= 0 || ref_index < 0 || ref_index >= frame_count) {
        return false;
    }
    if (params.width <= 0 || params.height <= 0 || params.channels < 3) {
        return false;
    }

    GpuScopedLock lock(g_cuda_lock);
    cudaStream_t stream = cuda_stream ? reinterpret_cast<cudaStream_t>(cuda_stream) : nullptr;

    init_dct_constants();

    const int w = params.width;
    const int h = params.height;
    const int step = std::max(1, params.step_size);
    const int num_bx = (w + 15) / 16;
    const int num_by = (h + 15) / 16;
    const size_t total_pixels = static_cast<size_t>(w) * static_cast<size_t>(h);

    if (!g_arena.ensure_capacity(total_pixels, frame_count, num_bx * num_by)) {
        return false;
    }

    GpuKernelConfig cfg{};
    cfg.width = w;
    cfg.height = h;
    cfg.stride_floats = params.stride_floats;
    cfg.channels = params.channels;
    cfg.step_size = step;
    cfg.frame_count = frame_count;
    cfg.ref_index = ref_index;
    cfg.num_bx = num_bx;
    cfg.num_by = num_by;
    cfg.early_exit_motion = params.early_exit_motion;
    cfg.hard_thresh_lambda = params.hard_thresh_lambda;
    cfg.temporal_weight = params.temporal_weight;
    cfg.spatial_weight = params.spatial_weight;
    cfg.detail_recovery = params.detail_recovery;
    cfg.grain_amount = params.grain_amount;
    cfg.mix_original = params.mix_original;
    cfg.chroma_scale_cb = params.chroma_scale_cb;
    cfg.chroma_scale_cr = params.chroma_scale_cr;
    cfg.preview_mode = params.preview_mode;
    cfg.force_opaque_alpha = params.force_opaque_alpha;
    for (int b = 0; b < 5; ++b) {
        cfg.band_weights[b] = params.band_weights[b];
        cfg.tonal_lut[b] = params.tonal_lut[b];
        for (int c = 0; c < 3; ++c) {
            cfg.psd_weights[c][b] = params.psd_weights[c][b];
        }
    }
    for (int c = 0; c < 3; ++c) {
        cfg.nlf_a[c] = params.nlf_a[c];
        cfg.nlf_b[c] = params.nlf_b[c];
    }

    dim3 block_2d(16, 16);
    dim3 grid_pixel((w + block_2d.x - 1) / block_2d.x, (h + block_2d.y - 1) / block_2d.y);

    // Stage 0: Luma extraction, exposure drift / scene-cut detection, and regularized 16x16 motion vectors
    cudaMemsetAsync(g_arena.d_frame_stats, 0, static_cast<size_t>(frame_count) * 4 * sizeof(float), stream);

    dim3 grid_luma(grid_pixel.x, grid_pixel.y, frame_count);
    k_extract_smoothed_luma<<<grid_luma, block_2d, 0, stream>>>(
        d_input_frames, cfg, g_arena.d_luma, g_arena.d_frame_stats
    );

    k_finalize_frame_stats<<<1, 32, 0, stream>>>(
        frame_count, ref_index, g_arena.d_frame_stats, g_arena.d_gain_corr, g_arena.d_scene_cut
    );

    if (frame_count > 1 && params.temporal_weight > 0.01f) {
        dim3 grid_mv(num_bx, num_by, frame_count);
        k_estimate_motion_luma<<<grid_mv, dim3(32, 1), 0, stream>>>(
            g_arena.d_luma, g_arena.d_gain_corr, g_arena.d_scene_cut, cfg, g_arena.d_mv_raw
        );

        dim3 block_mv_s(16, 16);
        dim3 grid_mv_s((num_bx + 15) / 16, (num_by + 15) / 16, frame_count);
        k_smooth_motion_vectors<<<grid_mv_s, block_mv_s, 0, stream>>>(
            g_arena.d_mv_raw, num_bx, num_by, frame_count, ref_index, g_arena.d_mv_smooth
        );
    }

    // Stage 1 & 2: Engine Dispatch
    if (params.engine_choice == 1) {
        // Engine 1: Motion-Compensated 5-Band Cascaded À-Trous Wavelet
        k_temporal_motion_average<<<grid_pixel, block_2d, 0, stream>>>(
            d_input_frames, g_arena.d_luma, g_arena.d_mv_smooth, g_arena.d_gain_corr, cfg,
            g_arena.d_pilot[0], g_arena.d_pilot[1], g_arena.d_pilot[2]
        );
        for (int c = 0; c < 3; ++c) {
            cudaMemsetAsync(g_arena.d_num[c], 0, total_pixels * sizeof(float), stream);
        }
        // Ping-pong 5 dyadic levels between d_pilot and d_den, accumulating Wiener-shrunk details in d_num
        for (int level = 0; level < 5; ++level) {
            float** src_p = (level % 2 == 0) ? g_arena.d_pilot : g_arena.d_den;
            float** dst_p = (level % 2 == 0) ? g_arena.d_den : g_arena.d_pilot;
            k_wavelet_atrous_level<<<grid_pixel, block_2d, 0, stream>>>(
                src_p[0], src_p[1], src_p[2], cfg, level,
                dst_p[0], dst_p[1], dst_p[2],
                g_arena.d_num[0], g_arena.d_num[1], g_arena.d_num[2]
            );
        }
        // After 5 levels (levels 0..4), final coarse approximation resides in d_den
        k_wavelet_atrous_reconstruct<<<grid_pixel, block_2d, 0, stream>>>(
            g_arena.d_den[0], g_arena.d_den[1], g_arena.d_den[2], w, h,
            g_arena.d_num[0], g_arena.d_den[0],
            g_arena.d_num[1], g_arena.d_den[1],
            g_arena.d_num[2], g_arena.d_den[2]
        );
    } else if (params.engine_choice == 2) {
        // Engine 2: Motion-Compensated Multi-Channel Non-Local Means
        k_temporal_motion_average<<<grid_pixel, block_2d, 0, stream>>>(
            d_input_frames, g_arena.d_luma, g_arena.d_mv_smooth, g_arena.d_gain_corr, cfg,
            g_arena.d_pilot[0], g_arena.d_pilot[1], g_arena.d_pilot[2]
        );
        k_nlm_spatial<<<grid_pixel, block_2d, 0, stream>>>(
            g_arena.d_pilot[0], g_arena.d_pilot[1], g_arena.d_pilot[2], cfg,
            g_arena.d_num[0], g_arena.d_den[0],
            g_arena.d_num[1], g_arena.d_den[1],
            g_arena.d_num[2], g_arena.d_den[2]
        );
    } else if (params.engine_choice == 3) {
        // Engine 3: Motion-Compensated 8x8 Frequency-Domain Spectral Wiener
        dim3 block_dct(8, 8);
        k_temporal_motion_average<<<grid_pixel, block_2d, 0, stream>>>(
            d_input_frames, g_arena.d_luma, g_arena.d_mv_smooth, g_arena.d_gain_corr, cfg,
            g_arena.d_pilot[0], g_arena.d_pilot[1], g_arena.d_pilot[2]
        );
        for (int c = 0; c < 3; ++c) {
            cudaMemsetAsync(g_arena.d_num[c], 0, total_pixels * sizeof(float), stream);
            cudaMemsetAsync(g_arena.d_den[c], 0, total_pixels * sizeof(float), stream);
        }
        int f_npx = (std::max(0, w - 8) + step - 1) / step + 1;
        int f_npy = (std::max(0, h - 8) + step - 1) / step + 1;
        k_freq3d_spectral_wiener<<<dim3(f_npx, f_npy), block_dct, 0, stream>>>(
            g_arena.d_pilot[0], g_arena.d_pilot[1], g_arena.d_pilot[2], cfg,
            g_arena.d_num[0], g_arena.d_den[0],
            g_arena.d_num[1], g_arena.d_den[1],
            g_arena.d_num[2], g_arena.d_den[2]
        );
    } else {
        // Engine 0 (Default): 2-Stage Spatiotemporal Collaborative 3D V-BM3D
        dim3 block_dct(8, 8);

        // Pass 1: Spatiotemporal 3D Hard-Thresholding Basic Estimate -> Pilot
        for (int c = 0; c < 3; ++c) {
            cudaMemsetAsync(g_arena.d_num[c], 0, total_pixels * sizeof(float), stream);
            cudaMemsetAsync(g_arena.d_den[c], 0, total_pixels * sizeof(float), stream);

            int c_step = (c > 0 && params.fast_chroma) ? std::min(step + 2, 6) : step;
            int npx = (std::max(0, w - 8) + c_step - 1) / c_step + 1;
            int npy = (std::max(0, h - 8) + c_step - 1) / c_step + 1;

            k_bm3d_stage1_hard<<<dim3(npx, npy), block_dct, 0, stream>>>(
                d_input_frames, g_arena.d_luma, g_arena.d_mv_smooth, g_arena.d_gain_corr,
                cfg, c, c_step, g_arena.d_num[c], g_arena.d_den[c]
            );
        }

        k_normalize_pilot<<<grid_pixel, block_2d, 0, stream>>>(
            d_input_frames, ref_index, w, h, params.stride_floats, params.channels,
            g_arena.d_num[0], g_arena.d_den[0],
            g_arena.d_num[1], g_arena.d_den[1],
            g_arena.d_num[2], g_arena.d_den[2],
            g_arena.d_pilot[0], g_arena.d_pilot[1], g_arena.d_pilot[2]
        );

        // Pass 2: Spatiotemporal Non-Local Collaborative 3D Empirical Wiener Polish
        for (int c = 0; c < 3; ++c) {
            cudaMemsetAsync(g_arena.d_num[c], 0, total_pixels * sizeof(float), stream);
            cudaMemsetAsync(g_arena.d_den[c], 0, total_pixels * sizeof(float), stream);
        }

        int w_npx = (std::max(0, w - 8) + step - 1) / step + 1;
        int w_npy = (std::max(0, h - 8) + step - 1) / step + 1;

        k_bm3d_stage2_wiener<<<dim3(w_npx, w_npy), block_dct, 0, stream>>>(
            d_input_frames,
            g_arena.d_pilot[0], g_arena.d_pilot[1], g_arena.d_pilot[2],
            g_arena.d_mv_smooth, g_arena.d_gain_corr, cfg,
            g_arena.d_num[0], g_arena.d_den[0],
            g_arena.d_num[1], g_arena.d_den[1],
            g_arena.d_num[2], g_arena.d_den[2]
        );
    }

    // Stage 3: Finalize (Detail Recovery, Calibrated Grain, Mix, HDR-safe Output)
    k_postprocess_finalize<<<grid_pixel, block_2d, 0, stream>>>(
        d_input_frames,
        g_arena.d_num[0], g_arena.d_den[0],
        g_arena.d_num[1], g_arena.d_den[1],
        g_arena.d_num[2], g_arena.d_den[2],
        cfg, d_output
    );

    cudaError_t sync_err = stream ? cudaStreamSynchronize(stream) : cudaDeviceSynchronize();
    return (sync_err == cudaSuccess);
}

} // namespace librenoise
