#include "librenoise/gpu_denoiser.hpp"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <algorithm>
#include <iostream>

namespace librenoise {

// Constant memory for 8x8 Orthonormal DCT-II basis matrix
__constant__ float c_dct_mat[8][8];

namespace {

// Host initialization of DCT matrix
void init_dct_matrix_constant() {
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

    cudaMemcpyToSymbol(c_dct_mat, h_mat, sizeof(h_mat));
    initialized = true;
}

// Pseudo-random hash for synthetic film grain
__device__ inline float hash_noise(int x, int y, uint32_t seed) {
    uint32_t n = static_cast<uint32_t>(x) + static_cast<uint32_t>(y) * 19849u + seed * 65537u;
    n = (n ^ (n >> 13)) * 1274126177u;
    n = n ^ (n >> 16);
    return (static_cast<float>(n & 0x00FFFFFF) / static_cast<float>(0x00FFFFFF)) * 2.0f - 1.0f;
}

// =========================================================================
// Kernel 1: Temporal Motion-Gated Pre-filtering (Channel-Parallel)
// =========================================================================
__global__ void k_temporal_accumulation(
    const float* const* __restrict__ d_frames,
    int frame_count,
    int ref_index,
    int width,
    int height,
    int stride_floats,
    float temporal_weight,
    const float* __restrict__ nlf_a,
    const float* __restrict__ nlf_b,
    float* __restrict__ d_temp_accum
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int pixel_idx = y * stride_floats + x * 4;
    const float* ref_frame = d_frames[ref_index];

    for (int c = 0; c < 3; ++c) {
        float ref_val = ref_frame[pixel_idx + c];
        float var = fmaxf(0.0f, nlf_a[c] * ref_val + nlf_b[c]);
        float sigma = sqrtf(var);
        float trust_limit = 3.5f * sigma;

        float sum = ref_val;
        float weight_sum = 1.0f;
        float inv_two_sig_sq = 1.0f / (2.0f * (sigma * temporal_weight) * (sigma * temporal_weight) + 1e-7f);

        for (int f = 0; f < frame_count; ++f) {
            if (f == ref_index) continue;
            float cand_val = d_frames[f][pixel_idx + c];
            float diff = fabsf(cand_val - ref_val);
            if (diff <= trust_limit) {
                float w = expf(-diff * diff * inv_two_sig_sq);
                sum += cand_val * w;
                weight_sum += w;
            }
        }

        d_temp_accum[pixel_idx + c] = sum / weight_sum;
    }

    // Preserve Alpha
    d_temp_accum[pixel_idx + 3] = ref_frame[pixel_idx + 3];
}

// =========================================================================
// Kernel 2: 2D 8x8 DCT Frequency Shrinkage (64 threads per 8x8 block)
// =========================================================================
__global__ void k_spatial_dct_shrinkage(
    const float* __restrict__ d_temp_accum,
    int width,
    int height,
    int stride_floats,
    int step_size,
    int channel,
    float spatial_weight,
    const float* __restrict__ band_weights,
    float nlf_a,
    float nlf_b,
    float* __restrict__ d_numerator,
    float* __restrict__ d_denominator
) {
    __shared__ float s_patch[8][8];
    __shared__ float s_temp[8][8];
    __shared__ float s_sigma;

    int bx = blockIdx.x * step_size;
    int by = blockIdx.y * step_size;

    int tx = threadIdx.x; // 0..7
    int ty = threadIdx.y; // 0..7

    int px = bx + tx;
    int py = by + ty;

    // Load patch into shared memory with edge clamping
    px = min(px, width - 1);
    py = min(py, height - 1);

    float val = d_temp_accum[py * stride_floats + px * 4 + channel];
    s_patch[ty][tx] = val;

    // Center pixel computes block noise sigma
    if (tx == 4 && ty == 4) {
        float var = fmaxf(0.0f, nlf_a * val + nlf_b);
        s_sigma = sqrtf(var) * spatial_weight;
    }
    __syncthreads();

    // 1D DCT across rows: s_temp[ty][k] = sum_n (c_dct_mat[k][n] * s_patch[ty][n])
    float row_sum = 0.0f;
    #pragma unroll
    for (int n = 0; n < 8; ++n) {
        row_sum += c_dct_mat[tx][n] * s_patch[ty][n];
    }
    s_temp[ty][tx] = row_sum;
    __syncthreads();

    // 1D DCT across columns: s_patch[k][tx] = sum_m (c_dct_mat[k][m] * s_temp[m][tx])
    float col_sum = 0.0f;
    #pragma unroll
    for (int m = 0; m < 8; ++m) {
        col_sum += c_dct_mat[ty][m] * s_temp[m][tx];
    }
    s_patch[ty][tx] = col_sum;
    __syncthreads();

    // 5-Band Radial Wiener Spectral Shrinkage
    float coeff = s_patch[ty][tx];
    if (!(tx == 0 && ty == 0)) { // Preserve DC
        float r = sqrtf(static_cast<float>(tx * tx + ty * ty));
        int band = (r < 1.5f) ? 0 : (r < 3.0f) ? 1 : (r < 5.0f) ? 2 : (r < 7.5f) ? 3 : 4;
        float noise_pwr = (s_sigma * s_sigma) * band_weights[band];
        float pwr = coeff * coeff;
        float gain = fmaxf(0.0f, (pwr - noise_pwr) / (pwr + 1e-7f));
        coeff *= gain;
    }
    s_patch[ty][tx] = coeff;
    __syncthreads();

    // 1D IDCT across columns: s_temp[m][tx] = sum_k (c_dct_mat[k][m] * s_patch[k][tx])
    float icol_sum = 0.0f;
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        icol_sum += c_dct_mat[k][ty] * s_patch[k][tx];
    }
    s_temp[ty][tx] = icol_sum;
    __syncthreads();

    // 1D IDCT across rows: s_patch[ty][n] = sum_k (c_dct_mat[k][n] * s_temp[ty][k])
    float irow_sum = 0.0f;
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        irow_sum += c_dct_mat[k][tx] * s_temp[ty][k];
    }

    // Overlap-add into numerator and denominator buffers
    int out_x = bx + tx;
    int out_y = by + ty;
    if (out_x < width && out_y < height) {
        int idx = out_y * width + out_x;
        atomicAdd(&d_numerator[idx], irow_sum);
        atomicAdd(&d_denominator[idx], 1.0f);
    }
}

// =========================================================================
// Kernel 3: Post-processing (Normalization, Detail Recovery, Synthetic Grain)
// =========================================================================
__global__ void k_postprocess_finalize(
    const float* const* __restrict__ d_frames,
    int ref_index,
    const float* __restrict__ d_temp_accum,
    const float* __restrict__ d_num_r,
    const float* __restrict__ d_den_r,
    const float* __restrict__ d_num_g,
    const float* __restrict__ d_den_g,
    const float* __restrict__ d_num_b,
    const float* __restrict__ d_den_b,
    int width,
    int height,
    int stride_floats,
    float detail_recovery,
    float grain_amount,
    int preview_mode,
    const float* __restrict__ nlf_a,
    const float* __restrict__ nlf_b,
    float* __restrict__ d_output
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int pixel_idx = y * stride_floats + x * 4;
    int flat_idx = y * width + x;

    const float* d_ref_frame = d_frames[ref_index];
    const float* nums[3] = {d_num_r, d_num_g, d_num_b};
    const float* dens[3] = {d_den_r, d_den_g, d_den_b};

    for (int c = 0; c < 3; ++c) {
        float ref_val = d_ref_frame[pixel_idx + c];
        float d = dens[c][flat_idx];
        float filtered = (d > 1e-6f) ? (nums[c][flat_idx] / d) : d_temp_accum[pixel_idx + c];

        float sigma = sqrtf(fmaxf(0.0f, nlf_a[c] * ref_val + nlf_b[c]));

        // Edge-protected micro-contrast recovery
        if (detail_recovery > 0.0f) {
            float residual = ref_val - filtered;
            if (fabsf(residual) > 2.0f * sigma) {
                filtered += residual * detail_recovery;
            }
        }

        // Calibrated synthetic film grain
        if (grain_amount > 0.0f) {
            float grain = hash_noise(x, y, static_cast<uint32_t>(c * 7919)) * sigma * grain_amount;
            filtered += grain;
        }

        // Preview Mode: 1 = Residual visual check
        if (preview_mode == 1) {
            filtered = (ref_val - filtered) * 2.0f + 0.5f;
        }

        d_output[pixel_idx + c] = fminf(1.0f, fmaxf(0.0f, filtered));
    }

    // Preserve Alpha
    d_output[pixel_idx + 3] = d_ref_frame[pixel_idx + 3];
}

} // anonymous namespace

bool is_cuda_available() noexcept {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    return (err == cudaSuccess && device_count > 0);
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

    cudaStream_t stream = cuda_stream ? reinterpret_cast<cudaStream_t>(cuda_stream) : nullptr;

    init_dct_matrix_constant();

    const int32_t w = params.width;
    const int32_t h = params.height;
    const int32_t stride = params.stride_floats;
    const size_t total_pixels = static_cast<size_t>(w * h);

    // Allocate temporary device buffers for temporal accumulator and DCT overlap-add
    float* d_temp_accum = nullptr;
    float* d_num_r = nullptr;
    float* d_den_r = nullptr;
    float* d_num_g = nullptr;
    float* d_den_g = nullptr;
    float* d_num_b = nullptr;
    float* d_den_b = nullptr;
    float* d_band_weights = nullptr;
    float* d_nlf_a = nullptr;
    float* d_nlf_b = nullptr;

    cudaMalloc(&d_temp_accum, static_cast<size_t>(stride * h * sizeof(float)));
    cudaMalloc(&d_num_r, total_pixels * sizeof(float));
    cudaMalloc(&d_den_r, total_pixels * sizeof(float));
    cudaMalloc(&d_num_g, total_pixels * sizeof(float));
    cudaMalloc(&d_den_g, total_pixels * sizeof(float));
    cudaMalloc(&d_num_b, total_pixels * sizeof(float));
    cudaMalloc(&d_den_b, total_pixels * sizeof(float));

    cudaMalloc(&d_band_weights, 5 * sizeof(float));
    cudaMalloc(&d_nlf_a, 3 * sizeof(float));
    cudaMalloc(&d_nlf_b, 3 * sizeof(float));

    cudaMemcpyAsync(d_band_weights, params.band_weights, 5 * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_nlf_a, params.nlf_a, 3 * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_nlf_b, params.nlf_b, 3 * sizeof(float), cudaMemcpyHostToDevice, stream);

    cudaMemsetAsync(d_num_r, 0, total_pixels * sizeof(float), stream);
    cudaMemsetAsync(d_den_r, 0, total_pixels * sizeof(float), stream);
    cudaMemsetAsync(d_num_g, 0, total_pixels * sizeof(float), stream);
    cudaMemsetAsync(d_den_g, 0, total_pixels * sizeof(float), stream);
    cudaMemsetAsync(d_num_b, 0, total_pixels * sizeof(float), stream);
    cudaMemsetAsync(d_den_b, 0, total_pixels * sizeof(float), stream);

    // 1. Launch Temporal Pre-filter Kernel
    dim3 block_2d(16, 16);
    dim3 grid_pixel((w + block_2d.x - 1) / block_2d.x, (h + block_2d.y - 1) / block_2d.y);

    k_temporal_accumulation<<<grid_pixel, block_2d, 0, stream>>>(
        d_input_frames,
        frame_count,
        ref_index,
        w,
        h,
        stride,
        params.temporal_weight,
        d_nlf_a,
        d_nlf_b,
        d_temp_accum
    );

    // 2. Launch 2D DCT Frequency-Domain Shrinkage for each channel
    dim3 block_dct(8, 8);
    const int step = std::max(1, params.step_size);
    dim3 grid_dct((w - 8 + step) / step, (h - 8 + step) / step);

    float* nums[3] = {d_num_r, d_num_g, d_num_b};
    float* dens[3] = {d_den_r, d_den_g, d_den_b};

    for (int c = 0; c < 3; ++c) {
        k_spatial_dct_shrinkage<<<grid_dct, block_dct, 0, stream>>>(
            d_temp_accum,
            w,
            h,
            stride,
            step,
            c,
            params.spatial_weight,
            d_band_weights,
            params.nlf_a[c],
            params.nlf_b[c],
            nums[c],
            dens[c]
        );
    }

    // 3. Launch Post-processing Finalize Kernel
    k_postprocess_finalize<<<grid_pixel, block_2d, 0, stream>>>(
        d_input_frames,
        ref_index,
        d_temp_accum,
        d_num_r, d_den_r,
        d_num_g, d_den_g,
        d_num_b, d_den_b,
        w,
        h,
        stride,
        params.detail_recovery,
        params.grain_amount,
        params.preview_mode,
        d_nlf_a,
        d_nlf_b,
        d_output
    );

    if (stream) {
        cudaStreamSynchronize(stream);
    } else {
        cudaDeviceSynchronize();
    }

    // Free temporary device memory
    cudaFree(d_temp_accum);
    cudaFree(d_num_r);
    cudaFree(d_den_r);
    cudaFree(d_num_g);
    cudaFree(d_den_g);
    cudaFree(d_num_b);
    cudaFree(d_den_b);
    cudaFree(d_band_weights);
    cudaFree(d_nlf_a);
    cudaFree(d_nlf_b);

    return true;
}

} // namespace librenoise
