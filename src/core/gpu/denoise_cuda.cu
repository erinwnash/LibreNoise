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
// Kernel 1: 3D Collaborative Spatiotemporal BM3D Transform Filter
// =========================================================================
__global__ void k_bm3d_collaborative_filter(
    const float* const* __restrict__ d_frames,
    int frame_count,
    int ref_index,
    int width,
    int height,
    int stride_floats,
    int step_size,
    int channel,
    float temporal_weight,
    float spatial_weight,
    float hard_thresh_lambda,
    const float* __restrict__ band_weights,
    float nlf_a,
    float nlf_b,
    float* __restrict__ d_numerator,
    float* __restrict__ d_denominator
) {
    __shared__ float s_group[8][8][8];
    __shared__ float s_temp[8][8];
    __shared__ float s_sigma;
    __shared__ int s_group_size;
    __shared__ int s_group_f[8];
    __shared__ int s_group_dx[8];
    __shared__ int s_group_dy[8];
    __shared__ int s_total_non_zero;
    __shared__ float s_warp_reduce[2];

    int tx = threadIdx.x; // 0..7
    int ty = threadIdx.y; // 0..7
    int tid = ty * 8 + tx; // 0..63
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    int bx = blockIdx.x * step_size;
    int by = blockIdx.y * step_size;

    // Edge clamping: ensure blocks at image boundary cover right and bottom edges
    if (bx + 8 > width) bx = max(0, width - 8);
    if (by + 8 > height) by = max(0, height - 8);

    int px = min(bx + tx, width - 1);
    int py = min(by + ty, height - 1);

    const float* ref_frame = d_frames[ref_index];
    float ref_val = ref_frame[py * stride_floats + px * 4 + channel];
    s_group[0][ty][tx] = ref_val;

    if (tx == 4 && ty == 4) {
        float var = fmaxf(1e-7f, nlf_a * ref_val + nlf_b);
        s_sigma = sqrtf(var) * spatial_weight;
        s_group_size = 1;
        s_group_f[0] = ref_index;
        s_group_dx[0] = 0;
        s_group_dy[0] = 0;
    }
    __syncthreads();

    float trust_dist = 4.0f * (s_sigma * s_sigma + 1e-6f);

    // Form 3D group by cohesive block-matching across temporal frames
    if (frame_count > 1 && temporal_weight > 0.01f) {
        for (int f = 0; f < frame_count; ++f) {
            if (f == ref_index) continue;
            if (s_group_size >= 8) break;

            const float* cand_frame = d_frames[f];

            // 1. Stationary offset (0, 0) check
            float cand_v0 = cand_frame[py * stride_floats + px * 4 + channel];
            float diff0 = ref_val - cand_v0;
            float sq0 = diff0 * diff0;

            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                sq0 += __shfl_down_sync(0xffffffff, sq0, offset);
            }
            if (lane_id == 0) s_warp_reduce[warp_id] = sq0;
            __syncthreads();

            float dist0 = (s_warp_reduce[0] + s_warp_reduce[1]) / 64.0f;

            int best_dx = 0;
            int best_dy = 0;
            bool accepted = false;

            if (dist0 <= trust_dist) {
                best_dx = 0;
                best_dy = 0;
                accepted = true;
            } else {
                // Coherent block motion search
                const int num_cands = 8;
                const int search_dx[8] = { 2, -2,  0,  0,  4, -4,  0,  0 };
                const int search_dy[8] = { 0,  0,  2, -2,  0,  0,  4, -4 };

                float min_dist = dist0;
                for (int s = 0; s < num_cands; ++s) {
                    int cx = min(max(bx + tx + search_dx[s], 0), width - 1);
                    int cy = min(max(by + ty + search_dy[s], 0), height - 1);
                    float cv = cand_frame[cy * stride_floats + cx * 4 + channel];
                    float d = ref_val - cv;
                    float sq = d * d;

                    #pragma unroll
                    for (int offset = 16; offset > 0; offset /= 2) {
                        sq += __shfl_down_sync(0xffffffff, sq, offset);
                    }
                    if (lane_id == 0) s_warp_reduce[warp_id] = sq;
                    __syncthreads();

                    float dist = (s_warp_reduce[0] + s_warp_reduce[1]) / 64.0f;
                    if (dist < min_dist) {
                        min_dist = dist;
                        best_dx = search_dx[s];
                        best_dy = search_dy[s];
                    }
                }

                if (min_dist <= trust_dist) {
                    accepted = true;
                }
            }

            if (accepted) {
                if (tid == 0 && s_group_size < 8) {
                    int idx = s_group_size++;
                    s_group_f[idx] = f;
                    s_group_dx[idx] = best_dx;
                    s_group_dy[idx] = best_dy;
                }
                __syncthreads();
            }
        }
    }

    // Load accepted candidate patches into s_group
    int K = s_group_size;
    for (int k = 1; k < K; ++k) {
        int f = s_group_f[k];
        int dx = s_group_dx[k];
        int dy = s_group_dy[k];
        int cx = min(max(bx + tx + dx, 0), width - 1);
        int cy = min(max(by + ty + dy, 0), height - 1);
        s_group[k][ty][tx] = d_frames[f][cy * stride_floats + cx * 4 + channel];
    }
    __syncthreads();

    // 2D Spatial DCT on each member k in [0, K-1]
    for (int k = 0; k < K; ++k) {
        // 1D DCT across rows
        float r_sum = 0.0f;
        #pragma unroll
        for (int n = 0; n < 8; ++n) {
            r_sum += c_dct_mat[tx][n] * s_group[k][ty][n];
        }
        s_temp[ty][tx] = r_sum;
        __syncthreads();

        // 1D DCT across columns
        float c_sum = 0.0f;
        #pragma unroll
        for (int m = 0; m < 8; ++m) {
            c_sum += c_dct_mat[ty][m] * s_temp[m][tx];
        }
        s_group[k][ty][tx] = c_sum;
        __syncthreads();
    }

    // 1D Temporal DCT across K members for frequency bin (ty, tx)
    float temp_in[8];
    float temp_out[8];
    for (int k = 0; k < K; ++k) {
        temp_in[k] = s_group[k][ty][tx];
    }

    if (K == 1) {
        temp_out[0] = temp_in[0];
    } else {
        float alpha0 = 1.0f / sqrtf(static_cast<float>(K));
        float alpha = sqrtf(2.0f / static_cast<float>(K));
        constexpr float pi = 3.141592653589793f;
        for (int k = 0; k < K; ++k) {
            float factor = (k == 0) ? alpha0 : alpha;
            float sum = 0.0f;
            for (int n = 0; n < K; ++n) {
                sum += temp_in[n] * cosf((pi / static_cast<float>(K)) * (n + 0.5f) * k);
            }
            temp_out[k] = factor * sum;
        }
    }

    // Collaborative Hard-Thresholding
    float r = sqrtf(static_cast<float>(tx * tx + ty * ty));
    int band = (r < 1.5f) ? 0 : (r < 3.0f) ? 1 : (r < 5.0f) ? 2 : (r < 7.5f) ? 3 : 4;
    float band_scale = sqrtf(band_weights[band]);
    float hard_thresh = hard_thresh_lambda * s_sigma * band_scale;

    int thread_nz = 0;
    for (int k = 0; k < K; ++k) {
        if (k == 0 && tx == 0 && ty == 0) {
            // DC of reference patch is always preserved
            thread_nz++;
            continue;
        }
        float v = temp_out[k];
        if (fabsf(v) < hard_thresh) {
            temp_out[k] = 0.0f;
        } else {
            thread_nz++;
        }
    }

    // Count non-zero coefficients across all 64 threads in the block
    int nz = thread_nz;
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        nz += __shfl_down_sync(0xffffffff, nz, offset);
    }
    if (lane_id == 0) s_warp_reduce[warp_id] = static_cast<float>(nz);
    __syncthreads();
    if (tid == 0) {
        s_total_non_zero = max(1, static_cast<int>(s_warp_reduce[0] + s_warp_reduce[1]));
    }
    __syncthreads();

    // Inverse 1D Temporal DCT (extracting reference member n = 0)
    float inv_temp_0 = 0.0f;
    if (K == 1) {
        inv_temp_0 = temp_out[0];
    } else {
        float alpha0 = 1.0f / sqrtf(static_cast<float>(K));
        float alpha = sqrtf(2.0f / static_cast<float>(K));
        constexpr float pi = 3.141592653589793f;
        inv_temp_0 = alpha0 * temp_out[0];
        for (int k = 1; k < K; ++k) {
            inv_temp_0 += alpha * temp_out[k] * cosf((pi / static_cast<float>(K)) * 0.5f * k);
        }
    }
    s_group[0][ty][tx] = inv_temp_0;
    __syncthreads();

    // Inverse 2D IDCT on s_group[0]
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

    // Overlap-add into numerator and denominator buffers with sparsity weighting
    float weight = 1.0f / static_cast<float>(s_total_non_zero);
    int out_x = bx + tx;
    int out_y = by + ty;
    if (out_x < width && out_y < height) {
        int idx = out_y * width + out_x;
        atomicAdd(&d_numerator[idx], irow_sum * weight);
        atomicAdd(&d_denominator[idx], weight);
    }
}

// =========================================================================
// Kernel 2: Post-processing Finalize (Gradient Edge Detail, Grain, Mix)
// =========================================================================
__global__ void k_postprocess_finalize(
    const float* const* __restrict__ d_frames,
    int ref_index,
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
    float mix_original,
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

    int left_x = max(0, x - 1);
    int right_x = min(width - 1, x + 1);
    int up_y = max(0, y - 1);
    int down_y = min(height - 1, y + 1);

    for (int c = 0; c < 3; ++c) {
        float ref_val = d_ref_frame[pixel_idx + c];
        float d = dens[c][flat_idx];
        float filtered = (d > 1e-6f) ? (nums[c][flat_idx] / d) : ref_val;

        float var = fmaxf(1e-7f, nlf_a[c] * filtered + nlf_b[c]);
        float sigma = sqrtf(var);

        // Edge-protected micro-contrast recovery matching DetailEngine
        if (detail_recovery > 0.0f) {
            float d_l = dens[c][y * width + left_x];
            float d_r = dens[c][y * width + right_x];
            float d_u = dens[c][up_y * width + x];
            float d_d = dens[c][down_y * width + x];

            float fl = (d_l > 1e-6f) ? (nums[c][y * width + left_x] / d_l) : d_ref_frame[y * stride_floats + left_x * 4 + c];
            float fr = (d_r > 1e-6f) ? (nums[c][y * width + right_x] / d_r) : d_ref_frame[y * stride_floats + right_x * 4 + c];
            float fu = (d_u > 1e-6f) ? (nums[c][up_y * width + x] / d_u) : d_ref_frame[up_y * stride_floats + x * 4 + c];
            float fd = (d_d > 1e-6f) ? (nums[c][down_y * width + x] / d_d) : d_ref_frame[down_y * stride_floats + x * 4 + c];

            float gx = (fr - fl) * 0.5f;
            float gy = (fd - fu) * 0.5f;
            float grad_mag = sqrtf(gx * gx + gy * gy);

            if (grad_mag > 1.5f * sigma) {
                float residual = ref_val - filtered;
                float boost = fminf(0.5f, fmaxf(0.0f, detail_recovery * (grad_mag / (grad_mag + sigma))));
                filtered = fminf(1.0f, fmaxf(0.0f, filtered + boost * residual));
            }
        }

        // Calibrated synthetic film grain
        if (grain_amount > 0.0f) {
            float grain = hash_noise(x, y, static_cast<uint32_t>(c * 7919)) * sigma * grain_amount;
            filtered = fminf(1.0f, fmaxf(0.0f, filtered + grain));
        }

        // Preview Mode: 1 = Residual visual check
        if (preview_mode == 1) {
            filtered = (ref_val - filtered) * 2.0f + 0.5f;
        } else if (mix_original > 0.0f) {
            filtered = filtered * (1.0f - mix_original) + ref_val * mix_original;
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

    // Allocate device buffers for DCT overlap-add
    float* d_num_r = nullptr;
    float* d_den_r = nullptr;
    float* d_num_g = nullptr;
    float* d_den_g = nullptr;
    float* d_num_b = nullptr;
    float* d_den_b = nullptr;
    float* d_band_weights = nullptr;
    float* d_nlf_a = nullptr;
    float* d_nlf_b = nullptr;

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

    // 1. Launch 3D Collaborative Spatiotemporal Filter for each channel
    dim3 block_dct(8, 8);
    const int step = std::max(1, params.step_size);
    dim3 grid_dct((w + step - 1) / step, (h + step - 1) / step);

    float* nums[3] = {d_num_r, d_num_g, d_num_b};
    float* dens[3] = {d_den_r, d_den_g, d_den_b};

    for (int c = 0; c < 3; ++c) {
        k_bm3d_collaborative_filter<<<grid_dct, block_dct, 0, stream>>>(
            d_input_frames,
            frame_count,
            ref_index,
            w,
            h,
            stride,
            step,
            c,
            params.temporal_weight,
            params.spatial_weight,
            params.hard_thresh_lambda,
            d_band_weights,
            params.nlf_a[c],
            params.nlf_b[c],
            nums[c],
            dens[c]
        );
    }

    // 2. Launch Post-processing Finalize Kernel
    dim3 block_2d(16, 16);
    dim3 grid_pixel((w + block_2d.x - 1) / block_2d.x, (h + block_2d.y - 1) / block_2d.y);

    k_postprocess_finalize<<<grid_pixel, block_2d, 0, stream>>>(
        d_input_frames,
        ref_index,
        d_num_r, d_den_r,
        d_num_g, d_den_g,
        d_num_b, d_den_b,
        w,
        h,
        stride,
        params.detail_recovery,
        params.grain_amount,
        params.mix_original,
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
