#include "librenoise/librenoise.hpp"
#include "librenoise/gpu_denoiser.hpp"
#if defined(LIBRENOISE_HAS_CUDA) && (LIBRENOISE_HAS_CUDA != 0)
#include <cuda_runtime.h>
#endif
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <string>

namespace {

using namespace librenoise;

struct SyntheticTestPattern {
    VideoFrame32F clean;
    VideoFrame32F noisy;
    NoiseLevelFunction nlf;
};

SyntheticTestPattern create_synthetic_clip(int32_t width, int32_t height, float noise_a, float noise_b, uint32_t seed) {
    SyntheticTestPattern test;
    test.clean = VideoFrame32F(width, height, 3);
    test.noisy = VideoFrame32F(width, height, 3);
    test.nlf.a = noise_a;
    test.nlf.b = noise_b;

    std::mt19937 rng(seed);
    std::normal_distribution<float> norm(0.0f, 1.0f);

    for (size_t c = 0; c < 3; ++c) {
        for (int32_t y = 0; y < height; ++y) {
            for (int32_t x = 0; x < width; ++x) {
                float cx = static_cast<float>(x - width / 2);
                float cy = static_cast<float>(y - height / 2);
                float dist = std::sqrt(cx * cx + cy * cy);
                float val = 0.5f + 0.35f * std::sin(dist * dist * 0.002f);
                if (x < width / 4) val = 0.2f;
                if (x > 3 * width / 4) val = 0.8f;

                test.clean.channels[c].at(x, y) = val;
                float sigma = test.nlf.sigma_at(val);
                float noisy_val = std::clamp(val + norm(rng) * sigma, 0.0f, 1.0f);
                test.noisy.channels[c].at(x, y) = noisy_val;
            }
        }
    }

    return test;
}

double compute_psnr(const ImagePlane32F& clean, const ImagePlane32F& test) {
    double mse = 0.0;
    int64_t count = 0;
    for (int32_t y = 0; y < clean.height(); ++y) {
        for (int32_t x = 0; x < clean.width(); ++x) {
            double diff = clean.at(x, y) - test.at(x, y);
            mse += diff * diff;
            count++;
        }
    }
    if (count == 0 || mse < 1e-12) return 100.0;
    mse /= static_cast<double>(count);
    return 10.0 * std::log10(1.0 / mse);
}

double compute_residual_energy(const ImagePlane32F& raw, const ImagePlane32F& denoised) {
    double energy = 0.0;
    int64_t count = 0;
    for (int32_t y = 0; y < raw.height(); ++y) {
        for (int32_t x = 0; x < raw.width(); ++x) {
            double res = raw.at(x, y) - denoised.at(x, y);
            energy += res * res;
            count++;
        }
    }
    return count > 0 ? (energy / count) : 0.0;
}

struct EngineBenchmarkResult {
    std::string name;
    double duration_ms;
    double denoised_psnr;
    double psnr_gain;
    double res_energy;
};

} // namespace

int main(int argc, char* argv[]) {
    (void)argc;
    (void)argv;
    std::cout << "====================================================================\n";
    std::cout << " LibreNoise Multi-Engine Comparative Benchmark v" << LIBRENOISE_VERSION_STRING << "\n";
    std::cout << "====================================================================\n\n";

    const int32_t test_w = 320;
    const int32_t test_h = 240;
    const size_t num_frames = 5;

    std::cout << "[*] Synthesizing 5 frames (" << test_w << "x" << test_h << ") with calibrated Poisson-Gaussian noise...\n";
    std::vector<VideoFrame32F> noisy_sequence;
    SyntheticTestPattern ground_truth = create_synthetic_clip(test_w, test_h, 0.0015f, 0.0001f, 1000);

    for (size_t i = 0; i < num_frames; ++i) {
        SyntheticTestPattern p = create_synthetic_clip(test_w, test_h, 0.0015f, 0.0001f, 1000 + static_cast<uint32_t>(i * 37));
        noisy_sequence.push_back(p.noisy);
    }

    double raw_psnr = compute_psnr(ground_truth.clean.channels[0], noisy_sequence[num_frames / 2].channels[0]);
    std::cout << "[-] Raw Noisy Reference Frame PSNR: " << std::fixed << std::setprecision(2) << raw_psnr << " dB\n\n";

    NoiseProfile profile;
    profile.profile_name = "CalibratedTestSensor";
    for (size_t c = 0; c < 3; ++c) profile.nlf_per_channel[c] = ground_truth.nlf;

    struct EngineCandidate {
        DenoiseEngine engine;
        std::string label;
    };

    std::vector<EngineCandidate> candidates = {
        {DenoiseEngine::Collaborative3D, "3D Collaborative Transform (BM3D)"},
        {DenoiseEngine::Wavelet5Band, "5-Band Wavelet Decomposition"},
        {DenoiseEngine::NonLocalMeans, "Non-Local Means (NLM)"},
        {DenoiseEngine::FrequencyDomain3D, "3D Frequency-Domain (DFTTest)"}
    };

    std::vector<EngineBenchmarkResult> results;

    for (const auto& cand : candidates) {
        std::cout << "[*] Benchmarking Engine: " << cand.label << " ...\n";

        VideoDenoiser::Config config;
        config.temporal_radius = 2;
        config.filter_params.engine = cand.engine;
        config.filter_params.patch_size = 8;
        config.filter_params.step_size = 3;
        config.detail_params.detail_recovery = 0.05f;
        config.detail_params.grain_amount = 0.0f;

        VideoDenoiser denoiser(config);

        auto start = std::chrono::high_resolution_clock::now();
        VideoFrame32F denoised = denoiser.process_window(noisy_sequence, &profile);
        auto end = std::chrono::high_resolution_clock::now();

        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        double psnr = compute_psnr(ground_truth.clean.channels[0], denoised.channels[0]);
        double gain = psnr - raw_psnr;
        double res = compute_residual_energy(noisy_sequence[num_frames / 2].channels[0], denoised.channels[0]);

        results.push_back({cand.label, ms, psnr, gain, res});
    }

#if defined(LIBRENOISE_HAS_CUDA) && (LIBRENOISE_HAS_CUDA != 0)
    if (is_cuda_available()) {
        std::cout << "[*] Benchmarking Engine: GPU Accelerated Pipeline (CUDA sm_86) ...\n";
        size_t rgba_count = static_cast<size_t>(test_w * test_h * 4);
        size_t rgba_bytes = rgba_count * sizeof(float);

        std::vector<float> h_rgba(rgba_count);
        std::vector<float*> d_frames(num_frames, nullptr);

        for (size_t f = 0; f < num_frames; ++f) {
            for (int y = 0; y < test_h; ++y) {
                for (int x = 0; x < test_w; ++x) {
                    size_t idx = (y * test_w + x) * 4;
                    h_rgba[idx + 0] = noisy_sequence[f].channels[0].at(x, y);
                    h_rgba[idx + 1] = noisy_sequence[f].channels[1].at(x, y);
                    h_rgba[idx + 2] = noisy_sequence[f].channels[2].at(x, y);
                    h_rgba[idx + 3] = 1.0f;
                }
            }
            cudaMalloc(&d_frames[f], rgba_bytes);
            cudaMemcpy(d_frames[f], h_rgba.data(), rgba_bytes, cudaMemcpyHostToDevice);
        }

        float* d_output = nullptr;
        cudaMalloc(&d_output, rgba_bytes);

        const float** d_frame_ptrs = nullptr;
        cudaMalloc(&d_frame_ptrs, num_frames * sizeof(float*));
        cudaMemcpy(d_frame_ptrs, d_frames.data(), num_frames * sizeof(float*), cudaMemcpyHostToDevice);

        CudaDenoiseParams cp;
        cp.width = test_w;
        cp.height = test_h;
        cp.stride_floats = test_w * 4;
        cp.channels = 4;
        cp.step_size = 4;
        cp.temporal_weight = 1.0f;
        cp.spatial_weight = 1.0f;
        for (int b = 0; b < 5; ++b) cp.band_weights[b] = 1.0f;
        for (int c = 0; c < 3; ++c) {
            cp.nlf_a[c] = ground_truth.nlf.a;
            cp.nlf_b[c] = ground_truth.nlf.b;
        }
        cp.detail_recovery = 0.05f;
        cp.grain_amount = 0.0f;
        cp.preview_mode = 0;

        // Warmup
        cuda_denoise_device(d_frame_ptrs, static_cast<int32_t>(num_frames), static_cast<int32_t>(num_frames / 2), d_output, cp);
        cudaDeviceSynchronize();

        // Benchmark timing
        auto start = std::chrono::high_resolution_clock::now();
        cuda_denoise_device(d_frame_ptrs, static_cast<int32_t>(num_frames), static_cast<int32_t>(num_frames / 2), d_output, cp);
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();

        double ms = std::chrono::duration<double, std::milli>(end - start).count();

        std::vector<float> h_out(rgba_count);
        cudaMemcpy(h_out.data(), d_output, rgba_bytes, cudaMemcpyDeviceToHost);

        ImagePlane32F gpu_plane(test_w, test_h);
        for (int y = 0; y < test_h; ++y) {
            for (int x = 0; x < test_w; ++x) {
                gpu_plane.at(x, y) = h_out[(y * test_w + x) * 4 + 0];
            }
        }

        double psnr = compute_psnr(ground_truth.clean.channels[0], gpu_plane);
        double gain = psnr - raw_psnr;
        double res = compute_residual_energy(noisy_sequence[num_frames / 2].channels[0], gpu_plane);

        results.push_back({"GPU Accelerated (CUDA sm_86)", ms, psnr, gain, res});

        cudaFree(d_output);
        cudaFree(d_frame_ptrs);
        for (auto p : d_frames) cudaFree(p);
    }
#endif

    std::cout << "\n========================================================================================\n";
    std::cout << "                       CANDIDATE ENGINE COMPARISON MATRIX\n";
    std::cout << "========================================================================================\n";
    std::cout << std::left << std::setw(38) << "Engine Candidate"
              << std::setw(14) << "Time (ms)"
              << std::setw(16) << "PSNR (dB)"
              << std::setw(16) << "Gain (dB)"
              << std::setw(14) << "Residual E." << "\n";
    std::cout << "----------------------------------------------------------------------------------------\n";

    for (const auto& r : results) {
        std::cout << std::left << std::setw(38) << r.name
                  << std::setw(14) << std::fixed << std::setprecision(1) << r.duration_ms
                  << std::setw(16) << std::fixed << std::setprecision(2) << r.denoised_psnr
                  << "+" << std::setw(15) << std::fixed << std::setprecision(2) << r.psnr_gain
                  << std::scientific << std::setprecision(2) << r.res_energy << "\n";
    }
    std::cout << "========================================================================================\n\n";

    return (results[0].psnr_gain >= 2.5) ? 0 : 1;
}
