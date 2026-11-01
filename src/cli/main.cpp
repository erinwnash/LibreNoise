#include "librenoise/librenoise.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <chrono>
#include <iomanip>

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
                // Zone plate / edge pattern for testing detail and flat regions
                float cx = static_cast<float>(x - width / 2);
                float cy = static_cast<float>(y - height / 2);
                float dist = std::sqrt(cx * cx + cy * cy);
                float val = 0.5f + 0.35f * std::sin(dist * dist * 0.002f);
                if (x < width / 4) val = 0.2f; // Flat dark zone
                if (x > 3 * width / 4) val = 0.8f; // Flat bright zone

                test.clean.channels[c].at(x, y) = val;

                // Calibrated heteroscedastic noise: Var(I) = a*I + b
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

} // namespace

int main(int argc, char* argv[]) {
    (void)argc;
    (void)argv;
    std::cout << "=========================================================\n";
    std::cout << " LibreNoise CLI Test & Benchmark Harness v" << LIBRENOISE_VERSION_STRING << "\n";
    std::cout << "=========================================================\n\n";

    const int32_t test_w = 320;
    const int32_t test_h = 240;
    const size_t num_frames = 5;

    std::cout << "[*] Synthesizing " << num_frames << " frames (" << test_w << "x" << test_h << ") with calibrated Poisson-Gaussian noise...\n";
    std::vector<VideoFrame32F> noisy_sequence;
    SyntheticTestPattern ground_truth = create_synthetic_clip(test_w, test_h, 0.0015f, 0.0001f, 1000);

    for (size_t i = 0; i < num_frames; ++i) {
        SyntheticTestPattern p = create_synthetic_clip(test_w, test_h, 0.0015f, 0.0001f, 1000 + static_cast<uint32_t>(i * 37));
        noisy_sequence.push_back(p.noisy);
    }

    double raw_psnr = compute_psnr(ground_truth.clean.channels[0], noisy_sequence[num_frames / 2].channels[0]);
    std::cout << "[-] Raw Noisy Reference Frame PSNR: " << std::fixed << std::setprecision(2) << raw_psnr << " dB\n";

    // Known ground truth noise profile
    NoiseProfile profile;
    profile.profile_name = "CalibratedTestSensor";
    for (size_t c = 0; c < 3; ++c) {
        profile.nlf_per_channel[c] = ground_truth.nlf;
    }

    // Configure LibreNoise Pipeline
    VideoDenoiser::Config config;
    config.temporal_radius = 2;
    config.filter_params.patch_size = 8;
    config.filter_params.step_size = 3;
    config.filter_params.hard_thresh_lambda = 2.8f;
    config.detail_params.detail_recovery = 0.05f;
    config.detail_params.grain_amount = 0.0f;

    VideoDenoiser denoiser(config);

    std::cout << "[*] Executing 5-Stage LibreNoise Spatiotemporal Filter...\n";
    auto start_time = std::chrono::high_resolution_clock::now();

    VideoFrame32F denoised = denoiser.process_window(noisy_sequence, &profile);

    auto end_time = std::chrono::high_resolution_clock::now();
    double duration_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();

    double denoised_psnr = compute_psnr(ground_truth.clean.channels[0], denoised.channels[0]);
    double psnr_gain = denoised_psnr - raw_psnr;
    double res_energy = compute_residual_energy(noisy_sequence[num_frames / 2].channels[0], denoised.channels[0]);

    std::cout << "\n================= BENCHMARK RESULTS =================\n";
    std::cout << " Processing Time    : " << std::fixed << std::setprecision(1) << duration_ms << " ms\n";
    std::cout << " Initial PSNR       : " << std::fixed << std::setprecision(2) << raw_psnr << " dB\n";
    std::cout << " Denoised PSNR      : " << std::fixed << std::setprecision(2) << denoised_psnr << " dB\n";
    std::cout << " PSNR Improvement   : +" << std::fixed << std::setprecision(2) << psnr_gain << " dB\n";
    std::cout << " Residual Energy    : " << std::scientific << res_energy << "\n";
    std::cout << " Gate 1 Status      : " << (psnr_gain >= 2.5 ? "PASSED [OK]" : "FAILED") << "\n";
    std::cout << "=====================================================\n\n";

    return (psnr_gain >= 2.5) ? 0 : 1;
}
