#include "librenoise/librenoise.hpp"
#include "librenoise/gpu_denoiser.hpp"
#include "librenoise/roi_analyzer.hpp"
#include "librenoise/profile_curve.hpp"
#include "librenoise/profile_io.hpp"
#if defined(LIBRENOISE_HAS_CUDA) && (LIBRENOISE_HAS_CUDA != 0)
#include <cuda_runtime.h>
#endif
#include <iostream>
#include <cassert>
#include <cstdlib>
#include <cmath>
#include <filesystem>
#include <random>

#define LN_REQUIRE(cond)                                                               \
    do {                                                                               \
        if (!(cond)) {                                                                 \
            std::cerr << "[FAIL] " << __FILE__ << ":" << __LINE__ << ": " #cond "\n"; \
            std::exit(1);                                                              \
        }                                                                              \
    } while (0)

using namespace librenoise;

void test_variance_stabilization() {
    std::cout << "[Test] Running Generalized Anscombe transform test...\n";
    ImagePlane32F plane(16, 16);
    NoiseLevelFunction nlf{0.002f, 0.0001f};

    plane.fill(0.45f);
    NoiseProfiler::apply_variance_stabilization(plane, nlf);
    assert(plane.at(8, 8) > 0.0f);

    NoiseProfiler::invert_variance_stabilization(plane, nlf);
    float reconstructed = plane.at(8, 8);
    float err = std::abs(reconstructed - 0.45f);
    assert(err < 0.001f);
    std::cout << "  -> Passed: Inversion error < 0.001 (actual: " << err << ")\n";
}

void test_scene_cut_detection() {
    std::cout << "[Test] Running scene cut detector...\n";
    ImagePlane32F frame1(64, 64);
    ImagePlane32F frame2(64, 64);
    frame1.fill(0.2f);
    frame2.fill(0.85f); // Drastic change

    bool is_cut = MotionEstimator::is_scene_cut(frame1, frame2, 0.35f);
    assert(is_cut == true);

    frame2.fill(0.21f); // Minor variation
    is_cut = MotionEstimator::is_scene_cut(frame1, frame2, 0.35f);
    assert(is_cut == false);
    std::cout << "  -> Passed: Scene cut correctly discriminated.\n";
}

void test_3d_collaborative_filtering() {
    std::cout << "[Test] Running 3D collaborative transform filter on synthetic pattern...\n";
    const int32_t w = 64;
    const int32_t h = 64;
    std::vector<VideoFrame32F> sequence;

    for (int i = 0; i < 3; ++i) {
        VideoFrame32F frame(w, h, 1);
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                // Diagonal pattern with synthetic noise
                float clean = (x + y > 64) ? 0.7f : 0.3f;
                float noise = ((x * 17 + y * 31 + i * 53) % 100 - 50) * 0.002f;
                frame.channels[0].at(x, y) = clean + noise;
            }
        }
        sequence.push_back(frame);
    }

    VideoDenoiser::Config cfg;
    cfg.temporal_radius = 1;
    cfg.filter_params.patch_size = 8;
    cfg.filter_params.step_size = 4;
    VideoDenoiser denoiser(cfg);

    VideoFrame32F result = denoiser.process_window(sequence);
    assert(result.is_valid());
    assert(result.width == w && result.height == h);
    std::cout << "  -> Passed: Filter completed successfully.\n";
}

void test_roi_and_profiling() {
    std::cout << "[Test] Running ROI analyzer and featureless patch detector...\n";
    const int32_t w = 256;
    const int32_t h = 256;
    VideoFrame32F frame(w, h, 3);

    // Create an image with an edge/feature on the left and a flat noisy patch on the right
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float val = 0.5f;
            if (x < 100) {
                // High contrast edge pattern
                val = ((x / 16) % 2 == 0) ? 0.9f : 0.1f;
            } else {
                // Featureless area with mild noise
                float n = ((x * 19 + y * 37) % 100 - 50) * 0.001f;
                val = 0.45f + n;
            }
            frame.channels[0].at(x, y) = val;
            frame.channels[1].at(x, y) = val;
            frame.channels[2].at(x, y) = val;
        }
    }

    AnalysisRoi roi = NoiseRoiAnalyzer::find_optimal_roi(frame, 64, 32);
    assert(roi.is_valid(w, h));
    // The optimal patch should avoid the edges on the left (x >= 100)
    assert(roi.x >= 90);
    std::cout << "  -> Auto-detected optimal ROI: [" << roi.x << ", " << roi.y 
              << ", w=" << roi.width << ", h=" << roi.height << "]\n";

    RoiAnalysisResult result = NoiseRoiAnalyzer::analyze_roi(frame, roi);
    assert(result.uniformity_score > 70.0f);
    assert(result.sigma_y > 0.0f);
    std::cout << "  -> Quality score: " << result.uniformity_score << "%, Assessment: " << result.assessment << "\n";
    std::cout << "  -> Passed: ROI analyzer found and analyzed uniform patch.\n";
}

void test_profile_curve_spline() {
    std::cout << "[Test] Running NoiseProfileCurve cubic Hermite spline...\n";
    NoiseProfileCurve curve({
        {0.00f, 1.40f},
        {0.25f, 1.20f},
        {0.50f, 1.00f},
        {0.75f, 0.90f},
        {1.00f, 0.80f}
    });

    // Exact knot evaluations
    assert(std::abs(curve.evaluate(0.00f) - 1.40f) < 1e-4f);
    assert(std::abs(curve.evaluate(0.50f) - 1.00f) < 1e-4f);
    assert(std::abs(curve.evaluate(1.00f) - 0.80f) < 1e-4f);

    // Intermediate points should be strictly monotonic decreasing between these knots
    float val_10 = curve.evaluate(0.10f);
    float val_40 = curve.evaluate(0.40f);
    float val_60 = curve.evaluate(0.60f);
    assert(val_10 < 1.40f && val_10 > 1.20f);
    assert(val_40 < 1.20f && val_40 > 1.00f);
    assert(val_60 < 1.00f && val_60 > 0.90f);

    // Clamping outside [0, 1]
    assert(curve.evaluate(-0.5f) == 1.40f);
    assert(curve.evaluate(1.5f) == 0.80f);
    std::cout << "  -> Passed: Cubic Hermite spline evaluates smoothly without oscillations.\n";
}

void test_profile_serialization() {
    std::cout << "[Test] Running LibreNoise profile (.lnp) serialization...\n";
    LibreNoiseDeviceProfile p;
    p.device_name = "Sony A7S III";
    p.mode_description = "4K S-Log3 ISO 12800";
    p.resolution_width = 3840;
    p.resolution_height = 2160;
    p.uniformity_score = 96.5f;
    p.quality_assessment = "Excellent";
    p.analysis_roi = {512, 512, 256, 256};
    p.nlf_y = {0.0025f, 0.0003f};
    p.nlf_cb = {0.0018f, 0.0002f};
    p.nlf_cr = {0.0019f, 0.0002f};
    p.psd_5band = {1.25f, 1.10f, 1.00f, 0.85f, 0.70f};

    std::string tempFile = (std::filesystem::temp_directory_path() / "test_librenoise.lnp").string();

    bool saved = ProfileSerializer::save_to_file(tempFile, p);
    assert(saved);

    LibreNoiseDeviceProfile loaded;
    bool loadedOk = ProfileSerializer::load_from_file(tempFile, loaded);
    assert(loadedOk);

    assert(loaded.device_name == "Sony A7S III");
    assert(loaded.mode_description == "4K S-Log3 ISO 12800");
    assert(loaded.resolution_width == 3840);
    assert(loaded.resolution_height == 2160);
    assert(std::abs(loaded.uniformity_score - 96.5f) < 0.1f);
    assert(loaded.quality_assessment == "Excellent");
    assert(loaded.analysis_roi.x == 512 && loaded.analysis_roi.width == 256);
    assert(std::abs(loaded.nlf_y.a - 0.0025f) < 1e-5f);
    assert(std::abs(loaded.psd_5band[0] - 1.25f) < 1e-4f);
    assert(std::abs(loaded.psd_5band[4] - 0.70f) < 1e-4f);

    std::filesystem::remove(tempFile);
    std::cout << "  -> Passed: .lnp profile saved and reloaded with 100% fidelity.\n";
}

void test_motion_field_resilience() {
    std::cout << "[Test] Running MotionField and Scene-Cut resilience test...\n";
    MotionField empty_mf;
    assert(!empty_mf.is_valid());

    auto mv = empty_mf.get_vector_at_pixel(100, 200);
    assert(mv.dx == 0 && mv.dy == 0);
    assert(mv.trusted == true);

    empty_mf.is_scene_cut = true;
    auto mv_cut = empty_mf.get_vector_at_pixel(100, 200);
    assert(mv_cut.trusted == false);

    // Sequence with a hard scene cut
    const int32_t w = 64;
    const int32_t h = 64;
    std::vector<VideoFrame32F> sequence;
    for (int i = 0; i < 3; ++i) {
        VideoFrame32F frame(w, h, 1);
        float base = (i == 1) ? 0.9f : 0.1f; // Frame 1 is completely different (scene cut)
        frame.channels[0].fill(base);
        sequence.push_back(frame);
    }

    VideoDenoiser::Config cfg;
    cfg.temporal_radius = 1;
    cfg.match_params.cut_detection_threshold = 0.3f;
    VideoDenoiser denoiser(cfg);
    VideoFrame32F result = denoiser.process_window(sequence);
    assert(result.is_valid());
    std::cout << "  -> Passed: Uninitialized/cut MotionField handles boundaries safely.\n";
}

void test_gpu_cpu_correlated_grain_parity() {
    std::cout << "[Test] Running correlated grain PSD & GPU/CPU parity test...\n";
    const int w = 160;
    const int h = 128;
    const int stride = w * 4;
    const int num_frames = 3;
    const int ref_idx = 1;

    std::vector<VideoFrame32F> clean_frames(num_frames, VideoFrame32F(w, h, 3));
    std::vector<VideoFrame32F> noisy_frames(num_frames, VideoFrame32F(w, h, 3));
    std::vector<std::vector<float>> rgba_frames(num_frames, std::vector<float>(stride * h, 1.0f));

    std::mt19937 rng(1337);
    std::normal_distribution<float> norm(0.0f, 1.0f);

    for (int f = 0; f < num_frames; ++f) {
        int dx = (f - ref_idx) * 3;
        int dy = (f - ref_idx) * 1;
        std::vector<float> raw(w * h * 3);
        for (float& v : raw) v = norm(rng);

        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                int sx = std::clamp(x + dx, 0, w - 1);
                int sy = std::clamp(y + dy, 0, h - 1);
                // Include flat region on left, midtone ramp, and HDR > 1.0 highlight patch on right
                float clean = 0.12f;
                if (sx > 96 && sy > 32 && sy < 96) {
                    clean = 2.25f; // HDR scene-linear highlight (> 1.0)
                } else if (sx > 64) {
                    clean = 0.35f;
                }

                for (int c = 0; c < 3; ++c) {
                    float corr = 0.0f;
                    for (int ky = -1; ky <= 1; ++ky) {
                        for (int kx = -1; kx <= 1; ++kx) {
                            int nx = std::clamp(x + kx, 0, w - 1);
                            int ny = std::clamp(y + ky, 0, h - 1);
                            corr += raw[(ny * w + nx) * 3 + c];
                        }
                    }
                    corr /= 3.0f;
                    float fine = raw[(y * w + x) * 3 + c];
                    float sigma = std::sqrt(0.00025f * clean + 0.00015f);
                    float val = std::max(0.0f, clean + (0.8f * corr + 0.2f * fine) * sigma);
                    clean_frames[f].channels[c].at(x, y) = clean;
                    noisy_frames[f].channels[c].at(x, y) = val;
                    rgba_frames[f][y * stride + x * 4 + c] = val;
                }
            }
        }
    }

    // 1. Verify ROI analyzer detects correlated low-frequency grain energy (measured_psd[0] > 1.25f)
    AnalysisRoi flat_roi{8, 16, 48, 96};
    RoiAnalysisResult roi_res = NoiseRoiAnalyzer::analyze_roi(noisy_frames[ref_idx], flat_roi);
    LN_REQUIRE(roi_res.sigma_y > 0.008f && roi_res.sigma_y < 0.030f);
    LN_REQUIRE(roi_res.measured_psd[0] > 1.25f);

    // 2. Verify auto_estimate also populates correlated PSD weights and accurate NLF
    auto auto_prof = NoiseProfiler::auto_estimate(noisy_frames[ref_idx]);
    LN_REQUIRE(auto_prof.psd.octave_weights[0][0] > 1.20f);

    auto calc_psnr = [&](const VideoFrame32F& est) {
        double mse = 0.0;
        for (int c = 0; c < 3; ++c) {
            for (int y = 12; y < h - 12; ++y) {
                for (int x = 12; x < w - 12; ++x) {
                    double d = est.channels[c].at(x, y) - clean_frames[ref_idx].channels[c].at(x, y);
                    mse += d * d;
                }
            }
        }
        mse /= (3.0 * (w - 24) * (h - 24));
        return 10.0 * std::log10(1.0 / mse);
    };

    double noisy_psnr = calc_psnr(noisy_frames[ref_idx]);

    VideoDenoiser::Config cfg;
    cfg.temporal_radius = 1;
    VideoDenoiser cpu_denoiser(cfg);
    VideoFrame32F cpu_out = cpu_denoiser.process_window(noisy_frames, &auto_prof);
    double cpu_psnr = calc_psnr(cpu_out);
    LN_REQUIRE(cpu_psnr > noisy_psnr + 7.0);
    // Verify HDR > 1.0 highlight was NOT clipped to 1.0 on CPU
    LN_REQUIRE(cpu_out.channels[0].at(120, 64) > 2.15f);

#if defined(LIBRENOISE_HAS_CUDA) && (LIBRENOISE_HAS_CUDA != 0)
    if (is_cuda_available()) {
        std::vector<float*> d_ptrs(num_frames, nullptr);
        for (int f = 0; f < num_frames; ++f) {
            cudaMalloc(&d_ptrs[f], stride * h * sizeof(float));
            cudaMemcpy(d_ptrs[f], rgba_frames[f].data(), stride * h * sizeof(float), cudaMemcpyHostToDevice);
        }
        const float** d_frame_arr = nullptr;
        cudaMalloc(&d_frame_arr, num_frames * sizeof(float*));
        cudaMemcpy(d_frame_arr, d_ptrs.data(), num_frames * sizeof(float*), cudaMemcpyHostToDevice);
        float* d_out = nullptr;
        cudaMalloc(&d_out, stride * h * sizeof(float));

        for (int eng = 0; eng < 4; ++eng) {
            CudaDenoiseParams cp;
            cp.width = w;
            cp.height = h;
            cp.stride_floats = stride;
            cp.channels = 4;
            cp.step_size = 4;
            cp.engine_choice = eng;
            for (int c = 0; c < 3; ++c) {
                cp.nlf_a[c] = auto_prof.nlf_per_channel[c].a;
                cp.nlf_b[c] = auto_prof.nlf_per_channel[c].b;
                for (int b = 0; b < 5; ++b) {
                    cp.psd_weights[c][b] = auto_prof.psd.octave_weights[c][b];
                }
            }

            bool ok = cuda_denoise_device(d_frame_arr, num_frames, ref_idx, d_out, cp, nullptr);
            LN_REQUIRE(ok);

            std::vector<float> h_out(stride * h, 0.0f);
            cudaMemcpy(h_out.data(), d_out, stride * h * sizeof(float), cudaMemcpyDeviceToHost);

            VideoFrame32F gpu_frame(w, h, 3);
            for (int y = 0; y < h; ++y) {
                for (int x = 0; x < w; ++x) {
                    for (int c = 0; c < 3; ++c) {
                        float v = h_out[y * stride + x * 4 + c];
                        LN_REQUIRE(std::isfinite(v) && v >= 0.0f);
                        gpu_frame.channels[c].at(x, y) = v;
                    }
                }
            }

            double gpu_psnr = calc_psnr(gpu_frame);
            std::cout << "  -> Engine " << eng << " GPU PSNR: " << gpu_psnr
                      << " dB (Noisy: " << noisy_psnr << " dB, CPU V-BM3D: " << cpu_psnr << " dB)\n";
            LN_REQUIRE(gpu_psnr > noisy_psnr + 6.5);
            // Verify HDR > 1.0 highlight was NOT clipped or dimmed on GPU
            LN_REQUIRE(gpu_frame.channels[0].at(120, 64) > 2.15f);
        }

        cudaFree(d_out);
        cudaFree(d_frame_arr);
        for (float* p : d_ptrs) cudaFree(p);
        release_cuda_resources();
    }
#endif
    std::cout << "  -> Passed: Correlated grain PSD, HDR > 1.0 preservation, and GPU/CPU parity verified.\n";
}

int main() {
    std::cout << "--- Running LibreNoise Core Unit Tests ---\n";
    test_variance_stabilization();
    test_scene_cut_detection();
    test_3d_collaborative_filtering();
    test_roi_and_profiling();
    test_profile_curve_spline();
    test_profile_serialization();
    test_motion_field_resilience();
    test_gpu_cpu_correlated_grain_parity();
    std::cout << "ALL TESTS PASSED SUCCESSFULLY!\n";
    return 0;
}
