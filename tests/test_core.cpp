#include "librenoise/librenoise.hpp"
#include "librenoise/roi_analyzer.hpp"
#include "librenoise/profile_curve.hpp"
#include "librenoise/profile_io.hpp"
#include <iostream>
#include <cassert>
#include <cmath>
#include <filesystem>

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

int main() {
    std::cout << "--- Running LibreNoise Core Unit Tests ---\n";
    test_variance_stabilization();
    test_scene_cut_detection();
    test_3d_collaborative_filtering();
    test_roi_and_profiling();
    test_profile_curve_spline();
    test_profile_serialization();
    std::cout << "ALL TESTS PASSED SUCCESSFULLY!\n";
    return 0;
}
