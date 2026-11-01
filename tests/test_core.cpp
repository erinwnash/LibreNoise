#include "librenoise/librenoise.hpp"
#include <iostream>
#include <cassert>
#include <cmath>

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

int main() {
    std::cout << "--- Running LibreNoise Core Unit Tests ---\n";
    test_variance_stabilization();
    test_scene_cut_detection();
    test_3d_collaborative_filtering();
    std::cout << "ALL TESTS PASSED SUCCESSFULLY!\n";
    return 0;
}
