// LibreNoise v0.2 core tests. Plain executable: prints each check and returns non-zero on any failure.
#include "librenoise/librenoise.hpp"
#include "../src/core/v2/kernels.hpp"

#if defined(LIBRENOISE_HAS_CUDA) && (LIBRENOISE_HAS_CUDA != 0)
#include <cuda_runtime.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

using namespace librenoise::v2;

namespace {

int g_failures = 0;

void check(bool ok, const std::string& what, const std::string& detail = "") {
    std::cout << (ok ? "  PASS  " : "  FAIL  ") << what;
    if (!detail.empty()) std::cout << "  (" << detail << ")";
    std::cout << "\n";
    if (!ok) ++g_failures;
}

std::string fmt(double v) { char b[64]; std::snprintf(b, sizeof(b), "%.4g", v); return b; }

// Smooth synthetic scene with edges, a gradient and some texture, values in [0.01, 1.2]
std::vector<float> make_scene(int w, int h) {
    std::vector<float> img(static_cast<size_t>(w) * h * 4);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            const float gx = static_cast<float>(x) / w, gy = static_cast<float>(y) / h;
            float base = 0.02f + 1.1f * gx * gx;
            if ((x / 64 + y / 64) % 2 == 0) base *= 0.6f;
            const float tex = 0.03f * std::sin(x * 0.21f) * std::sin(y * 0.17f);
            float* p = &img[(static_cast<size_t>(y) * w + x) * 4];
            p[0] = base * (0.9f + 0.2f * gy) + tex;
            p[1] = base + tex;
            p[2] = base * (1.1f - 0.2f * gy) + tex;
            p[3] = 1.0f;
        }
    return img;
}

// Film-like noise: sigma proportional to value, independent per channel, mildly correlated spatially.
std::vector<float> add_noise(const std::vector<float>& clean, int w, int h, float k, float floor, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> white(static_cast<size_t>(w) * h * 3);
    for (auto& v : white) v = nd(rng);
    std::vector<float> out = clean;
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            for (int c = 0; c < 3; ++c) {
                const int x1 = std::min(x + 1, w - 1), y1 = std::min(y + 1, h - 1);
                const float n = 0.5f * (white[(static_cast<size_t>(y) * w + x) * 3 + c] + white[(static_cast<size_t>(y) * w + x1) * 3 + c] +
                                        white[(static_cast<size_t>(y1) * w + x) * 3 + c] + white[(static_cast<size_t>(y1) * w + x1) * 3 + c]);
                const size_t o = (static_cast<size_t>(y) * w + x) * 4 + c;
                const float v = std::max(clean[o], 0.0f);
                out[o] = clean[o] + std::sqrt(floor * floor + k * k * v * v) * n;
            }
    return out;
}

double rmse(const std::vector<float>& a, const std::vector<float>& b, int w, int h, int margin = 16) {
    double s = 0;
    size_t n = 0;
    for (int y = margin; y < h - margin; ++y)
        for (int x = margin; x < w - margin; ++x)
            for (int c = 0; c < 3; ++c) {
                const size_t o = (static_cast<size_t>(y) * w + x) * 4 + c;
                const double d = static_cast<double>(a[o]) - b[o];
                s += d * d;
                ++n;
            }
    return std::sqrt(s / static_cast<double>(n));
}

} // namespace

int main() {
    const int W = 512, H = 384;
    std::cout << "LibreNoise " << LIBRENOISE_VERSION_STRING << " core tests\n";

    // -------------------------------------------------------------- transforms
    {
        NoiseModel m;
        m.valid = true;
        for (int c = 0; c < 3; ++c)
            for (int q = 0; q < kCurveKnots; ++q) m.sigma[c][q] = std::sqrt(1e-6f + std::pow(0.05f * curve_knot_x(q), 2.0f));
        auto lut = std::make_unique<VstLut>();
        build_vst(m, *lut);
        double worst = 0;
        for (float x : {-0.01f, 0.0f, 1e-5f, 0.001f, 0.01f, 0.18f, 0.5f, 1.0f, 4.0f, 15.9f, 20.0f}) {
            const float back = k::vst_inv(*lut, 1, k::vst_fwd(*lut, 1, x));
            worst = std::max(worst, std::abs(static_cast<double>(back - x)) / std::max(1e-4, std::abs(static_cast<double>(x))));
        }
        check(worst < 2e-3, "VST forward/inverse round trip", "worst relative error " + fmt(worst));
        const float x = 0.3f, dx = 1e-4f;
        const float slope = (k::vst_fwd(*lut, 0, x + dx) - k::vst_fwd(*lut, 0, x - dx)) / (2 * dx);
        const float expect = 1.0f / model_sigma_at(m, 0, x);
        check(std::abs(slope / expect - 1.0f) < 0.02f, "VST slope = 1/sigma (unit noise after stabilisation)", fmt(slope) + " vs " + fmt(expect));
        float Y, Cb, Cr, r, g, b;
        k::rgb_to_work(*lut, 0.2f, 0.5f, 0.9f, Y, Cb, Cr);
        k::work_to_rgb(*lut, Y, Cb, Cr, r, g, b);
        check(std::abs(r - 0.2f) < 1e-3f && std::abs(g - 0.5f) < 1e-3f && std::abs(b - 0.9f) < 2e-3f, "Y/Cb/Cr round trip");
        float in[64], co[64], back[64];
        for (int i = 0; i < 64; ++i) in[i] = std::sin(i * 0.7f);
        k::dct8x8(in, co);
        k::idct8x8(co, back);
        double e = 0;
        for (int i = 0; i < 64; ++i) e = std::max(e, static_cast<double>(std::abs(back[i] - in[i])));
        check(e < 1e-5, "8x8 DCT round trip", fmt(e));
    }

    // -------------------------------------------------------------- profiler on synthetic film-like noise
    const auto clean = make_scene(W, H);
    const auto noisy = add_noise(clean, W, H, 0.06f, 0.0005f, 1);
    const NoiseModel model = estimate_noise_model(FrameView{noisy.data(), W * 4, 4}, W, H);
    check(model.valid, "profiler produced a model", model.summary());
    check(model.fit_exponent[1] > 0.75f && model.fit_exponent[1] < 1.2f, "noise exponent ~1 for value-proportional noise", fmt(model.fit_exponent[1]));
    check(model.detected_type == SourceType::Film, "independent per-channel noise detected as film", source_type_name(model.detected_type));
    check(model.grid_phase_x < 0 && model.grid_phase_y < 0, "no compression grid on clean synthetic noise");
    {
        NoiseModel back;
        check(NoiseModel::deserialize(model.serialize(), back) && back.valid &&
              std::abs(back.level_sigma[2][0][9] - model.level_sigma[2][0][9]) < 1e-5f, "model serialise round trip");
        const auto path = (std::filesystem::temp_directory_path() / "librenoise_test.lnp").string();
        ProfileFile pf;
        pf.device_name = "Test cam";
        pf.model = model;
        ProfileFile pl;
        check(save_profile(path, pf) && load_profile(path, pl) && pl.device_name == "Test cam" && pl.model.valid, ".lnp v2 save/load");
        std::filesystem::remove(path);
    }

    // -------------------------------------------------------------- pipeline identity (all amounts 0)
    {
        Params p;
        p.temporal_radius = 0;
        for (float& a : p.amount_band) a = 0.0f;
        std::vector<float> out(noisy.size());
        const FrameView v{noisy.data(), W * 4, 4};
        const bool ok = denoise_cpu(&v, 1, 0, W, H, model, p, OutputView{out.data(), W * 4, 4});
        check(ok && rmse(out, noisy, W, H, 0) < 2e-4, "amount 0 reproduces the input", fmt(rmse(out, noisy, W, H, 0)));
    }

    // -------------------------------------------------------------- spatial denoising
    {
        Params p;
        p.temporal_radius = 0;
        std::vector<float> spatial(noisy.size());
        const FrameView v{noisy.data(), W * 4, 4};
        const bool ok = denoise_cpu(&v, 1, 0, W, H, model, p, OutputView{spatial.data(), W * 4, 4});
        const double before = rmse(noisy, clean, W, H), after = rmse(spatial, clean, W, H);
        check(ok && after < 0.5 * before, "spatial filter removes most noise", fmt(before) + " -> " + fmt(after));
    }

    // -------------------------------------------------------------- temporal denoising + repeated frames
    {
        std::vector<std::vector<float>> frames;
        for (unsigned s = 0; s < 5; ++s) frames.push_back(add_noise(clean, W, H, 0.06f, 0.0005f, 10 + s));
        std::vector<FrameView> views;
        for (auto& f : frames) views.push_back(FrameView{f.data(), W * 4, 4});
        Params p;
        p.temporal_radius = 2;
        std::vector<float> outT(noisy.size()), out0(noisy.size()), outDup(noisy.size());
        denoise_cpu(views.data(), 5, 2, W, H, model, p, OutputView{outT.data(), W * 4, 4});
        p.temporal_radius = 0;
        denoise_cpu(&views[2], 1, 0, W, H, model, p, OutputView{out0.data(), W * 4, 4});
        const double eT = rmse(outT, clean, W, H), e0 = rmse(out0, clean, W, H);
        check(eT < 0.85 * e0, "temporal radius 2 beats spatial only on a static scene", fmt(e0) + " -> " + fmt(eT));
        std::vector<FrameView> dup(5, views[2]);
        p.temporal_radius = 2;
        denoise_cpu(dup.data(), 5, 2, W, H, model, p, OutputView{outDup.data(), W * 4, 4});
        check(rmse(outDup, out0, W, H, 0) < 1e-6, "repeated frames are not counted as new samples", fmt(rmse(outDup, out0, W, H, 0)));

        // moving scene: 3 px per frame; temporal must still help (motion compensation)
        std::vector<std::vector<float>> mv;
        for (int i = 0; i < 5; ++i) {
            std::vector<float> shifted(clean.size());
            const int dx = (i - 2) * 3;
            for (int y = 0; y < H; ++y)
                for (int x = 0; x < W; ++x)
                    for (int c = 0; c < 4; ++c)
                        shifted[(static_cast<size_t>(y) * W + x) * 4 + c] = clean[(static_cast<size_t>(y) * W + std::clamp(x + dx, 0, W - 1)) * 4 + c];
            mv.push_back(add_noise(shifted, W, H, 0.06f, 0.0005f, 40 + i));
        }
        std::vector<FrameView> mviews;
        for (auto& f : mv) mviews.push_back(FrameView{f.data(), W * 4, 4});
        std::vector<float> outM(noisy.size()), outM0(noisy.size());
        denoise_cpu(mviews.data(), 5, 2, W, H, model, p, OutputView{outM.data(), W * 4, 4});
        p.temporal_radius = 0;
        denoise_cpu(&mviews[2], 1, 0, W, H, model, p, OutputView{outM0.data(), W * 4, 4});
        const double m0 = rmse(outM0, clean, W, H, 24), mT = rmse(outM, clean, W, H, 24);
        check(mT < 0.9 * m0, "motion-compensated temporal helps on a moving scene", fmt(m0) + " -> " + fmt(mT));

#if defined(LIBRENOISE_HAS_CUDA) && (LIBRENOISE_HAS_CUDA != 0)
        if (cuda_available()) {
            p.temporal_radius = 2;
            const size_t bytes = clean.size() * sizeof(float);
            std::vector<float*> d(5, nullptr);
            std::vector<FrameView> dv;
            for (int i = 0; i < 5; ++i) {
                cudaMalloc(&d[i], bytes);
                cudaMemcpy(d[i], mv[i].data(), bytes, cudaMemcpyHostToDevice);
                dv.push_back(FrameView{d[i], W * 4, 4});
            }
            float* dout = nullptr;
            cudaMalloc(&dout, bytes);
            const bool ok = denoise_cuda(dv.data(), 5, 2, W, H, model, p, OutputView{dout, W * 4, 4});
            std::vector<float> outG(clean.size());
            cudaMemcpy(outG.data(), dout, bytes, cudaMemcpyDeviceToHost);
            cudaFree(dout);
            for (float* q : d) cudaFree(q);
            const double diff = rmse(outG, outM, W, H, 0);
            check(ok && diff < 1e-4, "CUDA matches CPU", "rms difference " + fmt(diff));
        } else {
            std::cout << "  SKIP  CUDA parity (no compute 8.6+ device)\n";
        }
#endif
    }

    // -------------------------------------------------------------- compression grid detection
    {
        std::vector<float> blocky = add_noise(clean, W, H, 0.02f, 0.0005f, 77);
        std::mt19937 rng(5);
        std::uniform_real_distribution<float> ud(-0.01f, 0.01f);
        for (int by = 0; by < H / 8; ++by)
            for (int bx = 0; bx < W / 8; ++bx) {
                const float off = ud(rng);
                for (int y = 0; y < 8; ++y)
                    for (int x = 0; x < 8; ++x)
                        for (int c = 0; c < 3; ++c) {
                            const size_t o = (static_cast<size_t>(by * 8 + y) * W + bx * 8 + x) * 4 + c;
                            blocky[o] += off * (0.2f + clean[o]);
                        }
            }
        const NoiseModel bm = estimate_noise_model(FrameView{blocky.data(), W * 4, 4}, W, H);
        check(bm.grid_phase_x == 7 && bm.grid_phase_y == 7, "8x8 block grid detected at the right phase",
              "phase " + std::to_string(bm.grid_phase_x) + "," + std::to_string(bm.grid_phase_y) + " strength " + fmt(bm.compression));
    }

    std::cout << (g_failures == 0 ? "ALL TESTS PASSED\n" : "TESTS FAILED: " + std::to_string(g_failures) + "\n");
    return g_failures == 0 ? 0 : 1;
}
