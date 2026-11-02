// Stress benchmark: moving content, colored scene, optionally spatially-correlated noise and
// *estimated* (not oracle) noise profiles. Reports how much of the injected noise is actually
// removed (flat regions, per luma/chroma component) and how much structure is damaged (moving
// regions), for every CPU engine and for the CUDA pipeline.
//
// Usage: librenoise_cli --stress [--oracle] [--white]
#include "librenoise/librenoise.hpp"
#include "librenoise/gpu_denoiser.hpp"
#include "librenoise/roi_analyzer.hpp"
#if defined(LIBRENOISE_HAS_CUDA) && (LIBRENOISE_HAS_CUDA != 0)
#include <cuda_runtime.h>
#endif
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

using namespace librenoise;

constexpr int W = 384;
constexpr int H = 256;
constexpr int kFrames = 5;

struct FlatPatch { int x, y, w, h; };
const FlatPatch kFlats[] = {
    {16, 16, 48, 48},     // dark
    {16, 100, 48, 48},    // mid
    {16, 190, 48, 48},    // bright
};

void rgb_to_ycc(float r, float g, float b, float& y, float& cb, float& cr) {
    y = 0.2126f * r + 0.7152f * g + 0.0722f * b;
    cb = (b - y) / 1.8556f;
    cr = (r - y) / 1.5748f;
}

void clean_pixel(int x, int y, int t, float out[3]) {
    // Static background: smooth colored gradient
    float u = static_cast<float>(x) / W;
    float v = static_cast<float>(y) / H;
    out[0] = 0.30f + 0.25f * u;
    out[1] = 0.35f + 0.20f * v;
    out[2] = 0.30f + 0.15f * (1.0f - u);

    // Flat patches (noise measurement)
    const float levels[3][3] = {{0.06f, 0.08f, 0.05f}, {0.30f, 0.42f, 0.22f}, {0.75f, 0.70f, 0.60f}};
    for (int i = 0; i < 3; ++i) {
        const auto& p = kFlats[i];
        if (x >= p.x && x < p.x + p.w && y >= p.y && y < p.y + p.h) {
            out[0] = levels[i][0]; out[1] = levels[i][1]; out[2] = levels[i][2];
        }
    }

    // Moving textured rectangle (+3 px / frame)
    int rx = 120 + 3 * t;
    if (x >= rx && x < rx + 110 && y >= 40 && y < 120) {
        float tx = static_cast<float>(x - rx);
        float ty = static_cast<float>(y - 40);
        float tex = 0.5f + 0.25f * std::sin(tx * 0.9f) * std::sin(ty * 0.7f) + 0.15f * std::sin((tx + ty) * 0.25f);
        out[0] = tex;
        out[1] = tex * 0.8f + 0.1f;
        out[2] = 0.9f - tex * 0.5f;
    }

    // Moving bright disk (-2,+1 px / frame)
    float dx = static_cast<float>(x) - (300.0f - 2.0f * t);
    float dy = static_cast<float>(y) - (190.0f + 1.0f * t);
    if (dx * dx + dy * dy < 30.0f * 30.0f) {
        out[0] = 0.9f; out[1] = 0.25f; out[2] = 0.2f;
    }
}

VideoFrame32F make_clean(int t) {
    VideoFrame32F f(W, H, 3);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            float px[3];
            clean_pixel(x, y, t, px);
            for (int c = 0; c < 3; ++c) f.channels[c].at(x, y) = px[c];
        }
    return f;
}

VideoFrame32F add_noise(const VideoFrame32F& clean, const NoiseLevelFunction& nlf, bool correlated, uint32_t seed) {
    VideoFrame32F noisy = clean;
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (int c = 0; c < 3; ++c) {
        ImagePlane32F n(W, H);
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) n.at(x, y) = nd(rng);
        if (correlated) {
            // 3x3 binomial blur, renormalised to unit variance (spatially correlated grain / debayer noise)
            ImagePlane32F tmp(W, H), out(W, H);
            const float k[3] = {0.25f, 0.5f, 0.25f};
            for (int y = 0; y < H; ++y)
                for (int x = 0; x < W; ++x) {
                    float s = 0;
                    for (int i = -1; i <= 1; ++i) s += k[i + 1] * n.at(std::clamp(x + i, 0, W - 1), y);
                    tmp.at(x, y) = s;
                }
            for (int y = 0; y < H; ++y)
                for (int x = 0; x < W; ++x) {
                    float s = 0;
                    for (int i = -1; i <= 1; ++i) s += k[i + 1] * tmp.at(x, std::clamp(y + i, 0, H - 1));
                    out.at(x, y) = s;
                }
            const float norm = 1.0f / std::sqrt(0.375f * 0.375f); // sum(k^2)=0.375 per axis
            for (int y = 0; y < H; ++y)
                for (int x = 0; x < W; ++x) n.at(x, y) = out.at(x, y) * norm;
        }
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
                float v = clean.channels[c].at(x, y);
                noisy.channels[c].at(x, y) = v + n.at(x, y) * nlf.sigma_at(v);
            }
    }
    return noisy;
}

struct Metrics {
    double noise_left_y = 0, noise_left_c = 0; // ratio of residual noise std to input noise std (flat patches)
    double moving_psnr = 0;                    // PSNR on the moving objects' bounding area
    double full_psnr = 0;
    double ms = 0;
};

double region_std(const VideoFrame32F& test, const VideoFrame32F& clean, bool chroma) {
    double acc = 0; long n = 0;
    for (const auto& p : kFlats) {
        for (int y = p.y + 4; y < p.y + p.h - 4; ++y)
            for (int x = p.x + 4; x < p.x + p.w - 4; ++x) {
                float ty, tcb, tcr, cy, ccb, ccr;
                rgb_to_ycc(test.channels[0].at(x, y), test.channels[1].at(x, y), test.channels[2].at(x, y), ty, tcb, tcr);
                rgb_to_ycc(clean.channels[0].at(x, y), clean.channels[1].at(x, y), clean.channels[2].at(x, y), cy, ccb, ccr);
                double d2 = chroma ? 0.5 * ((tcb - ccb) * (tcb - ccb) + (tcr - ccr) * (tcr - ccr))
                                   : (ty - cy) * (ty - cy);
                acc += d2; n++;
            }
    }
    return std::sqrt(acc / std::max<long>(1, n));
}
double region_psnr(const VideoFrame32F& test, const VideoFrame32F& clean, int x0, int y0, int x1, int y1) {
    double mse = 0; long n = 0;
    for (int y = y0; y < y1; ++y)
        for (int x = x0; x < x1; ++x)
            for (int c = 0; c < 3; ++c) {
                double d = test.channels[c].at(x, y) - clean.channels[c].at(x, y);
                mse += d * d; n++;
            }
    mse /= std::max<long>(1, n);
    return mse < 1e-12 ? 100.0 : 10.0 * std::log10(1.0 / mse);
}

Metrics evaluate(const VideoFrame32F& out, const VideoFrame32F& noisy_ref, const VideoFrame32F& clean_ref, double ms) {
    Metrics m;
    double in_y = region_std(noisy_ref, clean_ref, false);
    double in_c = region_std(noisy_ref, clean_ref, true);
    m.noise_left_y = region_std(out, clean_ref, false) / std::max(1e-9, in_y);
    m.noise_left_c = region_std(out, clean_ref, true) / std::max(1e-9, in_c);
    m.moving_psnr = region_psnr(out, clean_ref, 100, 30, 280, 130);
    m.full_psnr = region_psnr(out, clean_ref, 0, 0, W, H);
    m.ms = ms;
    return m;
}

void print_row(const std::string& name, const Metrics& m) {
    std::cout << std::left << std::setw(34) << name << std::right << std::fixed
              << std::setw(10) << std::setprecision(1) << m.ms
              << std::setw(14) << std::setprecision(1) << (100.0 * m.noise_left_y) << "%"
              << std::setw(13) << std::setprecision(1) << (100.0 * m.noise_left_c) << "%"
              << std::setw(14) << std::setprecision(2) << m.moving_psnr
              << std::setw(12) << std::setprecision(2) << m.full_psnr << "\n";
}

} // namespace

int run_stress_benchmark(bool oracle, bool white_noise) {
    const bool correlated = !white_noise;
    NoiseLevelFunction truth;
    truth.a = 0.0020f;
    truth.b = 0.00015f;

    std::vector<VideoFrame32F> clean(kFrames), noisy(kFrames);
    for (int t = 0; t < kFrames; ++t) {
        clean[t] = make_clean(t);
        noisy[t] = add_noise(clean[t], truth, correlated, 4242u + 101u * t);
    }
    const int ref = kFrames / 2;

    NoiseProfile profile;
    if (oracle) {
        profile.profile_name = "Oracle";
        profile.nlf_per_channel = std::array<NoiseLevelFunction, 3>{truth, truth, truth};
        if (correlated) {
            // Analytic PSD of the 3x3-binomial-blurred noise, folded into the 5 radial DCT bands
            double be[5] = {0, 0, 0, 0, 0}; int bc[5] = {0, 0, 0, 0, 0}; double tot = 0; int tn = 0;
            for (int v = 0; v < 8; ++v)
                for (int u = 0; u < 8; ++u) {
                    if (u == 0 && v == 0) continue;
                    const double pi = 3.14159265358979323846;
                    double hu = 0.5 + 0.5 * std::cos(pi * u / 8.0), hv = 0.5 + 0.5 * std::cos(pi * v / 8.0);
                    double p = (hu * hu / 0.375) * (hv * hv / 0.375);
                    double r = std::sqrt(double(u * u + v * v));
                    int b = (r < 1.5) ? 0 : (r < 3.0) ? 1 : (r < 5.0) ? 2 : (r < 7.5) ? 3 : 4;
                    be[b] += p; bc[b]++; tot += p; tn++;
                }
            for (int c = 0; c < 3; ++c)
                for (int b = 0; b < 5; ++b)
                    profile.psd.octave_weights[c][b] = static_cast<float>(std::sqrt((be[b] / bc[b]) / (tot / tn)));
        }
    } else {
        profile = NoiseProfiler::auto_estimate(noisy[ref]);
    }

    std::cout << "\n=== LibreNoise stress benchmark (" << (correlated ? "correlated" : "white")
              << " noise, " << (oracle ? "oracle" : "auto-estimated") << " profile) ===\n";
    std::cout << "True NLF: a=" << truth.a << " b=" << truth.b
              << " | Estimated per-plane sigma @0.5: ";
    for (int c = 0; c < 3; ++c) std::cout << profile.nlf_per_channel[c].sigma_at(0.5f) << " ";
    std::cout << "(true " << truth.sigma_at(0.5f) << ")\n\n";
    std::cout << std::left << std::setw(34) << "Engine" << std::right << std::setw(10) << "ms"
              << std::setw(15) << "Luma noise left" << std::setw(14) << "Chroma left"
              << std::setw(14) << "Moving PSNR" << std::setw(12) << "Full PSNR" << "\n";
    std::cout << std::string(99, '-') << "\n";

    {
        Metrics m = evaluate(noisy[ref], noisy[ref], clean[ref], 0.0);
        m.moving_psnr = region_psnr(noisy[ref], clean[ref], 100, 30, 280, 130);
        m.full_psnr = region_psnr(noisy[ref], clean[ref], 0, 0, W, H);
        print_row("(input)", m);
    }

    struct E { DenoiseEngine e; const char* n; };
    const E engines[] = {
        {DenoiseEngine::Collaborative3D, "CPU 3D Collaborative"},
        {DenoiseEngine::Wavelet5Band, "CPU 5-Band Wavelet"},
        {DenoiseEngine::NonLocalMeans, "CPU Non-Local Means"},
        {DenoiseEngine::FrequencyDomain3D, "CPU Frequency-Domain 3D"},
    };
    for (const auto& e : engines) {
        VideoDenoiser::Config cfg;
        cfg.temporal_radius = 2;
        cfg.filter_params.engine = e.e;
        cfg.filter_params.quality = QualityPreset::Balanced;
        cfg.detail_params.detail_recovery = 0.15f;
        VideoDenoiser den(cfg);
        auto t0 = std::chrono::high_resolution_clock::now();
        VideoFrame32F out = den.process_window(noisy, &profile);
        auto t1 = std::chrono::high_resolution_clock::now();
        print_row(e.n, evaluate(out, noisy[ref], clean[ref], std::chrono::duration<double, std::milli>(t1 - t0).count()));
    }

#if defined(LIBRENOISE_HAS_CUDA) && (LIBRENOISE_HAS_CUDA != 0)
    if (is_cuda_available()) {
        const size_t count = static_cast<size_t>(W) * H * 4;
        const size_t bytes = count * sizeof(float);
        std::vector<float*> d_frames(kFrames, nullptr);
        std::vector<float> host(count);
        for (int f = 0; f < kFrames; ++f) {
            for (int y = 0; y < H; ++y)
                for (int x = 0; x < W; ++x) {
                    size_t i = (static_cast<size_t>(y) * W + x) * 4;
                    host[i + 0] = noisy[f].channels[0].at(x, y);
                    host[i + 1] = noisy[f].channels[1].at(x, y);
                    host[i + 2] = noisy[f].channels[2].at(x, y);
                    host[i + 3] = 1.0f;
                }
            cudaMalloc(&d_frames[f], bytes);
            cudaMemcpy(d_frames[f], host.data(), bytes, cudaMemcpyHostToDevice);
        }
        float* d_out = nullptr;
        cudaMalloc(&d_out, bytes);
        const float** d_arr = nullptr;
        cudaMalloc(&d_arr, kFrames * sizeof(float*));
        cudaMemcpy(d_arr, d_frames.data(), kFrames * sizeof(float*), cudaMemcpyHostToDevice);

        const char* gnames[] = {"GPU 3D Collaborative", "GPU 5-Band Wavelet", "GPU Non-Local Means", "GPU Frequency-Domain 3D"};
        for (int e = 0; e < 4; ++e) {
            CudaDenoiseParams cp;
            cp.width = W; cp.height = H; cp.stride_floats = W * 4; cp.channels = 4;
            cp.step_size = 4;
            cp.engine_choice = e;
            for (int c = 0; c < 3; ++c) {
                cp.nlf_a[c] = profile.nlf_per_channel[c].a;
                cp.nlf_b[c] = profile.nlf_per_channel[c].b;
                for (int b = 0; b < 5; ++b) cp.psd_weights[c][b] = profile.psd.octave_weights[c][b];
            }
            cp.detail_recovery = 0.15f;
            cuda_denoise_device(d_arr, kFrames, ref, d_out, cp, nullptr);
            cudaDeviceSynchronize();
            auto t0 = std::chrono::high_resolution_clock::now();
            cuda_denoise_device(d_arr, kFrames, ref, d_out, cp, nullptr);
            cudaDeviceSynchronize();
            auto t1 = std::chrono::high_resolution_clock::now();
            cudaMemcpy(host.data(), d_out, bytes, cudaMemcpyDeviceToHost);
            VideoFrame32F out(W, H, 3);
            for (int y = 0; y < H; ++y)
                for (int x = 0; x < W; ++x)
                    for (int c = 0; c < 3; ++c) out.channels[c].at(x, y) = host[(static_cast<size_t>(y) * W + x) * 4 + c];
            print_row(gnames[e], evaluate(out, noisy[ref], clean[ref], std::chrono::duration<double, std::milli>(t1 - t0).count()));
        }
        cudaFree(d_out);
        cudaFree(d_arr);
        for (auto p : d_frames) cudaFree(p);
        release_cuda_resources();
    }
#endif
    std::cout << "\n('noise left' = RMS error vs ground truth in flat patches divided by input RMS error; lower is better)\n";
    return 0;
}
