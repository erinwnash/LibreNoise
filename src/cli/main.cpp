// SPDX-License-Identifier: Apache-2.0
// LibreNoise v0.2 command-line renderer.
//
//   librenoise_cli --in E:/media/source/clip1.%04d.exr --out E:/out/clip1.%04d.exr --frames 1029 1031
//                  [--gpu] [--radius 2] [--type auto|film|digital|compressed] [--step 4]
//                  [--amount-y 1] [--amount-c 1] [--noise-level 1] [--sharpen 0] [--profile-frame N]
//                  [--model-out file] [--model-in file] [--quiet]
//
// Each output frame t is denoised with frames t-R..t+R (clamped to the given range; missing frames repeat the
// reference and are skipped as duplicates). The noise model is measured once from --profile-frame (default:
// the middle frame of the range) and reused for every frame, so strength does not change frame to frame.

#include "librenoise/librenoise.hpp"
#include "librenoise/v2/pipeline.hpp"

#include "tinyexr.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#if defined(LIBRENOISE_HAS_CUDA) && (LIBRENOISE_HAS_CUDA != 0)
#include <cuda_runtime.h>
#endif

using namespace librenoise::v2;

namespace {

struct Image {
    int w = 0, h = 0;
    std::vector<float> rgba;
};

std::string format_path(const std::string& pattern, int frame) {
    char buf[4096];
    std::snprintf(buf, sizeof(buf), pattern.c_str(), frame);
    return buf;
}

bool load_exr(const std::string& path, Image& img) {
    float* data = nullptr;
    const char* err = nullptr;
    if (LoadEXR(&data, &img.w, &img.h, path.c_str(), &err) != TINYEXR_SUCCESS) {
        std::cerr << "Failed to load " << path << ": " << (err ? err : "unknown") << "\n";
        if (err) FreeEXRErrorMessage(err);
        return false;
    }
    img.rgba.assign(data, data + static_cast<size_t>(img.w) * img.h * 4);
    std::free(data);
    return true;
}

bool save_exr(const std::string& path, const Image& img) {
    std::vector<float> rgb(static_cast<size_t>(img.w) * img.h * 3);
    for (size_t i = 0, n = static_cast<size_t>(img.w) * img.h; i < n; ++i)
        for (int c = 0; c < 3; ++c) rgb[i * 3 + c] = img.rgba[i * 4 + c];
    const char* err = nullptr;
    if (SaveEXR(rgb.data(), img.w, img.h, 3, 1, path.c_str(), &err) != TINYEXR_SUCCESS) {
        std::cerr << "Failed to save " << path << ": " << (err ? err : "unknown") << "\n";
        if (err) FreeEXRErrorMessage(err);
        return false;
    }
    return true;
}

SourceType parse_type(const std::string& s) {
    if (s == "film") return SourceType::Film;
    if (s == "digital") return SourceType::DigitalSensor;
    if (s == "compressed") return SourceType::CompressedVideo;
    return SourceType::Auto;
}

void usage() {
    std::cout << "LibreNoise " << LIBRENOISE_VERSION_STRING << " CLI\n"
              << "  --in <pattern.%04d.exr> --out <pattern.%04d.exr> --frames <first> <last>\n"
              << "  [--gpu] [--radius R] [--type auto|film|digital|compressed] [--step 2..6]\n"
              << "  [--amount-y A] [--amount-c A] [--noise-level L] [--sharpen S] [--flicker F]\n"
              << "  [--profile-frame N] [--model-out file] [--model-in file] [--only N] [--quiet]\n";
}

} // namespace

int main(int argc, char** argv) {
    std::string in_pat, out_pat, model_out, model_in;
    int first = 0, last = -1, profile_frame = -1000000, only = -1000000;
    bool gpu = false, quiet = false;
    Params P;
    ProfileOptions popt;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char* what) -> std::string {
            if (i + 1 >= argc) { std::cerr << "missing value for " << what << "\n"; std::exit(2); }
            return argv[++i];
        };
        if (a == "--in") in_pat = next("--in");
        else if (a == "--out") out_pat = next("--out");
        else if (a == "--frames") { first = std::atoi(next("--frames").c_str()); last = std::atoi(next("--frames").c_str()); }
        else if (a == "--gpu") gpu = true;
        else if (a == "--radius") P.temporal_radius = std::atoi(next("--radius").c_str());
        else if (a == "--type") { P.source_type = parse_type(next("--type")); popt.force_type = P.source_type; }
        else if (a == "--step") P.step = std::atoi(next("--step").c_str());
        else if (a == "--amount-y") P.amount_channel[0] = static_cast<float>(std::atof(next("--amount-y").c_str()));
        else if (a == "--amount-c") { P.amount_channel[1] = P.amount_channel[2] = static_cast<float>(std::atof(next("--amount-c").c_str())); }
        else if (a == "--noise-level") { const float v = static_cast<float>(std::atof(next("--noise-level").c_str())); for (float& n : P.noise_level) n = v; }
        else if (a == "--nl-band" || a == "--amt-band" || a == "--tonal") {
            // five comma-separated values: High, Mid, Low, VeryLow, UltraLow (tonal: shadows..highlights)
            float* dst = (a == "--nl-band") ? P.noise_level : (a == "--amt-band") ? P.amount_band : P.tonal;
            std::stringstream ss(next(a.c_str()));
            std::string tok;
            for (int b = 0; b < 5 && std::getline(ss, tok, ','); ++b) dst[b] = static_cast<float>(std::atof(tok.c_str()));
        }
        else if (a == "--temporal-strength") P.temporal_strength = static_cast<float>(std::atof(next("--temporal-strength").c_str()));
        else if (a == "--sharpen") P.sharpening = static_cast<float>(std::atof(next("--sharpen").c_str()));
        else if (a == "--flicker") P.flicker_reduction = static_cast<float>(std::atof(next("--flicker").c_str()));
        else if (a == "--artifacts") P.artifact_removal = static_cast<float>(std::atof(next("--artifacts").c_str()));
        else if (a == "--profile-frame") profile_frame = std::atoi(next("--profile-frame").c_str());
        else if (a == "--only") only = std::atoi(next("--only").c_str());
        else if (a == "--model-out") model_out = next("--model-out");
        else if (a == "--model-in") model_in = next("--model-in");
        else if (a == "--quiet") quiet = true;
        else if (a == "--help" || a == "-h") { usage(); return 0; }
        else { std::cerr << "unknown argument " << a << "\n"; usage(); return 2; }
    }
    if (in_pat.empty() || last < first) { usage(); return in_pat.empty() ? 0 : 2; }

    std::map<int, Image> cache;
    auto get = [&](int f) -> const Image* {
        auto it = cache.find(f);
        if (it != cache.end()) return &it->second;
        Image img;
        if (!load_exr(format_path(in_pat, f), img)) return nullptr;
        return &cache.emplace(f, std::move(img)).first->second;
    };

    NoiseModel model;
    if (!model_in.empty()) {
        std::ifstream mf(model_in);
        std::stringstream ss;
        ss << mf.rdbuf();
        if (!NoiseModel::deserialize(ss.str(), model)) { std::cerr << "bad model file\n"; return 1; }
    } else {
        const int pf = (profile_frame > -1000000) ? profile_frame : (first + last) / 2;
        const Image* im = get(pf);
        if (!im) return 1;
        const auto t0 = std::chrono::steady_clock::now();
        model = estimate_noise_model(FrameView{im->rgba.data(), im->w * 4, 4}, im->w, im->h, popt);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (!quiet) std::cout << "profile (" << pf << ", " << ms << " ms): " << model.summary() << "\n  " << model.assessment << "\n";
    }
    if (!model_out.empty()) { std::ofstream mf(model_out); mf << model.serialize(); }
    if (!model.valid) { std::cerr << "noise model invalid: " << model.assessment << "\n"; return 1; }
    if (out_pat.empty()) return 0;

    if (gpu && !cuda_available()) { std::cerr << "CUDA not available\n"; return 1; }

    const int R = std::max(0, P.temporal_radius);
    double total_ms = 0.0;
    int count = 0;
    for (int t = first; t <= last; ++t) {
        if (only > -1000000 && t != only) continue;
        std::vector<const Image*> win;
        for (int d = -R; d <= R; ++d) {
            const Image* im = get(std::clamp(t + d, first, last));
            if (!im) return 1;
            win.push_back(im);
        }
        const Image* refim = win[static_cast<size_t>(R)];
        const int w = refim->w, h = refim->h;
        Image out;
        out.w = w; out.h = h;
        out.rgba.assign(static_cast<size_t>(w) * h * 4, 0.0f);
        bool ok = false;
        const auto t0 = std::chrono::steady_clock::now();
        if (!gpu) {
            std::vector<FrameView> views;
            for (const Image* im : win) views.push_back(FrameView{im->rgba.data(), w * 4, 4});
            ok = denoise_cpu(views.data(), static_cast<int>(views.size()), R, w, h, model, P, OutputView{out.rgba.data(), w * 4, 4});
        } else {
#if defined(LIBRENOISE_HAS_CUDA) && (LIBRENOISE_HAS_CUDA != 0)
            const size_t bytes = static_cast<size_t>(w) * h * 4 * sizeof(float);
            std::vector<float*> dptr(win.size(), nullptr);
            std::vector<FrameView> views;
            for (size_t i = 0; i < win.size(); ++i) {
                cudaMalloc(&dptr[i], bytes);
                cudaMemcpy(dptr[i], win[i]->rgba.data(), bytes, cudaMemcpyHostToDevice);
                views.push_back(FrameView{dptr[i], w * 4, 4});
            }
            float* dout = nullptr;
            cudaMalloc(&dout, bytes);
            const auto tk = std::chrono::steady_clock::now();
            ok = denoise_cuda(views.data(), static_cast<int>(views.size()), R, w, h, model, P, OutputView{dout, w * 4, 4}, nullptr);
            cudaDeviceSynchronize();
            if (!quiet) std::cout << "  gpu processing " << std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tk).count() << " ms\n";
            cudaMemcpy(out.rgba.data(), dout, bytes, cudaMemcpyDeviceToHost);
            cudaFree(dout);
            for (float* p : dptr) cudaFree(p);
#endif
        }
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (!ok) { std::cerr << "denoise failed: " << last_error() << "\n"; return 1; }
        total_ms += ms;
        ++count;
        if (!quiet) std::cout << "frame " << t << ": " << ms << " ms\n";
        if (!save_exr(format_path(out_pat, t), out)) return 1;
        for (auto it = cache.begin(); it != cache.end();) {
            if (it->first < t + 1 - R) it = cache.erase(it); else ++it;
        }
    }
    if (count > 0) std::cout << "average " << (total_ms / count) << " ms/frame (" << (gpu ? "GPU" : "CPU") << ")\n";
    return 0;
}
