#include "librenoise/v2/model.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <sstream>

namespace librenoise::v2 {

const char* source_type_name(SourceType t) noexcept {
    switch (t) {
        case SourceType::Film: return "Film";
        case SourceType::DigitalSensor: return "Digital sensor";
        case SourceType::CompressedVideo: return "Compressed video";
        default: return "Auto";
    }
}

float model_sigma_at(const NoiseModel& m, int c, float x) noexcept {
    c = std::clamp(c, 0, 2);
    const auto& s = m.sigma[static_cast<size_t>(c)];
    const float lx = std::log2(std::max(x, 1e-12f));
    if (x <= 0.0f || lx <= kCurveLog2Min) return std::max(s[0], 1e-7f);
    if (lx >= kCurveLog2Max) {
        // continue with the fitted model's shape beyond the top knot
        const float p = std::clamp(m.fit_exponent[static_cast<size_t>(c)], 0.0f, 1.2f);
        return std::max(s[kCurveKnots - 1] * std::exp2(p * (lx - kCurveLog2Max)), 1e-7f);
    }
    const float t = (lx - kCurveLog2Min) / (kCurveLog2Max - kCurveLog2Min) * static_cast<float>(kCurveKnots - 1);
    const int i = std::clamp(static_cast<int>(t), 0, kCurveKnots - 2);
    const float fr = t - static_cast<float>(i);
    const float a = std::log2(std::max(s[static_cast<size_t>(i)], 1e-9f));
    const float b = std::log2(std::max(s[static_cast<size_t>(i) + 1], 1e-9f));
    return std::max(std::exp2(a + (b - a) * fr), 1e-7f);
}

void build_vst(const NoiseModel& m, VstLut& L) {
    L.log2x0 = kCurveLog2Min;
    L.log2x1 = kCurveLog2Max;
    L.x0 = std::exp2(L.log2x0);
    L.x1 = std::exp2(L.log2x1);
    const double du = static_cast<double>(L.log2x1 - L.log2x0) / static_cast<double>(kVstFwd - 1);
    for (int c = 0; c < 3; ++c) {
        L.sigma0[c] = model_sigma_at(m, c, L.x0);
        L.sigma1[c] = model_sigma_at(m, c, L.x1);
        double f = static_cast<double>(L.x0) / L.sigma0[c];
        L.f0[c] = static_cast<float>(f);
        L.fwd[c][0] = static_cast<float>(f);
        for (int i = 1; i < kVstFwd; ++i) {
            // integrate dx / sigma(x) over one log2 step with 4 midpoint sub-steps (dx = x ln2 du)
            const double u0 = L.log2x0 + du * (i - 1);
            double acc = 0.0;
            for (int s = 0; s < 4; ++s) {
                const double u = u0 + du * (s + 0.5) / 4.0;
                const double x = std::exp2(u);
                acc += x * 0.69314718056 * (du / 4.0) / model_sigma_at(m, c, static_cast<float>(x));
            }
            f += acc;
            L.fwd[c][i] = static_cast<float>(f);
        }
        L.f1[c] = L.fwd[c][kVstFwd - 1];
        // inverse table: log2(x) sampled uniformly in f
        int i = 0;
        for (int j = 0; j < kVstInv; ++j) {
            const double y = L.f0[c] + (static_cast<double>(L.f1[c]) - L.f0[c]) * j / (kVstInv - 1);
            while (i < kVstFwd - 2 && L.fwd[c][i + 1] < y) ++i;
            const double a = L.fwd[c][i], b = L.fwd[c][i + 1];
            const double fr = (b > a) ? std::clamp((y - a) / (b - a), 0.0, 1.0) : 0.0;
            L.inv[c][j] = static_cast<float>(L.log2x0 + du * (i + fr));
        }
    }
}

std::string NoiseModel::serialize() const {
    std::ostringstream o;
    o.precision(7);
    o << "LNM2\n";
    o << "valid " << (valid ? 1 : 0) << "\n";
    o << "type " << static_cast<int>(detected_type) << "\n";
    o << "film " << film_score << "\n";
    o << "comp " << compression << " " << grid_phase_x << " " << grid_phase_y << " " << grid_excess_x << " " << grid_excess_y << "\n";
    o << "quality " << quality << " " << windows_used << "\n";
    o << "sample " << sample_x << " " << sample_y << " " << sample_w << " " << sample_h << "\n";
    for (int c = 0; c < 3; ++c) o << "fit " << c << " " << fit_floor[c] << " " << fit_gain[c] << " " << fit_exponent[c] << "\n";
    o << "corr " << corr[0] << " " << corr[1] << " " << corr[2] << "\n";
    o << "pix " << pixel_sigma[0] << " " << pixel_sigma[1] << " " << pixel_sigma[2] << "\n";
    for (int c = 0; c < 3; ++c) {
        o << "sigma " << c;
        for (int k = 0; k < kCurveKnots; ++k) o << " " << sigma[c][k];
        o << "\nmeas " << c << " ";
        for (int k = 0; k < kCurveKnots; ++k) o << (measured[c][k] ? '1' : '0');
        o << "\n";
    }
    for (int l = 0; l < kLevels; ++l)
        for (int c = 0; c < 3; ++c) {
            o << "lsig " << l << " " << c;
            for (int q = 0; q < kCoefs; ++q) o << " " << level_sigma[l][c][q];
            o << "\n";
        }
    for (int c = 0; c < 3; ++c) {
        o << "psd " << c;
        for (float v : radial_psd[c]) o << " " << v;
        o << "\n";
    }
    std::string a = assessment;
    std::replace(a.begin(), a.end(), '\n', ' ');
    o << "assess " << a << "\n";
    return o.str();
}

bool NoiseModel::deserialize(const std::string& text, NoiseModel& m) {
    std::istringstream in(text);
    std::string line;
    if (!std::getline(in, line) || line.rfind("LNM2", 0) != 0) return false;
    NoiseModel r;
    while (std::getline(in, line)) {
        std::istringstream ls(line);
        std::string key;
        ls >> key;
        if (key == "valid") { int v = 0; ls >> v; r.valid = v != 0; }
        else if (key == "type") { int v = 1; ls >> v; r.detected_type = static_cast<SourceType>(std::clamp(v, 1, 3)); }
        else if (key == "film") ls >> r.film_score;
        else if (key == "comp") ls >> r.compression >> r.grid_phase_x >> r.grid_phase_y >> r.grid_excess_x >> r.grid_excess_y;
        else if (key == "quality") ls >> r.quality >> r.windows_used;
        else if (key == "sample") ls >> r.sample_x >> r.sample_y >> r.sample_w >> r.sample_h;
        else if (key == "fit") { int c = 0; ls >> c; c = std::clamp(c, 0, 2); ls >> r.fit_floor[c] >> r.fit_gain[c] >> r.fit_exponent[c]; }
        else if (key == "corr") ls >> r.corr[0] >> r.corr[1] >> r.corr[2];
        else if (key == "pix") ls >> r.pixel_sigma[0] >> r.pixel_sigma[1] >> r.pixel_sigma[2];
        else if (key == "sigma") { int c = 0; ls >> c; c = std::clamp(c, 0, 2); for (int k = 0; k < kCurveKnots; ++k) ls >> r.sigma[c][k]; }
        else if (key == "meas") {
            int c = 0; std::string bits; ls >> c >> bits; c = std::clamp(c, 0, 2);
            for (int k = 0; k < kCurveKnots && k < static_cast<int>(bits.size()); ++k) r.measured[c][k] = bits[k] == '1';
        }
        else if (key == "lsig") {
            int l = 0, c = 0; ls >> l >> c;
            if (l < 0 || l >= kLevels || c < 0 || c > 2) continue;
            for (int q = 0; q < kCoefs; ++q) ls >> r.level_sigma[l][c][q];
        }
        else if (key == "psd") { int c = 0; ls >> c; c = std::clamp(c, 0, 2); for (auto& v : r.radial_psd[c]) ls >> v; }
        else if (key == "assess") { std::getline(ls, r.assessment); if (!r.assessment.empty() && r.assessment[0] == ' ') r.assessment.erase(0, 1); }
    }
    if (!in.eof() && in.fail()) return false;
    m = r;
    return true;
}

std::string NoiseModel::summary() const {
    char buf[256];
    std::snprintf(buf, sizeof(buf), "%s | noise exponent %.2f/%.2f/%.2f | channel corr %.2f | compression %.0f%% | quality %.0f%%",
                  source_type_name(detected_type), fit_exponent[0], fit_exponent[1], fit_exponent[2],
                  (std::abs(corr[0]) + std::abs(corr[1]) + std::abs(corr[2])) / 3.0f, compression * 100.0f, quality);
    return buf;
}

} // namespace librenoise::v2
