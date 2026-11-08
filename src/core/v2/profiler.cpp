// LibreNoise v0.2 noise profiler: measures the noise model from a single frame. Pure algorithm.
//
//  1. Brightness curve. The frame is cut into 16x16 blocks; for each channel the high-pass std of every
//     valid block is binned by the block's mean value on a half-stop grid. In each bin the lowest 10% of
//     block sigmas (the flattest blocks) form the lower envelope. A parametric curve
//     sigma^2 = floor^2 + (gain * x^p)^2 is fitted robustly (texture can only push points up, so points
//     above the fit count less), and measured knots then correct the fit locally. Only the curve *shape*
//     matters: the absolute level is measured afterwards in the stabilised domain.
//  2. Spectrum. With the curve the frame is variance-stabilised and converted to Y/Cb/Cr. Flat 64x64
//     windows (lowest variance; or inside the user's box) give a 2D noise power spectrum per channel.
//  3. Per-level noise table. A large synthetic noise field with exactly that spectrum is run through the
//     same pyramid + 8x8 DCT as the filter, giving the noise std of every coefficient at every level.
//  4. Detection: cross-channel noise correlation and spectral shape (film vs demosaiced sensor) and an
//     8x8 block-grid detector (compressed video).

#include "librenoise/v2/pipeline.hpp"
#include "kernels.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <memory>
#include <numeric>
#include <random>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace librenoise::v2 {

namespace {

using cfloat = std::complex<float>;

void fft1d(cfloat* a, int n, bool inverse) {
    for (int i = 1, j = 0; i < n; ++i) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }
    for (int len = 2; len <= n; len <<= 1) {
        const double ang = 2.0 * 3.14159265358979323846 / len * (inverse ? 1.0 : -1.0);
        const cfloat wl(static_cast<float>(std::cos(ang)), static_cast<float>(std::sin(ang)));
        for (int i = 0; i < n; i += len) {
            cfloat w(1.0f, 0.0f);
            for (int j = 0; j < len / 2; ++j) {
                const cfloat u = a[i + j], v = a[i + j + len / 2] * w;
                a[i + j] = u + v;
                a[i + j + len / 2] = u - v;
                w *= wl;
            }
        }
    }
    if (inverse) {
        const float s = 1.0f / static_cast<float>(n);
        for (int i = 0; i < n; ++i) a[i] *= s;
    }
}

void fft2d(std::vector<cfloat>& a, int n, bool inverse) {
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < n; ++r) fft1d(a.data() + static_cast<size_t>(r) * n, n, inverse);
    #pragma omp parallel
    {
        std::vector<cfloat> col(static_cast<size_t>(n));
        #pragma omp for schedule(static)
        for (int c = 0; c < n; ++c) {
            for (int r = 0; r < n; ++r) col[r] = a[static_cast<size_t>(r) * n + c];
            fft1d(col.data(), n, inverse);
            for (int r = 0; r < n; ++r) a[static_cast<size_t>(r) * n + c] = col[r];
        }
    }
}

void gaussian_blur(const std::vector<float>& src, std::vector<float>& dst, int w, int h, float sigma) {
    const int r = std::max(1, static_cast<int>(std::ceil(sigma * 3.0f)));
    std::vector<float> kern(static_cast<size_t>(2 * r + 1));
    float s = 0.0f;
    for (int i = -r; i <= r; ++i) { kern[static_cast<size_t>(i + r)] = std::exp(-0.5f * i * i / (sigma * sigma)); s += kern[static_cast<size_t>(i + r)]; }
    for (auto& v : kern) v /= s;
    std::vector<float> tmp(src.size());
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            float a = 0.0f;
            for (int i = -r; i <= r; ++i) a += kern[static_cast<size_t>(i + r)] * src[static_cast<size_t>(y) * w + std::clamp(x + i, 0, w - 1)];
            tmp[static_cast<size_t>(y) * w + x] = a;
        }
    dst.resize(src.size());
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            float a = 0.0f;
            for (int i = -r; i <= r; ++i) a += kern[static_cast<size_t>(i + r)] * tmp[static_cast<size_t>(std::clamp(y + i, 0, h - 1)) * w + x];
            dst[static_cast<size_t>(y) * w + x] = a;
        }
}

struct CurveFit { float floor2 = 0, gain = 0, p = 0.5f, err = 1e30f; };

float fit_sigma(const CurveFit& f, float x) {
    const float s = f.gain * std::pow(std::max(x, 1e-9f), f.p);
    return std::sqrt(f.floor2 + s * s);
}

// Robust fit of sigma^2 = floor^2 + (gain x^p)^2 to (x_i, s_i, weight_i) points.
CurveFit fit_curve(const std::vector<float>& xs, const std::vector<float>& ss, const std::vector<float>& ws) {
    CurveFit best;
    if (xs.size() < 2) {
        if (!xs.empty()) { best.floor2 = 0.0f; best.p = 0.75f; best.gain = ss[0] / std::pow(xs[0], 0.75f); best.err = 0.0f; }
        return best;
    }
    const float smin = *std::min_element(ss.begin(), ss.end());
    for (float p = 0.25f; p <= 1.2501f; p += 0.025f) {
        for (int fi = 0; fi <= 24; ++fi) {
            const float floor2 = (fi == 0) ? 0.0f : smin * smin * std::pow(2.0f, static_cast<float>(fi - 20) * 0.5f);
            // gain from weighted log-median of the points above the floor
            std::vector<std::pair<float, float>> g;
            for (size_t i = 0; i < xs.size(); ++i) {
                const float r = ss[i] * ss[i] - floor2;
                if (r > 0.0f) g.emplace_back(std::log(std::sqrt(r) / std::pow(xs[i], p)), ws[i]);
            }
            if (g.empty()) continue;
            std::sort(g.begin(), g.end());
            float tot = 0.0f;
            for (auto& e : g) tot += e.second;
            float run = 0.0f, lg = g[0].first;
            for (auto& e : g) { run += e.second; if (run >= 0.5f * tot) { lg = e.first; break; } }
            CurveFit c{floor2, std::exp(lg), p, 0.0f};
            float err = 0.0f, wsum = 0.0f;
            for (size_t i = 0; i < xs.size(); ++i) {
                const float d = std::log(ss[i]) - std::log(fit_sigma(c, xs[i]));
                const float a = std::abs(d);
                const float hub = a < 0.2f ? 0.5f * a * a : 0.2f * (a - 0.1f);
                err += ws[i] * hub * (d > 0.0f ? 0.5f : 1.0f);   // above-fit points may be texture
                wsum += ws[i];
            }
            c.err = err / std::max(wsum, 1e-6f);
            if (c.err < best.err) best = c;
        }
    }
    return best;
}

} // namespace

NoiseModel estimate_noise_model(const FrameView& frame, int32_t w, int32_t h, const ProfileOptions& opt) {
    NoiseModel M;
    if (frame.data == nullptr || w < 64 || h < 64) {
        M.assessment = "Frame too small to profile";
        return M;
    }
    const size_t N = static_cast<size_t>(w) * h;
    std::vector<float> rgb[3];
    for (auto& p : rgb) p.resize(N);
    float cmax[3] = {-1e30f, -1e30f, -1e30f};
    for (int y = 0; y < h; ++y) {
        const float* row = frame.data + static_cast<size_t>(y) * frame.stride_floats;
        for (int x = 0; x < w; ++x)
            for (int c = 0; c < 3; ++c) {
                float v = row[static_cast<size_t>(x) * frame.channels + c];
                if (!std::isfinite(v)) v = 0.0f;
                rgb[c][static_cast<size_t>(y) * w + x] = v;
                cmax[c] = std::max(cmax[c], v);
            }
    }

    // ---------------------------------------------------------------- 1. brightness curve
    constexpr int BS = 16;
    const int nbx = w / BS, nby = h / BS;
    std::vector<float> hp[3];
    for (int c = 0; c < 3; ++c) {
        std::vector<float> bl;
        gaussian_blur(rgb[c], bl, w, h, 1.5f);
        hp[c].resize(N);
        for (size_t i = 0; i < N; ++i) hp[c][i] = rgb[c][i] - bl[i];
    }
    struct BlockStat { float mean[3]; float sd[3]; bool ok; };
    std::vector<BlockStat> blocks(static_cast<size_t>(nbx) * nby);
    #pragma omp parallel for schedule(static)
    for (int by = 0; by < nby; ++by)
        for (int bx = 0; bx < nbx; ++bx) {
            BlockStat& b = blocks[static_cast<size_t>(by) * nbx + bx];
            b.ok = true;
            for (int c = 0; c < 3; ++c) {
                double s = 0, s2 = 0, hs = 0, hs2 = 0;
                int clip = 0;
                float mn = 1e30f, mx = -1e30f;
                for (int j = 0; j < BS; ++j)
                    for (int i = 0; i < BS; ++i) {
                        const size_t o = static_cast<size_t>(by * BS + j) * w + (bx * BS + i);
                        const float v = rgb[c][o];
                        s += v; s2 += static_cast<double>(v) * v;
                        hs += hp[c][o]; hs2 += static_cast<double>(hp[c][o]) * hp[c][o];
                        mn = std::min(mn, v); mx = std::max(mx, v);
                        if (v >= cmax[c] * 0.999f) ++clip;
                    }
                const double n = BS * BS;
                b.mean[c] = static_cast<float>(s / n);
                b.sd[c] = static_cast<float>(std::sqrt(std::max(0.0, hs2 / n - (hs / n) * (hs / n))));
                if (mx - mn <= 1e-7f || b.sd[c] <= 1e-8f || clip > BS * BS / 8) b.ok = false;
            }
        }
    std::array<CurveFit, 3> fits{};
    int curve_points = 0;
    for (int c = 0; c < 3; ++c) {
        std::vector<std::vector<float>> bins(kCurveKnots);
        for (const auto& b : blocks) {
            if (!b.ok || b.mean[c] <= 0.0f) continue;
            const float l = std::log2(b.mean[c]);
            const float t = (l - kCurveLog2Min) / (kCurveLog2Max - kCurveLog2Min) * (kCurveKnots - 1);
            const int k = static_cast<int>(std::lround(t));
            if (k < 0 || k >= kCurveKnots) continue;
            bins[static_cast<size_t>(k)].push_back(b.sd[c]);
        }
        std::vector<float> xs, ss, ws;
        std::array<float, kCurveKnots> meas{};
        for (int k = 0; k < kCurveKnots; ++k) {
            auto& v = bins[static_cast<size_t>(k)];
            if (v.size() < 24) continue;
            std::sort(v.begin(), v.end());
            const size_t take = std::max<size_t>(5, v.size() / 10);
            double s = 0;
            for (size_t i = 0; i < take; ++i) s += v[i];
            meas[static_cast<size_t>(k)] = static_cast<float>(s / take);
            M.measured[c][k] = 1;
            xs.push_back(curve_knot_x(k));
            ss.push_back(meas[static_cast<size_t>(k)]);
            ws.push_back(std::sqrt(static_cast<float>(v.size())));
        }
        curve_points += static_cast<int>(xs.size());
        if (xs.empty()) {   // no usable blocks: flat curve
            for (int k = 0; k < kCurveKnots; ++k) M.sigma[c][k] = 0.01f;
            fits[c] = CurveFit{1e-4f, 0.0f, 0.5f, 0.0f};
            continue;
        }
        fits[c] = fit_curve(xs, ss, ws);
        M.fit_floor[c] = std::sqrt(fits[c].floor2);
        M.fit_gain[c] = fits[c].gain;
        M.fit_exponent[c] = fits[c].p;
        // local correction of the fit by the measured knots, smoothed
        std::array<float, kCurveKnots> lr{};
        for (int k = 0; k < kCurveKnots; ++k)
            if (M.measured[c][k]) lr[static_cast<size_t>(k)] = std::clamp(std::log(meas[static_cast<size_t>(k)] / fit_sigma(fits[c], curve_knot_x(k))), -0.35f, 0.22f);
        for (int k = 0; k < kCurveKnots; ++k) {
            float a = 0.0f, wsum = 0.0f;
            for (int d = -1; d <= 1; ++d) {
                const int q = k + d;
                if (q < 0 || q >= kCurveKnots || !M.measured[c][q]) continue;
                const float wt = (d == 0) ? 2.0f : 1.0f;
                a += wt * lr[static_cast<size_t>(q)]; wsum += wt;
            }
            const float corr = (wsum > 0.0f && M.measured[c][k]) ? a / wsum : 0.0f;
            M.sigma[c][k] = std::max(fit_sigma(fits[c], curve_knot_x(k)) * std::exp(corr), 1e-7f);
        }
    }
    hp[0].clear(); hp[1].clear(); hp[2].clear();
    M.valid = true;   // provisional; needed to build the VST

    // ---------------------------------------------------------------- 2. stabilised working planes
    auto lut = std::make_unique<VstLut>();
    build_vst(M, *lut);
    std::vector<float> wk[3], fv[3];
    for (int c = 0; c < 3; ++c) { wk[c].resize(N); fv[c].resize(N); }
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < static_cast<int64_t>(N); ++i) {
        float Y, Cb, Cr;
        k::rgb_to_work(*lut, rgb[0][i], rgb[1][i], rgb[2][i], Y, Cb, Cr);
        wk[0][i] = Y; wk[1][i] = Cb; wk[2][i] = Cr;
        for (int c = 0; c < 3; ++c) fv[c][i] = k::vst_fwd(*lut, c, rgb[c][i]);
    }

    // ---------------------------------------------------------------- compression grid (on stabilised luma)
    {
        // rows / columns that carry signal (skip letterbox)
        std::vector<uint8_t> row_ok(static_cast<size_t>(h), 0), col_ok(static_cast<size_t>(w), 0);
        for (int y = 0; y < h; ++y) {
            double s = 0, s2 = 0;
            for (int x = 0; x < w; x += 4) { const float v = wk[0][static_cast<size_t>(y) * w + x]; s += v; s2 += v * v; }
            const double n = (w + 3) / 4;
            row_ok[static_cast<size_t>(y)] = (s2 / n - (s / n) * (s / n)) > 1e-3;
        }
        for (int x = 0; x < w; ++x) {
            double s = 0, s2 = 0;
            for (int y = 0; y < h; y += 4) { const float v = wk[0][static_cast<size_t>(y) * w + x]; s += v; s2 += v * v; }
            const double n = (h + 3) / 4;
            col_ok[static_cast<size_t>(x)] = (s2 / n - (s / n) * (s / n)) > 1e-3;
        }
        double px[8] = {0}, py[8] = {0};
        long long cx[8] = {0}, cy[8] = {0};
        for (int y = 0; y < h; ++y) {
            if (!row_ok[static_cast<size_t>(y)]) continue;
            const float* r = wk[0].data() + static_cast<size_t>(y) * w;
            for (int x = 0; x + 1 < w; ++x) {
                const float d = std::abs(r[x + 1] - r[x]);
                if (d > 6.0f) continue;   // ignore strong edges
                px[x % 8] += d; ++cx[x % 8];
            }
        }
        for (int y = 0; y + 1 < h; ++y) {
            if (!row_ok[static_cast<size_t>(y)] || !row_ok[static_cast<size_t>(y) + 1]) continue;
            const float* r0 = wk[0].data() + static_cast<size_t>(y) * w;
            const float* r1 = r0 + w;
            for (int x = 0; x < w; ++x) {
                if (!col_ok[static_cast<size_t>(x)]) continue;
                const float d = std::abs(r1[x] - r0[x]);
                if (d > 6.0f) continue;
                py[y % 8] += d; ++cy[y % 8];
            }
        }
        auto analyse = [](const double* s, const long long* n, int& phase, float& excess) {
            float m[8];
            for (int p = 0; p < 8; ++p) m[p] = n[p] > 0 ? static_cast<float>(s[p] / n[p]) : 0.0f;
            float sorted[8];
            std::copy(m, m + 8, sorted);
            std::sort(sorted, sorted + 8);
            const float med = 0.5f * (sorted[3] + sorted[4]);
            const int best = static_cast<int>(std::max_element(m, m + 8) - m);
            const float second = sorted[6];
            excess = med > 0.0f ? m[best] / med - 1.0f : 0.0f;
            phase = (excess > 0.03f && m[best] > second * 1.02f) ? best : -1;
        };
        analyse(px, cx, M.grid_phase_x, M.grid_excess_x);
        analyse(py, cy, M.grid_phase_y, M.grid_excess_y);
        const float ex = M.grid_phase_x >= 0 ? M.grid_excess_x : 0.0f;
        const float ey = M.grid_phase_y >= 0 ? M.grid_excess_y : 0.0f;
        M.compression = std::clamp((std::max(ex, ey) - 0.02f) / 0.10f, 0.0f, 1.0f);
    }

    // ---------------------------------------------------------------- 3. flat windows -> noise spectrum
    int WS = 64;
    int rx0 = 0, ry0 = 0, rx1 = w, ry1 = h;
    const bool user_roi = opt.roi_w >= 32 && opt.roi_h >= 32;
    if (user_roi) {
        rx0 = std::clamp(opt.roi_x, 0, w - 32); ry0 = std::clamp(opt.roi_y, 0, h - 32);
        rx1 = std::clamp(opt.roi_x + opt.roi_w, rx0 + 32, w); ry1 = std::clamp(opt.roi_y + opt.roi_h, ry0 + 32, h);
        if (rx1 - rx0 < 64 || ry1 - ry0 < 64) WS = 32;
    }
    const int wstep = WS / 2;
    struct Win { int x, y; float score; };
    std::vector<Win> wins;
    for (int y = ry0; y + WS <= ry1; y += wstep)
        for (int x = rx0; x + WS <= rx1; x += wstep) {
            // reject windows touching invalid blocks
            bool ok = true;
            for (int by = y / BS; by <= std::min(nby - 1, (y + WS - 1) / BS) && ok; ++by)
                for (int bx = x / BS; bx <= std::min(nbx - 1, (x + WS - 1) / BS); ++bx)
                    if (!blocks[static_cast<size_t>(by) * nbx + bx].ok) { ok = false; break; }
            if (!ok) continue;
            double s = 0, s2 = 0;
            for (int j = 0; j < WS; ++j)
                for (int i = 0; i < WS; ++i) { const float v = wk[0][static_cast<size_t>(y + j) * w + x + i]; s += v; s2 += v * v; }
            const double n = static_cast<double>(WS) * WS;
            wins.push_back({x, y, static_cast<float>(s2 / n - (s / n) * (s / n))});
        }
    if (wins.empty()) {
        M.valid = false;
        M.assessment = "No usable area found (frame clipped or blank)";
        return M;
    }
    std::sort(wins.begin(), wins.end(), [](const Win& a, const Win& b) { return a.score < b.score; });
    size_t nsel = user_roi ? wins.size() : std::clamp<size_t>(wins.size() / 10, std::min<size_t>(12, wins.size()), 200);
    nsel = std::min(nsel, wins.size());
    M.windows_used = static_cast<int32_t>(nsel);
    M.sample_x = wins[0].x; M.sample_y = wins[0].y; M.sample_w = WS; M.sample_h = WS;

    std::vector<float> hann(static_cast<size_t>(WS));
    for (int i = 0; i < WS; ++i) hann[static_cast<size_t>(i)] = 0.5f - 0.5f * std::cos(2.0f * 3.14159265f * (i + 0.5f) / WS);
    double hann2 = 0.0;
    for (int j = 0; j < WS; ++j) for (int i = 0; i < WS; ++i) hann2 += static_cast<double>(hann[j] * hann[i]) * (hann[j] * hann[i]);
    std::vector<double> psd[3];
    for (auto& p : psd) p.assign(static_cast<size_t>(WS) * WS, 0.0);
    double cc[3] = {0, 0, 0};
    int ccn = 0;
    for (size_t wi = 0; wi < nsel; ++wi) {
        const Win& win = wins[wi];
        for (int c = 0; c < 3; ++c) {
            // plane detrend
            double s = 0, sx = 0, sy = 0;
            const double cen = 0.5 * (WS - 1);
            for (int j = 0; j < WS; ++j) for (int i = 0; i < WS; ++i) {
                const double v = wk[c][static_cast<size_t>(win.y + j) * w + win.x + i];
                s += v; sx += v * (i - cen); sy += v * (j - cen);
            }
            const double n = static_cast<double>(WS) * WS;
            double sxx = 0;
            for (int i = 0; i < WS; ++i) sxx += (i - cen) * (i - cen);
            sxx *= WS;
            const double mean = s / n, gx = sx / sxx, gy = sy / sxx;
            std::vector<cfloat> a(static_cast<size_t>(WS) * WS);
            for (int j = 0; j < WS; ++j) for (int i = 0; i < WS; ++i) {
                const double v = wk[c][static_cast<size_t>(win.y + j) * w + win.x + i] - mean - gx * (i - cen) - gy * (j - cen);
                a[static_cast<size_t>(j) * WS + i] = cfloat(static_cast<float>(v * hann[j] * hann[i]), 0.0f);
            }
            for (int r = 0; r < WS; ++r) fft1d(a.data() + static_cast<size_t>(r) * WS, WS, false);
            std::vector<cfloat> col(static_cast<size_t>(WS));
            for (int cI = 0; cI < WS; ++cI) {
                for (int r = 0; r < WS; ++r) col[r] = a[static_cast<size_t>(r) * WS + cI];
                fft1d(col.data(), WS, false);
                for (int r = 0; r < WS; ++r) a[static_cast<size_t>(r) * WS + cI] = col[r];
            }
            for (size_t q = 0; q < a.size(); ++q) psd[c][q] += std::norm(a[q]) / hann2;
        }
        // cross-channel correlation of high-passed stabilised R, G, B
        double sab[3] = {0, 0, 0}, saa[3] = {0, 0, 0};
        for (int j = 1; j < WS - 1; ++j) for (int i = 1; i < WS - 1; ++i) {
            float v[3];
            for (int c = 0; c < 3; ++c) {
                const float* p = fv[c].data() + static_cast<size_t>(win.y + j) * w + win.x + i;
                float box = 0.0f;
                for (int dj = -1; dj <= 1; ++dj) for (int di = -1; di <= 1; ++di) box += p[static_cast<ptrdiff_t>(dj) * w + di];
                v[c] = p[0] - box / 9.0f;
            }
            for (int c = 0; c < 3; ++c) saa[c] += static_cast<double>(v[c]) * v[c];
            sab[0] += static_cast<double>(v[0]) * v[1];
            sab[1] += static_cast<double>(v[1]) * v[2];
            sab[2] += static_cast<double>(v[0]) * v[2];
        }
        cc[0] += sab[0] / std::sqrt(saa[0] * saa[1] + 1e-20);
        cc[1] += sab[1] / std::sqrt(saa[1] * saa[2] + 1e-20);
        cc[2] += sab[2] / std::sqrt(saa[0] * saa[2] + 1e-20);
        ++ccn;
    }
    for (int i = 0; i < 3; ++i) M.corr[static_cast<size_t>(i)] = static_cast<float>(cc[i] / std::max(1, ccn));
    for (auto& p : psd) for (auto& v : p) v /= static_cast<double>(nsel);

    // radial profile; replace the detrend-damaged lowest ring with the next one (noise is white at low frequency)
    auto ring_of = [&](int qy, int qx) {
        const int fy = qy <= WS / 2 ? qy : qy - WS, fx = qx <= WS / 2 ? qx : qx - WS;
        return std::sqrt(static_cast<float>(fx * fx + fy * fy)) / static_cast<float>(WS);   // cycles / pixel
    };
    for (int c = 0; c < 3; ++c) {
        double low = 0; int lown = 0;
        for (int qy = 0; qy < WS; ++qy) for (int qx = 0; qx < WS; ++qx) {
            const float f = ring_of(qy, qx);
            if (f >= 1.5f / WS && f < 3.0f / WS) { low += psd[c][static_cast<size_t>(qy) * WS + qx]; ++lown; }
        }
        low /= std::max(1, lown);
        for (int qy = 0; qy < WS; ++qy) for (int qx = 0; qx < WS; ++qx)
            if (ring_of(qy, qx) < 1.5f / WS) psd[c][static_cast<size_t>(qy) * WS + qx] = low;
        std::array<double, 16> rs{}; std::array<int, 16> rn{};
        for (int qy = 0; qy < WS; ++qy) for (int qx = 0; qx < WS; ++qx) {
            const int b = std::min(15, static_cast<int>(ring_of(qy, qx) / 0.5f * 16.0f));
            rs[static_cast<size_t>(b)] += psd[c][static_cast<size_t>(qy) * WS + qx]; ++rn[static_cast<size_t>(b)];
        }
        for (int b = 0; b < 16; ++b) M.radial_psd[c][static_cast<size_t>(b)] = static_cast<float>(rs[static_cast<size_t>(b)] / std::max(1, rn[static_cast<size_t>(b)]));
    }

    // ---------------------------------------------------------------- 4. synthetic field -> per-level noise table
    constexpr int SN = 1024;
    for (int c = 0; c < 3; ++c) {
        std::mt19937 rng(12345u + static_cast<unsigned>(c));
        std::normal_distribution<float> nd(0.0f, 1.0f);
        std::vector<cfloat> f(static_cast<size_t>(SN) * SN);
        for (auto& v : f) v = cfloat(nd(rng), 0.0f);
        fft2d(f, SN, false);
        // sqrt(PSD) interpolated from the WS x WS grid (periodic bilinear in frequency)
        for (int qy = 0; qy < SN; ++qy) {
            const float fy = static_cast<float>(qy <= SN / 2 ? qy : qy - SN) / SN;
            for (int qx = 0; qx < SN; ++qx) {
                const float fx = static_cast<float>(qx <= SN / 2 ? qx : qx - SN) / SN;
                float gx = fx * WS, gy = fy * WS;
                const float rad = std::sqrt(fx * fx + fy * fy);
                double val;
                if (rad < 1.5f / WS) {
                    val = psd[c][0];
                } else {
                    if (gx < 0) gx += WS;
                    if (gy < 0) gy += WS;
                    const int x0 = static_cast<int>(std::floor(gx)) % WS, y0 = static_cast<int>(std::floor(gy)) % WS;
                    const int x1 = (x0 + 1) % WS, y1 = (y0 + 1) % WS;
                    const float ax = gx - std::floor(gx), ay = gy - std::floor(gy);
                    val = (psd[c][static_cast<size_t>(y0) * WS + x0] * (1 - ax) + psd[c][static_cast<size_t>(y0) * WS + x1] * ax) * (1 - ay) +
                          (psd[c][static_cast<size_t>(y1) * WS + x0] * (1 - ax) + psd[c][static_cast<size_t>(y1) * WS + x1] * ax) * ay;
                }
                f[static_cast<size_t>(qy) * SN + qx] *= static_cast<float>(std::sqrt(std::max(val, 0.0)));
            }
        }
        fft2d(f, SN, true);
        std::vector<float> g0(static_cast<size_t>(SN) * SN);
        double s2 = 0;
        for (size_t i = 0; i < g0.size(); ++i) { g0[i] = f[i].real(); s2 += static_cast<double>(g0[i]) * g0[i]; }
        M.pixel_sigma[static_cast<size_t>(c)] = static_cast<float>(std::sqrt(s2 / g0.size()));
        // pyramid + DCT statistics (same operators as the filter)
        std::vector<float> cur = g0;
        int cw = SN, ch = SN;
        for (int l = 0; l < kLevels; ++l) {
            const int nw = (cw + 1) / 2, nh = (ch + 1) / 2;
            std::vector<float> small(static_cast<size_t>(nw) * nh), band(static_cast<size_t>(cw) * ch);
            #pragma omp parallel for schedule(static)
            for (int y = 0; y < nh; ++y) for (int x = 0; x < nw; ++x) small[static_cast<size_t>(y) * nw + x] = k::pyr_down(cur.data(), cw, ch, x, y);
            #pragma omp parallel for schedule(static)
            for (int y = 0; y < ch; ++y) for (int x = 0; x < cw; ++x)
                band[static_cast<size_t>(y) * cw + x] = cur[static_cast<size_t>(y) * cw + x] - k::pyr_up(small.data(), nw, nh, x, y);
            double acc[kCoefs] = {0};
            long long cnt = 0;
            const int margin = 8;
            for (int by = margin; by + 8 <= ch - margin; by += 4)
                for (int bx = margin; bx + 8 <= cw - margin; bx += 4) {
                    float blk[64], co[64];
                    for (int j = 0; j < 8; ++j) for (int i = 0; i < 8; ++i) blk[j * 8 + i] = band[static_cast<size_t>(by + j) * cw + bx + i];
                    k::dct8x8(blk, co);
                    for (int q = 0; q < 64; ++q) acc[q] += static_cast<double>(co[q]) * co[q];
                    ++cnt;
                }
            for (int q = 0; q < 64; ++q)
                M.level_sigma[l][c][q] = cnt > 0 ? static_cast<float>(std::sqrt(acc[q] / cnt)) : 0.0f;
            cur.swap(small);
            cw = nw; ch = nh;
        }
    }

    // ---------------------------------------------------------------- 5. classification and quality
    const float mean_corr = (std::abs(M.corr[0]) + std::abs(M.corr[1]) + std::abs(M.corr[2])) / 3.0f;
    const float max_corr = std::max({std::abs(M.corr[0]), std::abs(M.corr[1]), std::abs(M.corr[2])});
    // Film grain: dye layers are independent, so noise is nearly uncorrelated across R, G, B.
    // Demosaiced / matrixed sensor noise is strongly correlated in at least one channel pair.
    M.film_score = std::clamp(1.0f - (std::max(mean_corr, 0.7f * max_corr) - 0.18f) / 0.25f, 0.0f, 1.0f);
    if (opt.force_type != SourceType::Auto) M.detected_type = opt.force_type;
    else if (M.compression >= 0.5f) M.detected_type = SourceType::CompressedVideo;
    else M.detected_type = (M.film_score >= 0.5f) ? SourceType::Film : SourceType::DigitalSensor;

    const float spread = (nsel >= 4) ? wins[nsel - 1].score / std::max(wins[0].score, 1e-9f) : 4.0f;
    const float q_count = std::clamp(static_cast<float>(nsel) / 40.0f, 0.2f, 1.0f);
    const float q_spread = std::clamp(1.0f - (spread - 1.3f) / 3.0f, 0.1f, 1.0f);
    const float q_curve = std::clamp(static_cast<float>(curve_points) / 24.0f, 0.2f, 1.0f);
    M.quality = 100.0f * std::cbrt(q_count * q_spread * q_curve);
    char buf[200];
    std::snprintf(buf, sizeof(buf), "%s | %d flat areas | %s", source_type_name(M.detected_type), static_cast<int>(nsel),
                  M.quality >= 75 ? "Good" : M.quality >= 50 ? "Usable - add a flatter sample area if noise remains" :
                  "Weak - draw the box on a flat, featureless area");
    M.assessment = buf;
    M.valid = true;
    return M;
}

} // namespace librenoise::v2
