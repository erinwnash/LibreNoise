// LibreNoise v0.2 processing pipeline, written once and instantiated for the CPU and CUDA back ends.
//
// Stages for one output frame:
//   1. Convert every frame to the working domain: per-channel variance stabilisation (measured noise curve)
//      followed by an orthonormal Y / Cb / Cr opponent transform. Noise is ~unit strength at all brightness.
//      Optional deblocking on the detected compression grid.
//   2. Motion: hierarchical block matching on stabilised luma, chained across distances.
//   3. Temporal: per-pixel motion-compensated weighted average over the window, weights from a 5x5 patch
//      distance normalised by the known noise level. Tracks the per-pixel residual noise factor rho.
//   4. Spatial: 5-level Laplacian pyramid per channel; each band-pass level is filtered by an 8x8 sliding DCT
//      (hard-threshold pilot + empirical Wiener) using the measured per-level, per-coefficient noise table
//      scaled by rho, the tonal equaliser and the user's Noise Level. Amounts blend each level back.
//   5. Back to RGB, mix / preview / alpha.
//
// Requirements on the back end B:
//   B::Buf alloc(size_t n); float* B::Buf::p
//   void zero(float*, size_t); void copy(float* dst, const float* src, size_t n)
//   B::Buf upload(const float* host, size_t n)
//   template<class F> void for2d(int w, int h, F f)       f(x, y)
//   template<class F> double sum2d(int w, int h, F f)     sum of f(x, y)
//   bool aborted()

#include "kernels.hpp"
#include <cmath>
#include <algorithm>
#include <vector>

namespace librenoise::v2::detail {

struct Tonal5 { float t[5]; };

struct DeblockSetup {
    int phx = -1, phy = -1;
    float strength = 0.0f;
    float thresh[3] = {0, 0, 0};
};

template <class B>
void deblock_plane(B& be, float* plane, float* tmp, int w, int h, const DeblockSetup& db, int c) {
    const float th = db.thresh[c], st = db.strength;
    if (db.phx >= 0) {
        const int ph = db.phx;
        be.for2d(w, h, LN_LAMBDA(int x, int y) { tmp[static_cast<size_t>(y) * w + x] = k::deblock_h(plane, w, x, y, ph, th, st); });
        be.copy(plane, tmp, static_cast<size_t>(w) * h);
    }
    if (db.phy >= 0) {
        const int ph = db.phy;
        be.for2d(w, h, LN_LAMBDA(int x, int y) { tmp[static_cast<size_t>(y) * w + x] = k::deblock_v(plane, w, h, x, y, ph, th, st); });
        be.copy(plane, tmp, static_cast<size_t>(w) * h);
    }
}

template <class B>
void convert_frame(B& be, const FrameView& f, int w, int h, const VstLut* L, float* Y, float* Cb, float* Cr,
                   float* tmp, const DeblockSetup& db) {
    const float* d = f.data;
    const int st = f.stride_floats, ch = f.channels;
    be.for2d(w, h, LN_LAMBDA(int x, int y) {
        const float* p = d + static_cast<size_t>(y) * st + static_cast<size_t>(x) * ch;
        float a, b, c;
        k::rgb_to_work(*L, p[0], p[1], p[2], a, b, c);
        const size_t i = static_cast<size_t>(y) * w + x;
        Y[i] = a; Cb[i] = b; Cr[i] = c;
    });
    if (db.strength > 0.0f) {
        deblock_plane(be, Y, tmp, w, h, db, 0);
        deblock_plane(be, Cb, tmp, w, h, db, 1);
        deblock_plane(be, Cr, tmp, w, h, db, 2);
    }
}

template <class B>
double frame_checksum(B& be, const FrameView& f, int w, int h) {
    const float* d = f.data;
    const int st = f.stride_floats, ch = f.channels;
    const int sw = (w + 3) / 4, sh = (h + 3) / 4;
    return be.sum2d(sw, sh, LN_LAMBDA(int x, int y) -> double {
        const int px = k::imin(x * 4, w - 1), py = k::imin(y * 4, h - 1);
        const float* p = d + static_cast<size_t>(py) * st + static_cast<size_t>(px) * ch;
        const double wt = 1.0 + static_cast<double>(k::hash32(static_cast<uint32_t>(px * 7919 + py * 104729)) & 1023u);
        return wt * (static_cast<double>(p[0]) + 3.0 * p[1] + 7.0 * p[2]);
    });
}

template <class B>
void box_blur(B& be, const float* src, float* dst, float* tmp, int w, int h, int r) {
    be.for2d(w, h, LN_LAMBDA(int x, int y) {
        float s = 0.0f;
        const size_t row = static_cast<size_t>(y) * w;
        for (int i = -r; i <= r; ++i) s += src[row + k::iclamp(x + i, 0, w - 1)];
        tmp[row + x] = s / static_cast<float>(2 * r + 1);
    });
    be.for2d(w, h, LN_LAMBDA(int x, int y) {
        float s = 0.0f;
        for (int j = -r; j <= r; ++j) s += tmp[static_cast<size_t>(k::iclamp(y + j, 0, h - 1)) * w + x];
        dst[static_cast<size_t>(y) * w + x] = s / static_cast<float>(2 * r + 1);
    });
}

struct LevelDims { int w[8], h[8]; int n; };

inline LevelDims make_dims(int w, int h, int levels) {
    LevelDims d{};
    d.n = levels;
    d.w[0] = w; d.h[0] = h;
    for (int l = 1; l < levels; ++l) { d.w[l] = (d.w[l - 1] + 1) / 2; d.h[l] = (d.h[l - 1] + 1) / 2; }
    return d;
}

// Motion pyramid of the stabilised luma (level 0 is the full-res plane, not owned).
template <class B>
struct MotionPyr {
    std::vector<typename B::Buf> lv;   // levels 1..3
    const float* p[4] = {nullptr, nullptr, nullptr, nullptr};
    LevelDims d{};
};

template <class B>
void build_motion_pyr(B& be, const float* Y, int w, int h, MotionPyr<B>& mp) {
    mp.d = make_dims(w, h, 4);
    mp.p[0] = Y;
    mp.lv.clear();
    for (int l = 1; l < 4; ++l) {
        mp.lv.push_back(be.alloc(static_cast<size_t>(mp.d.w[l]) * mp.d.h[l]));
        float* dst = mp.lv.back().p;
        const float* src = mp.p[l - 1];
        const int sw = mp.d.w[l - 1], sh = mp.d.h[l - 1], dw = mp.d.w[l];
        be.for2d(mp.d.w[l], mp.d.h[l], LN_LAMBDA(int x, int y) { dst[static_cast<size_t>(y) * dw + x] = k::box_down(src, sw, sh, x, y); });
        mp.p[l] = dst;
    }
}

// Estimates block vectors (8x8 at full res) from ref to cand. pred (may be null) = predictor field at level 0.
template <class B>
void estimate_motion(B& be, const MotionPyr<B>& R, const MotionPyr<B>& C, const float* pred, float pred_scale, float* out_v0) {
    typename B::Buf vb[4];
    int nbx[4], nby[4];
    for (int l = 0; l < 4; ++l) {
        nbx[l] = (R.d.w[l] + 7) / 8;
        nby[l] = (R.d.h[l] + 7) / 8;
        if (l > 0) vb[l] = be.alloc(static_cast<size_t>(nbx[l]) * nby[l] * 2);
    }
    // coarsest: full search
    {
        const float* rp = R.p[3]; const float* cp = C.p[3];
        const int w3 = R.d.w[3], h3 = R.d.h[3], nb = nbx[3];
        float* v = vb[3].p;
        be.for2d(nbx[3], nby[3], LN_LAMBDA(int bx, int by) {
            const k::Vec2 r = k::me_full(rp, cp, w3, h3, bx, by, 6);
            v[(static_cast<size_t>(by) * nb + bx) * 2] = r.x;
            v[(static_cast<size_t>(by) * nb + bx) * 2 + 1] = r.y;
        });
    }
    for (int l = 2; l >= 0; --l) {
        const float* rp = R.p[l]; const float* cp = C.p[l];
        const int wl = R.d.w[l], hl = R.d.h[l], nb = nbx[l], pnb = nbx[l + 1], pnby = nby[l + 1];
        const float* pv = vb[l + 1].p;
        float* v = (l == 0) ? out_v0 : vb[l].p;
        const float* pr = (l == 0) ? pred : nullptr;
        const float ps = pred_scale;
        const int row_step = (l == 0) ? 2 : 1;
        const bool half = (l == 0);
        be.for2d(nbx[l], nby[l], LN_LAMBDA(int bx, int by) {
            k::Vec2 c[7];
            int n = 0;
            const int pbx = k::imin(bx / 2, pnb - 1), pby = k::imin(by / 2, pnby - 1);
            const int offs[5][2] = {{0, 0}, {-1, 0}, {1, 0}, {0, -1}, {0, 1}};
            for (int o = 0; o < 5; ++o) {
                const int qx = k::iclamp(pbx + offs[o][0], 0, pnb - 1), qy = k::iclamp(pby + offs[o][1], 0, pnby - 1);
                const size_t qi = (static_cast<size_t>(qy) * pnb + qx) * 2;
                c[n].x = pv[qi] * 2.0f; c[n].y = pv[qi + 1] * 2.0f; ++n;
            }
            if (pr != nullptr) {
                const size_t qi = (static_cast<size_t>(by) * nb + bx) * 2;
                c[n].x = pr[qi] * ps; c[n].y = pr[qi + 1] * ps; ++n;
            }
            const k::Vec2 r = k::me_refine(rp, cp, wl, hl, bx, by, c, n, row_step, half);
            v[(static_cast<size_t>(by) * nb + bx) * 2] = r.x;
            v[(static_cast<size_t>(by) * nb + bx) * 2 + 1] = r.y;
        });
    }
}

// Warps cand into W (3 planes) with the block vectors, computes weights, accumulates. Returns mean weight.
template <class B>
double temporal_accumulate(B& be, int w, int h, const float* refY, const float* const cand[3], const float* v0,
                           float* W[3], float* dd, float* tmp, float* tmp2, float* acc[3], float* wsum, float* w2sum,
                           float sigmaY, float strength, float flicker) {
    const int nbx = (w + 7) / 8;
    const float* c0 = cand[0]; const float* c1 = cand[1]; const float* c2 = cand[2];
    float* w0 = W[0]; float* w1 = W[1]; float* w2 = W[2];
    be.for2d(w, h, LN_LAMBDA(int x, int y) {
        const size_t bi = (static_cast<size_t>(y >> 3) * nbx + (x >> 3)) * 2;
        const float sx = static_cast<float>(x) + v0[bi], sy = static_cast<float>(y) + v0[bi + 1];
        const size_t i = static_cast<size_t>(y) * w + x;
        w0[i] = k::sample_bilinear(c0, w, h, sx, sy);
        w1[i] = k::sample_bilinear(c1, w, h, sx, sy);
        w2[i] = k::sample_bilinear(c2, w, h, sx, sy);
        dd[i] = refY[i] - w0[i];
    });
    if (flicker > 0.0f) {
        box_blur(be, dd, tmp2, tmp, w, h, 8);
        const float fl = flicker;
        be.for2d(w, h, LN_LAMBDA(int x, int y) {
            const size_t i = static_cast<size_t>(y) * w + x;
            dd[i] -= fl * tmp2[i];
        });
    }
    be.for2d(w, h, LN_LAMBDA(int x, int y) { const size_t i = static_cast<size_t>(y) * w + x; dd[i] = dd[i] * dd[i]; });
    box_blur(be, dd, tmp2, tmp, w, h, 2);
    const float inv_e = 1.0f / (2.0f * sigmaY * sigmaY + 1e-9f);
    const float tol = fmaxf(strength, 0.0f);
    const float knee = 1.0f + 0.8f * tol;
    const float soft = fmaxf(0.6f * tol, 1e-3f);
    float* a0 = acc[0]; float* a1 = acc[1]; float* a2 = acc[2];
    const float* dm = tmp2;
    be.for2d(w, h, LN_LAMBDA(int x, int y) {
        const size_t i = static_cast<size_t>(y) * w + x;
        const float e = dm[i] * inv_e;
        float wt = (tol <= 0.0f) ? 0.0f : expf(-fmaxf(0.0f, e - knee) / soft);
        a0[i] += wt * w0[i];
        a1[i] += wt * w1[i];
        a2[i] += wt * w2[i];
        wsum[i] += wt;
        w2sum[i] += wt * wt;
        dd[i] = wt;
    });
    const float* wts = dd;
    const double s = be.sum2d(w, h, LN_LAMBDA(int x, int y) -> double { return static_cast<double>(wts[static_cast<size_t>(y) * w + x]); });
    return s / (static_cast<double>(w) * h);
}

// Sliding 8x8 DCT filter of one band-pass level (two passes). F = per-pixel threshold factor at this level.
template <class B>
void denoise_level(B& be, const float* band, int lw, int lh, const float* F, const float* sig, int step, float* out) {
    if (lw < 8 || lh < 8) {
        be.copy(out, band, static_cast<size_t>(lw) * lh);
        return;
    }
    const int pw = ((lw - 8 + step - 1) / step) * step + 8;
    const int ph = ((lh - 8 + step - 1) / step) * step + 8;
    const size_t PN = static_cast<size_t>(pw) * ph;
    auto P = be.alloc(PN), num = be.alloc(PN), den = be.alloc(PN), pilot = be.alloc(PN);
    float* pp = P.p; float* np = num.p; float* dp = den.p; float* pl = pilot.p;
    be.for2d(pw, ph, LN_LAMBDA(int x, int y) {
        pp[static_cast<size_t>(y) * pw + x] = band[static_cast<size_t>(k::reflect(y, lh)) * lw + k::reflect(x, lw)];
    });
    const int nbx = (pw - 8) / step + 1, nby = (ph - 8) / step + 1;
    const int colors = (8 + step - 1) / step;
    for (int pass = 0; pass < 2; ++pass) {
        be.zero(np, PN);
        be.zero(dp, PN);
        for (int cy = 0; cy < colors; ++cy) {
            for (int cx = 0; cx < colors; ++cx) {
                const int nx = (nbx - cx + colors - 1) / colors, ny = (nby - cy + colors - 1) / colors;
                if (nx <= 0 || ny <= 0) continue;
                const int ccx = cx, ccy = cy, ncol = colors, st = step;
                if (pass == 0) {
                    be.for2d(nx, ny, LN_LAMBDA(int i, int j) {
                        const int bx = (ccx + i * ncol) * st, by = (ccy + j * ncol) * st;
                        const float f = F[static_cast<size_t>(k::imin(by + 4, lh - 1)) * lw + k::imin(bx + 4, lw - 1)];
                        k::block_ht(pp, pw, bx, by, sig, f, 2.7f, np, dp);
                    });
                } else {
                    be.for2d(nx, ny, LN_LAMBDA(int i, int j) {
                        const int bx = (ccx + i * ncol) * st, by = (ccy + j * ncol) * st;
                        const float f = F[static_cast<size_t>(k::imin(by + 4, lh - 1)) * lw + k::imin(bx + 4, lw - 1)];
                        k::block_wiener(pp, pl, pw, bx, by, sig, f, np, dp);
                    });
                }
            }
        }
        if (pass == 0) {
            be.for2d(pw, ph, LN_LAMBDA(int x, int y) {
                const size_t i = static_cast<size_t>(y) * pw + x;
                pl[i] = dp[i] > 0.0f ? np[i] / dp[i] : pp[i];
            });
        }
    }
    be.for2d(lw, lh, LN_LAMBDA(int x, int y) {
        const size_t i = static_cast<size_t>(y) * pw + x;
        out[static_cast<size_t>(y) * lw + x] = dp[i] > 0.0f ? np[i] / dp[i] : pp[i];
    });
}

template <class B>
bool run_pipeline(B& be, const FrameView* frames, int n, int ref, int w, int h, const NoiseModel& M, const Params& P,
                  const OutputView& out, const VstLut* L) {
    const size_t N = static_cast<size_t>(w) * h;
    const SourceType type = (P.source_type == SourceType::Auto) ? M.detected_type : P.source_type;

    DeblockSetup db;
    if (M.grid_phase_x >= 0 || M.grid_phase_y >= 0 || type == SourceType::CompressedVideo) {
        float base = M.compression;
        if (type == SourceType::CompressedVideo) base = std::max(base, 0.5f);
        db.strength = std::clamp(base * P.artifact_removal, 0.0f, 1.5f);
        db.phx = M.grid_phase_x >= 0 ? M.grid_phase_x : (type == SourceType::CompressedVideo ? 7 : -1);
        db.phy = M.grid_phase_y >= 0 ? M.grid_phase_y : (type == SourceType::CompressedVideo ? 7 : -1);
        for (int c = 0; c < 3; ++c) db.thresh[c] = 8.0f * std::max(M.pixel_sigma[c], 0.05f);
    }

    // Reference frame in the working domain
    auto rY = be.alloc(N), rCb = be.alloc(N), rCr = be.alloc(N), scratch = be.alloc(N);
    convert_frame(be, frames[ref], w, h, L, rY.p, rCb.p, rCr.p, scratch.p, db);
    if (be.aborted()) return false;

    auto acc0 = be.alloc(N), acc1 = be.alloc(N), acc2 = be.alloc(N), wsum = be.alloc(N), w2sum = be.alloc(N);
    be.copy(acc0.p, rY.p, N);
    be.copy(acc1.p, rCb.p, N);
    be.copy(acc2.p, rCr.p, N);
    {
        float* ws = wsum.p; float* w2 = w2sum.p;
        be.for2d(w, h, LN_LAMBDA(int x, int y) { const size_t i = static_cast<size_t>(y) * w + x; ws[i] = 1.0f; w2[i] = 1.0f; });
    }

    const int radius = std::clamp(P.temporal_radius, 0, 5);
    if (radius > 0 && n > 1) {
        MotionPyr<B> rpyr;
        build_motion_pyr(be, rY.p, w, h, rpyr);
        auto kY = be.alloc(N), kCb = be.alloc(N), kCr = be.alloc(N);
        auto W0 = be.alloc(N), W1 = be.alloc(N), W2 = be.alloc(N), dd = be.alloc(N), t2 = be.alloc(N);
        const size_t nvec = static_cast<size_t>((w + 7) / 8) * ((h + 7) / 8) * 2;
        const double ref_sum = P.skip_repeated_frames ? frame_checksum(be, frames[ref], w, h) : 0.0;
        for (int side = -1; side <= 1; side += 2) {
            auto vprev = be.alloc(nvec), vcur = be.alloc(nvec);
            bool have_prev = false;
            double prev_sum = ref_sum;
            for (int d = 1; d <= radius; ++d) {
                const int kidx = ref + side * d;
                if (kidx < 0 || kidx >= n) break;
                if (be.aborted()) return false;
                if (P.skip_repeated_frames) {
                    const double s = frame_checksum(be, frames[kidx], w, h);
                    const bool dup = std::abs(s - prev_sum) <= 1e-9 * (std::abs(prev_sum) + 1.0) ||
                                     std::abs(s - ref_sum) <= 1e-9 * (std::abs(ref_sum) + 1.0);
                    prev_sum = s;
                    if (dup) continue;
                }
                convert_frame(be, frames[kidx], w, h, L, kY.p, kCb.p, kCr.p, scratch.p, db);
                MotionPyr<B> kpyr;
                build_motion_pyr(be, kY.p, w, h, kpyr);
                estimate_motion(be, rpyr, kpyr, have_prev ? vprev.p : nullptr,
                                static_cast<float>(d) / static_cast<float>(std::max(1, d - 1)), vcur.p);
                const float* cand[3] = {kY.p, kCb.p, kCr.p};
                float* Wp[3] = {W0.p, W1.p, W2.p};
                float* ap[3] = {acc0.p, acc1.p, acc2.p};
                const double mean_w = temporal_accumulate(be, w, h, rY.p, cand, vcur.p, Wp, dd.p, scratch.p, t2.p, ap,
                                                          wsum.p, w2sum.p, M.pixel_sigma[0], P.temporal_strength,
                                                          P.flicker_reduction);
                std::swap(vprev, vcur);
                have_prev = true;
                if (mean_w < 0.08) break;   // scene cut or unusable frame: stop on this side
            }
        }
    }
    if (be.aborted()) return false;

    // Normalise the temporal estimate and build the per-pixel threshold factor F0 = rho * tonal.
    auto F0 = be.alloc(N);
    {
        float* a0 = acc0.p; float* a1 = acc1.p; float* a2 = acc2.p;
        const float* ws = wsum.p; const float* w2 = w2sum.p;
        float* f0 = F0.p;
        const float* rd = frames[ref].data;
        const int st = frames[ref].stride_floats, ch = frames[ref].channels;
        Tonal5 tn;
        for (int i = 0; i < 5; ++i) tn.t[i] = P.tonal[i];
        be.for2d(w, h, LN_LAMBDA(int x, int y) {
            const size_t i = static_cast<size_t>(y) * w + x;
            const float inv = 1.0f / ws[i];
            a0[i] *= inv; a1[i] *= inv; a2[i] *= inv;
            const float rho = sqrtf(w2[i]) * inv;
            const float* p = rd + static_cast<size_t>(y) * st + static_cast<size_t>(x) * ch;
            const float luma = 0.2126f * p[0] + 0.7152f * p[1] + 0.0722f * p[2];
            f0[i] = rho * k::tonal_factor(tn.t, luma);
        });
        box_blur(be, F0.p, F0.p, scratch.p, w, h, 3);
    }
    rY = typename B::Buf(); rCb = typename B::Buf(); rCr = typename B::Buf();
    wsum = typename B::Buf(); w2sum = typename B::Buf();

    // Factor pyramid
    const LevelDims D = make_dims(w, h, kLevels + 1);
    std::vector<typename B::Buf> Fl;
    const float* Fp[kLevels + 1];
    Fp[0] = F0.p;
    for (int l = 1; l <= kLevels; ++l) {
        Fl.push_back(be.alloc(static_cast<size_t>(D.w[l]) * D.h[l]));
        float* dst = Fl.back().p;
        const float* src = Fp[l - 1];
        const int sw = D.w[l - 1], sh = D.h[l - 1], dw = D.w[l];
        be.for2d(D.w[l], D.h[l], LN_LAMBDA(int x, int y) { dst[static_cast<size_t>(y) * dw + x] = k::box_down(src, sw, sh, x, y); });
        Fp[l] = dst;
    }

    // Scaled per-level noise tables (host), uploaded once
    std::vector<float> sig_host(static_cast<size_t>(kLevels) * 3 * kCoefs);
    for (int l = 0; l < kLevels; ++l)
        for (int c = 0; c < 3; ++c)
            for (int q = 0; q < kCoefs; ++q)
                sig_host[(static_cast<size_t>(l) * 3 + c) * kCoefs + q] =
                    M.level_sigma[l][c][q] * P.noise_level[l] * P.channel_noise_level[c];
    auto sig_dev = be.upload(sig_host.data(), sig_host.size());

    // Spatial multi-scale filter, one channel at a time
    float* chans[3] = {acc0.p, acc1.p, acc2.p};
    const int step = std::clamp(P.step, 2, 7);
    for (int c = 0; c < 3; ++c) {
        if (be.aborted()) return false;
        std::vector<typename B::Buf> G, Bd;
        const float* Gp[kLevels + 1];
        Gp[0] = chans[c];
        for (int l = 1; l <= kLevels; ++l) {
            G.push_back(be.alloc(static_cast<size_t>(D.w[l]) * D.h[l]));
            float* dst = G.back().p;
            const float* src = Gp[l - 1];
            const int sw = D.w[l - 1], sh = D.h[l - 1], dw = D.w[l];
            be.for2d(D.w[l], D.h[l], LN_LAMBDA(int x, int y) { dst[static_cast<size_t>(y) * dw + x] = k::pyr_down(src, sw, sh, x, y); });
            Gp[l] = dst;
        }
        for (int l = 0; l < kLevels; ++l) {
            const int lw = D.w[l], lh = D.h[l];
            const size_t LN = static_cast<size_t>(lw) * lh;
            auto band = be.alloc(LN), den = be.alloc(LN);
            float* bp = band.p;
            const float* g = Gp[l]; const float* gs = Gp[l + 1];
            const int sw = D.w[l + 1], sh = D.h[l + 1];
            be.for2d(lw, lh, LN_LAMBDA(int x, int y) {
                bp[static_cast<size_t>(y) * lw + x] = g[static_cast<size_t>(y) * lw + x] - k::pyr_up(gs, sw, sh, x, y);
            });
            const float a = P.amount_band[l] * P.amount_channel[c];
            const bool skip = (a <= 0.0f && P.sharpening <= 0.0f);
            if (!skip) {
                denoise_level(be, bp, lw, lh, Fp[l], sig_dev.p + (static_cast<size_t>(l) * 3 + c) * kCoefs, step, den.p);
                const float sh_amt = P.sharpening * (l == 0 ? 0.8f : (l == 1 ? 0.5f : 0.0f));
                const float* dpp = den.p;
                be.for2d(lw, lh, LN_LAMBDA(int x, int y) {
                    const size_t i = static_cast<size_t>(y) * lw + x;
                    bp[i] = bp[i] - a * (bp[i] - dpp[i]) + sh_amt * dpp[i];
                });
            }
            Bd.push_back(std::move(band));
        }
        // Reconstruct from the coarsest residual upward
        auto rec = be.alloc(static_cast<size_t>(D.w[kLevels]) * D.h[kLevels]);
        be.copy(rec.p, Gp[kLevels], static_cast<size_t>(D.w[kLevels]) * D.h[kLevels]);
        for (int l = kLevels - 1; l >= 0; --l) {
            const int lw = D.w[l], lh = D.h[l], sw = D.w[l + 1], sh = D.h[l + 1];
            typename B::Buf nxt;
            float* dst;
            if (l == 0) dst = chans[c];
            else { nxt = be.alloc(static_cast<size_t>(lw) * lh); dst = nxt.p; }
            const float* bp = Bd[l].p; const float* rp = rec.p;
            be.for2d(lw, lh, LN_LAMBDA(int x, int y) {
                dst[static_cast<size_t>(y) * lw + x] = bp[static_cast<size_t>(y) * lw + x] + k::pyr_up(rp, sw, sh, x, y);
            });
            if (l > 0) rec = std::move(nxt);
        }
    }
    if (be.aborted()) return false;

    // Output
    {
        const float* a0 = acc0.p; const float* a1 = acc1.p; const float* a2 = acc2.p;
        const float* rd = frames[ref].data;
        const int st = frames[ref].stride_floats, ich = frames[ref].channels;
        float* od = out.data;
        const int ost = out.stride_floats, och = out.channels;
        const float mix = std::clamp(P.mix_original, 0.0f, 1.0f);
        const int preview = P.preview;
        const float grain = P.grain;
        const uint32_t seed = P.grain_seed;
        const float gs0 = M.pixel_sigma[0], gs1 = M.pixel_sigma[1], gs2 = M.pixel_sigma[2];
        be.for2d(w, h, LN_LAMBDA(int x, int y) {
            const size_t i = static_cast<size_t>(y) * w + x;
            float Y = a0[i], Cb = a1[i], Cr = a2[i];
            if (grain > 0.0f) {
                Y += grain * gs0 * k::hash_gauss(x, y, seed);
                Cb += grain * gs1 * k::hash_gauss(x, y, seed + 101u);
                Cr += grain * gs2 * k::hash_gauss(x, y, seed + 202u);
            }
            float r, g, b;
            k::work_to_rgb(*L, Y, Cb, Cr, r, g, b);
            const float* p = rd + static_cast<size_t>(y) * st + static_cast<size_t>(x) * ich;
            if (preview == 1) {
                r = (p[0] - r) * 2.0f + 0.5f; g = (p[1] - g) * 2.0f + 0.5f; b = (p[2] - b) * 2.0f + 0.5f;
            } else if (mix > 0.0f) {
                r += (p[0] - r) * mix; g += (p[1] - g) * mix; b += (p[2] - b) * mix;
            }
            float* o = od + static_cast<size_t>(y) * ost + static_cast<size_t>(x) * och;
            o[0] = r; o[1] = g; o[2] = b;
            if (och >= 4) o[3] = (ich >= 4) ? p[3] : 1.0f;
        });
    }
    return true;
}

} // namespace librenoise::v2::detail
