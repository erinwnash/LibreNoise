#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Score denoiser renders against the source and a reference render (for example another denoiser).

No reference images are stored in the repository; point the script at your own frames.

    python tools/compare_renders.py --source D:/media/source --ref D:/media/reference \
        --cand v01=D:/media/ln01 --cand v02=D:/media/ln02 --clip clip1 --frame 1030

Each directory holds <clip>.<frame:04d>.exr (scene-linear RGB). Frame+1 is used for the flicker column when present.

Columns (all averaged over R, G, B):
  flat <band>   power kept in flat areas per frequency band (1 = untouched, 0 = removed): how much noise is left
  tex <band>    power kept in textured areas: how much detail (and noise) survives
  detail        correlation between what was removed and the reference's fine detail (lower = less detail removed)
  flicker       RMS frame-to-frame change in flat areas (lower = more stable)
  curve         removed-noise sigma relative to the reference in six brightness bins (1.00 = same strength)

Requires numpy and opencv-python (OpenEXR support is enabled through OPENCV_IO_ENABLE_OPENEXR).
"""
import argparse
import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2  # noqa: E402
import numpy as np  # noqa: E402

TILE = 128
BANDS = [("VLow", 1 / 32, 1 / 16), ("Low", 1 / 16, 1 / 8), ("Mid", 1 / 8, 1 / 4), ("High", 1 / 4, 0.75)]


def load(directory, clip, frame):
    path = os.path.join(directory, f"{clip}.{frame:04d}.exr")
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    return np.ascontiguousarray(img[:, :, 2::-1].astype(np.float32))


def luma(img):
    return 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True)
    ap.add_argument("--ref", required=True, help="reference render directory (used for flat/texture classification)")
    ap.add_argument("--cand", action="append", default=[], help="name=directory, repeatable")
    ap.add_argument("--clip", required=True)
    ap.add_argument("--frame", type=int, required=True)
    a = ap.parse_args()

    src = load(a.source, a.clip, a.frame)
    ref = load(a.ref, a.clip, a.frame)
    if src is None or ref is None:
        raise SystemExit("source or reference frame missing")
    h, w, _ = src.shape

    fy = np.fft.fftfreq(TILE)[:, None]
    fx = np.fft.fftfreq(TILE)[None, :]
    fr = np.sqrt(fx * fx + fy * fy)
    masks = [(fr >= lo) & (fr < hi) for _, lo, hi in BANDS]
    win = np.outer(np.hanning(TILE), np.hanning(TILE))

    def band_power(t):
        f = np.fft.fft2((t - t.mean()) * win)
        p = f.real ** 2 + f.imag ** 2
        return np.array([p[m].mean() for m in masks])

    y_ref = luma(ref)
    yb = cv2.GaussianBlur(np.log2(np.maximum(y_ref, 1e-4)), (0, 0), 3)
    grad = np.sqrt(cv2.Sobel(yb, cv2.CV_32F, 1, 0) ** 2 + cv2.Sobel(yb, cv2.CV_32F, 0, 1) ** 2)
    tiles = [(y, x) for y in range(0, h - TILE + 1, TILE) for x in range(0, w - TILE + 1, TILE)]
    score = np.array([1e9 if src[y:y + TILE, x:x + TILE].std() < 1e-4 else grad[y:y + TILE, x:x + TILE].mean() for y, x in tiles])
    live = score < 1e8
    flat = [t for t, s in zip(tiles, score) if s <= np.percentile(score[live], 20)]
    tex = [t for t, s in zip(tiles, score) if np.percentile(score[live], 80) <= s < 1e8]
    detail_ref = y_ref - cv2.GaussianBlur(y_ref, (0, 0), 2)
    tile_luma = np.array([y_ref[y:y + TILE, x:x + TILE].mean() for y, x in tiles])
    bins = np.quantile(tile_luma[live], [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1])
    flatter = [t for t, s, ok in zip(tiles, score, live) if ok and s <= np.percentile(score[live], 50)]

    def binned_sigma(removed):
        out = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            sel = [(y, x) for (y, x) in flatter if lo <= y_ref[y:y + TILE, x:x + TILE].mean() <= hi]
            out.append(np.sqrt(np.mean([removed[y:y + TILE, x:x + TILE].var() for y, x in sel])) if sel else np.nan)
        return np.array(out)

    sig_ref = binned_sigma(luma(src - ref))
    candidates = [("ref", a.ref)] + [tuple(c.split("=", 1)) for c in a.cand]
    print(f"{a.clip} frame {a.frame}: " + " ".join(f"flat{n:>5s}" for n, _, _ in BANDS) + " | " +
          " ".join(f"tex{n:>5s}" for n, _, _ in BANDS) + " | detail  flicker  | curve vs ref (dark..bright)")
    for name, directory in candidates:
        out = load(directory, a.clip, a.frame)
        if out is None:
            print(f"  {name}: missing")
            continue

        def kept(tlist):
            ps = np.zeros(len(BANDS))
            po = np.zeros(len(BANDS))
            for y, x in tlist:
                for c in range(3):
                    ps += band_power(src[y:y + TILE, x:x + TILE, c])
                    po += band_power(out[y:y + TILE, x:x + TILE, c])
            return po / ps

        removed = luma(src - out)
        num = d1 = d2 = 0.0
        for y, x in tex:
            r = removed[y:y + TILE, x:x + TILE]
            d = detail_ref[y:y + TILE, x:x + TILE]
            r = r - r.mean()
            d = d - d.mean()
            num += (r * d).sum()
            d1 += (r * r).sum()
            d2 += (d * d).sum()
        nxt = load(directory, a.clip, a.frame + 1)
        flick = np.sqrt(np.mean([((nxt[y:y + TILE, x:x + TILE] - out[y:y + TILE, x:x + TILE]) ** 2).mean()
                                 for y, x in flat])) if nxt is not None else float("nan")
        curve = binned_sigma(removed) / sig_ref
        print(f"  {name:10s}" + " ".join(f"{v:9.3f}" for v in kept(flat)) + " | " + " ".join(f"{v:8.3f}" for v in kept(tex)) +
              f" | {num / np.sqrt(d1 * d2):+.3f}  {flick:.5f} | " + " ".join(f"{v:.2f}" for v in curve))


if __name__ == "__main__":
    main()
