# LibreNoise v0.2: design and results

## Why v0.2

Measured against reference renders on five real clips (three 4K film scans, one Alexa ProRes 1080p, one H.264 1080p),
v0.1 had problems that no slider could fix:

- **Wrong noise model.** v0.1 assumed `variance = a·x + b` (raw sensor). Scanned film grain grows almost linearly with
  brightness in linear light, and a curve fitted to one sample box was badly wrong elsewhere: up to ~9× too much noise
  assumed in shadows, ~3× too little in highlights. Result: over-filtered shadows, grain left in highlights.
- **Tonal equaliser on the wrong scale.** It clamped linear values to 0-1, so mid-grey fell in the 1.3× shadow band.
- **Single scale.** 8×8 patches cannot separate coarse grain from image detail, so blotchy low-frequency noise
  stayed (2-4× more than the reference).
- **Weak temporal filtering.** Frame-to-frame change in flat areas was 1.4-4.6× the reference (boiling).
- **RGB processing.** Noisy channels (often blue on film) were filtered on their own and lost detail.
- **Compressed footage barely touched** (14-25% of the reference's removal on H.264).

## Pipeline

All stages live in `src/core/v2/pipeline_impl.inl`, a template compiled for both back ends (`backend_cpu.cpp`,
`backend_cuda.cu`). The per-pixel / per-block math is in `kernels.hpp` as host+device functions.

1. **Noise model** (`profiler.cpp`, measured once per clip, stored in the project):
   - *Curve*: the frame is cut into 16×16 blocks; each block's high-pass sigma is binned by its mean value on a
     half-stop grid; the lowest 10% of each bin is the noise envelope. A robust fit of
     `sigma² = floor² + (gain·x^p)²` (texture only pushes points up, so those count less) is corrected locally by the
     measured knots. Only the curve's shape matters at this stage.
   - *Stabilisation*: `f(x) = ∫ dx / sigma(x)` per channel (lookup tables, linear below 2^-14 so negatives and zero
     are handled) makes noise unit strength at every brightness.
   - *Working space*: orthonormal Y = (R+G+B)/√3, Cb = (2B-R-G)/√6, Cr = (R-G)/√2 on the stabilised channels.
   - *Spectrum*: 2D noise power spectrum per channel from the flattest 64×64 windows (or windows inside the user's
     box). Because noise is unit strength after stabilisation, flat areas at any brightness can be pooled; this is
     why one flat area is enough.
   - *Per-scale table*: a 1024² synthetic field with exactly that spectrum is run through the same pyramid and 8×8 DCT
     as the filter, giving the noise std of every coefficient at every level.
   - *Detection*: mean / max cross-channel correlation of the noise (film grain ~0-0.3; demosaiced and matrixed
     sensor noise 0.4-0.8 in at least one pair) and an 8×8 block-boundary detector that also finds the grid phase.
2. **Temporal**: hierarchical block matching (8×8 blocks, 4 levels, half-pel) on stabilised luma, chained across
   distance; per-pixel weights from a 5×5 patch distance normalised by the known noise; scene-cut stop when the mean
   weight collapses; exact duplicate frames skipped. The per-pixel residual factor `rho = sqrt(Σw²)/Σw` tells the
   spatial stage how much noise is left at each pixel.
3. **Spatial**: 5-level Laplacian pyramid per channel. Each band-pass level gets a sliding 8×8 DCT filter
   (hard-threshold pilot, then empirical Wiener) with thresholds = table × rho × tonal × Noise Level.
   `out = band − Amount·(band − denoised)` per level and channel; sharpening boosts the denoised High/Mid levels.
4. **Compression**: when a grid is found, a boundary-only deblocking filter runs in the stabilised domain before
   matching and filtering.

## Results (tools/compare_renders.py, frame 1030 / 20)

Power kept in flat areas (1 = untouched, lower = more noise removed), removed-noise strength vs the reference across
brightness (dark … bright), and frame-to-frame change in flat areas. Reference = renders of the same frames from another denoiser (not distributed with this repository).

| Clip | Version | Flat Low | Flat Mid | Flat High | Strength vs reference | Flicker |
|---|---|---|---|---|---|---|
| clip1 film, bright | reference | 0.080 | 0.039 | 0.026 | 1.00 | 0.0080 |
| | v0.1 CPU | 0.215 | 0.061 | 0.054 | 1.39 … 0.79 | 0.0112 |
| | **v0.2** | 0.059 | 0.047 | 0.026 | 1.12 … 1.03 | 0.0090 |
| clip2 film, medium | reference | 0.366 | 0.092 | 0.024 | 1.00 | 0.0019 |
| | v0.1 CPU | 0.590 | 0.269 | 0.160 | 1.12 … 0.73 | 0.0085 |
| | **v0.2** | 0.415 | 0.146 | 0.063 | 0.94 … 0.90 | 0.0051 |
| clip3 film, dark | reference | 0.017 | 0.012 | 0.010 | 1.00 | 0.0035 |
| | v0.1 CPU | 0.291 | 0.129 | 0.089 | 1.23 … 0.72 | 0.0092 |
| | **v0.2** | 0.032 | 0.028 | 0.014 | 1.18 … 1.03 | 0.0043 |
| clip4 Alexa ProRes | reference | 0.064 | 0.025 | 0.027 | 1.00 | (camera move) |
| | v0.1 CPU | 0.251 | 0.030 | 0.027 | 1.16 … 0.96 | |
| | **v0.2** | 0.020 | 0.007 | 0.004 | 1.18 … 1.30 | |
| clip5 H.264 | reference | 0.945 | 0.765 | 0.367 | 1.00 | (camera move) |
| | v0.1 CPU | 0.998 | 0.982 | 0.881 | 1.06 … 0.17 | |
| | **v0.2** | 0.953 | 0.784 | 0.291 | 0.91 … 0.98 | |

CPU and GPU outputs are identical to within float rounding (about 110 dB PSNR between them).

Notes:
- The test set has three frames per clip, so v0.2 could only use one frame on each side; the reference renders were
  made from the full clips (two frames each side). Flicker numbers favour the reference for that reason.
- v0.2 is a little stronger than the reference on the Alexa clip; lower Noise Level or Amount if that matters.
- Detail removal in textured areas is still somewhat higher than the reference on clips 1 and 5 (the High-frequency
  Noise Level defaults to 0.8 to limit this).

## Speed

| | 1080p | 4K (4096×3112) |
|---|---|---|
| GPU (sm_86), processing only | ~37 ms | ~55 ms |
| CPU (Threadripper 2990WX, 32 threads) | ~0.35 s | ~2.2 s |

The CPU path is memory-bandwidth bound on this machine (scaling stops at ~16 threads). Next step for CPU speed is
tiling the spatial stage so a tile's pyramid stays in cache.

## Not done yet

- Artifact Removal for residual dot / line noise, dust and scratches, jitter of details.
- Adapting to changing noise inside a clip (the profile is measured once).
- Caching motion between renders, and OpenCL.
