# LibreNoise: Feature Parity

Source: `docs/reference/` (sections 1.2, 5.5-5.8, 6, 7, 8, 10). Compared against v0.2.
Legend: **Have** = implemented, **Partial** = exists but weaker or differently shaped, **Missing** = not implemented.

## Profiling

| Feature | Status | Notes |
|---|---|---|
| Auto Profile | Have | Whole-frame brightness curve + spectrum from the flattest areas, measured once per clip |
| Semi-automatic (draw box, Build Profile) | Have | Build Profile from Box measures the spectrum inside the box; the curve is still whole-frame |
| Noise level across brightness (Fine-Tune / Auto Complete) | Have (different) | Measured automatically over the whole frame as a curve, so no manual Fine-Tune pass is needed |
| Generic Profile (no clean area needed) | Partial | Auto mode pools many small flat areas; a frame with no flat area at all still gives a weak profile (quality shown) |
| Save / load device profile | Have | `.lnp` v2 (text), device and mode labels |
| Profile-quality warnings | Partial | Quality %, plain-language assessment; no per-problem warnings yet |
| Per-channel Y / Cr / Cb noise levels | Have | Per-channel, per-scale, per-DCT-coefficient table |
| Per-frequency noise spectrum | Have | Full 2D spectrum per channel |
| Input gamma / gain | Have (different) | Not needed: the measured curve adapts to any transfer function (linear, log, display) |
| Source type (film / digital / compressed) | Have (extra) | Auto-detected with manual override |
| Progressive / interlaced scan type | **Missing** | |

## Temporal filter

| Feature | Status | Notes |
|---|---|---|
| Radius | Have | 0-5 frames each side, default 2 |
| Quality Mode Normal / High | Partial | Quality preset changes spatial overlap; motion search is fixed |
| Temporal noise level | Have | Temporal Strength |
| Flicker reduction | Have | Flicker Reduction (local, slow brightness changes) |
| Repeated frames | Have (automatic) | Exact duplicates are detected and skipped |
| Motion compensation, scene-cut handling | Have | Hierarchical half-pel block matching, scene-cut stop |
| Jitter of details | **Missing** | |
| Dust and scratches | **Missing** | |
| Adapt to changing noise | **Missing** | Profile measured once per clip |

## Spatial filter

| Feature | Status | Notes |
|---|---|---|
| Noise Reduction Amount (Luminance, Cr, Cb) | Have | |
| Noise Level (threshold) separate from Amount | Have | Master, per channel (Luma / Chroma) and per frequency |
| Per-frequency amounts (High / Mid / Low / Very Low / Ultra Low) | Have | 5 pyramid levels; Amount = per channel x per frequency |
| Tonal equaliser | Have (extra) | In stops around 18% grey; neutral by default |
| Compression artifact reduction | Have | Grid detection + deblocking (Artifact Removal) |
| Artifact Removal (dot / line residue) | **Missing** | |
| Edge Smoothing | **Missing** | |
| Filter Tuning Assist | Partial | "Removed Noise" view |

## Sharpening and finishing

| Feature | Status | Notes |
|---|---|---|
| Sharpening | Have | Boosts denoised High/Mid levels only (no noise amplification); single amount |
| Mix with Original | Have | |
| Grain re-injection | Have (extra) | At the measured noise strength |

## UI and workflow

| Feature | Status | Notes |
|---|---|---|
| Dedicated plug-in window | **Missing** | Native host parameters plus viewer box overlay |
| Beginner / Advanced mode | Partial | Advanced groups (Frequencies, Tonal, Processing) start collapsed |
| Variants, presets | **Missing** | Profile files only |
| Tooltips | Have | Every parameter |

## Performance and platform

| Feature | Status | Notes |
|---|---|---|
| CUDA | Have | sm_86+, same result as the CPU |
| OpenCL (AMD) | **Missing** | |
| Multi-GPU | **Missing** | |
| CPU | Have | OpenMP + AVX2 |

## Next priorities

1. Tile the CPU spatial stage (memory bandwidth), cache motion between renders.
2. Residual dot / line Artifact Removal and Edge Smoothing.
3. Adapt to changing noise (re-measure on scene changes), dust and scratches.
4. Presets / variants and a simpler beginner panel.
