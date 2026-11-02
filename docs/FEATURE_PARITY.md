# LibreNoise: Feature Parity

Source: `docs/reference/` (sections 1.2, 5.5-5.8, 6, 7, 8, 10). Compared against the current tree.
Legend: **Have** = implemented, **Partial** = exists but weaker or differently shaped, **Missing** = not implemented.

## Profiling

| Feature | Status | Notes |
|---|---|---|
| Auto Profile | Have | `NoiseProfiler::auto_estimate`, stratified flat-patch search |
| Semi-automatic (draw box, Build Profile) | Have | Analysis box + Build Profile in OFX |
| Auto-detect featureless area | Have | `RoiAnalyzer`, quality score and assessment string |
| Manual Fine-Tune / Auto Complete (add samples from other brightness ranges) | **Missing** | Only one sample region per profile. Biggest profiling gap. |
| Generic Profile + "Generic Noise" slider (no clean area needed) | **Missing** | No fallback when a frame has no flat area except fixed defaults |
| Save / load device profile (.dnp) | Have | `.lnp` format, device and mode labels |
| Profile-quality warnings about the selected area | Partial | Quality score and assessment text; no per-problem warnings (too dark, too small, contains detail) |
| Per-channel Y / Cb / Cr noise levels | Have | Profile is now measured in the same Y/Cb/Cr space the filters use |
| Per-frequency noise spectrum | Have | 5-band DCT PSD measured |
| Input gamma / gain override | **Missing** | Noise model assumes linear or already-consistent data; no gain pre-adjust |
| Progressive / interlaced scan type | **Missing** | No field-aware processing |
| Default profile and preset in Preferences | **Missing** | |

## Temporal filter

| Feature | Status | Notes |
|---|---|---|
| Radius (frames each side) | Have | 0-4 |
| Quality Mode Normal / High | Have | Quality Preset (Fast / Balanced / High) |
| Temporal noise level | Partial | "Temporal Strength" scales the match threshold; not separate from spatial Noise Level |
| Flicker reduction (local and frame) | **Missing** | Only exposure-drift compensation inside motion matching, no flicker control |
| Repeated frames (Repeat Rate) | **Missing** | Duplicate frames from FPS conversion are treated as independent samples, so noise averages less than it should |
| Jitter of details ("heat haze") | **Missing** | |
| Dust and scratches (Normal / Aggressive) | **Missing** | |
| Adapt to changing noise | **Missing** | Profile is fixed, or re-estimated per frame when none is built |
| Motion compensation, scene-cut handling | Have | Block matching with trust map, cut detection |

## Spatial filter

| Feature | Status | Notes |
|---|---|---|
| Noise Reduction Amount, Luminance | **Have (new)** | Luma (Y) Amount added this session. It was missing before. |
| Noise Reduction Amount, Cb and Cr | Have | |
| Per-frequency amount (High / Mid / Low / Very Low / Ultra Low) | Partial | 5 bands exist, but they scale the noise threshold relative to the measured spectrum, applied to all channels together. |
| Noise Level (threshold) separate from Amount | **Missing** | LibreNoise has one multiplier per band, no separate "what counts as noise" and "how much of it to remove" |
| Tonal (brightness) equalizer | Have | 5 tonal bands. |
| Artifact Removal (dot / line residue) | **Missing** | |
| Edge Smoothing | **Missing** | |
| Filter Tuning Assist (exaggerate setting effect in preview) | **Missing** | Residual preview exists but is not an amplified-effect view |
| Multiple engines | Have (extra) | Collaborative 3D, Wavelet, NLM, Frequency-Domain. |

## Sharpening and finishing

| Feature | Status | Notes |
|---|---|---|
| Smart sharpening (per frequency and channel) | **Missing** | "Detail Recovery" re-adds lost detail but is not a sharpener |
| Mix with Original | Have | |
| Calibrated grain re-injection | Have (extra) | Not common in other tools |

## UI and workflow

| Feature | Status | Notes |
|---|---|---|
| Dedicated plug-in window with live zoom and pan preview | **Missing** | LibreNoise uses native host parameters plus a viewer overlay for the sample box |
| Beginner / Advanced mode | **Missing** | All controls are always visible |
| Variants A/B/C/D | **Missing** | |
| Filter presets | Partial | Settings travel inside the `.lnp` profile only |
| Click-to-compare original | Partial | Mix with Original and Residual preview; no click-to-toggle |
| Viewer modes (YCrCb + enhanced contrast, channel views) | **Missing** | |
| Global bypass | **Missing** | Use the host's effect bypass |
| Tooltips | **Have (new)** | Every parameter has a hint as of this session |

## Performance and platform

| Feature | Status | Notes |
|---|---|---|
| CUDA | Have | sm_86+ only. |
| OpenCL (AMD) | **Missing** | |
| Multi-GPU | **Missing** | |
| Auto / manual device selection and Optimize Settings benchmark | **Missing** | Only Auto / Force GPU / Force CPU |
| Fast vs full floating-point ("high-precision") | Partial | Always 32-bit float |
| Preview cache and Preferences panel | **Missing** | |
| Multiple hosts | Partial | OFX plus an After Effects plug-in |

## Suggested priorities

1. **Per-channel x per-frequency amounts** (the core advanced mode of professional tools), with a separate Noise Level threshold. This is the real answer to "what should the band sliders do".
2. **Manual Fine-Tune / Auto Complete** profiling and a **Generic Profile** fallback. Weak profiles are the main reason noise survives.
3. **Cache or smooth the auto-profile across frames.** With no built profile, every frame is re-estimated, so strength varies frame to frame (visible as flicker).
4. **Flicker reduction** and **Repeated Frames**, the cheapest temporal wins.
5. **Artifact Removal**, **Edge Smoothing**, **Sharpening**.
6. **Dust and Scratches**, **Jitter of Details**, **Adapt to Changing Noise**.
7. Presets / Variants, and an Advanced / Beginner split of the parameter list.
8. Interlaced and gamma / gain input handling, OpenCL, lower CUDA compute floor.
