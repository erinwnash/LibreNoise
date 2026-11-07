# LibreNoise
**High-Performance Open-Source OFX Video Denoiser**

LibreNoise is a production-grade, open-source video denoising plugin for visual effects artists and colorists. It is designed to achieve equal or superior visual fidelity and performance compared to commercial denoisers across modern GPUs.

---

## Version 0.2

v0.2 replaces the v0.1 engines (tag `v0.1.0`) with one pipeline, written once and compiled for both the CPU
(OpenMP + AVX2) and CUDA, so both paths give the same result. See [`docs/V0_2_DESIGN.md`](docs/V0_2_DESIGN.md).

1. **Noise model measured from the footage** (nothing stored or looked up): a noise-vs-brightness curve per
   R, G, B over the whole frame, a variance-stabilising transform built from that curve, the noise spectrum
   from flat areas, and a per-scale noise table.
2. **Source detection**: film grain (independent per colour layer) vs digital sensor (channel-correlated noise)
   vs compressed video (8x8 block grid). Manual override in the plug-in.
3. **Temporal filter**: motion-compensated averaging over up to 5 frames each side, scene-cut stop,
   repeated-frame skip, optional flicker reduction.
4. **Spatial filter** in Luma / Cb / Cr on a 5-level pyramid (High, Mid, Low, Very Low, Ultra Low), with separate
   Noise Level and Amount per frequency and channel, tonal equaliser, deblocking and sharpening.

Targets:
- **`librenoise_core` (`src/core/v2`)**: the pipeline.
- **`librenoise_cli` (`src/cli`)**: renders EXR sequences, e.g.
  `librenoise_cli --in src/clip.%04d.exr --out out/clip.%04d.exr --frames 1001 1100 [--gpu]`.
- **`LibreNoise_v02.ofx.bundle` (`src/ofx`)**: OpenFX plug-in. New identifier `org.librenoise.denoiser_v02`, so it
  installs next to v0.1 for side-by-side comparison.
- **`tools/compare_renders.py`**: scores renders against the source and a reference render (noise left per
  frequency, detail removed, flicker, strength vs brightness). Bring your own frames; none are stored here.

---

## Clean-Room Provenance & Governance
LibreNoise is an original, clean-room implementation developed exclusively from peer-reviewed scientific literature (*Maggioni et al., IEEE TIP 2012*; *Abdelhamed et al., CVPR 2018*). 

- **Two-Party Clean-Room Defense**: Implementers code strictly from formal specifications in `docs/specs/` and **never view or execute non-commercial code (such as TAU's MATLAB/Python code)**.
- **Licensing**: Licensed under the **GNU General Public License v3.0 (GPLv3)**.
- **Host Integration**: In accordance with VFX industry standards, the OpenFX bundle is loaded dynamically at runtime via the OpenFX C API.

See [`PROVENANCE.md`](PROVENANCE.md) for detailed citations, origin tracking, and developer certification protocols.

---

## Building from Source

### Prerequisites
- CMake 3.20+
- C++20 compliant compiler (MSVC 2022/2026, Clang 16+, GCC 12+)
- Visual Studio on Windows or GCC/Clang on Linux/macOS

### Build Commands
```bash
# Configure
cmake -B build -S .

# Build all targets (Release)
cmake --build build --config Release

# Run Unit Tests & Benchmark Harness
ctest --test-dir build -C Release --output-on-failure
```

## Docs

- [v0.2 design and test results](docs/V0_2_DESIGN.md)
- [Feature status](docs/FEATURES.md)
- [Third-party notices](THIRD_PARTY_NOTICES.md)
