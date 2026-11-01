# LibreNoise
**High-Performance Open-Source OFX Video Denoiser**

LibreNoise is a production-grade, open-source video denoising plugin for visual effects artists and colorists. It is designed to achieve equal or superior visual fidelity and performance compared to commercial denoisers across modern GPUs.

---

## Architecture
The project is decoupled into:
- **`liblibre_noise` (`src/core`)**: A pure, host-free C++20 core engine implementing:
  1. *Calibrated Noise Modeling*: Frequency- and channel-resolved noise power spectrum density (PSD), heteroscedastic noise level functions ($\sigma^2(I) = a \cdot I + b$), and variance-stabilizing transforms.
  2. *Motion Alignment & Trust Gating*: Hierarchical sub-pixel block matching with noise-profile residual trust gating (defending against ghosting) and gain-compensated matching across dissolves and fades.
  3. *3D Collaborative Spatiotemporal Shrinkage*: Multi-frame patch grouping along motion trajectories with collaborative DCT/Haar thresholding and smooth fallback to 2D spatial filtering.
  4. *Detail & Look Preservation*: Edge-protected micro-contrast recovery and calibrated synthetic grain re-injection (preventing plastic/waxy skin).
- **`librenoise_cli` (`src/cli`)**: Headless command-line test harness for automated regression, PSNR/SSIM/LPIPS scoring, and residual analysis.
- **`LibreNoise.ofx` (`src/ofx`)**: OpenFX 1.5 standard plugin module compatible with DaVinci Resolve, Natron, Nuke, and other OFX hosts.

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
