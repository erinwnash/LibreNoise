# LibreNoise Provenance & Clean-Room Governance Ledger

## 1. Clean-Room Implementation Mandate
LibreNoise is an independent, clean-room implementation of spatio-temporal video filtering. It is developed from published mathematical papers, academic literature, and open specifications.

### 1.1 Separation of Spec and Code (Two-Party Defense)
- **Specification Writers**: Analyze peer-reviewed academic literature (e.g., Maggioni et al., 2012; Dabov et al., 2007) to write independent, self-contained mathematical functional specifications in `docs/specs/`.
- **Implementers**: Write C++20 code strictly guided by these functional specifications. Implementers **must never inspect, decompile, execute, or reference non-commercial or proprietary source code** (specifically Tampere University TAU BM3D/V-BM4D MATLAB or Python repositories).
- **Contamination Policy**: Any contributor who has inspected Tampere University source code is strictly disqualified from authoring or reviewing LibreNoise core shrinkage kernels.

### 1.2 Independent Gap Filling & Parameter Divergence
- Low-level implementation details omitted in academic publications (boundary padding, memory layout, SIMD vectorization paths, threshold tuning, and color-space variance stabilization) are developed independently and tuned exclusively on our own calibrated test data.
- **TAU parameter tables, constant dictionaries, or internal file structures must not be copied.**
- The project avoids proprietary names: we do **not** name the code, libraries, or functions `V-BM4D` or `BM3D`. Instead, terms such as `CollaborativePatchFilter3D` and `SpatiotemporalShrinkageEngine` are used.

### 1.3 Developer Certificate of Origin (DCO)
Every commit to this repository must include a `Signed-off-by:` trailer certifying that the contributor has the right to submit the work under the GPLv3 license and has adhered to the clean-room rules above.

---

## 2. Module Provenance Registry

| Module Path | Scientific / Literature Basis | Original Paper / Standard Citation | Inbound License | Authors / Implementers | Clean-Room Attestation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `include/librenoise/noise_model.hpp`<br>`src/core/noise_model.cpp` | Heteroscedastic noise level function (NLF) & Poisson-Gaussian variance stabilization | Abdelhamed et al. (CVPR 2018), *High-Quality Exposure Tracking and Denoising via SIDD Benchmark* | Original Clean Code (GPLv3) | LibreNoise Core Team | No TAU code viewed; spec-driven. |
| `include/librenoise/alignment.hpp`<br>`src/core/alignment.cpp` | Hierarchical block matching with noise-profile trust gating | Standard motion estimation literature & independent residual trust metric | Original Clean Code (GPLv3) | LibreNoise Core Team | Independent implementation. |
| `include/librenoise/filter.hpp`<br>`src/core/filter.cpp` | 3D transform shrinkage over motion-aligned patch groups | Maggioni, Boracchi, Foi, Egiazarian (IEEE TIP 2012), *Video Denoising, Deblurring, and Highly Displaced PCA Reconstruction* | Original Clean Code (GPLv3) | LibreNoise Core Team | Strictly coded from `docs/specs/clean_room_vbm4d_spec.md`; zero TAU code exposure. |
| `include/librenoise/detail.hpp`<br>`src/core/detail.cpp` | Edge-protected high-frequency restoration and profile-matched grain synthesis | Independent algorithm design | Original Clean Code (GPLv3) | LibreNoise Core Team | Original work. |
| `src/ofx/` | OpenFX 1.5 Image Effect Plugin API | Academy Software Foundation OpenFX 1.5 Specification | BSD-3-Clause (API headers) | LibreNoise Core Team | Standard OFX host integration. |

---

## 3. Host Dynamic Linking Statement
LibreNoise is licensed under the GNU General Public License v3.0 (GPLv3). In accordance with standard visual effects industry practice (such as OpenFX plugins loaded by commercial NLEs and grading suites like DaVinci Resolve), the plugin operates as an independent dynamic shared library loaded at runtime through the standardized OpenFX C API.
