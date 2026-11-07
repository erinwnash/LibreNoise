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
| `src/core/v2/profiler.cpp` | Noise level estimation from the lower envelope of block statistics; variance-stabilising transform built by integrating 1/sigma(x); Welch power-spectrum estimate | Liu, Szeliski et al. (CVPR 2006), *Noise Estimation from a Single Image*; Foi (2009), *Clipped noisy images: heteroskedastic modeling*; Welch (1967) | Original Clean Code (GPLv3) | LibreNoise Core Team | Independent implementation; no stored camera or film tables. |
| `src/core/v2/kernels.hpp`, `pipeline_impl.inl` | Laplacian pyramid; sliding-window 8x8 DCT hard-threshold + empirical Wiener shrinkage per pyramid level; motion-compensated temporal weighted averaging; block-boundary deblocking | Burt & Adelson (1983), *The Laplacian Pyramid as a Compact Image Code*; Yu & Sapiro (IPOL 2011), *DCT Image Denoising*; Buades, Coll, Morel (2005) non-local / temporal averaging; List et al. (2003), *Adaptive deblocking filter* | Original Clean Code (GPLv3) | LibreNoise Core Team | Written from the papers; no reference code used. |
| `src/core/v2/backend_cpu.cpp`, `backend_cuda.cu` | OpenMP and CUDA execution of the shared kernels | Engineering | Original Clean Code (GPLv3) | LibreNoise Core Team | Original work. |
| `third_party/tinyexr/` | OpenEXR reader/writer for the CLI | tinyexr v1.0.13 (BSD-3-Clause), miniz (MIT) | Third party, see `THIRD_PARTY_NOTICES.md` | Syoyo Fujita et al.; Rich Geldreich et al. | Unmodified upstream files. |
| v0.1 modules (`noise_model`, `alignment`, `filter`, `detail`) | 3D collaborative filtering (Maggioni et al., IEEE TIP 2012) and NLF modelling | See tag `v0.1.0` | Original Clean Code (GPLv3) | LibreNoise Core Team | Removed in v0.2; history kept in git. |
| `src/ofx/` | OpenFX 1.5 Image Effect Plugin API | Academy Software Foundation OpenFX 1.5 Specification | BSD-3-Clause (API headers) | LibreNoise Core Team | Standard OFX host integration. |

---

## 3. Host Dynamic Linking Statement
LibreNoise is licensed under the GNU General Public License v3.0 (GPLv3). In accordance with standard visual effects industry practice (such as OpenFX plugins loaded by commercial NLEs and grading suites like DaVinci Resolve), the plugin operates as an independent dynamic shared library loaded at runtime through the standardized OpenFX C API.
