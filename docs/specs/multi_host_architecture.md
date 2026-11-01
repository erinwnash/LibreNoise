# LibreNoise Multi-Host Architecture & Integration Specification

## 1. Executive Summary & Design Philosophy
LibreNoise is designed from first principles with a decoupled, host-agnostic architecture. The core compute pipeline (`librenoise_core`), comprising the 5-stage denoiser, CUDA sm_86 GPU kernels, ROI uniformity analyzer, cubic Hermite tonal spline model, and `.lnp` profile I/O, is completely independent of any video host SDK.

Host-specific adapters expose native interfaces, viewer overlays, and parameter layouts while sharing the exact same underlying GPU/CPU denoiser and profile calibration files:

```mermaid
flowchart TD
    subgraph Core Engine [librenoise_core]
        CUDA[CUDA sm_86 GPU Pipeline]
        CPU[AVX2 / Multi-Threaded CPU Engine]
        ROI[ROI Uniformity & Edge Rejection Analyzer]
        Spline[Monotonic Cubic Hermite Spline Model]
        LNP[Profile Serializer .lnp]
    end

    subgraph Host Integrations
        OFX[OpenFX 1.5 Plugin Bundle<br/>LibreNoise.ofx.bundle]
        Adobe[Adobe MediaCore Plugin<br/>LibreNoise.aex]
        FxPlug[Apple Final Cut Pro<br/>FxPlug 4 Metal Extension]
        AVX[Avid Media Composer<br/>AVX2 Plugin]
        VDub[VirtualDub Filter<br/>LibreNoise.vdf]
        Edius[Grass Valley EDIUS<br/>Custom Effect Filter]
    end

    OFX --> Core Engine
    Adobe --> Core Engine
    FxPlug --> Core Engine
    AVX --> Core Engine
    VDub --> Core Engine
    Edius --> Core Engine
```

---

## 2. Multi-Host Compatibility Matrix

### 2.1 OpenFX (OFX 1.5) Standard Hosts
A single unified binary bundle (`LibreNoise.ofx.bundle`) natively serves all OpenFX-compliant video applications. It implements the standard OpenFX 1.5 specification including GPU rendering (`kOfxImageEffectPropCudaRenderSupported`, `kOfxImageEffectPropCudaStreamSupported`) and viewer overlays (`kOfxImageEffectPluginPropOverlayInteractV1`).

| Host Application | Vendor | Context | GPU Rendering | Interactive Viewer Overlay | Installation Path (Windows) |
|---|---|---|---|---|---|
| **Nuke / NukeX / Nuke Studio** | The Foundry | Filter / General | CUDA sm_86 Zero-Copy | Native OpenGL Bounding Box | `C:\Program Files\Common Files\OFX\Plugins\LibreNoise.ofx.bundle` |
| **DaVinci Resolve / Studio** | Blackmagic Design | Color / Fusion | CUDA Direct Stream | Native OpenFX Interact Overlay | `C:\Program Files\Common Files\OFX\Plugins\LibreNoise.ofx.bundle` |
| **Fusion Studio** | Blackmagic Design | Tool Node | CUDA / CPU Fallback | Native OpenFX Interact Overlay | `C:\Program Files\Common Files\OFX\Plugins\LibreNoise.ofx.bundle` |
| **Flame** | Autodesk | Batch / Action | CUDA / CPU Fallback | Overlay Interact Supported | `/usr/discreet/openfx/LibreNoise.ofx.bundle` |
| **Baselight** | FilmLight | Grading Layer | CUDA / CPU Fallback | Viewer HUD & Overlay | `/usr/fl/plugins/openfx/LibreNoise.ofx.bundle` |
| **Mistika** | SGO | Node Graph | CUDA / CPU Fallback | Viewer Interact Box | `C:\Program Files\Common Files\OFX\Plugins\LibreNoise.ofx.bundle` |
| **Scratch** | Assimilate | Process Construct | CUDA / CPU Fallback | Viewer Overlay Construct | `C:\Program Files\Common Files\OFX\Plugins\LibreNoise.ofx.bundle` |
| **Natron** | Open Source | Node Graph | CPU / OpenGL | Viewer Overlay Bounding Box | `C:\Program Files\Common Files\OFX\Plugins\LibreNoise.ofx.bundle` |
| **DustBuster+** | HS-ART | Restoration Filter | CUDA / CPU Fallback | Viewer ROI Selector | `C:\Program Files\Common Files\OFX\Plugins\LibreNoise.ofx.bundle` |
| **Silhouette** | BorisFX | Node Filter | CUDA / CPU Fallback | Viewer Overlay ROI | `C:\Program Files\Common Files\OFX\Plugins\LibreNoise.ofx.bundle` |
| **HitFilm Pro** | FXHOME | Effect Stack | CUDA / CPU Fallback | Viewer Overlay Handles | `C:\Program Files\Common Files\OFX\Plugins\LibreNoise.ofx.bundle` |
| **Vegas Pro** | Magix / Sony | Video FX / Event FX | CUDA / CPU Fallback | Video Preview Window Overlay | `C:\Program Files\Common Files\OFX\Plugins\LibreNoise.ofx.bundle` |
| **Video Pro X** | Magix | Effect Track | CUDA / CPU Fallback | Preview Window Box Drag | `C:\Program Files\Common Files\OFX\Plugins\LibreNoise.ofx.bundle` |

### 2.2 Adobe MediaCore Hosts
Adobe Premiere Pro and Adobe After Effects share the Adobe MediaCore architecture (`LibreNoise.aex`).

| Host Application | Vendor | Plugin Format | Acceleration | Viewer Interaction | Installation Path (Windows) |
|---|---|---|---|---|---|
| **Adobe After Effects** | Adobe Inc. | Adobe Effect (`.aex`) | CUDA sm_86 / AVX2 | Composition Viewer Overlay (`PF_Cmd_EVENT`) | `C:\Program Files\Adobe\Common\Plug-ins\7.0\MediaCore\LibreNoise.aex` |
| **Adobe Premiere Pro** | Adobe Inc. | Adobe Effect (`.aex`) | CUDA sm_86 / AVX2 | Program Monitor Overlay (`PF_Cmd_EVENT`) | `C:\Program Files\Adobe\Common\Plug-ins\7.0\MediaCore\LibreNoise.aex` |

### 2.3 Apple Hosts (macOS)
| Host Application | Vendor | Architecture | Acceleration | Installation Path |
|---|---|---|---|---|
| **Final Cut Pro** | Apple Inc. | FxPlug 4 App Extension | Apple Silicon Metal Compute | `/Library/Plug-Ins/FxPlug/LibreNoise.fxplug` |

### 2.4 Avid Hosts
| Host Application | Vendor | Plugin Architecture | Installation Path |
|---|---|---|---|
| **Avid Media Composer** | Avid Technology | AVX2 (Avid Video Extension 2) | `C:\Program Files\Avid\AVX2_Plug-ins\LibreNoise.avx` |

### 2.5 Specialized & Classic Hosts
| Host Application | Architecture | Support Mechanism |
|---|---|---|
| **Grass Valley EDIUS** | EDIUS Custom Filter API | C++ Video Filter wrapper calling `librenoise_core` |
| **VirtualDub** | VirtualDub Filter Definition (`.vdf`) | Native VDF module wrapping `librenoise::VideoDenoiser` |

---

## 3. Profiling Workflow

LibreNoise provides an interactive profiling workflow while maintaining a 100% clean-room open-source codebase:

### 3.1 Interactive Viewer ROI Selection
- **Visual Display**: Renders a neon green rectangular bounding box (`#00FF80`) on the host viewer with a semi-transparent tint (`12% alpha`) for unobscured visibility of image texture.
- **Corner Handles**: 4 corner grab handles (`12x12 px`) allowing intuitive resizing.
- **Center Crosshair**: Indicates the exact centroid of the analysis patch.
- **Direct Viewer Dragging**: Click-and-drag directly on the viewer window to reposition the selection box or drag out an entirely new analysis region.

### 3.2 Automated Featureless Patch Detection ("Auto Profile")
A dedicated 3-stage heuristic algorithm sweeps the frame to find the ideal calibration region:
1. **Sobel Gradient Magnitude Check**: Computes local directional gradients $|\nabla I| = |G_x| + |G_y|$; rejects areas containing sharp edges, object boundaries, or prominent scene texture.
2. **Quadrant Disparity Variance**: Divides candidate patches into 4 equal quadrants and measures inter-quadrant mean variance $\text{Var}(\mu_1, \mu_2, \mu_3, \mu_4)$; rejects patches spanning gradients or lighting falloff.
3. **Exposure Gating**: Ensures sample intensity resides between $0.08$ and $0.92$, avoiding clipped black sensor floor or blown highlights.
4. **Quality Scoring**: Returns a calibrated 0–100% uniformity score and descriptive assessment (*"Excellent"*, *"Good"*, *"Unsuitable: Edges Detected"*).

### 3.3 Frequency & Tonal Profile Equalizer
- **5-Band Radial Spatial Frequency Equalizer**:
  - `Very Low` (Macro luma blotches / sensor pattern noise)
  - `Low` (Coarse grain)
  - `Mid` (Film grain / digital sensor noise)
  - `High` (Fine grain)
  - `Very High` (Sub-pixel high-frequency chroma and luma speckle)
- **Tonal Equalizer Spline**:
  - Monotonic cubic Hermite spline model ensuring smooth non-oscillating interpolation across image intensity.
  - Granular control over **Shadows (0–20%)**, **Dark Midtones (20–40%)**, **Midtones (40–60%)**, **Light Midtones (60–80%)**, and **Highlights (80–100%)**.

### 3.4 Profile Preset Serialization (`.lnp`)
- Standardized human-readable JSON profile format (`.lnp` - LibreNoise Profile).
- Stores camera device metadata, ISO/exposure mode, 5-band PSD spectrum, heteroscedastic noise level parameters ($a \cdot I + b$), and tonal curve knots.
- 100% interoperable across Nuke, DaVinci Resolve, Adobe Premiere Pro, After Effects, and command-line automated pipelines.
- Standard cross-platform storage directory:
  - Windows: `%APPDATA%\LibreNoise\Profiles\`
  - Linux: `~/.config/librenoise/profiles/`
  - macOS: `~/Library/Application Support/LibreNoise/Profiles/`
