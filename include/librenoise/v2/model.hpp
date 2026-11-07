#pragma once
// LibreNoise v0.2 noise model and processing parameters.
//
// The noise model has three measured parts, none of them taken from a stored camera or film-stock table:
//   1. A brightness curve per R, G, B channel: noise sigma as a function of the pixel value, measured over
//      the whole frame. It drives a variance-stabilising transform (VST) that makes noise unit-strength at
//      every brightness, so everything downstream works with one noise level.
//   2. A noise spectrum per working channel (Y, Cb, Cr) measured from flat areas in the stabilised domain,
//      converted to a per-scale, per-DCT-coefficient noise table for the multi-scale filter.
//   3. Source classification (film / digital sensor / compressed video) and compression-grid detection.

#include <array>
#include <cmath>
#include <cstdint>
#include <string>

namespace librenoise::v2 {

enum class SourceType : int32_t {
    Auto = 0,
    Film = 1,
    DigitalSensor = 2,
    CompressedVideo = 3
};

const char* source_type_name(SourceType t) noexcept;

constexpr int kCurveKnots = 37;                 // log2 grid, 0.5 stop spacing
constexpr float kCurveLog2Min = -14.0f;         // 2^-14 ~ 0.00006
constexpr float kCurveLog2Max = 4.0f;           // 2^4 = 16 (scene-linear highlights)
constexpr int kLevels = 5;                      // band-pass pyramid levels: High, Mid, Low, VeryLow, UltraLow
constexpr int kCoefs = 64;                      // 8x8 DCT

inline float curve_knot_log2(int k) noexcept {
    return kCurveLog2Min + (kCurveLog2Max - kCurveLog2Min) * static_cast<float>(k) / static_cast<float>(kCurveKnots - 1);
}
inline float curve_knot_x(int k) noexcept { return std::exp2(curve_knot_log2(k)); }

struct NoiseModel {
    bool valid = false;

    // Brightness curve: sigma at each knot x_k, per R, G, B channel.
    std::array<std::array<float, kCurveKnots>, 3> sigma{};
    // Which knots were measured (1) rather than filled from the fitted model (0), per channel.
    std::array<std::array<uint8_t, kCurveKnots>, 3> measured{};
    // Fitted model per channel: sigma^2 = floor^2 + (gain * x^exponent)^2
    std::array<float, 3> fit_floor{};
    std::array<float, 3> fit_gain{};
    std::array<float, 3> fit_exponent{};

    // Noise correlation between channels in the stabilised domain: RG, GB, RB.
    std::array<float, 3> corr{};

    // Noise std per pyramid level, per working channel (Y, Cb, Cr), per 8x8 DCT coefficient,
    // in stabilised units (1.0 = the brightness-curve noise level).
    float level_sigma[kLevels][3][kCoefs]{};
    // Per-pixel noise std of each working channel at full resolution (stabilised units).
    std::array<float, 3> pixel_sigma{1.0f, 1.0f, 1.0f};
    // Radial noise power spectrum per working channel (16 bins over 0..0.5 cycles/pixel), diagnostics only.
    std::array<std::array<float, 16>, 3> radial_psd{};

    // Classification
    SourceType detected_type = SourceType::Film;
    float film_score = 0.0f;       // 0..1, evidence for film grain over a demosaiced sensor
    float compression = 0.0f;      // 0..1 strength of block-grid artifacts
    int32_t grid_phase_x = -1;     // boundary sits between x and x+1 where x % 8 == phase; -1 = none
    int32_t grid_phase_y = -1;
    float grid_excess_x = 0.0f;    // relative step excess at the grid boundary
    float grid_excess_y = 0.0f;

    // Profile quality
    float quality = 0.0f;          // 0..100
    int32_t windows_used = 0;
    int32_t sample_x = 0, sample_y = 0, sample_w = 0, sample_h = 0; // flattest area found (for the UI box)
    std::string assessment;

    std::string serialize() const;
    static bool deserialize(const std::string& text, NoiseModel& out);
    std::string summary() const;   // one line for UI / logs
};

// Variance-stabilising lookup tables built from a NoiseModel. Plain arrays so the struct can be copied to a GPU.
constexpr int kVstFwd = 2048;
constexpr int kVstInv = 8192;

struct VstLut {
    float fwd[3][kVstFwd];   // f(x) sampled on a log2 grid between x0 and x1
    float inv[3][kVstInv];   // log2(x) sampled uniformly in f between f0 and f1
    float x0, x1, log2x0, log2x1;
    float sigma0[3];         // sigma below x0 (linear region, handles zero and negative values)
    float sigma1[3];         // sigma above x1 (linear extension)
    float f0[3], f1[3];
};

void build_vst(const NoiseModel& model, VstLut& lut);
float model_sigma_at(const NoiseModel& model, int channel, float x) noexcept;

// Processing parameters. Frequency arrays are indexed High, Mid, Low, VeryLow, UltraLow.
struct Params {
    int32_t temporal_radius = 2;           // frames each side, 0..5
    float temporal_strength = 1.0f;        // tolerance of the temporal match (higher = more averaging)
    float flicker_reduction = 0.0f;        // 0..1: ignore slow brightness changes when matching (smooths flicker)
    bool skip_repeated_frames = true;      // exact duplicate frames do not count as new samples

    // Threshold multiplier per frequency. High/Mid default slightly below 1 to protect fine detail
    // (tuned against reference renders on film, Alexa ProRes and H.264 test footage).
    float noise_level[5] = {0.8f, 0.9f, 1.0f, 1.0f, 1.0f};
    float channel_noise_level[3] = {1.0f, 1.0f, 1.0f};          // Y, Cb, Cr
    float amount_band[5] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};      // fraction of detected noise removed per frequency
    float amount_channel[3] = {1.0f, 1.0f, 1.0f};               // Y, Cb, Cr
    float tonal[5] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};            // threshold multiplier: shadows .. highlights

    float sharpening = 0.0f;               // 0..1, boosts denoised High/Mid detail
    float artifact_removal = 1.0f;         // 0..2 deblocking strength when a compression grid is present
    SourceType source_type = SourceType::Auto;  // Auto uses the model's detected type

    int32_t step = 4;                      // DCT block step (2 = best, 4 = balanced, 6 = fast)
    float grain = 0.0f;                    // re-add measured-spectrum grain (0..1)
    uint32_t grain_seed = 1337;
    float mix_original = 0.0f;
    int32_t preview = 0;                   // 0 = output, 1 = removed noise (x2 + 0.5)

    // Host cooperation
    bool (*abort_fn)(void*) = nullptr;
    void* abort_user = nullptr;
    int32_t max_threads = 0;
    bool aborted() const noexcept { return abort_fn != nullptr && abort_fn(abort_user); }
};

} // namespace librenoise::v2
