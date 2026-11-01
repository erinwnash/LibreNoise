#pragma once

#include "types.hpp"
#include "noise_model.hpp"
#include "roi_analyzer.hpp"
#include "profile_curve.hpp"

#include <string>
#include <vector>

namespace librenoise {

struct LibreNoiseDeviceProfile {
    std::string format_version = "1.0";
    std::string device_name = "Generic Camera";
    std::string mode_description = "ISO 3200";
    int32_t resolution_width = 1920;
    int32_t resolution_height = 1080;
    
    float uniformity_score = 90.0f;
    std::string quality_assessment = "Good";
    AnalysisRoi analysis_roi{0, 0, 128, 128};

    // Heteroscedastic noise level parameters per channel
    NoiseLevelFunction nlf_y{0.001f, 0.0001f};
    NoiseLevelFunction nlf_cb{0.0008f, 0.00008f};
    NoiseLevelFunction nlf_cr{0.0008f, 0.00008f};

    // 5-Band measured spatial frequency PSD
    std::array<float, 5> psd_5band{1.0f, 1.0f, 1.0f, 1.0f, 1.0f};

    // Tonal curve tuning knots
    NoiseProfileCurve tonal_curve;
};

class ProfileSerializer {
public:
    /**
     * @brief Save device noise profile to a .lnp (JSON) file.
     */
    static bool save_to_file(const std::string& filepath, const LibreNoiseDeviceProfile& profile);

    /**
     * @brief Load device noise profile from a .lnp (JSON) file.
     */
    static bool load_from_file(const std::string& filepath, LibreNoiseDeviceProfile& profile);

    /**
     * @brief Returns default user profile storage directory path on the current OS.
     * (%APPDATA%/LibreNoise/Profiles on Windows, ~/.librenoise/profiles on Linux/macOS)
     */
    static std::string get_default_profiles_directory();
};

} // namespace librenoise
