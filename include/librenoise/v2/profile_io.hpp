// SPDX-License-Identifier: Apache-2.0
#pragma once
// .lnp v2 profile files: a small text header (device / mode labels) followed by the serialised NoiseModel.

#include "librenoise/v2/model.hpp"
#include <string>

namespace librenoise::v2 {

struct ProfileFile {
    std::string device_name = "Unspecified device";
    std::string mode_description;
    NoiseModel model;
};

bool save_profile(const std::string& path, const ProfileFile& profile);
bool load_profile(const std::string& path, ProfileFile& profile);

// %APPDATA%/LibreNoise/Profiles on Windows, ~/.librenoise/profiles elsewhere.
std::string default_profiles_directory();

} // namespace librenoise::v2
