// SPDX-License-Identifier: Apache-2.0
#include "librenoise/v2/profile_io.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace librenoise::v2 {

namespace {
std::string one_line(std::string s) {
    for (char& c : s) if (c == '\n' || c == '\r') c = ' ';
    return s;
}
}

bool save_profile(const std::string& path, const ProfileFile& p) {
    std::error_code ec;
    const auto parent = std::filesystem::path(path).parent_path();
    if (!parent.empty()) std::filesystem::create_directories(parent, ec);
    std::ofstream o(path, std::ios::binary);
    if (!o) return false;
    o << "LNP2\n";
    o << "device " << one_line(p.device_name) << "\n";
    o << "mode " << one_line(p.mode_description) << "\n";
    o << "model\n" << p.model.serialize();
    return static_cast<bool>(o);
}

bool load_profile(const std::string& path, ProfileFile& p) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    std::string line;
    if (!std::getline(in, line) || line.rfind("LNP2", 0) != 0) return false;
    ProfileFile r;
    while (std::getline(in, line)) {
        if (line.rfind("device ", 0) == 0) r.device_name = line.substr(7);
        else if (line.rfind("mode ", 0) == 0) r.mode_description = line.substr(5);
        else if (line == "model") {
            std::stringstream rest;
            rest << in.rdbuf();
            if (!NoiseModel::deserialize(rest.str(), r.model)) return false;
            break;
        }
    }
    p = r;
    return true;
}

std::string default_profiles_directory() {
#if defined(_WIN32)
    const char* appdata = std::getenv("APPDATA");
    return std::string(appdata ? appdata : ".") + "\\LibreNoise\\Profiles";
#else
    const char* home = std::getenv("HOME");
    return std::string(home ? home : ".") + "/.librenoise/profiles";
#endif
}

} // namespace librenoise::v2
