#include "librenoise/profile_io.hpp"

#include <fstream>
#include <sstream>
#include <iomanip>
#include <cstdlib>
#include <filesystem>
#include <iostream>

namespace librenoise {

namespace {

std::string escape_json_string(const std::string& input) {
    std::string out;
    for (char c : input) {
        if (c == '"') out += "\\\"";
        else if (c == '\\') out += "\\\\";
        else if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else if (c == '\t') out += "\\t";
        else out += c;
    }
    return out;
}

// Simple key extraction helper from JSON
std::string extract_json_string(const std::string& json, const std::string& key) {
    std::string pattern = "\"" + key + "\"";
    size_t pos = json.find(pattern);
    if (pos == std::string::npos) return "";
    pos = json.find(':', pos);
    if (pos == std::string::npos) return "";
    size_t start = json.find('"', pos);
    if (start == std::string::npos) return "";
    size_t end = json.find('"', start + 1);
    if (end == std::string::npos) return "";
    return json.substr(start + 1, end - start - 1);
}

float extract_json_float(const std::string& json, const std::string& key, float default_val = 0.0f) {
    std::string pattern = "\"" + key + "\"";
    size_t pos = json.find(pattern);
    if (pos == std::string::npos) return default_val;
    pos = json.find(':', pos);
    if (pos == std::string::npos) return default_val;
    size_t val_start = json.find_first_not_of(" \t\r\n", pos + 1);
    if (val_start == std::string::npos) return default_val;
    try {
        return std::stof(json.substr(val_start));
    } catch (...) {
        return default_val;
    }
}

int32_t extract_json_int(const std::string& json, const std::string& key, int32_t default_val = 0) {
    std::string pattern = "\"" + key + "\"";
    size_t pos = json.find(pattern);
    if (pos == std::string::npos) return default_val;
    pos = json.find(':', pos);
    if (pos == std::string::npos) return default_val;
    size_t val_start = json.find_first_not_of(" \t\r\n", pos + 1);
    if (val_start == std::string::npos) return default_val;
    try {
        return std::stoi(json.substr(val_start));
    } catch (...) {
        return default_val;
    }
}

} // anonymous namespace

bool ProfileSerializer::save_to_file(const std::string& filepath, const LibreNoiseDeviceProfile& profile) {
    try {
        std::filesystem::path p(filepath);
        if (p.has_parent_path()) {
            std::filesystem::create_directories(p.parent_path());
        }

        std::ofstream ofs(filepath);
        if (!ofs.is_open()) return false;

        ofs << "{\n";
        ofs << "  \"format\": \"LibreNoiseProfile\",\n";
        ofs << "  \"version\": \"" << escape_json_string(profile.format_version) << "\",\n";
        ofs << "  \"metadata\": {\n";
        ofs << "    \"device_name\": \"" << escape_json_string(profile.device_name) << "\",\n";
        ofs << "    \"mode_description\": \"" << escape_json_string(profile.mode_description) << "\",\n";
        ofs << "    \"resolution_width\": " << profile.resolution_width << ",\n";
        ofs << "    \"resolution_height\": " << profile.resolution_height << "\n";
        ofs << "  },\n";
        ofs << "  \"calibration\": {\n";
        ofs << "    \"uniformity_score\": " << std::fixed << std::setprecision(2) << profile.uniformity_score << ",\n";
        ofs << "    \"quality_assessment\": \"" << escape_json_string(profile.quality_assessment) << "\",\n";
        ofs << "    \"roi_x\": " << profile.analysis_roi.x << ",\n";
        ofs << "    \"roi_y\": " << profile.analysis_roi.y << ",\n";
        ofs << "    \"roi_w\": " << profile.analysis_roi.width << ",\n";
        ofs << "    \"roi_h\": " << profile.analysis_roi.height << ",\n";
        ofs << "    \"nlf\": {\n";
        ofs << "      \"y_a\": " << std::scientific << std::setprecision(6) << profile.nlf_y.a << ",\n";
        ofs << "      \"y_b\": " << profile.nlf_y.b << ",\n";
        ofs << "      \"cb_a\": " << profile.nlf_cb.a << ",\n";
        ofs << "      \"cb_b\": " << profile.nlf_cb.b << ",\n";
        ofs << "      \"cr_a\": " << profile.nlf_cr.a << ",\n";
        ofs << "      \"cr_b\": " << profile.nlf_cr.b << "\n";
        ofs << "    },\n";
        ofs << "    \"psd_5band\": [\n";
        for (int i = 0; i < 5; ++i) {
            ofs << "      " << std::fixed << std::setprecision(4) << profile.psd_5band[i] << (i < 4 ? ",\n" : "\n");
        }
        ofs << "    ],\n";
        ofs << "    \"tonal_curve_points\": [\n";
        const auto& pts = profile.tonal_curve.points();
        for (size_t i = 0; i < pts.size(); ++i) {
            ofs << "      {\"x\": " << std::fixed << std::setprecision(3) << pts[i].x
                << ", \"y\": " << std::setprecision(3) << pts[i].y << "}" << (i + 1 < pts.size() ? ",\n" : "\n");
        }
        ofs << "    ]\n";
        ofs << "  }\n";
        ofs << "}\n";

        return true;
    } catch (...) {
        return false;
    }
}

bool ProfileSerializer::load_from_file(const std::string& filepath, LibreNoiseDeviceProfile& profile) {
    try {
        std::ifstream ifs(filepath);
        if (!ifs.is_open()) return false;

        std::stringstream buffer;
        buffer << ifs.rdbuf();
        std::string json = buffer.str();

        if (json.find("\"LibreNoiseProfile\"") == std::string::npos) {
            return false;
        }

        profile.format_version = extract_json_string(json, "version");
        profile.device_name = extract_json_string(json, "device_name");
        profile.mode_description = extract_json_string(json, "mode_description");
        profile.resolution_width = extract_json_int(json, "resolution_width", 1920);
        profile.resolution_height = extract_json_int(json, "resolution_height", 1080);
        profile.uniformity_score = extract_json_float(json, "uniformity_score", 85.0f);
        profile.quality_assessment = extract_json_string(json, "quality_assessment");

        profile.analysis_roi.x = extract_json_int(json, "roi_x", 0);
        profile.analysis_roi.y = extract_json_int(json, "roi_y", 0);
        profile.analysis_roi.width = extract_json_int(json, "roi_w", 128);
        profile.analysis_roi.height = extract_json_int(json, "roi_h", 128);

        profile.nlf_y.a = extract_json_float(json, "y_a", 0.001f);
        profile.nlf_y.b = extract_json_float(json, "y_b", 0.0001f);
        profile.nlf_cb.a = extract_json_float(json, "cb_a", 0.0008f);
        profile.nlf_cb.b = extract_json_float(json, "cb_b", 0.00008f);
        profile.nlf_cr.a = extract_json_float(json, "cr_a", 0.0008f);
        profile.nlf_cr.b = extract_json_float(json, "cr_b", 0.00008f);

        // PSD 5-Band array extraction
        size_t psd_pos = json.find("\"psd_5band\"");
        if (psd_pos != std::string::npos) {
            size_t bracket = json.find('[', psd_pos);
            size_t close_bracket = json.find(']', bracket);
            if (bracket != std::string::npos && close_bracket != std::string::npos) {
                std::string arr_str = json.substr(bracket + 1, close_bracket - bracket - 1);
                std::stringstream ss(arr_str);
                std::string token;
                int idx = 0;
                while (std::getline(ss, token, ',') && idx < 5) {
                    try {
                        profile.psd_5band[idx++] = std::stof(token);
                    } catch (...) {}
                }
            }
        }

        return true;
    } catch (...) {
        return false;
    }
}

std::string ProfileSerializer::get_default_profiles_directory() {
#if defined(_WIN32) || defined(__WIN32__)
    const char* appdata = std::getenv("APPDATA");
    if (appdata) {
        return std::string(appdata) + "\\LibreNoise\\Profiles\\";
    }
    return "C:\\ProgramData\\LibreNoise\\Profiles\\";
#else
    const char* home = std::getenv("HOME");
    if (home) {
        return std::string(home) + "/.librenoise/profiles/";
    }
    return "/etc/librenoise/profiles/";
#endif
}

} // namespace librenoise
