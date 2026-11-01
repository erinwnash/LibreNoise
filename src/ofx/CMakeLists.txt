add_library(librenoise_ofx MODULE
    plugin_main.cpp
)

target_link_libraries(librenoise_ofx PRIVATE librenoise_core)
target_compile_features(librenoise_ofx PRIVATE cxx_std_20)

target_include_directories(librenoise_ofx PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

# OpenFX standard naming and extension
set_target_properties(librenoise_ofx PROPERTIES
    PREFIX ""
    SUFFIX ".ofx"
    OUTPUT_NAME "LibreNoise"
)
