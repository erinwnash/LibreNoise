add_library(librenoise_ofx MODULE
    plugin_main.cpp
)

target_link_libraries(librenoise_ofx PRIVATE librenoise_core)
target_compile_features(librenoise_ofx PRIVATE cxx_std_20)

target_include_directories(librenoise_ofx PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

set_target_properties(librenoise_ofx PROPERTIES
    PREFIX ""
    SUFFIX ".ofx"
    OUTPUT_NAME "LibreNoise"
)

# OpenFX standard bundle packaging
if(WIN32)
    set(OFX_ARCH "Win64")
elseif(APPLE)
    set(OFX_ARCH "MacOS")
else()
    set(OFX_ARCH "Linux-x86-64")
endif()

set(BUNDLE_DIR "${CMAKE_BINARY_DIR}/LibreNoise.ofx.bundle/Contents/${OFX_ARCH}")

add_custom_command(TARGET librenoise_ofx POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${BUNDLE_DIR}"
    COMMAND ${CMAKE_COMMAND} -E copy "$<TARGET_FILE:librenoise_ofx>" "${BUNDLE_DIR}/LibreNoise.ofx"
    COMMENT "Packaging LibreNoise into standard OpenFX bundle (${BUNDLE_DIR})"
)
