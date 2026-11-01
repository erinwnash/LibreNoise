add_library(librenoise_core STATIC
    noise_model.cpp
    alignment.cpp
    filter.cpp
    detail.cpp
    core_pipeline.cpp
)

target_include_directories(librenoise_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../include>
        $<INSTALL_INTERFACE:include>
)

target_compile_features(librenoise_core PUBLIC cxx_std_20)

if(MSVC)
    target_compile_options(librenoise_core PRIVATE /W4 /arch:AVX2 /fp:fast)
else()
    target_compile_options(librenoise_core PRIVATE -Wall -Wextra -mavx2 -mfma -ffast-math)
endif()
