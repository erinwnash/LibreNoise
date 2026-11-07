cmake_minimum_required(VERSION 3.20)
project(LibreNoise
    VERSION 0.2.0
    DESCRIPTION "High-Performance Open-Source OFX Video Denoiser"
    LANGUAGES C CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
    # Maintain full binary ABI compatibility when loaded into VFX hosts (Nuke 13-15, AE, Resolve)
    # that bundle pre-14.40 MSVCP140.dll in their application directory.
    add_compile_definitions(_DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR)
endif()

# CUDA Toolkit Detection
find_package(CUDAToolkit)
if(CUDAToolkit_FOUND)
    message(STATUS "CUDA Toolkit found: ${CUDAToolkit_VERSION} (include: ${CUDAToolkit_INCLUDE_DIRS})")
    set(LIBRENOISE_HAS_CUDA ON)
    add_compile_definitions(LIBRENOISE_HAS_CUDA=1)
else()
    message(STATUS "CUDA Toolkit not found, building CPU-only")
    set(LIBRENOISE_HAS_CUDA OFF)
endif()

# Global options
option(LIBRENOISE_BUILD_CLI "Build the headless CLI test harness" ON)
option(LIBRENOISE_BUILD_OFX "Build the OpenFX 1.5 plugin module" ON)
option(LIBRENOISE_BUILD_AE "Build the Adobe After Effects / Premiere Pro (.aex) plugin" ON)
option(LIBRENOISE_BUILD_TESTS "Build test suite" ON)

include_directories(include)

add_subdirectory(src/core)

if(LIBRENOISE_BUILD_CLI)
    add_subdirectory(src/cli)
endif()

if(LIBRENOISE_BUILD_OFX)
    add_subdirectory(src/ofx)
endif()

if(LIBRENOISE_BUILD_AE)
    add_subdirectory(src/ae)
endif()

if(LIBRENOISE_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
