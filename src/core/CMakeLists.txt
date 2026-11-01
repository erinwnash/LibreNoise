find_package(OpenMP)

set(CORE_SOURCES
    noise_model.cpp
    alignment.cpp
    filter.cpp
    detail.cpp
    core_pipeline.cpp
    roi_analyzer.cpp
    profile_curve.cpp
    profile_io.cpp
)

if(LIBRENOISE_HAS_CUDA)
    set(CUDA_NVCC_EXECUTABLE "${CUDAToolkit_BIN_DIR}/nvcc.exe")
    set(CUDA_OBJ "${CMAKE_CURRENT_BINARY_DIR}/denoise_cuda.obj")

    add_custom_command(
        OUTPUT ${CUDA_OBJ}
        COMMAND ${CUDA_NVCC_EXECUTABLE}
            -c ${CMAKE_CURRENT_SOURCE_DIR}/gpu/denoise_cuda.cu
            -o ${CUDA_OBJ}
            -O3
            -allow-unsupported-compiler
            -arch=sm_86
            -Xcompiler="/MD"
            -ccbin "${CMAKE_CXX_COMPILER}"
            -I${CMAKE_CURRENT_SOURCE_DIR}/../../include
            -I${CUDAToolkit_INCLUDE_DIRS}
        DEPENDS
            ${CMAKE_CURRENT_SOURCE_DIR}/gpu/denoise_cuda.cu
            ${CMAKE_CURRENT_SOURCE_DIR}/../../include/librenoise/gpu_denoiser.hpp
        COMMENT "Compiling CUDA GPU kernels (sm_86 Ampere)"
    )
    list(APPEND CORE_SOURCES ${CUDA_OBJ})
else()
    list(APPEND CORE_SOURCES gpu/denoise_cuda_stub.cpp)
endif()

add_library(librenoise_core STATIC
    ${CORE_SOURCES}
)

target_include_directories(librenoise_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../include>
        $<INSTALL_INTERFACE:include>
)

target_compile_features(librenoise_core PUBLIC cxx_std_20)

if(MSVC)
    target_compile_options(librenoise_core PRIVATE /W4 /arch:AVX2 /fp:fast /openmp)
else()
    target_compile_options(librenoise_core PRIVATE -Wall -Wextra -mavx2 -mfma -ffast-math -fopenmp)
endif()

if(OpenMP_CXX_FOUND)
    target_link_libraries(librenoise_core PUBLIC OpenMP::OpenMP_CXX)
endif()

if(LIBRENOISE_HAS_CUDA)
    target_link_libraries(librenoise_core PUBLIC CUDA::cudart)
endif()
