find_package(OpenMP)

set(CORE_SOURCES
    v2/model.cpp
    v2/profiler.cpp
    v2/backend_cpu.cpp
    v2/profile_io.cpp
)

if(LIBRENOISE_HAS_CUDA)
    if(WIN32)
        set(CUDA_NVCC_EXECUTABLE "${CUDAToolkit_BIN_DIR}/nvcc.exe")
        set(CUDA_OBJ "${CMAKE_CURRENT_BINARY_DIR}/backend_cuda.obj")
        set(CUDA_HOST_FLAGS "-Xcompiler=/MD,/O2" -D_DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR)
    else()
        set(CUDA_NVCC_EXECUTABLE "${CUDAToolkit_BIN_DIR}/nvcc")
        set(CUDA_OBJ "${CMAKE_CURRENT_BINARY_DIR}/backend_cuda.o")
        set(CUDA_HOST_FLAGS "-Xcompiler=-fPIC,-O3")
    endif()

    add_custom_command(
        OUTPUT ${CUDA_OBJ}
        COMMAND ${CUDA_NVCC_EXECUTABLE}
            -c ${CMAKE_CURRENT_SOURCE_DIR}/v2/backend_cuda.cu
            -o ${CUDA_OBJ}
            -O3
            -std=c++20
            --extended-lambda
            --use_fast_math
            -allow-unsupported-compiler
            -arch=sm_86
            ${CUDA_HOST_FLAGS}
            -ccbin "${CMAKE_CXX_COMPILER}"
            -I${CMAKE_CURRENT_SOURCE_DIR}/../../include
            -I${CMAKE_CURRENT_SOURCE_DIR}/v2
            -I${CUDAToolkit_INCLUDE_DIRS}
        DEPENDS
            ${CMAKE_CURRENT_SOURCE_DIR}/v2/backend_cuda.cu
            ${CMAKE_CURRENT_SOURCE_DIR}/v2/pipeline_impl.inl
            ${CMAKE_CURRENT_SOURCE_DIR}/v2/kernels.hpp
            ${CMAKE_CURRENT_SOURCE_DIR}/../../include/librenoise/v2/model.hpp
            ${CMAKE_CURRENT_SOURCE_DIR}/../../include/librenoise/v2/pipeline.hpp
        COMMENT "Compiling LibreNoise v0.2 CUDA back end (sm_86)"
    )
    list(APPEND CORE_SOURCES ${CUDA_OBJ})
else()
    list(APPEND CORE_SOURCES v2/backend_cuda_stub.cpp)
endif()

add_library(librenoise_core STATIC ${CORE_SOURCES})

target_include_directories(librenoise_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../include>
        $<INSTALL_INTERFACE:include>
)

target_compile_features(librenoise_core PUBLIC cxx_std_20)

if(MSVC)
    target_compile_options(librenoise_core PRIVATE /W4 /arch:AVX2 /fp:fast /openmp /O2)
else()
    target_compile_options(librenoise_core PRIVATE -Wall -Wextra -ffast-math -O3)
    if(LIBRENOISE_X86)
        target_compile_options(librenoise_core PRIVATE -mavx2 -mfma)
    endif()
endif()

if(OpenMP_CXX_FOUND)
    target_link_libraries(librenoise_core PUBLIC OpenMP::OpenMP_CXX)
endif()

if(LIBRENOISE_HAS_CUDA)
    target_link_libraries(librenoise_core PUBLIC CUDA::cudart)
endif()
