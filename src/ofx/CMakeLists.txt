add_library(librenoise_ofx MODULE
    plugin_main.cpp
    ofx_interact.cpp
)

target_link_libraries(librenoise_ofx PRIVATE librenoise_core)
target_compile_features(librenoise_ofx PRIVATE cxx_std_20)

if(MSVC)
    target_compile_options(librenoise_ofx PRIVATE /arch:AVX2 /fp:fast /openmp)
else()
    target_compile_options(librenoise_ofx PRIVATE -mavx2 -mfma -ffast-math -fopenmp)
endif()

if(WIN32)
    target_link_libraries(librenoise_ofx PRIVATE opengl32)
elseif(APPLE)
    target_link_libraries(librenoise_ofx PRIVATE "-framework OpenGL")
else()
    target_link_libraries(librenoise_ofx PRIVATE GL)
endif()

target_include_directories(librenoise_ofx PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

if(LIBRENOISE_HAS_CUDA)
    target_include_directories(librenoise_ofx PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(librenoise_ofx PRIVATE CUDA::cudart)
endif()

set_target_properties(librenoise_ofx PROPERTIES
    PREFIX ""
    SUFFIX ".ofx"
    OUTPUT_NAME "LibreNoise_v02"
)

# OpenFX standard bundle packaging
if(WIN32)
    set(OFX_ARCH "Win64")
elseif(APPLE)
    set(OFX_ARCH "MacOS")
else()
    set(OFX_ARCH "Linux-x86-64")
endif()

set(BUNDLE_DIR "${CMAKE_BINARY_DIR}/LibreNoise_v02.ofx.bundle/Contents/${OFX_ARCH}")

add_custom_command(TARGET librenoise_ofx POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${BUNDLE_DIR}"
    COMMAND ${CMAKE_COMMAND} -E copy "$<TARGET_FILE:librenoise_ofx>" "${BUNDLE_DIR}/LibreNoise_v02.ofx"
    COMMENT "Packaging LibreNoise into standard OpenFX bundle (${BUNDLE_DIR})"
)

if(WIN32 AND LIBRENOISE_HAS_CUDA)
    file(GLOB CUDART_DLLS "${CUDAToolkit_BIN_DIR}/cudart64_*.dll")
    foreach(CUDART_DLL ${CUDART_DLLS})
        add_custom_command(TARGET librenoise_ofx POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CUDART_DLL}" "${BUNDLE_DIR}/"
            COMMENT "Bundling CUDA runtime DLL (${CUDART_DLL}) into OpenFX bundle"
        )
    endforeach()
endif()
