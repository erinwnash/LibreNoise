add_library(librenoise_ae MODULE
    ae_plugin.cpp
)

target_link_libraries(librenoise_ae PRIVATE librenoise_core)
target_compile_features(librenoise_ae PRIVATE cxx_std_20)

if(WIN32)
    target_link_libraries(librenoise_ae PRIVATE opengl32)
endif()

target_include_directories(librenoise_ae PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

if(LIBRENOISE_HAS_CUDA)
    target_include_directories(librenoise_ae PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(librenoise_ae PRIVATE CUDA::cudart)
endif()

set_target_properties(librenoise_ae PROPERTIES
    PREFIX ""
    SUFFIX ".aex"
    OUTPUT_NAME "LibreNoise"
)
