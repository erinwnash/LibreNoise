add_executable(librenoise_cli
    main.cpp
    exr_impl.cpp
    ${CMAKE_SOURCE_DIR}/third_party/tinyexr/miniz.c
)

target_include_directories(librenoise_cli PRIVATE ${CMAKE_SOURCE_DIR}/third_party/tinyexr)
target_link_libraries(librenoise_cli PRIVATE librenoise_core)
target_compile_features(librenoise_cli PRIVATE cxx_std_20)
if(MSVC)
    target_compile_options(librenoise_cli PRIVATE /openmp)
    set_source_files_properties(${CMAKE_SOURCE_DIR}/third_party/tinyexr/miniz.c exr_impl.cpp PROPERTIES COMPILE_OPTIONS "/W0")
endif()

if(LIBRENOISE_HAS_CUDA)
    target_include_directories(librenoise_cli PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(librenoise_cli PRIVATE CUDA::cudart)
endif()
