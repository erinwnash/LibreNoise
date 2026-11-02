add_executable(librenoise_cli
    main.cpp
    stress.cpp
)

target_link_libraries(librenoise_cli PRIVATE librenoise_core)
target_compile_features(librenoise_cli PRIVATE cxx_std_20)

if(LIBRENOISE_HAS_CUDA)
    target_include_directories(librenoise_cli PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(librenoise_cli PRIVATE CUDA::cudart)
endif()
