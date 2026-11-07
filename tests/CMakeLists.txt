add_executable(librenoise_tests
    test_core.cpp
)

target_link_libraries(librenoise_tests PRIVATE librenoise_core)
target_compile_features(librenoise_tests PRIVATE cxx_std_20)
if(MSVC)
    target_compile_options(librenoise_tests PRIVATE /openmp /fp:fast)
endif()
if(LIBRENOISE_HAS_CUDA)
    target_include_directories(librenoise_tests PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(librenoise_tests PRIVATE CUDA::cudart)
endif()

enable_testing()
add_test(NAME CoreUnitTests COMMAND librenoise_tests)
add_test(NAME CliUsage COMMAND librenoise_cli --help)
