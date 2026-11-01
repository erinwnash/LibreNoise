add_executable(librenoise_tests
    test_core.cpp
)

target_link_libraries(librenoise_tests PRIVATE librenoise_core)
target_compile_features(librenoise_tests PRIVATE cxx_std_20)

enable_testing()
add_test(NAME CoreUnitTests COMMAND librenoise_tests)
add_test(NAME CliBenchmarkTest COMMAND librenoise_cli)
