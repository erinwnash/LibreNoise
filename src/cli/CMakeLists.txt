add_executable(librenoise_cli
    main.cpp
)

target_link_libraries(librenoise_cli PRIVATE librenoise_core)
target_compile_features(librenoise_cli PRIVATE cxx_std_20)
