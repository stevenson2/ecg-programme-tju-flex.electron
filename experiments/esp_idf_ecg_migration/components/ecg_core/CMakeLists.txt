idf_component_register(
    SRCS
        "src/filter.cpp"
        "src/heartrate.cpp"
        "src/rhythm_safety.cpp"
        "src/af_detect.cpp"
        "src/vf_detect.cpp"
        "src/signal_generator/ecg_simulator.cpp"
        "src/signal_generator/ecg_replay.cpp"
    INCLUDE_DIRS "include"
    REQUIRES esp_timer
)
target_compile_options(${COMPONENT_LIB} PRIVATE -Wno-error=missing-field-initializers)
