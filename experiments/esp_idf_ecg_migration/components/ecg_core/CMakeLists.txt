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

# M2 噪声语料段 (30 case × 30s, int16 ~880KB): 仅 16MB flash 板 (N16R8) 编入。
# 4MB 板 (SuperMini) app 分区余量不足, 保持 3 段 (0/1/2)。
if(CONFIG_ESPTOOLPY_FLASHSIZE_16MB)
    target_compile_definitions(${COMPONENT_LIB} PUBLIC ECG_REPLAY_CORPUS=1)
endif()
