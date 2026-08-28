idf_component_register(
    SRCS "src/ecg_recorder.cpp"
    INCLUDE_DIRS "include"
    REQUIRES spiffs esp_timer
)
