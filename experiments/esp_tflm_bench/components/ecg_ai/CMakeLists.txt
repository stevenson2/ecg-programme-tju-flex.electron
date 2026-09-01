idf_component_register(
    SRCS "src/ecg_ai.cpp"
    INCLUDE_DIRS "include"
    REQUIRES esp-tflite-micro esp_timer
)
