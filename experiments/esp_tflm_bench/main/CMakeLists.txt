idf_component_register(
    SRCS "main.cc" "model.cc"
    INCLUDE_DIRS "."
    REQUIRES esp-tflite-micro esp_timer ecg_ai)
