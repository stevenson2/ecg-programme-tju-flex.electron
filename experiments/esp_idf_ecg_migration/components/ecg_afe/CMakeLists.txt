idf_component_register(
    SRCS "src/afe_hal.cpp"
    INCLUDE_DIRS "include"
    REQUIRES esp_adc esp_driver_gpio)
target_compile_options(${COMPONENT_LIB} PRIVATE -Wno-error=missing-field-initializers)
