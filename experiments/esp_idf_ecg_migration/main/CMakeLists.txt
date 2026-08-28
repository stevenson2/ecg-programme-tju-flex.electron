idf_component_register(
    SRCS "main.cc" "model.cc"
    INCLUDE_DIRS "."
    REQUIRES ecg_ai ecg_core ecg_storage ecg_wifi ecg_ble ecg_afe esp_timer esp_driver_gpio)
