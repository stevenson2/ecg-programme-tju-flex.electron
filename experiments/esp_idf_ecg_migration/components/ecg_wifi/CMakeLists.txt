idf_component_register(
    SRCS "src/ecg_wifi.cpp"
    INCLUDE_DIRS "include"
    REQUIRES esp_wifi esp_netif esp_event esp_http_server nvs_flash ecg_storage)

# Inject WiFi AP password from environment (same mechanism as the Arduino
# scripts/inject_build_flags.py: $env:ECG_WIFI_AP_PASSWORD="...").
# ecg_wifi.h falls back to the dev default "12345678" when not set.
if(DEFINED ENV{ECG_WIFI_AP_PASSWORD})
    target_compile_definitions(${COMPONENT_LIB} PRIVATE
        ECG_WIFI_AP_PASSWORD="$ENV{ECG_WIFI_AP_PASSWORD}")
endif()
