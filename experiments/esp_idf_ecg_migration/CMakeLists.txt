cmake_minimum_required(VERSION 3.5)

# Generated sdkconfig entries can still follow SDKCONFIG_DEFAULTS. Select the
# board defaults before IDF loads them, rather than silently applying 16MB ones.
get_filename_component(_ecg_config_name "${SDKCONFIG}" NAME)
if(_ecg_config_name MATCHES "^sdkconfig\\.supermini($|\\.)")
    set(_ecg_supermini_config TRUE)
    if(NOT SDKCONFIG_DEFAULTS)
        set(SDKCONFIG_DEFAULTS "${CMAKE_CURRENT_LIST_DIR}/sdkconfig.defaults.supermini")
    endif()
endif()
option(ECG_FIXED_INPUT_BOARD_TEST "Replay only; no AFE, storage or radio initialization" OFF)
include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(esp_idf_ecg_migration)

if(_ecg_supermini_config AND
   (NOT CONFIG_ESPTOOLPY_FLASHSIZE_4MB OR
    NOT CONFIG_PARTITION_TABLE_CUSTOM_FILENAME STREQUAL "partitions_ecg_supermini.csv"))
    message(FATAL_ERROR "SuperMini requires 4MB flash and partitions_ecg_supermini.csv; use sdkconfig.defaults.supermini")
endif()
