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
option(ECG_AI_MODEL_XLSMALL "XL_small 2-model int8 N/S/V/X ensemble (default ON since v0.2.0; OFF = legacy v3-A)" ON)
include($ENV{IDF_PATH}/tools/cmake/project.cmake)
if(ECG_AI_MODEL_XLSMALL)
    idf_build_set_property(COMPILE_DEFINITIONS "ECG_AI_MODEL_XLSMALL=1" APPEND)
endif()
project(esp_idf_ecg_migration)

if(_ecg_supermini_config AND
   (NOT CONFIG_ESPTOOLPY_FLASHSIZE_4MB OR
    NOT CONFIG_PARTITION_TABLE_CUSTOM_FILENAME STREQUAL "partitions_ecg_supermini.csv"))
    message(FATAL_ERROR "SuperMini requires 4MB flash and partitions_ecg_supermini.csv; use sdkconfig.defaults.supermini")
endif()
