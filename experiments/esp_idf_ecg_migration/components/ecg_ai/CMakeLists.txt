set(_ecg_ai_srcs "src/ecg_ai.cpp")
if(ECG_AI_MODEL_XLSMALL)
    list(APPEND _ecg_ai_srcs "src/xl_pipeline.cpp" "src/xl_model.cpp" "src/xl_d1.cpp" "src/model_xl_s42.cc" "src/model_xl_s43.cc")
endif()
idf_component_register(
    SRCS ${_ecg_ai_srcs}
    INCLUDE_DIRS "include" "src"
    REQUIRES esp-tflite-micro esp_timer
)
