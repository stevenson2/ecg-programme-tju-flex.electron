idf_component_register(
    SRCS "src/ble.cpp"
    INCLUDE_DIRS "include"
    PRIV_REQUIRES bt nvs_flash)
target_compile_options(${COMPONENT_LIB} PRIVATE -Wno-error=missing-field-initializers)
