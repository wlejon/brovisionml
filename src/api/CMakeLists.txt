add_library(brovisionml_api STATIC
    api.cpp
    host_class.cpp
    native_vision_models.cpp
    native_vision_ops.cpp
)

target_include_directories(brovisionml_api PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../include>
)

target_link_libraries(brovisionml_api PUBLIC
    brovisionml
    brotensor
    bronze_runtime_shared
)

target_compile_features(brovisionml_api PUBLIC cxx_std_20)

if(MSVC)
    target_compile_options(brovisionml_api PRIVATE /W4)
else()
    target_compile_options(brovisionml_api PRIVATE -Wall -Wextra)
endif()
