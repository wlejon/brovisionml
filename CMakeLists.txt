cmake_minimum_required(VERSION 3.24)
project(brovisionml LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(MSVC)
    add_compile_options(
        $<$<COMPILE_LANGUAGE:CXX>:/MP>
        $<$<COMPILE_LANGUAGE:CXX>:/permissive->
        $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
    )
    if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
        set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    endif()
endif()

# --- GPU backend selection ---
#
# brovisionml stays pure C++20: it composes brotensor ops and lowers its small
# set of domain-specific GPU kernels (DPT bicubic preprocess, DSINE RayReLU and
# AngMF propagate) from single-source Brass MIR through brotensor::interop — no
# CUDA, Objective-C++, or GLSL/glslc build toolchain required here.
option(BROTENSOR_WITH_CUDA  "Build the CUDA GPU backend"  OFF)
option(BROTENSOR_WITH_METAL "Build the Metal GPU backend" OFF)
option(BROTENSOR_WITH_VULKAN "Build the Vulkan GPU backend" OFF)

# --- Dependencies: bromath (header-only), bronze (+ brass), brotensor, broimage ---
#
# Each resolves through cmake/bro_deps.cmake, like every repo in the bro
# ecosystem: an existing target (a superbuild such as bro already added it, and
# its choice wins) first; otherwise the ../<name> working tree beside the top-level
# project; otherwise the head of its main, fetched at configure.
#
# Order matters: leaves first, so broimage (which links bromath and brotensor,
# and brass for its JIT) reuses these targets rather than adding a second copy.
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/bro_deps.cmake)

bro_dependency(bromath GITHUB wlejon/bromath TARGET bromath::bromath)

# bronze, the JavaScript runtime the api binds (it brings brass, its code
# generator). The api is bound across bronze's C++ embed boundary, so it has to
# build against the bronze the surrounding tree builds against.
bro_dependency(bronze GITHUB wlejon/bronze TARGET bronze_runtime_shared
               OPTIONS BRONZE_BUILD_SHARED_RUNTIME=ON BRONZE_BUILD_TESTS=OFF)

bro_dependency(brotensor GITHUB wlejon/brotensor)

# broimage — host-side image decode + geometric resampling + normalize presets
# (CLIP/ImageNet/SAM). It is the preprocessing front-end every vision model in
# this repo feeds: decode → resize/letterbox → normalize → NHWC->NCHW. Its own
# CUDA/Metal image preproc forwards into brotensor, so no GPU language here.
bro_dependency(broimage GITHUB wlejon/broimage)

# --- brovisionml library ---

add_library(brovisionml STATIC)
add_library(brovisionml::brovisionml ALIAS brovisionml)

target_include_directories(brovisionml PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(brovisionml PUBLIC cxx_std_20)
target_link_libraries(brovisionml PUBLIC
    bromath::bromath
    brotensor::brotensor
    broimage::broimage
)
target_link_libraries(brovisionml PRIVATE $<BUILD_INTERFACE:brass>)

if(MSVC)
    target_compile_options(brovisionml PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W4>)
else()
    target_compile_options(brovisionml PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall>
        $<$<COMPILE_LANGUAGE:CXX>:-Wextra>)
endif()

target_sources(brovisionml PRIVATE
    src/version.cpp
    src/profile.cpp
    src/tile_runner.cpp
    src/dpt_preprocess.cpp
    src/dsine_preprocess.cpp
    src/dsine_encoder.cpp
    src/dsine_decoder.cpp
    src/dsine_refine.cpp
    src/dsine_ops.cpp
    src/dsine.cpp
    src/hed_preprocess.cpp
    src/hed.cpp
    src/lineart_preprocess.cpp
    src/lineart.cpp
    src/mlsd_preprocess.cpp
    src/mlsd.cpp
    src/openpose_preprocess.cpp
    src/openpose.cpp
    src/segformer_preprocess.cpp
    src/segformer.cpp
    src/dinov2.cpp
    src/dinov3.cpp
    src/dinov3_preprocess.cpp
    src/dpt_head.cpp
    src/depth_anything.cpp
    src/sam_preprocess.cpp
    src/sam_image_encoder.cpp
    src/sam_prompt_encoder.cpp
    src/sam_mask_decoder.cpp
    src/sam.cpp
    src/sam_amg.cpp
    src/birefnet.cpp
    src/stylegan3_mapping.cpp
    src/stylegan3_filter.cpp
    src/stylegan3_synthesis.cpp
    src/stylegan3.cpp
    src/sam2_config.cpp
    src/sam2_ops.cpp
    src/sam2_image_encoder.cpp
    src/sam2_decoder.cpp
    src/sam2_memory.cpp
    src/sam2.cpp
    src/sam2_tracker.cpp
)

# OWLv2 open-vocabulary detector (its CLIP towers, tokenizer and preprocessing
# are local: brovisionml carries no brolm dependency).
target_sources(brovisionml PRIVATE
    src/gpu_mir_ops.cpp
    src/owlv2_tokenizer.cpp
    src/owlv2_preprocess.cpp
    src/owlv2.cpp
)

# --- Install ---
#
# Off by default: brovisionml is consumed as a sibling repo via the same
# add_subdirectory pattern bro uses for its other libraries, not via
# find_package. When enabled, this installs only the static library and the
# public headers — no CMake package config, since the sibling libraries it
# links are EXCLUDE_FROM_ALL add_subdirectory targets that are not themselves
# exported. Consume brovisionml with add_subdirectory().
option(BROVISIONML_INSTALL "Install the brovisionml static library and headers" OFF)
if(BROVISIONML_INSTALL)
    include(GNUInstallDirs)
    install(TARGETS brovisionml
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
    install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
endif()

# --- Standalone Bronze JavaScript API ---
option(BROVISIONML_ENABLE_API "Build Bronze JavaScript API" ON)
if(BROVISIONML_ENABLE_API)
    add_subdirectory(src/api)
endif()

# --- Tools ---
option(BROVISIONML_TOOLS "Build brovisionml CLI tools" ON)
if(BROVISIONML_TOOLS AND CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    foreach(tool sam_segment sam_amg depth_estimate normal_estimate hed_edges lineart mlsd_lines openpose_pose segformer_seg stylegan3_generate bench)
        add_executable(${tool} tools/${tool}.cpp)
        target_link_libraries(${tool} PRIVATE brovisionml)
        if(MSVC)
            target_compile_options(${tool} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W4>)
        else()
            target_compile_options(${tool} PRIVATE
                $<$<COMPILE_LANGUAGE:CXX>:-Wall>
                $<$<COMPILE_LANGUAGE:CXX>:-Wextra>)
        endif()
    endforeach()
endif()

# --- Tests ---
option(BROVISIONML_TESTS "Build brovisionml tests" ON)
if(BROVISIONML_TESTS AND CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    enable_testing()
    add_subdirectory(tests)
endif()
