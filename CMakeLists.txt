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
# brovisionml composes brotensor ops — it ships no GPU kernels of its own, so it
# enables no CUDA/Metal language. It only forwards the backend choice so a
# standalone GPU build resolves brotensor's CUDA/Metal backend. Built inside
# bro, brotensor is already a target (added transitively) and the
# if(NOT TARGET ...) guards below reuse it, backend and all.
option(BROTENSOR_WITH_CUDA  "Build the CUDA GPU backend"  OFF)
option(BROTENSOR_WITH_METAL "Build the Metal GPU backend" OFF)

# Unlike the rest of the stack (which composes brotensor ops), brovisionml ships
# a small number of its own CUDA kernels for DSINE's surface-normal *domain* math
# — RayReLU and the fused AngMF propagate — which have no brotensor primitive.
# Enable the CUDA language only when the GPU backend is requested, mirroring
# brotensor's own setup. The .cu sources are added to the brovisionml target
# below, guarded by the same option.
if(BROTENSOR_WITH_CUDA)
    if(NOT CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES native)
    endif()
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    find_package(CUDAToolkit REQUIRED)
    if(MSVC AND DEFINED CMAKE_MSVC_RUNTIME_LIBRARY)
        set(CMAKE_CUDA_RUNTIME_LIBRARY "${CMAKE_MSVC_RUNTIME_LIBRARY}")
    endif()
endif()

# --- Siblings: bromath (header-only) + brotensor (tensor + ops) + broimage ---
#
# Prefer standalone repos at ../<name>; fall back to third_party/ submodules.
# Matches the pattern documented in bro/docs/multi-repo-workflow.md.
#
# Order matters: bromath and brotensor are resolved first so broimage (which
# links both, guarded with its own if(NOT TARGET ...)) reuses these targets
# rather than adding a second copy.

set(BROMATH_DIR "${CMAKE_SOURCE_DIR}/../bromath" CACHE PATH
    "Path to standalone bromath repo (set to a missing path to force submodule fallback)")
if(NOT TARGET bromath)
    if(EXISTS "${BROMATH_DIR}/CMakeLists.txt")
        add_subdirectory("${BROMATH_DIR}" "${CMAKE_BINARY_DIR}/bromath" EXCLUDE_FROM_ALL)
    elseif(EXISTS "${CMAKE_SOURCE_DIR}/third_party/bromath/CMakeLists.txt")
        add_subdirectory(third_party/bromath EXCLUDE_FROM_ALL)
    else()
        message(FATAL_ERROR
            "bromath not found at ${BROMATH_DIR} or third_party/bromath. "
            "Clone the bromath repo next to this repo, or override with -DBROMATH_DIR=...")
    endif()
endif()

set(BROTENSOR_DIR "${CMAKE_SOURCE_DIR}/../brotensor" CACHE PATH
    "Path to standalone brotensor repo (set to a missing path to force submodule fallback)")
if(NOT TARGET brotensor)
    if(EXISTS "${BROTENSOR_DIR}/CMakeLists.txt")
        add_subdirectory("${BROTENSOR_DIR}" "${CMAKE_BINARY_DIR}/brotensor" EXCLUDE_FROM_ALL)
    elseif(EXISTS "${CMAKE_SOURCE_DIR}/third_party/brotensor/CMakeLists.txt")
        add_subdirectory(third_party/brotensor EXCLUDE_FROM_ALL)
    else()
        message(FATAL_ERROR
            "brotensor not found at ${BROTENSOR_DIR} or third_party/brotensor. "
            "Clone the brotensor repo next to this repo, or override with -DBROTENSOR_DIR=...")
    endif()
endif()

# broimage — host-side image decode + geometric resampling + normalize presets
# (CLIP/ImageNet/SAM). It is the preprocessing front-end every vision model in
# this repo feeds: decode → resize/letterbox → normalize → NHWC->NCHW. Its own
# CUDA/Metal image preproc forwards into brotensor, so no GPU language here.
set(BROIMAGE_DIR "${CMAKE_SOURCE_DIR}/../broimage" CACHE PATH
    "Path to standalone broimage repo (set to a missing path to force submodule fallback)")
if(NOT TARGET broimage)
    if(EXISTS "${BROIMAGE_DIR}/CMakeLists.txt")
        add_subdirectory("${BROIMAGE_DIR}" "${CMAKE_BINARY_DIR}/broimage" EXCLUDE_FROM_ALL)
    elseif(EXISTS "${CMAKE_SOURCE_DIR}/third_party/broimage/CMakeLists.txt")
        add_subdirectory(third_party/broimage EXCLUDE_FROM_ALL)
    else()
        message(FATAL_ERROR
            "broimage not found at ${BROIMAGE_DIR} or third_party/broimage. "
            "Clone the broimage repo next to this repo, or override with -DBROIMAGE_DIR=...")
    endif()
endif()

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

if(MSVC)
    target_compile_options(brovisionml PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W4>)
else()
    target_compile_options(brovisionml PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall>
        $<$<COMPILE_LANGUAGE:CXX>:-Wextra>)
endif()

target_sources(brovisionml PRIVATE
    src/version.cpp
    src/dpt_preprocess.cpp
    src/dsine_preprocess.cpp
    src/dsine_encoder.cpp
    src/dsine_decoder.cpp
    src/dsine_refine.cpp
    src/dsine_ops.cpp
    src/dsine.cpp
    src/hed_preprocess.cpp
    src/hed.cpp
    src/dinov2.cpp
    src/dpt_head.cpp
    src/depth_anything.cpp
    src/sam_preprocess.cpp
    src/sam_image_encoder.cpp
    src/sam_prompt_encoder.cpp
    src/sam_mask_decoder.cpp
    src/sam.cpp
    src/sam_amg.cpp
)

# brovisionml's own CUDA kernels (DSINE RayReLU + AngMF propagate). Compiled into
# the library only when the GPU backend is on; BROVISIONML_WITH_CUDA gates the
# device-dispatch branch in src/dsine_ops.cpp that calls them. Everything else in
# the library stays device-agnostic by composing brotensor ops.
if(BROTENSOR_WITH_CUDA)
    target_sources(brovisionml PRIVATE src/dsine_ops.cu)
    target_compile_definitions(brovisionml PRIVATE BROVISIONML_WITH_CUDA=1)
    target_link_libraries(brovisionml PUBLIC CUDA::cudart)
    # dsine_ops.cu is self-contained (kernels + host launchers in one TU, no
    # cross-TU device symbols), so it uses whole-program compilation — no
    # separable/relocatable device code, which would otherwise need a device-link
    # step (__cudaRegisterLinkedBinary) the static library does not perform.
    set_target_properties(brovisionml PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_SEPARABLE_COMPILATION OFF)
endif()

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

# --- Tools ---
option(BROVISIONML_TOOLS "Build brovisionml CLI tools" ON)
if(BROVISIONML_TOOLS AND CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    foreach(tool sam_segment sam_amg depth_estimate normal_estimate hed_edges)
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
