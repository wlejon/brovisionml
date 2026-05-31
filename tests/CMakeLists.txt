function(brovisionml_add_test name)
    add_executable(${name} ${ARGN})
    target_link_libraries(${name} PRIVATE brovisionml)
    if(MSVC)
        target_compile_options(${name} PRIVATE /W4)
    else()
        target_compile_options(${name} PRIVATE -Wall -Wextra)
    endif()
    add_test(NAME ${name} COMMAND ${name})
endfunction()

brovisionml_add_test(brovisionml_test_smoke            test_smoke.cpp)
brovisionml_add_test(brovisionml_test_dpt_preprocess   test_dpt_preprocess.cpp)
brovisionml_add_test(brovisionml_test_dsine_preprocess test_dsine_preprocess.cpp)
brovisionml_add_test(brovisionml_test_dsine_encoder    test_dsine_encoder.cpp)
brovisionml_add_test(brovisionml_test_dinov2           test_dinov2.cpp)
brovisionml_add_test(brovisionml_test_dpt_head         test_dpt_head.cpp)
brovisionml_add_test(brovisionml_test_depth_anything   test_depth_anything.cpp)
brovisionml_add_test(brovisionml_test_sam_preprocess   test_sam_preprocess.cpp)
brovisionml_add_test(brovisionml_test_sam_image_encoder test_sam_image_encoder.cpp)
brovisionml_add_test(brovisionml_test_sam_prompt_encoder test_sam_prompt_encoder.cpp)
brovisionml_add_test(brovisionml_test_sam_mask_decoder   test_sam_mask_decoder.cpp)
brovisionml_add_test(brovisionml_test_sam               test_sam.cpp)
brovisionml_add_test(brovisionml_test_sam_amg           test_sam_amg.cpp)

# Real-checkpoint end-to-end test: loads actual facebook/sam-vit-* weights from
# weights/ (downloaded via scripts/download-weights.sh) and skips cleanly when
# absent. The weights dir is passed in at configure time; an env var of the same
# name overrides it at run time.
brovisionml_add_test(brovisionml_test_sam_weights        test_sam_weights.cpp)
target_compile_definitions(brovisionml_test_sam_weights PRIVATE
    BROVISIONML_WEIGHTS_DIR="${CMAKE_SOURCE_DIR}/weights")

# The automatic-mask-generator test also exercises the real checkpoint when
# present (a small grid for runtime), and skips that part cleanly when absent.
target_compile_definitions(brovisionml_test_sam_amg PRIVATE
    BROVISIONML_WEIGHTS_DIR="${CMAKE_SOURCE_DIR}/weights")

# Depth-Anything-V2 end-to-end test: structural checks always run; the
# real-checkpoint run is gated on a Depth-Anything-V2 checkpoint under weights/
# (download via scripts/download-weights.sh depth-anything-v2-small) and skips
# cleanly when absent.
target_compile_definitions(brovisionml_test_depth_anything PRIVATE
    BROVISIONML_WEIGHTS_DIR="${CMAKE_SOURCE_DIR}/weights")

# Depth-Anything-V2 HF numeric-parity test: compares brovisionml's output to a
# golden dump of the actual HuggingFace model (generated out-of-repo, never
# committed). Gated on both the checkpoint and the golden_*.bin under
# weights/Depth-Anything-V2-Small/golden; skips cleanly when either is absent.
brovisionml_add_test(brovisionml_test_depth_parity       test_depth_parity.cpp)
target_compile_definitions(brovisionml_test_depth_parity PRIVATE
    BROVISIONML_WEIGHTS_DIR="${CMAKE_SOURCE_DIR}/weights")

# DSINE preprocessing test: structural CPU checks always run; the numeric-parity
# part is gated on golden_dsine_*.bin under weights/dsine (generated out-of-repo,
# never committed) and skips cleanly when absent.
target_compile_definitions(brovisionml_test_dsine_preprocess PRIVATE
    BROVISIONML_WEIGHTS_DIR="${CMAKE_SOURCE_DIR}/weights")

# DSINE EfficientNet-B5 encoder parity test: gated on the real checkpoint
# (weights/dsine/model.safetensors) and the golden_dsine_*.bin dumps under
# weights/dsine (generated out-of-repo, never committed); skips cleanly when
# either is absent. Compares the three encoder taps against the golden.
target_compile_definitions(brovisionml_test_dsine_encoder PRIVATE
    BROVISIONML_WEIGHTS_DIR="${CMAKE_SOURCE_DIR}/weights")
