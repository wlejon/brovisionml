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
brovisionml_add_test(brovisionml_test_tile_runner      test_tile_runner.cpp)
brovisionml_add_test(brovisionml_test_dpt_preprocess   test_dpt_preprocess.cpp)
brovisionml_add_test(brovisionml_test_dsine_preprocess test_dsine_preprocess.cpp)
brovisionml_add_test(brovisionml_test_dsine_encoder    test_dsine_encoder.cpp)
brovisionml_add_test(brovisionml_test_dsine_decoder    test_dsine_decoder.cpp)
brovisionml_add_test(brovisionml_test_dsine_refine     test_dsine_refine.cpp)
brovisionml_add_test(brovisionml_test_dsine             test_dsine.cpp)
brovisionml_add_test(brovisionml_test_hed              test_hed.cpp)
brovisionml_add_test(brovisionml_test_lineart          test_lineart.cpp)
brovisionml_add_test(brovisionml_test_mlsd             test_mlsd.cpp)
brovisionml_add_test(brovisionml_test_openpose         test_openpose.cpp)
brovisionml_add_test(brovisionml_test_segformer        test_segformer.cpp)
brovisionml_add_test(brovisionml_test_dinov2           test_dinov2.cpp)
brovisionml_add_test(brovisionml_test_dinov3           test_dinov3.cpp)
brovisionml_add_test(brovisionml_test_birefnet_backbone test_birefnet_backbone.cpp)
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

# DINOv3 ViT-H backbone test: structural CPU checks always run; the HF
# numeric-parity part is gated on the real checkpoint
# (weights/triposplat/clip_vision/dino_v3_vit_h.safetensors) and the
# golden_dinov3.bin under weights/triposplat/clip_vision/golden (generated
# out-of-repo, never committed) and skips cleanly when either is absent.
target_compile_definitions(brovisionml_test_dinov3 PRIVATE
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

# DSINE v02 decoder parity test: gated on the real checkpoint and the
# golden_dsine_*.bin dumps under weights/dsine; skips cleanly when absent.
# Compares the initial normal map (dec_normal record) against the golden.
target_compile_definitions(brovisionml_test_dsine_decoder PRIVATE
    BROVISIONML_WEIGHTS_DIR="${CMAKE_SOURCE_DIR}/weights")

# DSINE v02 iterative refinement (NRN) END-TO-END parity test: gated on the real
# checkpoint and the golden_dsine_*.bin dumps under weights/dsine; skips cleanly
# when absent. Runs encoder->decoder->refiner and compares the final full-res
# normals against the golden `final` record (the LAST record in the dump).
target_compile_definitions(brovisionml_test_dsine_refine PRIVATE
    BROVISIONML_WEIGHTS_DIR="${CMAKE_SOURCE_DIR}/weights")

# DSINE NormalEstimator orchestrator END-TO-END test: gated on the real
# checkpoint and the golden_dsine_sq256.bin dump under weights/dsine; skips
# cleanly when absent. Loads via the single-class NormalEstimator, runs the full
# preprocess->encoder->decoder->refiner pipeline, asserts unit-length finite
# normals, and compares against the golden `final` record.
target_compile_definitions(brovisionml_test_dsine PRIVATE
    BROVISIONML_WEIGHTS_DIR="${CMAKE_SOURCE_DIR}/weights")

# HED soft-edge END-TO-END parity test: gated on the real checkpoint
# (weights/hed/model.safetensors) and the golden_hed_*.bin dumps under
# weights/hed (generated out-of-repo, never committed); skips cleanly when
# either is absent. Loads via SoftEdgeDetector, runs the full conv trunk at
# native resolution, and compares the edge map against the golden `edge` record.
target_compile_definitions(brovisionml_test_hed PRIVATE
    BROVISIONML_WEIGHTS_DIR="${CMAKE_SOURCE_DIR}/weights")

# Lineart END-TO-END parity test: gated on the real checkpoint
# (weights/lineart/model.safetensors) and the golden_lineart_*.bin dumps under
# weights/lineart (generated out-of-repo, never committed); skips cleanly when
# either is absent. Loads via LineartDetector, runs the generator at native
# resolution (dims divisible by 4 so it round-trips), and compares the raw
# (un-inverted) line map against the golden.
target_compile_definitions(brovisionml_test_lineart PRIVATE
    BROVISIONML_WEIGHTS_DIR="${CMAKE_SOURCE_DIR}/weights")

# MLSD END-TO-END parity test: gated on the real checkpoint
# (weights/mlsd/model.safetensors) and the golden_mlsd_*.bin dump under
# weights/mlsd (generated out-of-repo, never committed); skips cleanly when
# either is absent. Compares the raw TP map (tight) and the decoded segment set.
target_compile_definitions(brovisionml_test_mlsd PRIVATE
    BROVISIONML_WEIGHTS_DIR="${CMAKE_SOURCE_DIR}/weights")

# OpenPose body-pose END-TO-END parity test: gated on the real checkpoint
# (weights/openpose/model.safetensors) and the golden_openpose.bin dump under
# weights/openpose (generated out-of-repo, never committed); skips cleanly when
# either is absent. Gate 1 compares the raw stage-6 PAF + heatmap (tight conv
# parity); gate 2 matches decoded people + keypoint recall against the golden.
target_compile_definitions(brovisionml_test_openpose PRIVATE
    BROVISIONML_WEIGHTS_DIR="${CMAKE_SOURCE_DIR}/weights")

# SegFormer semantic-seg END-TO-END parity test: gated on the real checkpoint
# (weights/segformer-b0-ade/model.safetensors) and the golden_segformer.bin dump
# under weights/segformer-b0-ade (generated out-of-repo, never committed); skips
# cleanly when either is absent. Gate 1 compares the decode-head logits (tight
# transformer parity); gate 2 compares the per-pixel class map agreement.
target_compile_definitions(brovisionml_test_segformer PRIVATE
    BROVISIONML_WEIGHTS_DIR="${CMAKE_SOURCE_DIR}/weights")
