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
