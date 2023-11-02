function test_night() {
    local MODEL_NAME="$1"
    local CONFIG_SUFFIX="$2"  # Suffix for the configuration file, e.g., "night", "night_05x"
    
    # Construct the configuration filename using the constant prefix and the passed suffix.
    local CONFIG_FILE="configs/bdd100k_test_${CONFIG_SUFFIX}.yaml"
    
    CUDA_VISIBLE_DEVICES=2 \
    python train_net.py \
        --num-gpus 1 \
        --eval-only \
        --config "${CONFIG_FILE}" \
        MODEL.WEIGHTS "outputs/${MODEL_NAME}/model_final.pth" \
        OUTPUT_DIR "outputs/${MODEL_NAME}/${CONFIG_SUFFIX}"
}



# test_night "pretrained"
# test_night "warp_aug_9_12"
# test_night "warp_aug_8_2"
# test_night "bdd100k_9_22_v1"


# test_night_05x "bdd100k_10_9_05x"
# test_night_05x "bdd100k_bbox_05x_retrain"
# test_night_05x "bdd100k_fovea_05x_retrain"
# test_night_05x "bdd100k_tpp_05x_retrain"

# test_rainy "bdd100k_10_18_baseline"
# test_rainy "bdd100k_10_18_fovea"
# test_rainy "bdd100k_10_18_tpp"
# test_rainy "bdd100k_10_18_bbox"


# TODO: rainy (0.5x)