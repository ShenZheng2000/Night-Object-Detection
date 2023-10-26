function test_night() {
    local MODEL_NAME="$1"
    
    CUDA_VISIBLE_DEVICES=2 \
    python train_net.py \
        --num-gpus 1 \
        --eval-only \
        --config "configs/bdd100k_test_night.yaml" \
        MODEL.WEIGHTS "outputs/${MODEL_NAME}/model_final.pth" \
        OUTPUT_DIR "outputs/${MODEL_NAME}/night"
}


function test_night_05x() {
    local MODEL_NAME="$1"
    
    CUDA_VISIBLE_DEVICES=2 \
    python train_net.py \
        --num-gpus 1 \
        --eval-only \
        --config "configs/bdd100k_test_night_05.yaml" \
        MODEL.WEIGHTS "outputs/${MODEL_NAME}/model_final.pth" \
        OUTPUT_DIR "outputs/${MODEL_NAME}/night_05"
}

function test_rainy() {
    local MODEL_NAME="$1"
    
    CUDA_VISIBLE_DEVICES=2 \
    python train_net.py \
        --num-gpus 1 \
        --eval-only \
        --config "configs/bdd100k_test_rainy.yaml" \
        MODEL.WEIGHTS "outputs/${MODEL_NAME}/model_final.pth" \
        OUTPUT_DIR "outputs/${MODEL_NAME}/rainy"
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