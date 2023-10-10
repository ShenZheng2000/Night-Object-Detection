#!/bin/bash

# NOTE: let's test all on day images!
test_net_wrapper(){
    local model_name="$1"
    
    python train_net.py \
      --num-gpus 3 \
      --eval-only \
      --config "configs/bdd100k_test_day.yaml" \
      MODEL.WEIGHTS "outputs/${model_name}/model_final.pth" \
      OUTPUT_DIR "outputs/${model_name}"
}

# test_net_wrapper "warp_aug_8_2"

# test_net_wrapper "warp_aug_9_12"

test_net_wrapper "bdd100k_9_22_v1"

# # NOTE: for testing baseline model on day images
# python train_net.py \
#       --num-gpus 3 \
#         --eval-only \
#       --config configs/bdd100k_test_day.yaml \
#       MODEL.WEIGHTS outputs/pretrained/model_final.pth \
#       OUTPUT_DIR outputs/pretrained \