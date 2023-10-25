#!/bin/bash

# NOTE: let's test all on day images!
# test_net_wrapper(){
#     local model_name="$1"
    
#     python train_net.py \
#       --num-gpus 3 \
#       --eval-only \
#       --config "configs/bdd100k_test_day.yaml" \
#       MODEL.WEIGHTS "outputs/${model_name}/model_final.pth" \
#       OUTPUT_DIR "outputs/${model_name}"
# }


test_net_wrapper_05x(){
    local model_name="$1"
    
    python train_net.py \
      --num-gpus 3 \
      --eval-only \
      --config "configs/bdd100k_test_day_05x.yaml" \
      MODEL.WEIGHTS "outputs/${model_name}/model_final.pth" \
      OUTPUT_DIR "outputs/${model_name}"
}

# test_net_wrapper "warp_aug_8_2"

# test_net_wrapper "warp_aug_9_12"

# test_net_wrapper "bdd100k_9_22_v1"

# test_net_wrapper_05x "bdd100k_bbox_05x_retrain"

# test_net_wrapper_05x "bdd100k_10_9_05x"

# # NOTE: for testing baseline model on day images
# python train_net.py \
#       --num-gpus 3 \
#         --eval-only \
#       --config configs/bdd100k_test_day.yaml \
#       MODEL.WEIGHTS outputs/pretrained/model_final.pth \
#       OUTPUT_DIR outputs/pretrained \



# Let's test on night images!
# python train_net.py \
#       --num-gpus 3 \
#       --eval-only \
#       --config configs/bdd100k_10_9_05x.yaml \
#       MODEL.WEIGHTS "outputs/bdd100k_10_9_05x/model_final.pth" \
#       OUTPUT_DIR outputs/bdd100k_10_9_05x \


python train_net.py \
  --num-gpus 3 \
  --eval-only \
  --config "configs/bdd100k_10_18_bbox.yaml" \
  MODEL.WEIGHTS "outputs/bdd100k_10_18_bbox/model_0114999.pth" \
  OUTPUT_DIR "outputs/bdd100k_10_18_bbox"