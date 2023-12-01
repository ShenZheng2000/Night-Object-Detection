# function test_night() {
#     local MODEL_NAME="$1"
#     local CONFIG_SUFFIX="$2"  # Suffix for the configuration file, e.g., "night", "night_05x"
    
#     # Construct the configuration filename using the constant prefix and the passed suffix.
#     local CONFIG_FILE="configs/bdd100k_test_${CONFIG_SUFFIX}.yaml"
    
#     CUDA_VISIBLE_DEVICES=2 \
#     python train_net.py \
#         --num-gpus 1 \
#         --eval-only \
#         --config "${CONFIG_FILE}" \
#         MODEL.WEIGHTS "outputs/${MODEL_NAME}/model_final.pth" \
#         OUTPUT_DIR "outputs/${MODEL_NAME}/${CONFIG_SUFFIX}"
# }



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


# nohup \
# python train_net.py \
#       --num-gpus 3 \
#       --config configs/bdd100k_warp_aug_9_12.yaml \
#       OUTPUT_DIR outputs/warp_aug_9_12 \
#       > warp_aug_9_12.out 2>&1 &

# TODO: write a new test config, and use the function above
# TODO: configs/bdd100k_test_warp_night.yaml, where CONFIG_SUFFIX="warp_night"
# NAME="pretrained"
# NAME="warp_aug_8_2"
# NAME="warp_aug_9_12"
# # NAME="bdd100k_9_22_v1"

# TOD='clear'
# TOD='day'
# TOD='day_bad_weather'
TOD='clear_night'

# List of NAME values
NAMES=(
  # "acdc_11_6_baseline"
  # "acdc_11_6_fovea"
  # "acdc_11_6_tpp"
  # "acdc_11_6_bbox"
  # "pretrained"
  # "warp_aug_9_12"
  # "warp_aug_8_2"
  # "bdd100k_9_22_v1"
  "bdd100k_10_18_baseline"
  # "bdd100k_10_18_fovea"
  # "bdd100k_10_18_tpp"
  "bdd100k_10_18_bbox"
  )

# # Loop over each NAME and run the command
export CUDA_VISIBLE_DEVICES=0
for NAME in "${NAMES[@]}"; do
  python train_net.py \
    --num-gpus 1 \
    --eval-only \
    --config "configs/bdd100k_test_${TOD}.yaml" \
    MODEL.WEIGHTS "/longdata/anurag_storage/2PCNet/outputs_11_14_det_ckpts/${NAME}/model_final.pth" \
    OUTPUT_DIR "outputs/${NAME}/${TOD}"
done

