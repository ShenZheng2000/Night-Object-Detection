# NOTE: the original paper use 3 GPUs!
# NOTE: use GPU=3,4,5 only!

# export CUDA_VISIBLE_DEVICES=3,4,5

# nohup \
# python train_net.py \
#       --num-gpus 3 \
#       --config configs/faster_rcnn_R50_bdd100k.yaml \
#       OUTPUT_DIR output/bdd100k \
#       > file_5_22_v1.out 2>&1 &

# # For testing with pretrained model (NOTE: must specify OUTPUT_DIR to save preds)
# python train_net.py \
#       --eval-only \
#       --config configs/faster_rcnn_R50_bdd100k.yaml \
#       MODEL.WEIGHTS outputs/pretrained/model_final.pth \
#       OUTPUT_DIR outputs/pretrained

# # # For testing with pretrained model on cutsom images
# python train_net.py \
#       --eval-only \
#       --config configs/faster_rcnn_R50_20230112_o.yaml \
#       MODEL.WEIGHTS pretrained/model_final.pth \
#       OUTPUT_DIR pretrained_20230112_o

# NOTE: for nighttime
# Test on all (for debug)
# python train_net.py \
#       --eval-only \
#       --config configs/bdd100k_all.yaml \
#       MODEL.WEIGHTS outputs/pretrained/model_final.pth \
#       OUTPUT_DIR outputs/pretrained_all

# # # Test on rainy
# python train_net.py \
#       --eval-only \
#       --config configs/bdd100k_rainy.yaml \
#       MODEL.WEIGHTS outputs/pretrained/model_final.pth \
#       OUTPUT_DIR outputs/pretrained_rainy

# # # Test on snowy
# python train_net.py \
#       --eval-only \
#       --config configs/bdd100k_snowy.yaml \
#       MODEL.WEIGHTS outputs/pretrained/model_final.pth \
#       OUTPUT_DIR outputs/pretrained_snowy

# # # Test on foggy
# python train_net.py \
#       --eval-only \
#       --config configs/bdd100k_foggy.yaml \
#       MODEL.WEIGHTS outputs/pretrained/model_final.pth \
#       OUTPUT_DIR outputs/pretrained_foggy


# Train on night (i.e. oracle)
# nohup \
# python train_net.py \
#       --num-gpus 3 \
#       --config configs/bdd100k_oracle.yaml \
#       OUTPUT_DIR output/oracle \
#       > file_5_27_v1.out 2>&1 &


################## oracle models ##################
# nohup \
# python train_net.py \
#       --resume \
#       --num-gpus 3 \
#       --config configs/bdd100k_oracle.yaml \
#       MODEL.WEIGHTS output/oracle/model_0049999.pth \
#       OUTPUT_DIR output/oracle \
#       > file_5_28_v1.out 2>&1 &

# python train_net.py \
#       --eval-only \
#       --config configs/bdd100k_oracle.yaml \
#       MODEL.WEIGHTS output/oracle/model_final.pth \
#       OUTPUT_DIR output/oracle


# ################## one-stage models ##################
# nohup \
# python train_net.py \
#       --num-gpus 3 \
#       --config configs/bdd100k_one_stage.yaml \
#       OUTPUT_DIR outputs/one_stage \
#       > file_5_29_v1.out 2>&1 &


### bbox-wise depth training
# nohup \
# python train_net.py \
#       --num-gpus 3 \
#       --config configs/bdd100k_depth.yaml \
#       OUTPUT_DIR outputs/depth_bbox \
#       > depth_bbox_6_3_v1.out 2>&1 &

# nohup \
# python train_net.py \
#       --resume \
#       --num-gpus 3 \
#       --config configs/bdd100k_depth.yaml \
#       MODEL.WEIGHTS outputs/depth_bbox/model_0049999.pth \
#       > depth_bbox_6_3_v2.out 2>&1 &

 
# # # NOTE: test on all (depth_bbox)
# python train_net.py \
#       --eval-only \
#       --config configs/bdd100k_depth.yaml \
#       MODEL.WEIGHTS outputs/depth_bbox/model_final.pth \
#       OUTPUT_DIR outputs/depth_bbox

# # NOTE: test on video 20230112_o (depth_bbox)
# python train_net.py \
#       --eval-only \
#       --config configs/bdd100k_depth.yaml \
#       MODEL.WEIGHTS outputs/depth_bbox/model_final.pth \
#       OUTPUT_DIR outputs/depth_bbox_20230112_o

# NOTE: test on all (depth_bbox), visualization only
# python train_net.py \
#       --vis-only \
#       --config configs/bdd100k_depth.yaml \
#       MODEL.WEIGHTS outputs/depth_bbox/model_final.pth \
#       OUTPUT_DIR outputs/depth_bbox


# NOTE: for debug only (depth_bbox)
# python train_net.py \
#       --num-gpus 1 \
#       --config configs/bdd100k_mask_first.yaml \
#       OUTPUT_DIR outputs/mask_debug \

# NOTE: even for resume, must specify OUTPUT_DIR

###################### Mask ##################
# nohup \
# python train_net.py \
#       --resume \
#       --num-gpus 3 \
#       --config configs/bdd100k_mask.yaml \
#       MODEL.WEIGHTS outputs/mask_6_18/model_0049999.pth \
#       OUTPUT_DIR outputs/mask_6_18 \
#       > mask_6_18.out 2>&1 &

# python train_net.py \
#       --eval-only \
#       --config configs/bdd100k_mask.yaml \
#       MODEL.WEIGHTS outputs/mask_6_18/model_final.pth \
#       OUTPUT_DIR outputs/mask_6_18

###################### Mask+Depth_bbox ##################
# nohup \
# python train_net.py \
#       --resume \
#       --num-gpus 3 \
#       --config configs/bdd100k_mask_depth.yaml \
#       MODEL.WEIGHTS outputs/mask_depth_6_19/model_0049999.pth \
#       OUTPUT_DIR outputs/mask_depth_6_19 \
#       > mask_depth_6_19.out 2>&1 &


################## Training source-mask models from scractch ##################
# nohup \
# python train_net.py \
#       --num-gpus 3 \
#       --config configs/bdd100k_mask_src.yaml \
#       OUTPUT_DIR outputs/mask_src_6_18 \
#       > mask_src_debug_6_18.out 2>&1 &


# ################## Mask (Cons) ##################
# nohup \
# python train_net.py \
#       --resume \
#       --num-gpus 3 \
#       --config configs/bdd100k_mask_cons.yaml \
#       MODEL.WEIGHTS outputs/mask_cons_6_22/model_0049999.pth \
#       OUTPUT_DIR outputs/mask_cons_6_22 \
#       > mask_cons_6_22.out 2>&1 &


# ################## Motion Blur ##################
# nohup \
# python train_net.py \
#       --resume \
#       --num-gpus 3 \
#       --config configs/bdd100k_mask_blur.yaml \
#       MODEL.WEIGHTS outputs/mask_blur_6_23/model_0049999.pth \
#       OUTPUT_DIR outputs/mask_blur_6_23 \
#       > mask_blur_6_23.out 2>&1 &


# ################## Motion Blur Random (REDO) ##################
# nohup \
# python train_net.py \
#       --resume \
#       --num-gpus 3 \
#       --config configs/bdd100k_mask_blur_rand.yaml \
#       MODEL.WEIGHTS outputs/mask_blur_rand_6_25/model_0049999.pth \
#       OUTPUT_DIR outputs/mask_blur_rand_6_25 \
#       > mask_blur_rand_6_25.out 2>&1 &


# # ################## Light Rendering (white, s=100) ##################
# nohup \
# python train_net.py \
#       --num-gpus 3 \
#       --config configs/bdd100k_light_white_100.yaml \
#       OUTPUT_DIR outputs/light_white_100_7_1 \
#       > light_white_100_7_1.out 2>&1 &


# ################## Light Rendering (white, s=100) ##################
# # NOTE: stop mid, resume from model_0029999.pth
# nohup \
# python train_net.py \
#       --resume \
#       --num-gpus 3 \
#       --config configs/bdd100k_light_white_100.yaml \
#       MODEL.WEIGHTS outputs/light_white_100_7_1/model_0094999.pth.pth \
#       OUTPUT_DIR outputs/light_white_100_7_1 \
#       > light_white_100_7_1.out 2>&1 &


# # ################## Path Blur (s=15,constant) ##################
# # TODO: change configs and use 3 gpu for training
# nohup \
# python train_net.py \
#       --num-gpus 3 \
#       --config configs/bdd100k_path_blur_cons.yaml \
#       OUTPUT_DIR outputs/path_blur_cons_7_5 \
#       > path_blur_cons_7_5.out 2>&1 &


# TODO later
# nohup \
# python train_net.py \
#       --num-gpus 3 \
#       --config configs/bdd100k_path_blur_cons_prob05_nopc.yaml \
#       OUTPUT_DIR outputs/path_blur_cons_7_8_v1 \
#       > path_blur_cons_7_8_v1.out 2>&1 &

# # DOING
# nohup \
# python train_net.py \
#       --num-gpus 3 \
#       --config configs/bdd100k_path_blur_cons_prob05_yespc.yaml \
#       OUTPUT_DIR outputs/path_blur_cons_7_8_v2 \
#       > path_blur_cons_7_8_v2.out 2>&1 &

# # TODO: reflection rendering experiments
# nohup \
# python train_net.py \
#       --num-gpus 3 \
#       --config configs/bdd100k_reflect_white_100.yaml \
#       OUTPUT_DIR outputs/reflect_white_100_7_9_v1 \
#       > reflect_white_100_7_9_v1.out 2>&1 &


# # TODO later: masking on source AND target
# nohup \
# python train_net.py \
#       --num-gpus 1 \
#       --config configs/bdd100k_mask_src_tgt.yaml \
#       OUTPUT_DIR outputs/mask_src_7_10_v1 \
#       > mask_src_7_10_v1.out 2>&1 &


# # NOTE: retrain this to debug
# nohup \
# python train_net.py \
#       --num-gpus 3 \
#       --config configs/bdd100k_mask_src.yaml \
#       OUTPUT_DIR outputs/mask_src_6_18_v2 \
#       > mask_src_6_18_v2.out 2>&1 &


# # TODO: retrain baseline models
# nohup \
# python train_net.py \
#       --num-gpus 3 \
#       --config configs/faster_rcnn_R50_bdd100k.yaml \
#       OUTPUT_DIR outputs/pretrained_7_12_v1 \
#       > pretrained_7_12_v1.out 2>&1 &


# # train vertical motion blur
# nohup \
# python train_net.py \
#       --num-gpus 3 \
#       --config configs/bdd100k_mask_blur_vet.yaml \
#       OUTPUT_DIR outputs/mask_blur_vet_7_13_v1 \
#       > mask_blur_vet_7_13_v1.out 2>&1 &



# ################## Mask (Sparse, Target) TODO ##################
# TODO: set ckpts in directory
# nohup \
# python train_net.py \
#       --resume \
#       --num-gpus 3 \
#       --config configs/bdd100k_mask_sparse.yaml \
#       MODEL.WEIGHTS outputs/mask_sparse_7_14/model_0049999.pth \
#       OUTPUT_DIR outputs/mask_sparse_7_14 \
#       > mask_sparse_7_14.out 2>&1 &


# # NOTE: train model with cur learning (M1)
# nohup \
# python train_net.py \
#       --resume \
#       --num-gpus 3 \
#       --config configs/bdd100k_cur.yaml \
#       MODEL.WEIGHTS outputs/mask_cur_7_15/model_0094999.pth \
#       OUTPUT_DIR outputs/mask_cur_7_15 \
#       > mask_cur_7_15.out 2>&1 &

# # TODO: train model with cur learning (M2)
# nohup \
# python train_net.py \
#       --num-gpus 3 \
#       --config configs/bdd100k_ddd.yaml \
#       OUTPUT_DIR outputs/mask_ddd_7_17 \
#       > mask_ddd_7_17.out 2>&1 &

# # train model with cons loss (weight = 0.5) (DOING another machine)
# nohup \
# python train_net.py \
#       --resume \
#       --num-gpus 3 \
#       --config configs/bdd100k_mask_cons_05.yaml \
#       MODEL.WEIGHTS outputs/mask_cons_05_7_16/model_0049999.pth \
#       OUTPUT_DIR outputs/mask_cons_05_7_16 \
#       > mask_cons_05_7_16.out 2>&1 &


# TODO: train with LLIE_preprocessed images (pretrained model)
# CUDA_VISIBLE_DEVICES=1 \
# python train_net.py \
#       --eval-only \
#       --config configs/faster_rcnn_R50_bdd100k_SGZ.yaml \
#       MODEL.WEIGHTS outputs/pretrained_SGZ/model_final.pth \
#       OUTPUT_DIR outputs/pretrained_SGZ

# CUDA_VISIBLE_DEVICES=1 \
# python train_net.py \
#       --eval-only \
#       --config configs/faster_rcnn_R50_bdd100k_CLAHE.yaml \
#       MODEL.WEIGHTS outputs/pretrained_CLAHE/model_final.pth \
#       OUTPUT_DIR outputs/pretrained_CLAHE


# new path blur
# nohup \
# python train_net.py \
#       --resume \
#       --num-gpus 3 \
#       --config configs/bdd100k_path_blur_7_22.yaml \
#       MODEL.WEIGHTS outputs/path_blur_7_22/model_0049999.pth \
#       OUTPUT_DIR outputs/path_blur_7_22 \
#       > path_blur_7_22.out 2>&1 &

# TODO: train this model
# NOTE: switch to 3 GPU once having model
# nohup \
# python train_net.py \
#       --resume \
#       --num-gpus 2 \
#       --config configs/bdd100k_path_blur_7_25.yaml \
#       MODEL.WEIGHTS outputs/path_blur_7_25/model_0084999.pth \
#       OUTPUT_DIR outputs/path_blur_7_25 \
#       > path_blur_7_25.out 2>&1 &

# NOTE: use gpu now for debug vis
# nohup \
# python train_net.py \
#       --resume \
#       --num-gpus 3 \
#       --config configs/bdd100k_warp_aug_7_26.yaml \
#       MODEL.WEIGHTS outputs/warp_aug_7_26/model_0044999.pth \
#       OUTPUT_DIR outputs/warp_aug_7_26 \
#       > warp_aug_7_26.out 2>&1 &

# TODO: train this later
# nohup \
# python train_net.py \
#       --resume \
#       --num-gpus 3 \
#       --config configs/bdd100k_path_blur_7_29.yaml \
#       MODEL.WEIGHTS outputs/path_blur_7_29/model_0019999.pth \
#       OUTPUT_DIR outputs/path_blur_7_29 \
#       > path_blur_7_29.out 2>&1 &


# TODO: wait code to finish
# nohup \
# python train_net.py \
#       --resume \
#       --num-gpus 3 \
#       --config configs/bdd100k_warp_aug_8_2.yaml \
#       MODEL.WEIGHTS outputs/warp_aug_8_2/model_0029999.pth \
#       OUTPUT_DIR outputs/warp_aug_8_2 \
#       > warp_aug_8_2.out 2>&1 &

# nohup \
# python train_net.py \
#       --num-gpus 3 \
#       --config configs/bdd100k_warp_aug_blur_8_5.yaml \
#       OUTPUT_DIR outputs/warp_aug_blur_8_5 \
#       > warp_aug_blur_8_5.out 2>&1 &

# NOTE: not working well
# nohup \
# python train_net.py \
#       --num-gpus 2 \
#       --config configs/bdd100k_cur_TPSeNCE_8_7.yaml \
#       OUTPUT_DIR outputs/cur_TPSeNCE_8_7 \
#       > cur_TPSeNCE_8_7.out 2>&1 &

# NOTE: add TPSeNCE_aug as additional augmentation => failed training
# nohup \
# python train_net.py \
#       --num-gpus 2 \
#       --config configs/bdd100k_cur_TPSeNCE_8_8.yaml \
#       OUTPUT_DIR outputs/cur_TPSeNCE_8_8 \
#       > cur_TPSeNCE_8_8.out 2>&1 &

# Thought: if vp outside => blur too strong.
  # Therefore, to improve path blur, we have to think about more balanced blurs
  # For example, more for close-to-vp regions, and less for far-from-vp regions

# # # NOTE: testing on enhanced images
# train_model() {
#     local model_name="$1"
#     echo "Testing with $model_name"
#     python train_net.py \
#       --num-gpus 2 \
#       --eval-only \
#       --config "configs/${model_name}.yaml" \
#       MODEL.WEIGHTS "outputs/pretrained/model_final.pth" \
#       OUTPUT_DIR "outputs/${model_name}"
# }

# train_model "LLFlow"
# train_model "RetinexNet"
# train_model "RUAS"
# train_model "SCI"
# train_model "SGZ"
# train_model "URetinexNet"
# train_model "ZeroDCE"


# # TODO: change config after debu
# nohup \
# python train_net.py \
#       --num-gpus 2 \
#       --config configs/bdd100k_path_blur_8_23.yaml \
#       OUTPUT_DIR outputs/path_blur_8_23 \
#       > path_blur_8_23.out 2>&1 &


# train current warpping with all images
# nohup \
# python train_net.py \
#       --num-gpus 3 \
#       --config configs/bdd100k_warp_aug_debug_9_6.yaml \
#       OUTPUT_DIR outputs/warp_aug_debug_9_6 \
#       > warp_aug_debug_9_6.out 2>&1 &

# # train current warpping with all images (except both vp oob cases)
# nohup \
# python train_net.py \
#       --resume \
#       --num-gpus 2 \
#       --config configs/bdd100k_warp_aug_debug_9_9.yaml \
#       MODEL.WEIGHTS outputs/warp_aug_debug_9_9/model_0104999.pth \
#       OUTPUT_DIR outputs/warp_aug_debug_9_9 \
#       > warp_aug_debug_9_9.out 2>&1 &


# # For WARP_DEBUG visualization 
# CUDA_VISIBLE_DEVICES=2 \
# nohup \
# python train_net.py \
#       --num-gpus 1 \
#       --config configs/bdd100k_warp_aug_9_9.yaml \
#       OUTPUT_DIR outputs/warp_aug_9_9 \
#       > warp_aug_9_9.out 2>&1 &

# NOTE: fix night aug non-added error, and train again
# train current warpping with all images (except both vp oob cases)
# nohup \
# python train_net.py \
#       --resume \
#       --num-gpus 3 \
#       --config configs/bdd100k_warp_aug_9_11_4090.yaml \
#       MODEL.WEIGHTS outputs/path_blur_7_29/model_0064999.pth \
#       OUTPUT_DIR outputs/warp_aug_9_11_4090 \
#       > warp_aug_9_11_4090_resume.out 2>&1 &


# # TODO: train model with fovea grid
# nohup \
# python train_net.py \
#       --num-gpus 3 \
#       --config configs/bdd100k_warp_aug_9_12.yaml \
#       OUTPUT_DIR outputs/warp_aug_9_12 \
#       > warp_aug_9_12.out 2>&1 &

# python train_net.py \
#       --num-gpus 3 \
#       --eval-only \
#       --config configs/bdd100k_warp_aug_9_13.yaml \
#       MODEL.WEIGHTS outputs/warp_aug_9_11_3090/model_final.pth \
#       OUTPUT_DIR outputs/warp_aug_9_13


# nohup \
# python train_net.py \
#       --resume \
#       --num-gpus 3 \
#       --config configs/bdd100k_warp_aug_9_15.yaml \
#       MODEL.WEIGHTS outputs/warp_aug_9_11_3090/model_0059999.pth \
#       OUTPUT_DIR outputs/warp_aug_9_15 \
#       > warp_aug_9_15.out 2>&1 &


# nohup \
# python train_net.py \
#       --resume \
#       --num-gpus 3 \
#       --config configs/bdd100k_warp_aug_9_15_v2.yaml \
#       MODEL.WEIGHTS outputs/warp_aug_9_11_3090/model_0059999.pth \
#       OUTPUT_DIR outputs/warp_aug_9_15_v2 \
#       > warp_aug_9_15_v2.out 2>&1 &

# # NOTE: train AT baseline
# nohup \
# python train_net.py \
#       --resume \
#       --num-gpus 3 \
#       --config configs/bdd100k_AT_9_16.yaml \
#       MODEL.WEIGHTS outputs/9_16/model_0059999.pth \
#       OUTPUT_DIR outputs/9_16 \
#       > '9_16.out' 2>&1 &


# nohup \
# python train_net.py \
#       --num-gpus 3 \
#       --config configs/bdd100k_AT_9_17.yaml \
#       OUTPUT_DIR outputs/9_17 \
#       > '9_17.out' 2>&1 &

# nohup \
# python train_net.py \
#       --num-gpus 3 \
#       --config configs/bdd100k_9_18_v1.yaml \
#       OUTPUT_DIR outputs/9_18_v1 \
#       > '9_18_v1.out' 2>&1 &

# nohup \
# python train_net.py \
#       --num-gpus 3 \
#       --config configs/bdd100k_9_18_v2.yaml \
#       OUTPUT_DIR outputs/9_18_v2 \
#       > '9_18_v2_debug.out' 2>&1 &

# # TODO: resume later
# nohup \
# python train_net.py \
#       --resume \
#       --num-gpus 3 \
#       --config configs/bdd100k_9_21_v1.yaml \
#       MODEL.WEIGHTS outputs/bdd100k_9_21_v1/model_0074999.pth \
#       OUTPUT_DIR outputs/bdd100k_9_21_v1 \
#       > bdd100k_9_21_v1_resume.out 2>&1 &

# NOTE: training with instance-level warp
# nohup \
# python train_net.py \
#       --num-gpus 3 \
#       --config configs/bdd100k_9_22_v1.yaml \
#       OUTPUT_DIR outputs/bdd100k_9_22_v1 \
#       > bdd100k_9_22_v1.out 2>&1 &

# # NOTE: training with instance-level and image-level warp
# nohup \
# python train_net.py \
#       --num-gpus 3 \
#       --config configs/bdd100k_9_22_v2.yaml \
#       OUTPUT_DIR outputs/bdd100k_9_22_v2 \
#       > bdd100k_9_22_v2.out 2>&1 &

# nohup \
# python train_net.py \
#       --num-gpus 3 \
#       --config configs/bdd100k_9_23_v1.yaml \
#       OUTPUT_DIR outputs/bdd100k_9_23_v1 \
#       > bdd100k_9_23_v1.out 2>&1 &

# nohup \
# python train_net.py \
#       --num-gpus 3 \
#       --config configs/bdd100k_9_23_v2.yaml \
#       OUTPUT_DIR outputs/bdd100k_9_23_v2 \
#       > bdd100k_9_23_v2.out 2>&1 &

# nohup \
# python train_net.py \
#       --num-gpus 3 \
#       --config configs/bdd100k_10_1.yaml \
#       OUTPUT_DIR outputs/bdd100k_10_1 \
#       > bdd100k_10_1.out 2>&1 &

# nohup \
# python train_net.py \
#       --num-gpus 3 \
#       --config configs/bdd100k_10_4.yaml \
#       OUTPUT_DIR outputs/bdd100k_10_4 \
#       > bdd100k_10_4.out 2>&1 &

# # # TODO: resume training later
# nohup \
# python train_net.py \
#       --num-gpus 3 \
#       --config configs/bdd100k_10_4_v2.yaml \
#       OUTPUT_DIR outputs/bdd100k_10_4_v2 \
#       > bdd100k_10_4_v2.out 2>&1 &

# NOTE: train with (max_theta=150, max_theta_top=240) => still not working
# nohup \
# python train_net.py \
#       --num-gpus 3 \
#       --config configs/bdd100k_10_6_tpp_mid.yaml \
#       OUTPUT_DIR outputs/bdd100k_10_6_tpp_mid \
#       > bdd100k_10_6_tpp_mid.out 2>&1 &

# NOTE: train with (max_theta=170, max_theta_top=230) => still not working
# nohup \
# python train_net.py \
#       --num-gpus 3 \
#       --config configs/bdd100k_10_7_tpp_mid.yaml \
#       OUTPUT_DIR outputs/bdd100k_10_7_tpp_mid \
#       > bdd100k_10_7_tpp_mid.out 2>&1 &

# TODO: train with 0.5x images (baseline, fovea, tpp, bbox-level)
# baseline
# nohup \
# python train_net.py \
#       --num-gpus 3 \
#       --config configs/bdd100k_10_9_05x.yaml \
#       OUTPUT_DIR outputs/bdd100k_10_9_05x \
#       > bdd100k_10_9_05x.out 2>&1 &


# # fovea (retrain with correct warping)
# nohup \
# python train_net.py \
#       --num-gpus 3 \
#       --config configs/bdd100k_fovea_05x.yaml \
#       OUTPUT_DIR outputs/bdd100k_fovea_05x_retrain \
#       > bdd100k_fovea_05x_retrain.out 2>&1 &


# # tpp (retrain with correct warping) => TODO: try this once other training is done
# nohup \
# python train_net.py \
#       --num-gpus 3 \
#       --config configs/bdd100k_tpp_05x.yaml \
#       OUTPUT_DIR outputs/bdd100k_tpp_05x_retrain \
#       > bdd100k_tpp_05x_retrain.out 2>&1 &


# # bbox-level (retrain with correct warping)
# nohup \
#   python train_net.py \
#   --num-gpus 3 \
#   --config configs/bdd100k_bbox_05x.yaml \
#   OUTPUT_DIR outputs/bdd100k_bbox_05x_retrain \
#   > bdd100k_bbox_05x_retrain.out 2>&1 &


# python train_net.py --num-gpus 3 --config configs/bdd100k_9_18_v2.yaml OUTPUT_DIR outputs/bdd100k_9_18_v2_debug

# train_bdd() {
#   local name="$1"

#   nohup \
#     python train_net.py \
#     --num-gpus 3 \
#     --config "configs/${name}.yaml" \
#     OUTPUT_DIR "outputs/${name}" \
#     > "${name}.out" 2>&1 &
# }


# train_bdd "bdd100k_10_18_baseline"

# train_bdd "bdd100k_10_18_tpp"

# train_bdd "bdd100k_10_18_bbox"

# TODO: one gpu for debug only! Use 3 GPU for training!
# NOTE: temporal change the configs for debug
# python train_net.py \
#   --resume \
#   --num-gpus 1 \
#   --config configs/warp_aug_8_2.yaml \
#   MODEL.WEIGHTS outputs/warp_aug_8_2/model_final.pth \
#   OUTPUT_DIR outputs/warp_aug_8_2_debug

# python train_net.py \
#   --num-gpus 3 \
#   --config configs/warp_aug_9_12.yaml \
#   OUTPUT_DIR outputs/warp_aug_9_12_debug


# BDD to ACDC (resume from BDD supervised stage ends, and train ACDC unsupervised)
# NOTE: use smaller iteration (e.g. 10% of origin) for ACDC, maybe, to see if it converges

# bdd100k_10_18_baseline => acdc_11_6_baseline
# nohup \
#   python train_net.py \
#   --resume \
#   --num-gpus 3 \
#   --config configs/acdc_11_6_baseline.yaml \
#   MODEL.WEIGHTS outputs/bdd100k_10_18_baseline/model_0059999.pth \
#   OUTPUT_DIR outputs/acdc_11_6_baseline \
#   > acdc_11_6_baseline.out 2>&1 &

# # # bdd100k_10_18_fovea => acdc_11_6_fovea
# nohup \
#   python train_net.py \
#   --resume \
#   --num-gpus 3 \
#   --config configs/acdc_11_6_fovea.yaml \
#   MODEL.WEIGHTS outputs/acdc_11_6_fovea/model_0062499.pth \
#   OUTPUT_DIR outputs/acdc_11_6_fovea \
#   > acdc_11_6_fovea.out 2>&1 &


# # # bdd100k_10_18_tpp => acdc_11_6_tpp
# nohup \
#   python train_net.py \
#   --resume \
#   --num-gpus 3 \
#   --config configs/acdc_11_6_tpp.yaml \
#   MODEL.WEIGHTS outputs/acdc_11_6_tpp/model_0061999.pth \
#   OUTPUT_DIR outputs/acdc_11_6_tpp \
#   > acdc_11_6_tpp.out 2>&1 &


# # # bdd100k_10_18_bbox => acdc_11_6_bbox
# nohup \
#   python train_net.py \
#   --resume \
#   --num-gpus 3 \
#   --config configs/acdc_11_6_bbox.yaml \
#   MODEL.WEIGHTS outputs/bdd100k_10_18_bbox/model_0059999.pth \
#   OUTPUT_DIR outputs/acdc_11_6_bbox \
#   > acdc_11_6_bbox.out 2>&1 &

# python train_net.py \
#   --num-gpus 1 \
#   --config configs/bdd100k_11_7_new_unwarp.yaml \
#   OUTPUT_DIR outputs/bdd100k_11_7_new_unwarp


# train_bdd() {
#   local name="$1"

#   nohup \
#     python train_net.py \
#     --num-gpus 3 \
#     --config "configs/${name}.yaml" \
#     OUTPUT_DIR "outputs/${name}" \
#     > "${name}.out" 2>&1 &
# }


# # train_bdd "bdd100k_10_18_baseline_small"

# train_bdd "bdd100k_10_18_bbox_small"


# # bdd100k_10_18_baseline => dense_foggy_12_12_baseline
# nohup \
#   python train_net.py \
#   --resume \
#   --num-gpus 3 \
#   --config configs/dense_foggy_12_12_baseline.yaml \
#   MODEL.WEIGHTS outputs/bdd100k_10_18_baseline/model_0059999.pth \
#   OUTPUT_DIR outputs/dense_foggy_12_12_baseline \
#   > dense_foggy_12_12_baseline.out 2>&1 &


# bdd100k_10_18_bbox => dense_foggy_12_12_bbox
# nohup \
#   python train_net.py \
#   --resume \
#   --num-gpus 3 \
#   --config configs/dense_foggy_12_12_bbox.yaml \
#   MODEL.WEIGHTS outputs/bdd100k_10_18_bbox/model_0059999.pth \
#   OUTPUT_DIR outputs/dense_foggy_12_12_bbox \
#   > dense_foggy_12_12_bbox.out 2>&1 &

# bdd100k_10_18_baseline => dense_snow_12_12_baseline
# nohup \
#   python train_net.py \
#   --resume \
#   --num-gpus 3 \
#   --config configs/dense_snow_12_12_baseline.yaml \
#   MODEL.WEIGHTS outputs/bdd100k_10_18_baseline/model_0059999.pth \
#   OUTPUT_DIR outputs/dense_snow_12_12_baseline \
#   > dense_snow_12_12_baseline.out 2>&1 &

# # bdd100k_10_18_bbox => dense_snow_12_12_bbox
# NOTE: use this because currently no GPU left
# nohup \
#   python train_net.py \
#   --resume \
#   --num-gpus 2 \
#   --config configs/dense_snow_12_12_bbox.yaml \
#   MODEL.WEIGHTS outputs/dense_snow_12_12_bbox/model_0089999.pth \ # use this since train are interruptsed
#   OUTPUT_DIR outputs/dense_snow_12_12_bbox \
#   > dense_snow_12_12_bbox_resume_v2.out 2>&1 &

# NOTE: be careful with GPU=3!!!!. Use GPU=2 for training unless emergent cases

# run_training() {
#   echo "Training $1 to $2"
#     local src="$1"
#     local tgt="$2"
#     nohup \
#     python train_net.py \
#     --resume \
#     --num-gpus 2 \
#     --config configs/${tgt}.yaml \
#     MODEL.WEIGHTS outputs/${src}/model_0059999.pth \
#     OUTPUT_DIR outputs/${tgt} \
#     > ${tgt}.out 2>&1 &
# }

# NOTE: remove --eval-only and setup --resume for training!!!!

########### Boreas Snowy Experiments ############

# # bdd100k_10_18_baseline => boreas_snow_12_16_baseline
# run_training 'bdd100k_10_18_baseline' 'boreas_snow_12_16_baseline'

# bdd100k_10_18_bbox => boreas_snow_12_16_bbox
# run_training 'bdd100k_10_18_bbox' 'boreas_snow_12_16_bbox'

# # bdd100k_10_18_fovea => boreas_snow_12_16_fovea
# run_training 'bdd100k_10_18_fovea' 'boreas_snow_12_16_fovea'

# bdd100k_10_18_tpp => boreas_snow_12_16_tpp
# run_training 'bdd100k_10_18_tpp' 'boreas_snow_12_16_tpp'

############ Dense Foggy Experiments ############

# # bdd100k_10_18_baseline => dense_foggy_12_12_baseline
# run_training 'bdd100k_10_18_baseline' 'dense_foggy_12_12_baseline'

# bdd100k_10_18_bbox => dense_foggy_12_12_bbox
# run_training 'bdd100k_10_18_bbox' 'dense_foggy_12_12_bbox'

# # bdd100k_10_18_fovea => dense_foggy_12_12_fovea
# run_training 'bdd100k_10_18_fovea' 'dense_foggy_12_12_fovea'

# bdd100k_10_18_tpp => dense_foggy_12_12_tpp
# run_training 'bdd100k_10_18_tpp' 'dense_foggy_12_12_tpp'


# # # construction zone experiments
# run_training() {
#     local tgt="$1"
#     CUDA_VISIBLE_DEVICES=3 nohup \
#     python train_net.py \
#     --num-gpus 1 \
#     --config configs/construct/${tgt}.yaml \
#     OUTPUT_DIR outputs/${tgt} \
#     > outs/${tgt}_v1.out 2>&1 &
# }

# # # DA (baseline)
# run_training '1_12_v1'

# Upperbound (baseline)
# run_training '1_13_v1'

# Sup (baseline)
# run_training '1_11_v1'

# DA (baseline + Instance Warp) => TODO: train this after sem seg are done
# run_training '1_14_v1'

# CUDA_VISIBLE_DEVICES=0,1 \
#   nohup \
#   python train_net.py \
#   --num-gpus 2 \
#   --config configs/ANURAG_no_warp_no_adapt_bdd_tesla_FINAL.yaml \
#   OUTPUT_DIR outputs/ANURAG_no_warp_no_adapt_bdd_tesla_FINAL \
#   > ANURAG_no_warp_no_adapt_bdd_tesla_FINAL.out 2>&1 &

# CUDA_VISIBLE_DEVICES=2,3 \
#   nohup \
#   python train_net.py \
#   --num-gpus 2 \
#   --config configs/ANURAG_no_warp_no_adapt_bdd_FINAL.yaml \
#   --dist-url tcp://0.0.0.0:12345 \
#   OUTPUT_DIR outputs/ANURAG_no_warp_no_adapt_bdd_FINAL \
#   > ANURAG_no_warp_no_adapt_bdd_FINAL.out 2>&1 &