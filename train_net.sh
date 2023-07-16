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


# NOTE: train model with cur learning (M1)
nohup \
python train_net.py \
      --num-gpus 3 \
      --config configs/bdd100k_cur.yaml \
      OUTPUT_DIR outputs/mask_cur_7_15 \
      > mask_cur_7_15.out 2>&1 &

# TODO: train model with cur learning (M2)