## Train on BDD100K (clear) + Tesla Annotated. [Full-size experiments for Tesla]
## Test on Tesla Annotated Test.
# CUDA_VISIBLE_DEVICES=0,1,3 \
#     nohup python train_net.py \
#     --num-gpus 3 \
#     --config configs/ANURAG_no_warp_no_adapt_train_BDD_Clear_And_Tesla_Annotated.yaml \
#     OUTPUT_DIR /shortdata/aghosh/2PCNet/outputs_7_24_v1/fs > outs/fs.out 2>&1 &


## Train on BDD100K (clear) + Tesla Annotated.
## Test on Tesla Annotated Test.
# CUDA_VISIBLE_DEVICES=0,1,3 \
#     nohup python train_net.py \
#     --num-gpus 3 \
#     --config configs/ANURAG_no_warp_no_adapt_train_BDD_Clear_And_Tesla_Annotated.yaml \
#     OUTPUT_DIR /shortdata/aghosh/2PCNet/outputs_7_24_v1/fs > outs/fs.out 2>&1 &


## Train on Tesla Annotated.
## Test on Tesla Annotated Test.
# CUDA_VISIBLE_DEVICES=0,1,3 \
#     nohup python train_net.py \
#     --num-gpus 3 \
#     --config configs/ANURAG_no_warp_no_adapt_train_Tesla_Annotated.yaml \
#     OUTPUT_DIR /shortdata/aghosh/2PCNet/outputs_7_24_v1/sup > outs/sup.out 2>&1 &


# ## Train on BDD100K (clear). Adapt to Tesla Annotated.
# ## Test on Tesla Annotated Test.
# ## 2PCNet Baseline
# CUDA_VISIBLE_DEVICES=0,1,3 \
#     nohup python train_net.py \
#     --resume \
#     --num-gpus 3 \
#     --config configs/ANURAG_bdd100k_10_18_baseline_Tesla_Annotated.yaml \
#     MODEL.WEIGHTS /shortdata/aghosh/2PCNet/outputs_7_24_v1/pretrain/model_0059999.pth \
#     OUTPUT_DIR /shortdata/aghosh/2PCNet/outputs_7_24_v1/da > outs/da.out 2>&1 &


# ## Train on BDD100K (clear). Adapt to Tesla Annotated.
# ## Test on Tesla Annotated Test.
# ## 2PCNet + Instance Warp
# CUDA_VISIBLE_DEVICES=0,1,3 \
#     nohup python train_net.py \
#     --resume \
#     --num-gpus 3 \
#     --config configs/ANURAG_bdd100k_10_18_bbox_Tesla_Annotated.yaml \
#     MODEL.WEIGHTS /shortdata/aghosh/2PCNet/outputs_7_24_v1/pretrain/model_0059999.pth \
#     OUTPUT_DIR /shortdata/aghosh/2PCNet/outputs_7_24_v1/da_bbox > outs/da_bbox.out 2>&1 &


## Train on BDD100K (clear).
## No Warping, No Adaptation

# CUDA_VISIBLE_DEVICES=0,1,3 \
#     nohup python train_net.py \
#     --num-gpus 3 \
#     --config configs/ANURAG_bdd100k_Test_Only_Tesla_Annotated.yaml \
#     OUTPUT_DIR /shortdata/aghosh/2PCNet/outputs_7_24_v1/pretrain > outs/pretrain.out 2>&1 &





















# NOTE: this for testing only! Usually no need to run again, since when train ends, test starts automatically.
# python train_net.py \
#     --num-gpus 3 \
#     --eval-only \
#     --config configs/ANURAG_bdd100k_Test_Only_Tesla_Annotated.yaml \
#     MODEL.WEIGHTS /shortdata/aghosh/2PCNet/outputs_7_24_v1/pretrain/model_final.pth \
#     OUTPUT_DIR /shortdata/aghosh/2PCNet/outputs_7_24_v1/pretrain

# ## pretrain test on Tesla Annotated V2 Test (All)
# python train_net.py \
#     --num-gpus 2 \
#     --eval-only \
#     --config configs/ANURAG_bdd100k_Test_Only_Tesla_AnnotatedV2_All.yaml \
#     MODEL.WEIGHTS /shortdata/aghosh/2PCNet/outputs_7_24_v1/pretrain/model_final.pth \
#     OUTPUT_DIR /shortdata/aghosh/2PCNet/outputs_6_8_v1/pretrain_all

# ## pretrain test on Tesla Annotated V2 Test (Night)
# python train_net.py \
#     --num-gpus 2 \
#     --eval-only \
#     --config configs/ANURAG_bdd100k_Test_Only_Tesla_AnnotatedV2_Night.yaml \
#     MODEL.WEIGHTS /shortdata/aghosh/2PCNet/outputs_7_24_v1/pretrain/model_final.pth \
#     OUTPUT_DIR /shortdata/aghosh/2PCNet/outputs_6_8_v1/pretrain_night

# ## pretrain test on Tesla Annotated V2 Test (Snow)
# python train_net.py \
#     --num-gpus 2 \
#     --eval-only \
#     --config configs/ANURAG_bdd100k_Test_Only_Tesla_AnnotatedV2_Snow.yaml \
#     MODEL.WEIGHTS /shortdata/aghosh/2PCNet/outputs_7_24_v1/pretrain/model_final.pth \
#     OUTPUT_DIR /shortdata/aghosh/2PCNet/outputs_6_8_v1/pretrain_snow

# ## pretrain test on Tesla Annotated V2 Test (Rain)
# python train_net.py \
#     --num-gpus 2 \
#     --eval-only \
#     --config configs/ANURAG_bdd100k_Test_Only_Tesla_AnnotatedV2_Rain.yaml \
#     MODEL.WEIGHTS /shortdata/aghosh/2PCNet/outputs_7_24_v1/pretrain/model_final.pth \
#     OUTPUT_DIR /shortdata/aghosh/2PCNet/outputs_6_8_v1/pretrain_rain

## test on Tesla Annotated V2 Test (All) - fully supervised (best model)
# python train_net.py \
#     --num-gpus 2 \
#     --eval-only \
#     --config configs/ANURAG_bdd100k_Test_Only_Tesla_AnnotatedV2_All.yaml \
#     MODEL.WEIGHTS /shortdata/aghosh/2PCNet/outputs_7_24_v1/fs/model_final.pth \
#     OUTPUT_DIR /shortdata/aghosh/2PCNet/outputs_6_8_v1/fs_all

# # ## test on Tesla Annotated V2 Test (Night) - fully supervised (best model)
# python train_net.py \
#     --num-gpus 2 \
#     --eval-only \
#     --config configs/ANURAG_bdd100k_Test_Only_Tesla_AnnotatedV2_Night.yaml \
#     MODEL.WEIGHTS /shortdata/aghosh/2PCNet/outputs_7_24_v1/fs/model_final.pth \
#     OUTPUT_DIR /shortdata/aghosh/2PCNet/outputs_6_8_v1/fs_night

# # ## test on Tesla Annotated V2 Test (Snow) - fully supervised (best model)
# python train_net.py \
#     --num-gpus 2 \
#     --eval-only \
#     --config configs/ANURAG_bdd100k_Test_Only_Tesla_AnnotatedV2_Snow.yaml \
#     MODEL.WEIGHTS /shortdata/aghosh/2PCNet/outputs_7_24_v1/fs/model_final.pth \
#     OUTPUT_DIR /shortdata/aghosh/2PCNet/outputs_6_8_v1/fs_snow

# # ## test on Tesla Annotated V2 Test (Rain) - fully supervised (best model)
# python train_net.py \
#     --num-gpus 2 \
#     --eval-only \
#     --config configs/ANURAG_bdd100k_Test_Only_Tesla_AnnotatedV2_Rain.yaml \
#     MODEL.WEIGHTS /shortdata/aghosh/2PCNet/outputs_7_24_v1/fs/model_final.pth \
#     OUTPUT_DIR /shortdata/aghosh/2PCNet/outputs_6_8_v1/fs_rain

# ## test on Tesla Annotated V2 Test (All) - domain adaptation baseline (best model)
# python train_net.py \
#     --num-gpus 2 \
#     --eval-only \
#     --config configs/ANURAG_bdd100k_Test_Only_Tesla_AnnotatedV2_All.yaml \
#     MODEL.WEIGHTS /shortdata/aghosh/2PCNet/outputs_7_24_v1/da/model_final.pth \
#     OUTPUT_DIR /shortdata/aghosh/2PCNet/outputs_6_8_v1/da_all

## test on Tesla Annotated V2 Test (Night) - domain adaptation baseline (best model)
# python train_net.py \
#     --num-gpus 2 \
#     --eval-only \
#     --config configs/ANURAG_bdd100k_Test_Only_Tesla_AnnotatedV2_Night.yaml \
#     MODEL.WEIGHTS /shortdata/aghosh/2PCNet/outputs_7_24_v1/da/model_final.pth \
#     OUTPUT_DIR /shortdata/aghosh/2PCNet/outputs_6_8_v1/da_night

## test on Tesla Annotated V2 Test (Rain) - domain adaptation baseline (best model)
# python train_net.py \
#     --num-gpus 2 \
#     --eval-only \
#     --config configs/ANURAG_bdd100k_Test_Only_Tesla_AnnotatedV2_Rain.yaml \
#     MODEL.WEIGHTS /shortdata/aghosh/2PCNet/outputs_7_24_v1/da/model_final.pth \
#     OUTPUT_DIR /shortdata/aghosh/2PCNet/outputs_6_8_v1/da_rain

# ## test on Tesla Annotated V2 Test (Snow) - domain adaptation baseline (best model)
# python train_net.py \
#     --num-gpus 3 \
#     --eval-only \
#     --config configs/ANURAG_bdd100k_Test_Only_Tesla_AnnotatedV2_Snow.yaml \
#     MODEL.WEIGHTS /shortdata/aghosh/2PCNet/outputs_7_24_v1/da/model_final.pth \
#     OUTPUT_DIR /shortdata/aghosh/2PCNet/outputs_6_8_v1/da_snow

## test on Tesla Annotated V2 Test (All) - domain adaptation bbox (best model)
# python train_net.py \
#     --num-gpus 2 \
#     --eval-only \
#     --config configs/ANURAG_bdd100k_Test_Only_Tesla_AnnotatedV2_All.yaml \
#     MODEL.WEIGHTS /shortdata/aghosh/2PCNet/outputs_7_24_v1/da_bbox/model_final.pth \
#     OUTPUT_DIR /shortdata/aghosh/2PCNet/outputs_6_8_v1/da_bbox_all

# ## test on Tesla Annotated V2 Test (Night) - domain adaptation bbox (best model)
# python train_net.py \
#     --num-gpus 2 \
#     --eval-only \
#     --config configs/ANURAG_bdd100k_Test_Only_Tesla_AnnotatedV2_Night.yaml \
#     MODEL.WEIGHTS /shortdata/aghosh/2PCNet/outputs_7_24_v1/da_bbox/model_final.pth \
#     OUTPUT_DIR /shortdata/aghosh/2PCNet/outputs_6_8_v1/da_bbox_night

# ## test on Tesla Annotated V2 Test (Rain) - domain adaptation bbox (best model)
# python train_net.py \
#     --num-gpus 2 \
#     --eval-only \
#     --config configs/ANURAG_bdd100k_Test_Only_Tesla_AnnotatedV2_Rain.yaml \
#     MODEL.WEIGHTS /shortdata/aghosh/2PCNet/outputs_7_24_v1/da_bbox/model_final.pth \
#     OUTPUT_DIR /shortdata/aghosh/2PCNet/outputs_6_8_v1/da_bbox_rain

# ## test on Tesla Annotated V2 Test (Snow) - domain adaptation bbox (best model)
# python train_net.py \
#     --num-gpus 2 \
#     --eval-only \
#     --config configs/ANURAG_bdd100k_Test_Only_Tesla_AnnotatedV2_Snow.yaml \
#     MODEL.WEIGHTS /shortdata/aghosh/2PCNet/outputs_7_24_v1/da_bbox/model_final.pth \
#     OUTPUT_DIR /shortdata/aghosh/2PCNet/outputs_6_8_v1/da_bbox_snow