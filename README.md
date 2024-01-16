# Environment Setup

Create a new conda env and install torch and detectron2

```shell
conda create -n 2pcnetnew python=3.8
conda activate 2pcnetnew
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

# Dataset Setup

Download [[BDD100K](https://bdd-data.berkeley.edu/)]

Download [[ACDC](https://acdc.vision.ee.ethz.ch/)]

Download [[DENSE](https://www.uni-ulm.de/in/iui-drive-u/projekte/dense-datasets/)]

Download [[Boreas](https://www.boreas.utias.utoronto.ca/#/)]

# BDD100K Label Preparation

Download sample COCO format jsons from [[here](https://drive.google.com/drive/folders/1KV3NqNbRqzBrQ_ZN2rI0jsurPUUgRkKX?usp=drive_link)] or generate COCO format jsons using the scripts available [[here](https://github.com/ShenZheng2000/Instance-Warp-Scripts)]


# Dataset Structures

<details>
  <summary>Click Here</summary>
```
Datasets/
    └── bdd100k/
        ├── images
            ├── 100k
                ├── train/ 
                    ├── img00001.jpg
                    ├──...
                ├── val/ 
                    ├── img00003.jpg
                    ├──...
        ├── coco_labels
            ├── train_day.json
            ├── train_night.json
            ├── val_night.json
            ├──...
    └── acdc/
        ├── rgb_anon
            ├── train
            ├── val
            ├── test
        ├── gt
            ├── train
            ├── val
            ├── test
        ├── gt_detection
            ├── train.json
            ├── val.json
    └── dense/
        ├── cam_stereo_left_lut
            ├── ***.png
            ├── ...
        ├── coco_labels
            ├── train_dense_fog.json
            ├── val_dense_fog.json
            ├──...
    └── boreas/
        ├── images
            ├── train
                ├── ***.png
                ├── ...
            ├── test
                ├── ***.png
                ├── ...
        ├── coco_labels
            ├── train_snowy.json
            ├── test_snowy.json
```
</details>

# Dataset Path

* Go to `twophase/data/datasets/builtin.py`

* Change `data_path` to the absolute path of your dataset folder.


# Configure Path

* Go to `configs/faster_rcnn_R50_bdd100k.yaml` (or your custom yaml file)

* Specify the following

```
TRAIN_LABEL (supervised training images)

TRAIN_LABEL (unsupervised training images)

TEST (testing images)

NIGHTAUG (night augmentation: only useful for day2night domain adaptation)

MAX_ITER (training iterations)

IMG_PER_BATCH_LABEL (batch size for supervised training)

IMG_PER_BATCH_UNLABEL (batch size for unsupervised training)

```

# Training

```
bash train_net.sh
```

or 

```
python train_net.py \
        --num-gpus 3 \
        --config configs/faster_rcnn_R50_bdd100k.yaml \
        OUTPUT_DIR output/bdd100k \
```

# Resume Training

```
python train_net.py \
        --resume \
        --num-gpus 3 \
        --config configs/faster_rcnn_R50_bdd100k.yaml \
        MODEL.WEIGHTS <your weight>.pth  \
        OUTPUT_DIR output/bdd100k \
```


# Testing

```
bash test_net.sh
```

or 

```
python train_net.py \
      --eval-only \
      --config configs/faster_rcnn_R50_bdd100k.yaml \
      MODEL.WEIGHTS <your weight>.pth
```

# Checkpoints
Download all checkpoints from [[here](https://drive.google.com/drive/folders/1PfG6vwMMebGB31cGRzt1nDYwvP2FjJ1h?usp=drive_link)]

# Specific Configs

## BDD100K (Day -> Night)

| Experiments | Configs |
|----------|----------|
| 2PCNet | [pretrained.yaml](https://github.com/ShenZheng2000/Night-Object-Detection/blob/master/configs/pretrained.yaml) |
| 2PCNet + Sta. Prior | [warp_aug_9_12.yaml](https://github.com/ShenZheng2000/Night-Object-Detection/blob/master/configs/warp_aug_9_12.yaml) | 
| 2PCNet + Geo. Prior | [warp_aug_8_2.yaml](https://github.com/ShenZheng2000/Night-Object-Detection/blob/master/configs/warp_aug_8_2.yaml) |
| 2PCNet + Ours | [bdd100k_9_22_v1.yaml](https://github.com/ShenZheng2000/Night-Object-Detection/blob/master/configs/bdd100k_9_22_v1.yaml) |


## BDD100K (Clear -> Rainy)

| Experiments | Configs |
|----------|----------|
| 2PCNet | [bdd100k_10_18_baseline.yaml](https://github.com/ShenZheng2000/Night-Object-Detection/blob/master/configs/bdd100k_10_18_baseline.yaml) |
| 2PCNet + Sta. Prior | [bdd100k_10_18_fovea.yaml](https://github.com/ShenZheng2000/Night-Object-Detection/blob/master/configs/bdd100k_10_18_fovea.yaml) |
| 2PCNet + Geo. Prior | [bdd100k_10_18_tpp.yaml](https://github.com/ShenZheng2000/Night-Object-Detection/blob/master/configs/bdd100k_10_18_tpp.yaml) |
| 2PCNet + Ours | [bdd100k_10_18_bbox.yaml](https://github.com/ShenZheng2000/Night-Object-Detection/blob/master/configs/bdd100k_10_18_bbox.yaml) |


## BDD100K Clear -> ACDC

| Experiments | Configs |
|----------|----------|
| 2PCNet | [acdc_11_6_baseline.yaml](https://github.com/ShenZheng2000/Night-Object-Detection/blob/master/configs/acdc_11_6_baseline.yaml) |
| 2PCNet + Sta. Prior | [acdc_11_6_fovea.yaml](https://github.com/ShenZheng2000/Night-Object-Detection/blob/master/configs/acdc_11_6_fovea.yaml) | 
| 2PCNet + Geo. Prior | [acdc_11_6_tpp.yaml](https://github.com/ShenZheng2000/Night-Object-Detection/blob/master/configs/acdc_11_6_tpp.yaml) |
| 2PCNet + Ours | [acdc_11_6_bbox.yaml](https://github.com/ShenZheng2000/Night-Object-Detection/blob/master/configs/acdc_11_6_bbox.yaml) |


## BDD100K Clear -> DENSE Foggy

| Experiments | Configs |
|----------|----------|
| 2PCNet | [dense_foggy_12_12_baseline.yaml](https://github.com/ShenZheng2000/Night-Object-Detection/blob/master/configs/dense_foggy_12_12_baseline.yaml) |
| 2PCNet + Ours | [dense_foggy_12_12_bbox.yaml](https://github.com/ShenZheng2000/Night-Object-Detection/blob/master/configs/dense_foggy_12_12_bbox.yaml) |


## BDD100K Clear -> Boreas Snowy

| Experiments | Configs |
|----------|----------|
| 2PCNet | [boreas_snow_12_16_baseline.yaml](https://github.com/ShenZheng2000/Night-Object-Detection/blob/master/configs/boreas_snow_12_16_baseline.yaml) |
| 2PCNet + Ours | [boreas_snow_12_16_bbox.yaml](https://github.com/ShenZheng2000/Night-Object-Detection/blob/master/configs/boreas_snow_12_16_bbox.yaml) |


# Acknowledgements

Code is adapted from [[Detectron2](https://github.com/facebookresearch/detectron2)] and [[2PCNet](https://github.com/mecarill/2pcnet)], and is motivated by [[FOVEA](https://github.com/tchittesh/fovea)], [[TPP](https://github.com/geometriczoom/two-plane-prior)], and [[LZU](https://github.com/tchittesh/lzu)].


<!-- TODO: upload link for dense and boreas datasets, since we preprocessed some 3D stuffs into 2D labels -->