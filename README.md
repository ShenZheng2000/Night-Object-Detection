# Environment Setup

Python >= 3.6
PyTorch >= 1.5
[[Detectron2==0.6](https://detectron2.readthedocs.io/en/latest/tutorials/install.html)]

# Dataset Setup

Download [[BDD100K](https://bdd-data.berkeley.edu/)]

Download [[ACDC](https://acdc.vision.ee.ethz.ch/)]


# Dataset Structures

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
```

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


# Acknowledgements

Code is adapted from [[Detectron2](https://github.com/facebookresearch/detectron2)] and [[2PCNet](https://github.com/mecarill/2pcnet)]