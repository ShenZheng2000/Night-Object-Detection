import os
import contextlib
from detectron2.data import DatasetCatalog, MetadataCatalog
from fvcore.common.timer import Timer
from iopath.common.file_io import PathManager

from detectron2.data.datasets.pascal_voc import register_pascal_voc
from detectron2.data.datasets.builtin_meta import _get_builtin_metadata
from detectron2.data.datasets import register_coco_instances
import io
import logging

logger = logging.getLogger(__name__)

JSON_ANNOTATIONS_DIR = ""
_SPLITS_COCO_FORMAT = {}
_SPLITS_COCO_FORMAT["coco"] = {
    "coco_2017_unlabel": (
        "coco/unlabeled2017",
        "coco/annotations/image_info_unlabeled2017.json",
    ),
    "coco_2017_for_voc20": (
        "coco",
        "coco/annotations/google/instances_unlabeledtrainval20class.json",
    ),
}


def register_coco_unlabel(root):
    for _, splits_per_dataset in _SPLITS_COCO_FORMAT.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            meta = {}
            register_coco_unlabel_instances(
                key, meta, os.path.join(root, json_file), os.path.join(root, image_root)
            )


def register_coco_unlabel_instances(name, metadata, json_file, image_root):
    """
    Register a dataset in COCO's json annotation format for
    instance detection, instance segmentation and keypoint detection.
    (i.e., Type 1 and 2 in http://cocodataset.org/#format-data.
    `instances*.json` and `person_keypoints*.json` in the dataset).

    This is an example of how to register a new dataset.
    You can do something similar to this function, to register new datasets.

    Args:
        name (str): the name that identifies a dataset, e.g. "coco_2014_train".
        metadata (dict): extra metadata associated with this dataset.  You can
            leave it as an empty dict.
        json_file (str): path to the json instance annotation file.
        image_root (str or path-like): directory which contains all the images.
    """
    assert isinstance(name, str), name
    assert isinstance(json_file, (str, os.PathLike)), json_file
    assert isinstance(image_root, (str, os.PathLike)), image_root

    # 1. register a function which returns dicts
    DatasetCatalog.register(
        name, lambda: load_coco_unlabel_json(json_file, image_root, name)
    )

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root, evaluator_type="coco", **metadata
    )


def load_coco_unlabel_json(
    json_file, image_root, dataset_name=None, extra_annotation_keys=None
):
    from pycocotools.coco import COCO

    timer = Timer()
    json_file = PathManager.get_local_path(json_file)
    with contextlib.redirect_stdout(io.StringIO()):
        coco_api = COCO(json_file)
    if timer.seconds() > 1:
        logger.info(
            "Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds())
        )

    id_map = None
    # sort indices for reproducible results
    img_ids = sorted(coco_api.imgs.keys())

    imgs = coco_api.loadImgs(img_ids)

    logger.info("Loaded {} images in COCO format from {}".format(len(imgs), json_file))

    dataset_dicts = []

    for img_dict in imgs:
        record = {}
        record["file_name"] = os.path.join(image_root, img_dict["file_name"])
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        image_id = record["image_id"] = img_dict["id"]

        dataset_dicts.append(record)

    return dataset_dicts


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_coco_unlabel(_root)


# ==== Predefined splits for raw cityscapes foggy images ===========
_RAW_CITYSCAPES_SPLITS = {
    # "cityscapes_foggy_{task}_train": ("cityscape_foggy/leftImg8bit/train/", "cityscape_foggy/gtFine/train/"),
    # "cityscapes_foggy_{task}_val": ("cityscape_foggy/leftImg8bit/val/", "cityscape_foggy/gtFine/val/"),
    # "cityscapes_foggy_{task}_test": ("cityscape_foggy/leftImg8bit/test/", "cityscape_foggy/gtFine/test/"),
    "cityscapes_foggy_train": ("cityscapes_foggy/leftImg8bit/train/", "cityscapes_foggy/gtFine/train/"),
    "cityscapes_foggy_val": ("cityscapes_foggy/leftImg8bit/val/", "cityscapes_foggy/gtFine/val/"),
    "cityscapes_foggy_test": ("cityscapes_foggy/leftImg8bit/test/", "cityscapes_foggy/gtFine/test/"),
}


def register_all_cityscapes_foggy(root):
    # root = "manifold://mobile_vision_dataset/tree/yujheli/dataset"
    for key, (image_dir, gt_dir) in _RAW_CITYSCAPES_SPLITS.items():
        meta = _get_builtin_metadata("cityscapes")
        image_dir = os.path.join(root, image_dir)
        gt_dir = os.path.join(root, gt_dir)

        # inst_key = key.format(task="instance_seg")
        inst_key = key
        # DatasetCatalog.register(
        #     inst_key,
        #     lambda x=image_dir, y=gt_dir: load_cityscapes_instances(
        #         x, y, from_json=True, to_polygons=True
        #     ),
        # )
        DatasetCatalog.register(
            inst_key,
            lambda x=image_dir, y=gt_dir: load_cityscapes_instances(
                x, y, from_json=False, to_polygons=False
            ),
        )
        # MetadataCatalog.get(inst_key).set(
        #     image_dir=image_dir, gt_dir=gt_dir, evaluator_type="cityscapes_instance", **meta
        # )
        # MetadataCatalog.get(inst_key).set(
        #     image_dir=image_dir, gt_dir=gt_dir, evaluator_type="pascal_voc", **meta
        # )
        MetadataCatalog.get(inst_key).set(
            image_dir=image_dir, gt_dir=gt_dir, evaluator_type="coco", **meta
        )

# ==== Predefined splits for Clipart (PASCAL VOC format) ===========
def register_all_clipart(root):
    # root = "manifold://mobile_vision_dataset/tree/yujheli/dataset"
    SPLITS = [
        ("Clipart1k_train", "clipart", "train"),
        ("Clipart1k_test", "clipart", "test"),
    ]
    for name, dirname, split in SPLITS:
        year = 2012
        register_pascal_voc(name, os.path.join(root, dirname), split, year)
        MetadataCatalog.get(name).evaluator_type = "pascal_voc"
        # MetadataCatalog.get(name).evaluator_type = "coco"

# ==== Predefined splits for Watercolor (PASCAL VOC format) ===========
def register_all_water(root):
    # root = "manifold://mobile_vision_dataset/tree/yujheli/dataset"
    SPLITS = [
        ("Watercolor_train", "watercolor", "train"),
        ("Watercolor_test", "watercolor", "test"),
    ]
    for name, dirname, split in SPLITS:
        year = 2012
        # register_pascal_voc(name, os.path.join(root, dirname), split, year, class_names=["person", "dog","bicycle", "bird", "car", "cat"])
        register_pascal_voc(name, os.path.join(root, dirname), split, year)
        MetadataCatalog.get(name).evaluator_type = "pascal_voc_water"
        # MetadataCatalog.get(name).thing_classes = ["person", "dog","bike", "bird", "car", "cat"]
        # MetadataCatalog.get(name).thing_classes = ["person", "dog","bicycle", "bird", "car", "cat"]
        # MetadataCatalog.get(name).evaluator_type = "coco"

register_all_cityscapes_foggy(_root)
register_all_clipart(_root)
register_all_water(_root)

data_path = '/home/aghosh/Projects/2PCNet/Datasets'
enhance_path = '/home/aghosh/Projects/2PCNet/LLIE/Results'

# NOTE: warp for LLIE enhanced images
def register_retinex_model(model_name):
    register_coco_instances(f"bdd100k_night_val_night_{model_name}", 
                            {}, 
                            f'{data_path}/bdd100k/coco_labels/val_night.json', 
                            f'{enhance_path}/{model_name}/val_night')


# Register datasets

# NOTE: change to current path
# TODO: use the correct label and #GPU for training!
# NOTE: change according to each server


# NOTE: add complete bdd train and test
register_coco_instances("bdd100k_train",
{}, 
f'{data_path}/bdd100k/coco_labels/bdd100k_labels_images_train.json', 
f'{data_path}/bdd100k/images/100k/train')

register_coco_instances("bdd100k_val",
{}, 
f'{data_path}/bdd100k/coco_labels/bdd100k_labels_images_val.json', 
f'{data_path}/bdd100k/images/100k/val')



register_coco_instances("bdd100k_day_train",
{}, 
f'{data_path}/bdd100k/coco_labels/train_day.json', 
f'{data_path}/bdd100k/images/100k/train')

register_coco_instances("bdd100k_day_train_mid",
{}, 
f'{data_path}/bdd100k/coco_labels/train_day.json', 
f'{data_path}/bdd100k/images/100k/TPSeNCE_night')

register_coco_instances("bdd100k_day_train_valid_vp",
{}, 
f'{data_path}/bdd100k/coco_labels/train_day_valid_vp.json', 
f'{data_path}/bdd100k/images/100k/train')



# NOTE: 
register_coco_instances("bdd100k_clear_train_valid_vp",
{}, 
f'{data_path}/bdd100k/coco_labels/train_clear_valid_vp.json', 
f'{data_path}/bdd100k/images/100k/train')

register_coco_instances("bdd100k_clear_train_small_valid_vp",
{}, 
f'{data_path}/bdd100k/coco_labels/train_clear_small_valid_vp.json', 
f'{data_path}/bdd100k/images/100k/train')

register_coco_instances("bdd100k_rainy_train",
{}, 
f'{data_path}/bdd100k/coco_labels/train_rainy.json', 
f'{data_path}/bdd100k/images/100k/train')


# NOTE: 
register_coco_instances("bdd100k_rainy_val",
{}, 
f'{data_path}/bdd100k/coco_labels/val_rainy.json', 
f'{data_path}/bdd100k/images/100k/val')

# NOTE: 
register_coco_instances("bdd100k_rainy_val_100",
{}, 
f'{data_path}/bdd100k/coco_labels/val_rainy_100.json', 
f'{data_path}/bdd100k/images/100k/val')

register_coco_instances("bdd100k_rainy_val_day",
{}, 
f'{data_path}/bdd100k/coco_labels/val_rainy_day.json', 
f'{data_path}/bdd100k/images/100k/val')

register_coco_instances("bdd100k_rainy_val_night",
{}, 
f'{data_path}/bdd100k/coco_labels/val_rainy_night.json', 
f'{data_path}/bdd100k/images/100k/val')

register_coco_instances("bdd100k_rainy_val_SAPNet",
{}, 
f'{data_path}/bdd100k/coco_labels/val_rainy.json', 
f'{data_path}/bdd100k/images/100k/val_enhance/SAPNet')

register_coco_instances("bdd100k_rainy_val_100_SAPNet",
{}, 
f'{data_path}/bdd100k/coco_labels/val_rainy_100.json', 
f'{data_path}/bdd100k/images/100k/val_enhance/SAPNet')

register_coco_instances("bdd100k_rainy_val_day_SAPNet",
{}, 
f'{data_path}/bdd100k/coco_labels/val_rainy_day.json', 
f'{data_path}/bdd100k/images/100k/val_enhance/SAPNet')

register_coco_instances("bdd100k_rainy_val_night_SAPNet",
{}, 
f'{data_path}/bdd100k/coco_labels/val_rainy_night.json', 
f'{data_path}/bdd100k/images/100k/val_enhance/SAPNet')

# NOTE: 
register_coco_instances("bdd100k_clear_val",
{}, 
f'{data_path}/bdd100k/coco_labels/val_clear.json', 
f'{data_path}/bdd100k/images/100k/val')

# NOTE: 
register_coco_instances("bdd100k_test_day_good_weather",
{}, 
f'{data_path}/bdd100k/coco_labels/val_day_good_weather.json', 
f'{data_path}/bdd100k/images/100k/val')

# NOTE: 
register_coco_instances("bdd100k_test_day_bad_weather",
{}, 
f'{data_path}/bdd100k/coco_labels/val_day_bad_weather.json', 
f'{data_path}/bdd100k/images/100k/val')

register_coco_instances("bdd100k_test_clear_night",
{}, 
f'{data_path}/bdd100k/coco_labels/val_clear_night.json', 
f'{data_path}/bdd100k/images/100k/val')

register_coco_instances("bdd100k_test_bad_night",
{}, 
f'{data_path}/bdd100k/coco_labels/val_bad_night.json', 
f'{data_path}/bdd100k/images/100k/val')


# NOTE: add acdc here
register_coco_instances("acdc_train",
{}, 
f'{data_path}/acdc/gt_detection/train.json', 
f'{data_path}/acdc/rgb_anon')

register_coco_instances("acdc_val",
{}, 
f'{data_path}/acdc/gt_detection/val.json', 
f'{data_path}/acdc/rgb_anon')

# TODO: add weather of tod specific here!!!!!!!!!!
register_coco_instances("acdc_val_fog",
{}, 
f'{data_path}/acdc/gt_detection/val_fog.json', 
f'{data_path}/acdc/rgb_anon')

register_coco_instances("acdc_val_night",
{}, 
f'{data_path}/acdc/gt_detection/val_night.json', 
f'{data_path}/acdc/rgb_anon')

register_coco_instances("acdc_val_rain",
{}, 
f'{data_path}/acdc/gt_detection/val_rain.json', 
f'{data_path}/acdc/rgb_anon')

register_coco_instances("acdc_val_snow",
{}, 
f'{data_path}/acdc/gt_detection/val_snow.json', 
f'{data_path}/acdc/rgb_anon')


# NOTE: add dense here
register_coco_instances("dense_foggy_train",
{},
f'{data_path}/dense/coco_labels/train_dense_fog.json',
f'{data_path}/dense/cam_stereo_left_lut')

register_coco_instances("dense_foggy_val",
{},
f'{data_path}/dense/coco_labels/val_dense_fog.json',
f'{data_path}/dense/cam_stereo_left_lut')

register_coco_instances("dense_foggy_val_dehazeformer",
{},
f'{data_path}/dense/coco_labels/val_dense_fog.json',
f'{data_path}/dense/cam_stereo_left_lut_fog_val_dehazeformer')

register_coco_instances("dense_snow_train",
{},
f'{data_path}/dense/coco_labels/train_snow.json',
f'{data_path}/dense/cam_stereo_left_lut')

register_coco_instances("dense_snow_val",
{},
f'{data_path}/dense/coco_labels/val_snow.json',
f'{data_path}/dense/cam_stereo_left_lut')

# NOTE: add boreas snowy here
register_coco_instances("boreas_snowy_train",
{},
f'{data_path}/boreas/coco_labels/train_snowy.json',
f'{data_path}/boreas/images/train')

register_coco_instances("boreas_snowy_test",
{},
f'{data_path}/boreas/coco_labels/test_snowy.json',
f'{data_path}/boreas/images/test')


register_coco_instances("bdd100k_night_train", 
{}, 
f'{data_path}/bdd100k/coco_labels/train_night.json', 
f'{data_path}/bdd100k/images/100k/train')

register_coco_instances("bdd100k_night_train_valid_vp", 
{}, 
f'{data_path}/bdd100k/coco_labels/train_night_valid_vp.json', 
f'{data_path}/bdd100k/images/100k/train')


register_coco_instances("bdd100k_night_train_TPSeNCE", 
{}, 
f'{data_path}/bdd100k/coco_labels/train_day.json', 
f'{data_path}/bdd100k/images/100k/TPSeNCE_night')

# NOTE: add labels for curriculum learning
register_coco_instances("bdd100k_train_dd", 
{}, 
f'{data_path}/bdd100k/coco_labels/train_dawn_dusk_cur.json', 
f'{data_path}/bdd100k/images/100k/train')

register_coco_instances("bdd100k_train_n", 
{}, 
f'{data_path}/bdd100k/coco_labels/train_night_cur.json', 
f'{data_path}/bdd100k/images/100k/train')

register_coco_instances("bdd100k_train_n_png", 
{}, 
f'{data_path}/bdd100k/coco_labels/train_night_cur_png.json', 
f'{data_path}/Depth/train__night_dawn_dusk')

# register for c.l. (w/ simple swap)
register_coco_instances("bdd100k_train_ddd", 
{}, 
f'{data_path}/bdd100k/coco_labels/train_day_dawn_dusk_cur.json', 
f'{data_path}/bdd100k/images/100k/train')

register_coco_instances("bdd100k_train_ddn", 
{}, 
f'{data_path}/bdd100k/coco_labels/train_dawn_dusk_night_cur.json', 
f'{data_path}/bdd100k/images/100k/train')

register_coco_instances("bdd100k_night_val", 
{}, 
f'{data_path}/bdd100k/coco_labels/val_night.json', 
f'{data_path}/bdd100k/images/100k/val')

register_coco_instances("bdd100k_day_val", 
{}, 
f'{data_path}/bdd100k/coco_labels/val_day.json', 
f'{data_path}/bdd100k/images/100k/val')

register_coco_instances("bdd100k_night_val_valid_vp", 
{}, 
f'{data_path}/bdd100k/coco_labels/val_night_valid_vp.json', 
f'{data_path}/bdd100k/images/100k/val')

# NOTE: add enhanced images
register_retinex_model("LLFlow")
register_retinex_model("RetinexNet")
register_retinex_model("RUAS")
register_retinex_model("SCI")
register_retinex_model("SGZ")
register_retinex_model("URetinexNet")
register_retinex_model("ZeroDCE")

# SGZ
register_coco_instances("bdd100k_night_val_SGZ", 
{}, 
f'{data_path}/bdd100k/coco_labels/val_night.json', 
f'{data_path}/bdd100k/images/100k/val_enhance/SGZ')

register_coco_instances("bdd100k_night_val_clear_SGZ", 
{}, 
f'{data_path}/bdd100k/coco_labels/val_clear_night.json', 
f'{data_path}/bdd100k/images/100k/val_enhance/SGZ')

register_coco_instances("bdd100k_night_val_bad_SGZ", 
{}, 
f'{data_path}/bdd100k/coco_labels/val_bad_night.json', 
f'{data_path}/bdd100k/images/100k/val_enhance/SGZ')

register_coco_instances("bdd100k_night_val_CLAHE", 
{}, 
f'{data_path}/bdd100k/coco_labels/val_night.json', 
f'{data_path}/bdd100k/images/100k/val_enhance/CLAHE')

register_coco_instances("20230112_o", 
{}, 
f'{data_path}/rain_night01/20230112_o.json',
f'{data_path}/rain_night01/20230112_o')

register_coco_instances("bdd100k_night_val_all", 
{}, 
f'{data_path}/bdd100k/coco_labels_new/all/bdd100k_labels_images_det_coco_val_night_dawn_dusk.json', 
f'{data_path}/bdd100k/images/100k/val')

register_coco_instances("bdd100k_night_val_rainy", 
{}, 
f'{data_path}/bdd100k/coco_labels_new/rainy/bdd100k_labels_images_det_coco_val_rainy_night_dawn_dusk.json', 
f'{data_path}/bdd100k/images/100k/val')

register_coco_instances("bdd100k_night_val_snowy", 
{}, 
f'{data_path}/bdd100k/coco_labels_new/snowy/bdd100k_labels_images_det_coco_val_snowy_night_dawn_dusk.json', 
f'{data_path}/bdd100k/images/100k/val')

register_coco_instances("bdd100k_night_val_foggy", 
{}, 
f'{data_path}/bdd100k/coco_labels_new/foggy/bdd100k_labels_images_det_coco_val_foggy_night_dawn_dusk.json', 
f'{data_path}/bdd100k/images/100k/val')

register_coco_instances("bdd100k_night_train_depth", 
{}, 
f'{data_path}/bdd100k/coco_labels/train_night_png.json',
f'{data_path}/Depth/train__night_dawn_dusk')

register_coco_instances("sim_day_train",
{},
'datasets/shift/train_day.json', 
'datasets/shift/train')
register_coco_instances( "sim_night_train", 
{}, 
'datasets/shift/train_night.json', 
'datasets/shift/train')
register_coco_instances("sim_night_val", 
{}, 
'datasets/shift/val_night.json', 
'datasets/shift/val')


# NOTE: add gm videos here
register_coco_instances("gm_rainy_day", 
{}, 
f'{data_path}/gm/coco_labels/rainy_day.json', 
f'{data_path}/gm/rainy/day/images')


# NOTE: add gm videos here
register_coco_instances("gm_rainy_night", 
{}, 
f'{data_path}/gm/coco_labels/rainy_night.json', 
f'{data_path}/gm/rainy/night/images')


# NOTE: add gm videos here
register_coco_instances("gm_foggy_day", 
{}, 
f'{data_path}/gm/coco_labels/foggy_day.json', 
f'{data_path}/gm/foggy/day/images')



# NOTE; add construction-zone here
register_coco_instances("construct_trainA", 
{}, 
f'{data_path}/data/annotations/geographic_da/instances_pretrain.json', 
f'{data_path}/data/images')

register_coco_instances("construct_trainB", 
{}, 
f'{data_path}/data/annotations/geographic_da/instances_unsupervised_with_gt.json', 
f'{data_path}/data/images')

register_coco_instances("construct_testB", 
{}, 
f'{data_path}/data/annotations/geographic_da/instances_test.json', 
f'{data_path}/data/images')

register_coco_instances("construct_trainAB", 
{}, 
f'{data_path}/data/annotations/geographic_da/instances_all.json', 
f'{data_path}/data/images')