#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
A main training script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

import logging
import os
from collections import OrderedDict

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import (
    default_argument_parser,
    default_setup,
    DefaultTrainer,
    hooks,
    launch,
)
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.modeling import GeneralizedRCNNWithTTA

from detectron2.data.datasets import register_coco_instances

data_path = '/home/aghosh/Projects/2PCNet/Datasets'

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

register_coco_instances("bdd100k_night_val", 
{}, 
f'{data_path}/bdd100k/coco_labels/val_night.json', 
f'{data_path}/bdd100k/images/100k/val')

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


def build_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
        evaluator_list.append(
            SemSegEvaluator(
                dataset_name,
                distributed=True,
                output_dir=output_folder,
            )
        )
    if evaluator_type in ["coco", "coco_panoptic_seg"]:
        evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
    if evaluator_type == "coco_panoptic_seg":
        evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
    if evaluator_type == "cityscapes_instance":
        return CityscapesInstanceEvaluator(dataset_name)
    if evaluator_type == "cityscapes_sem_seg":
        return CityscapesSemSegEvaluator(dataset_name)
    elif evaluator_type == "pascal_voc":
        return PascalVOCDetectionEvaluator(dataset_name)
    elif evaluator_type == "lvis":
        return LVISEvaluator(dataset_name, output_dir=output_folder)
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(
                dataset_name, evaluator_type
            )
        )
    elif len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return build_evaluator(cfg, dataset_name, output_folder)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()


def invoke_main() -> None:
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )


if __name__ == "__main__":
    invoke_main()  # pragma: no cover