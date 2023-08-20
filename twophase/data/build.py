import logging
import numpy as np
import operator
import json
import torch.utils.data
from detectron2.utils.comm import get_world_size
from detectron2.data.common import (
    DatasetFromList,
    MapDataset,
)
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data.samplers import (
    InferenceSampler,
    RepeatFactorTrainingSampler,
    TrainingSampler,
)
from detectron2.data.build import (
    trivial_batch_collator,
    worker_init_reset_seed,
    get_detection_dataset_dicts,
    build_batch_data_loader,
)
from twophase.data.common import (
    AspectRatioGroupedSemiSupDatasetTwoCrop,
)


"""
This file contains the default logic to build a dataloader for training or testing.
"""


# def divide_label_unlabel(
#     dataset_dicts, SupPercent, random_data_seed, random_data_seed_path
# ):
#     num_all = len(dataset_dicts)
#     num_label = int(SupPercent / 100.0 * num_all)

#     # read from pre-generated data seed
#     with open(random_data_seed_path) as COCO_sup_file:
#         coco_random_idx = json.load(COCO_sup_file)

#     labeled_idx = np.array(coco_random_idx[str(SupPercent)][str(random_data_seed)])
#     assert labeled_idx.shape[0] == num_label, "Number of READ_DATA is mismatched."

#     label_dicts = []
#     unlabel_dicts = []
#     labeled_idx = set(labeled_idx)

#     for i in range(len(dataset_dicts)):
#         if i in labeled_idx:
#             label_dicts.append(dataset_dicts[i])
#         else:
#             unlabel_dicts.append(dataset_dicts[i])

#     return label_dicts, unlabel_dicts

# NOTE: add if-else for fully supervised learning
def divide_label_unlabel(dataset_dicts, SupPercent, random_data_seed=None, random_data_seed_path=None):
    if SupPercent == 100.0:
        label_dicts = dataset_dicts
        unlabel_dicts = []
    else:
        num_all = len(dataset_dicts)
        num_label = int(SupPercent / 100.0 * num_all)

        # read from pre-generated data seed
        with open(random_data_seed_path) as COCO_sup_file:
            coco_random_idx = json.load(COCO_sup_file)

        labeled_idx = np.array(coco_random_idx[str(SupPercent)][str(random_data_seed)])
        assert labeled_idx.shape[0] == num_label, "Number of READ_DATA is mismatched."

        label_dicts = []
        unlabel_dicts = []
        labeled_idx = set(labeled_idx)

        for i in range(len(dataset_dicts)):
            if i in labeled_idx:
                label_dicts.append(dataset_dicts[i])
            else:
                unlabel_dicts.append(dataset_dicts[i])

    return label_dicts, unlabel_dicts


# uesed by supervised-only baseline trainer
def build_detection_semisup_train_loader(cfg, mapper=None):
    dataset_dicts = get_detection_dataset_dicts(
        cfg.DATASETS.TRAIN,
        filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
        min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
        if cfg.MODEL.KEYPOINT_ON
        else 0,
        proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN
        if cfg.MODEL.LOAD_PROPOSALS
        else None,
    )

    # Divide into labeled and unlabeled sets according to supervision percentage
    label_dicts, unlabel_dicts = divide_label_unlabel(
        dataset_dicts,
        cfg.DATALOADER.SUP_PERCENT,
        cfg.DATALOADER.RANDOM_DATA_SEED,
        cfg.DATALOADER.RANDOM_DATA_SEED_PATH,
    )

    dataset = DatasetFromList(label_dicts, copy=False)

    if mapper is None:
        mapper = DatasetMapper(cfg, True)
    dataset = MapDataset(dataset, mapper)

    sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
    logger = logging.getLogger(__name__)
    logger.info("Using training sampler {}".format(sampler_name))

    if sampler_name == "TrainingSampler":
        sampler = TrainingSampler(len(dataset))
    elif sampler_name == "RepeatFactorTrainingSampler":
        repeat_factors = (
            RepeatFactorTrainingSampler.repeat_factors_from_category_frequency(
                label_dicts, cfg.DATALOADER.REPEAT_THRESHOLD
            )
        )
        sampler = RepeatFactorTrainingSampler(repeat_factors)
    else:
        raise ValueError("Unknown training sampler: {}".format(sampler_name))

    # list num of labeled and unlabeled
    logger.info("Number of training samples " + str(len(dataset)))
    logger.info("Supervision percentage " + str(cfg.DATALOADER.SUP_PERCENT))

    return build_batch_data_loader(
        dataset,
        sampler,
        cfg.SOLVER.IMS_PER_BATCH,
        aspect_ratio_grouping=cfg.DATALOADER.ASPECT_RATIO_GROUPING,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
    )


# uesed by evaluation
def build_detection_test_loader(cfg, dataset_name, mapper=None):
    dataset_dicts = get_detection_dataset_dicts(
        [dataset_name],
        filter_empty=False,
        proposal_files=[
            cfg.DATASETS.PROPOSAL_FILES_TEST[
                list(cfg.DATASETS.TEST).index(dataset_name)
            ]
        ]
        if cfg.MODEL.LOAD_PROPOSALS
        else None,
    )
    dataset = DatasetFromList(dataset_dicts)
    if mapper is None:
        mapper = DatasetMapper(cfg, False)
    dataset = MapDataset(dataset, mapper)

    sampler = InferenceSampler(len(dataset))
    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, 1, drop_last=False)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        batch_sampler=batch_sampler,
        collate_fn=trivial_batch_collator,
    )
    return data_loader


# Helper function to reduce code redund. 
def build_dataset_dicts(cfg, dataset_name, filter_empty=False):
    return get_detection_dataset_dicts(
        dataset_name,
        filter_empty=filter_empty,
        min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
            if cfg.MODEL.KEYPOINT_ON
            else 0,
        proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN
            if cfg.MODEL.LOAD_PROPOSALS
            else None,
    )


# Helper function to reduce code redund. 
def build_unlabel_dataset(cfg, dataset_name, mapper, copy=False):
    dataset_dicts = build_dataset_dicts(cfg, dataset_name)
    dataset = DatasetFromList(dataset_dicts, copy=copy)
    return MapDataset(dataset, mapper)


# uesed by unbiased teacher trainer
def build_detection_semisup_train_loader_two_crops(cfg, mapper=None):
    if cfg.DATASETS.CROSS_DATASET:
        label_dicts = build_dataset_dicts(cfg, cfg.DATASETS.TRAIN_LABEL, filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS)
        # unlabel_dicts = build_dataset_dicts(cfg, cfg.DATASETS.TRAIN_UNLABEL)
        # unlabel_dep_dicts = build_dataset_dicts(cfg, cfg.DATASETS.TRAIN_UNLABEL_DEPTH)
    else:
        print("CROSS_DATASET must be True for now")

    if mapper is None:
        mapper = DatasetMapper(cfg, True)

    label_dataset = DatasetFromList(label_dicts, copy=False)
    label_dataset = MapDataset(label_dataset, mapper)

    if cfg.DATASETS.CUR_LEARN_SEQ or cfg.DATASETS.CUR_LEARN_MIX:
        label_mid_dicts = build_dataset_dicts(cfg, cfg.DATASETS.TRAIN_LABEL_MID, 
                                              filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS)
        label_mid_dataset = DatasetFromList(label_mid_dicts, copy=False)
        label_mid_dataset = MapDataset(label_mid_dataset, mapper)

    unlabel_dataset = build_unlabel_dataset(cfg, cfg.DATASETS.TRAIN_UNLABEL, mapper)
    # unlabel_dep_dataset = build_unlabel_dataset(cfg, cfg.DATASETS.TRAIN_UNLABEL_DEPTH, mapper)

    sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
    logger = logging.getLogger(__name__)
    logger.info("Using training sampler {}".format(sampler_name))

    if sampler_name == "TrainingSampler":
        label_sampler = TrainingSampler(len(label_dataset))
        unlabel_sampler = TrainingSampler(len(unlabel_dataset))
        # unlabel_dep_sampler = TrainingSampler(len(unlabel_dep_dataset))
        
        if cfg.DATASETS.CUR_LEARN_SEQ or cfg.DATASETS.CUR_LEARN_MIX:
            label_mid_sampler = TrainingSampler(len(label_mid_dataset))

        if cfg.DATASETS.CUR_LEARN:
            unlabel_dataset_mid = build_unlabel_dataset(cfg, cfg.DATASETS.TRAIN_UNLABEL_MID, mapper)
            unlabel_dataset_last = build_unlabel_dataset(cfg, cfg.DATASETS.TRAIN_UNLABEL_LAST, mapper)
            unlabel_sampler_mid = TrainingSampler(len(unlabel_dataset_mid))
            unlabel_sampler_last = TrainingSampler(len(unlabel_dataset_last))
    elif sampler_name == "RepeatFactorTrainingSampler":
        raise NotImplementedError("{} not yet supported.".format(sampler_name))
    else:
        raise ValueError("Unknown training sampler: {}".format(sampler_name))

    # datasets = [label_dataset, unlabel_dataset, unlabel_dep_dataset]
    # samplers = [label_sampler, unlabel_sampler, unlabel_dep_sampler]
    datasets = [label_dataset, unlabel_dataset]
    samplers = [label_sampler, unlabel_sampler]

    if cfg.DATASETS.CUR_LEARN_SEQ or cfg.DATASETS.CUR_LEARN_MIX:
        datasets.insert(1, label_mid_dataset)
        samplers.insert(1, label_mid_sampler)

    if cfg.DATASETS.CUR_LEARN:
        datasets.extend([unlabel_dataset_mid, unlabel_dataset_last])
        samplers.extend([unlabel_sampler_mid, unlabel_sampler_last])

    return build_semisup_batch_data_loader_two_crop(
        datasets,
        samplers,
        cfg.SOLVER.IMG_PER_BATCH_LABEL,
        cfg.SOLVER.IMG_PER_BATCH_UNLABEL,
        aspect_ratio_grouping=cfg.DATALOADER.ASPECT_RATIO_GROUPING,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
    )

# batch data loader
def build_data_loader(dataset, sampler, num_workers):
    return torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        num_workers=num_workers,
        batch_sampler=None,
        collate_fn=operator.itemgetter(0),  # don't batch, but yield individual elements
        worker_init_fn=worker_init_reset_seed,
    )  # yield individual mapped dict


def build_semisup_batch_data_loader_two_crop(
    dataset_tuple,
    sampler_tuple,
    total_batch_size_label,
    total_batch_size_unlabel,
    *,
    aspect_ratio_grouping=False,
    num_workers=0
):
    world_size = get_world_size()
    assert (
        total_batch_size_label > 0 and total_batch_size_label % world_size == 0
    ), "Total label batch size ({}) must be divisible by the number of gpus ({}).".format(
        total_batch_size_label, world_size
    )

    assert (
        total_batch_size_unlabel > 0 and total_batch_size_unlabel % world_size == 0
    ), "Total unlabel batch size ({}) must be divisible by the number of gpus ({}).".format(
        total_batch_size_unlabel, world_size
    )

    batch_size_label = total_batch_size_label // world_size
    batch_size_unlabel = total_batch_size_unlabel // world_size

    if aspect_ratio_grouping:
        data_loaders = [build_data_loader(dataset, sampler, num_workers) 
                        for dataset, sampler in zip(dataset_tuple, sampler_tuple)]
        
        batch_sizes = [batch_size_label] + [batch_size_unlabel] * (len(data_loaders) - 1)

        return AspectRatioGroupedSemiSupDatasetTwoCrop(data_loaders, batch_sizes)
    else:
        raise NotImplementedError("ASPECT_RATIO_GROUPING = False is not supported yet")

# def build_semisup_batch_data_loader_two_crop(
#     dataset,
#     sampler,
#     total_batch_size_label,
#     total_batch_size_unlabel,
#     *,
#     aspect_ratio_grouping=False,
#     num_workers=0
# ):
#     world_size = get_world_size()
#     assert (
#         total_batch_size_label > 0 and total_batch_size_label % world_size == 0
#     ), "Total label batch size ({}) must be divisible by the number of gpus ({}).".format(
#         total_batch_size_label, world_size
#     )

#     assert (
#         total_batch_size_unlabel > 0 and total_batch_size_unlabel % world_size == 0
#     ), "Total unlabel batch size ({}) must be divisible by the number of gpus ({}).".format(
#         total_batch_size_unlabel, world_size
#     )

#     batch_size_label = total_batch_size_label // world_size
#     batch_size_unlabel = total_batch_size_unlabel // world_size

#     label_dataset, unlabel_dataset, unlabel_dep_dataset = dataset
#     label_sampler, unlabel_sampler, unlabel_dep_sampler = sampler

#     if aspect_ratio_grouping:
#         label_data_loader = build_data_loader(label_dataset, label_sampler, num_workers)
#         unlabel_data_loader = build_data_loader(unlabel_dataset, unlabel_sampler, num_workers)
#         unlabel_depth_data_loader = build_data_loader(unlabel_dep_dataset, unlabel_dep_sampler, num_workers)
        
#         return AspectRatioGroupedSemiSupDatasetTwoCrop(
#             (label_data_loader, unlabel_data_loader, unlabel_depth_data_loader),
#             (batch_size_label, batch_size_unlabel, batch_size_unlabel),
#         )
#     else:
#         raise NotImplementedError("ASPECT_RATIO_GROUPING = False is not supported yet")