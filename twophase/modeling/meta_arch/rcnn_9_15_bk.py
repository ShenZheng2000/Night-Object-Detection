
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
from detectron2.config import configurable
import logging
from typing import Dict, Tuple, List, Optional
from collections import OrderedDict
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.modeling.backbone import build_backbone, Backbone
from detectron2.modeling.roi_heads import build_roi_heads
from detectron2.utils.events import get_event_storage
from detectron2.structures import ImageList

from twophase.data.transforms.grid_generator import CuboidGlobalKDEGrid, FixedKDEGrid

# import sys
# sys.path.append('/root/autodl-tmp/Methods/Night-Object-Detection')

from twophase.data.transforms.fovea import process_and_update_features

import torchvision.utils as vutils
import matplotlib.pyplot as plt
import time
#######################

import json
import os

# def build_grid_net(cfg, opt=''):
#     # TODO: hardcode for now, write with cfg later
#     if opt == 'train':
#         my_shape = (600, 1067)
#     elif opt == 'test':
#         my_shape = (750, 1333)
    
#     # Based on the configuration, decide which type of grid_net to create.
#     if cfg.WARP_AUG or cfg.WARP_AUG_LZU:
#         grid_net = CuboidGlobalKDEGrid(separable=True, 
#                                        anti_crop=True, 
#                                        input_shape=my_shape, 
#                                        output_shape=my_shape)
#     elif cfg.WARP_FOVEA:
#         saliency_file = 'dataset_saliency.pkl'
#         grid_net = FixedKDEGrid(saliency_file,
#                                 separable=True, 
#                                 anti_crop=True, 
#                                 input_shape=my_shape, 
#                                 output_shape=my_shape)
#     else:
#         grid_net = None
        
#     return grid_net


def build_vp_dict(cfg):
    # If the VANISHING_POINT is set in the configuration, load it.
    if cfg.VANISHING_POINT:
        with open(cfg.VANISHING_POINT, 'r') as f:
            vp_dict = json.load(f)
        vp_dict = {os.path.basename(k): v for k, v in vp_dict.items()}
    else:
        vp_dict = None
        
    return vp_dict


@META_ARCH_REGISTRY.register()
class DAobjTwoStagePseudoLabGeneralizedRCNN(GeneralizedRCNN):

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        proposal_generator: nn.Module,
        roi_heads: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        input_format: Optional[str] = None,
        vis_period: int = 0,
        # grid_net_train: Optional[nn.Module] = None,       # Added this line
        # grid_net_test: Optional[nn.Module] = None,
        vp_dict: Optional[dict] = None,     # Added this line
        warp_test: bool = False,                     # Added this line for warp_test
        warp_debug: bool = False,
        warp_image_norm: bool = False,
        warp_aug_lzu: bool = False,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            roi_heads: a ROI head that performs per-region computation
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            input_format: describe the meaning of channels of input. Needed by visualization
            vis_period: the period to run visualization. Set to 0 to disable.
        """
        super(GeneralizedRCNN, self).__init__()
        self.backbone = backbone
        self.proposal_generator = proposal_generator
        self.roi_heads = roi_heads

        self.input_format = input_format
        self.vis_period = vis_period
        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization!"

        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"

        # self.grid_net_train = grid_net_train
        # self.grid_net_test = grid_net_test
        self.vp_dict = vp_dict
        self.warp_test = warp_test
        self.warp_debug = warp_debug
        self.warp_image_norm = warp_image_norm
        self.warp_aug_lzu = warp_aug_lzu

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        # grid_net_train = build_grid_net(cfg, 'train')               # Call the function to build grid_net_train
        # grid_net_test = build_grid_net(cfg, 'test')               # Call the function to build grid_net_test
        vp_dict = build_vp_dict(cfg)                 # Call the function to build vp_dict
        return {
            "backbone": backbone,
            "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
            "roi_heads": build_roi_heads(cfg, backbone.output_shape()),
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            # "grid_net_train": grid_net_train,                      # Include it in the returned dict
            # "grid_net_test": grid_net_test,
            "vp_dict": vp_dict,                        # Include it in the returned dict
            "warp_test": cfg.WARP_TEST,                 # Fetch the value from config and include it
            "warp_debug": cfg.WARP_AUG,
            "warp_image_norm": cfg.WARP_IMAGE_NORM,
            "warp_aug_lzu": cfg.WARP_AUG_LZU,
        }

    def preprocess_image_train(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)

        images_t = [x["image_unlabeled"].to(self.device) for x in batched_inputs]
        images_t = [(x - self.pixel_mean) / self.pixel_std for x in images_t]
        images_t = ImageList.from_tensors(images_t, self.backbone.size_divisibility)

        return images, images_t

    def inference(
        self,
        batched_inputs: List[Dict[str, torch.Tensor]],
        detected_instances = None,
        do_postprocess: bool = True,

    ):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            When do_postprocess=True, same as in :meth:`forward`.
            Otherwise, a list[Instances] containing raw network outputs.
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)

        # # NOTE: add zoom-unzoom here (NOT working for now)
        # if self.warp_test:
        #     # print("Hello! Running model inference with warpping!")
        #     features = process_and_update_features(batched_inputs, 
        #                                             images, 
        #                                             self.vp_dict, 
        #                                             self.grid_net,
        #                                             self.backbone)
        # else:
        #     features = self.backbone(images.tensor)
        features = self.backbone(images.tensor)

        if detected_instances is None:
            if self.proposal_generator is not None:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            results, _ = self.roi_heads(images, features, proposals, None)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            return GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)
        return results


    def forward(
        self, batched_inputs, branch="supervised", given_proposals=None, val_mode=False, proposal_index = None,
        grid_net = None,
    ):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        if (not self.training) and (not val_mode):  # only conduct when testing mode
            # NOTE: seems like not using warp-unwarp during inference time, which still make sense
            return self.inference(batched_inputs = batched_inputs)


        images = self.preprocess_image(batched_inputs)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        # print(f"batched_inputs len is {len(batched_inputs)}, [0].keys() is {batched_inputs[0].keys()}") 
            # 12, dict_keys(['file_name', 'height', 'width', 'image_id', 'transform', 'instances', 'image'])
        # print("images.tensor shape", images.tensor.shape) # [10, 3, 600, 1067]
        # print("images is", images) # <detectron2.structures.image_list.ImageList object at 0x7fc5ab256fd0>
        # .to(self.device)

        # NOTE: add zoom-unzoom here
        if self.warp_aug_lzu:
            # print("warppping image!!!!!!!!!")
            # print("grid_net is", grid_net)
            features = process_and_update_features(batched_inputs, 
                                                    images,
                                                    self.warp_aug_lzu,
                                                   self.vp_dict, 
                                                   grid_net, 
                                                   self.backbone, 
                                                   self.warp_debug, 
                                                   self.warp_image_norm)
        else:
            features = self.backbone(images.tensor)
        # first_key = next(iter(features))
        # first_value = features[first_key]
        # print(f"len {len(features)} all_keys {features.keys()} first_key {first_key}, first_value {first_value.shape}")
            # 1, dict_keys(['res5']), res5, torch.Size([BS, 2048, 38, 67])

        # TODO: remove the usage of if else here. This needs to be re-organized
        if branch == "supervised":
            
            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances
            )

            # roi_head lower branch
            _, detector_losses = self.roi_heads(
                images,
                features,
                proposals_rpn,
                compute_loss=True,
                targets=gt_instances,
                branch=branch,
            )

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses, [], [], None

        elif branch == "unsup_data_weak":
            """
            unsupervised weak branch: input image without any ground-truth label; output proposals of rpn and roi-head
            """
            # Region proposal network
            proposals_rpn, _ = self.proposal_generator(
                images, features, None, compute_loss=False
            )

            # roi_head lower branch (keep this for further production)
            # notice that we do not use any target in ROI head to do inference!
            proposals_roih, ROI_predictions = self.roi_heads(
                images,
                features,
                proposals_rpn,
                targets=None,
                compute_loss=False,
                branch=branch,
            )

            return {}, proposals_rpn, proposals_roih, ROI_predictions

        elif branch == "consistency_target":
            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances
            )

            # roi_head lower branch
            proposals_roih, proposals_into_roih, proposal_index = self.roi_heads(
                images,
                features,
                proposals_rpn,
                targets=gt_instances,
                compute_loss=False,
                branch=branch,
            )

            return proposal_losses,proposals_into_roih, proposals_rpn, proposals_roih, proposal_index

        elif branch == "unsup_data_consistency":
            """
            unsupervised weak branch: input image without any ground-truth label; output proposals of rpn and roi-head
            """
            # roi_head lower branch (keep this for further production)
            # notice that we do not use any target in ROI head to do inference!
            proposals_roih, ROI_predictions = self.roi_heads(
                images,
                features,
                given_proposals,
                targets=None,
                compute_loss=False,
                branch=branch,
                proposal_index=proposal_index,
            )

            return {}, [], proposals_roih, ROI_predictions
        elif branch == "val_loss":
            raise NotImplementedError()

    def visualize_training(self, batched_inputs, proposals, branch=""):
        """
        This function different from the original one:
        - it adds "branch" to the `vis_name`.

        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 predicted object
        proposals on the original image. Users can implement different
        visualization functions for different models.

        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        from detectron2.utils.visualizer import Visualizer

        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = (
                "Left: GT bounding boxes "
                + branch
                + ";  Right: Predicted proposals "
                + branch
            )
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch



@META_ARCH_REGISTRY.register()
class TwoStagePseudoLabGeneralizedRCNN(GeneralizedRCNN):
    def forward(
        self, batched_inputs, branch="supervised", given_proposals=None, val_mode=False
    ):
        if (not self.training) and (not val_mode):
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)

        if branch == "supervised":
            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances
            )

            # # roi_head lower branch
            _, detector_losses = self.roi_heads(
                images, features, proposals_rpn, gt_instances, branch=branch
            )

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses, [], [], None

        elif branch == "unsup_data_weak":
            # Region proposal network
            proposals_rpn, _ = self.proposal_generator(
                images, features, None, compute_loss=False
            )

            # roi_head lower branch (keep this for further production)  # notice that we do not use any target in ROI head to do inference !
            proposals_roih, ROI_predictions = self.roi_heads(
                images,
                features,
                proposals_rpn,
                targets=None,
                compute_loss=False,
                branch=branch,
            )

            return {}, proposals_rpn, proposals_roih, ROI_predictions

        elif branch == "val_loss":

            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances, compute_val_loss=True
            )

            # roi_head lower branch
            _, detector_losses = self.roi_heads(
                images,
                features,
                proposals_rpn,
                gt_instances,
                branch=branch,
                compute_val_loss=True,
            )

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses, [], [], None


