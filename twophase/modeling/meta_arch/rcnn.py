
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

from twophase.data.transforms.grid_generator import CuboidGlobalKDEGrid, FixedKDEGrid, PlainKDEGrid, MixKDEGrid, MidKDEGrid
from twophase.data.transforms.fovea import build_grid_net

# import sys
# sys.path.append('/root/autodl-tmp/Methods/Night-Object-Detection')

from twophase.data.transforms.fovea import process_and_update_features

import torchvision.utils as vutils
import matplotlib.pyplot as plt
import time
import json
import os

import matplotlib.pyplot as plt

def build_vp_dict(cfg):
    # If the VANISHING_POINT is set in the configuration, load it.
    if cfg.VANISHING_POINT:
        with open(cfg.VANISHING_POINT, 'r') as f:
            vp_dict = json.load(f)
        vp_dict = {os.path.basename(k): v for k, v in vp_dict.items()}
    else:
        vp_dict = None
        
    return vp_dict

############### Image discriminator ##############
class FCDiscriminator_img(nn.Module):
    def __init__(self, num_classes, ndf1=256, ndf2=128):
        super(FCDiscriminator_img, self).__init__()

        self.conv1 = nn.Conv2d(num_classes, ndf1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(ndf1, ndf2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(ndf2, ndf2, kernel_size=3, padding=1)
        self.classifier = nn.Conv2d(ndf2, 1, kernel_size=3, padding=1)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.classifier(x)
        return x
#################################

################ Gradient reverse function
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()

def grad_reverse(x):
    return GradReverse.apply(x)

#######################

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
        AT: bool = False,
        dis_type: str,
        # NOTE: add this for building grid net
        vp_dict: Optional[dict] = None,     # Added this line
        warp_aug: bool = False,
        warp_aug_lzu: bool = False,
        warp_fovea: bool = False,
        warp_fovea_inst: bool = False,
        warp_fovea_mix: bool = False,
        warp_middle: bool = False,
        warp_scale: float = 1.0,
        warp_fovea_inst_scale: bool = False,
        warp_fovea_inst_scale_l2: bool = False,
        fusion_method: str = "max",
        pyramid_layer: int = 2,
        warp_debug: bool = False,
        warp_image_norm: bool = False,
        warp_test: bool = False,
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

        self.AT = AT
        self.dis_type = dis_type
        self.D_img = None

        self.vp_dict = vp_dict

        self.warp_aug_lzu = warp_aug_lzu
        self.warp_aug = warp_aug
        self.warp_scale = warp_scale
        self.warp_fovea = warp_fovea
        self.warp_fovea_inst = warp_fovea_inst
        self.warp_fovea_mix = warp_fovea_mix
        self.warp_middle = warp_middle
        self.warp_debug = warp_debug
        self.warp_image_norm = warp_image_norm

        self.warp_test = warp_test

        # NOTE: define grid_net here (instead of in train.py)
        self.grid_net = build_grid_net(warp_aug_lzu=warp_aug_lzu, 
                                       warp_fovea=warp_fovea, 
                                       warp_fovea_inst=warp_fovea_inst, 
                                       warp_fovea_mix=warp_fovea_mix, 
                                       warp_middle=warp_middle, 
                                       warp_scale=warp_scale,
                                       warp_fovea_inst_scale=warp_fovea_inst_scale,
                                       warp_fovea_inst_scale_l2=warp_fovea_inst_scale_l2,
                                       fusion_method=fusion_method,
                                        pyramid_layer=pyramid_layer,
                                       )

    def build_discriminator(self):
        self.D_img = FCDiscriminator_img(self.backbone._out_feature_channels[self.dis_type]).to(self.device) # Need to know the channel

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        vp_dict = build_vp_dict(cfg)                 # Call the function to build vp_dict
        return {
            "backbone": backbone,
            "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
            "roi_heads": build_roi_heads(cfg, backbone.output_shape()),
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "AT": cfg.AT,
            "dis_type": cfg.SEMISUPNET.DIS_TYPE,
            # NOTE: add this for building grid net
            "vp_dict": vp_dict,                        # Include it in the returned dict
            "warp_aug": cfg.WARP_AUG,
            "warp_aug_lzu": cfg.WARP_AUG_LZU,
            "warp_fovea": cfg.WARP_FOVEA,
            "warp_fovea_inst": cfg.WARP_FOVEA_INST,
            "warp_fovea_mix": cfg.WARP_FOVEA_MIX,
            "warp_middle": cfg.WARP_MIDDLE,
            "warp_scale": cfg.WARP_SCALE,
            "warp_fovea_inst_scale": cfg.WARP_FOVEA_INST_SCALE,
            "warp_fovea_inst_scale_l2": cfg.WARP_FOVEA_INST_SCALE_L2,
            "fusion_method": cfg.FUSION_METHOD,
            "pyramid_layer": cfg.PYRAMID_LAYER,
            # NOTE: add below for debug
            "warp_debug": cfg.WARP_DEBUG,
            "warp_test": cfg.WARP_TEST,
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
        # print("Inference: batched_inputs", batched_inputs[0]['instances'])
        images = self.preprocess_image(batched_inputs)

        # NOTE: add zoom-unzoom here
        # if self.warp_aug_lzu:
        #     # print("Hello! Running model inference with warp_aug_lzu!")
        #     features = process_and_update_features(batched_inputs, images, self.warp_aug_lzu, 
        #                                             self.vp_dict, self.grid_net, self.backbone, warp_aug=self.warp_aug)
        # NOTE: hardcode non-warping during testing stage for now
        # print("images.tensor shape", images.tensor.shape) # [1, 3, 750, 1333]
        # print("images.tensor mean", images.tensor.mean()) # -98.3018
        # print("images.tensor min", images.tensor.min()) # -123.6750
        # print("images.tensor max", images.tensor.max()) # 151.4700

        # NOTE: add a warp_test command to manage this part of the code
        if self.warp_test and self.warp_aug_lzu and not self.warp_fovea_inst:
            print("use warping at test-time")
            features, images = process_and_update_features(batched_inputs, 
                                                            images, 
                                                            self.warp_aug_lzu, 
                                                            self.vp_dict, 
                                                            self.grid_net, 
                                                            self.backbone, 
                                                            self.warp_debug, 
                                                            self.warp_image_norm, 
                                                            self.warp_aug)
        else:
            features = self.backbone(images.tensor)
        # print("features shaoe", features['res5'].shape) # [1, 2048, 47, 84]
        # print("features mean", features['res5'].mean()) # 0.06
        # print("features min", features['res5'].min()) # 0
        # print("features max", features['res5'].max()) # 15.3375
        # exit()
        
        # # calculate the feature maps and visualize them
        # visualize_grad_cam(backbone=self.backbone, images=images.tensor)

        # features = self.backbone(images.tensor) # NOTE: this is the default code!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # print("features shape", features['res5'].shape) # [1, 2048, 47, 84]
        # print("features mean", features['res5'].mean()) # 0.0632
        # print("features min", features['res5'].min()) # 0
        # print("features max", features['res5'].max()) # 21.3976

        if self.warp_test:
            # NOTE: use them for debug vis only
            old_features = self.backbone(images.tensor)
            feat_vis_dir = os.path.join("feat_vis", self.grid_net.__class__.__name__)
            os.makedirs(feat_vis_dir, exist_ok=True)
            save_compared_feature_maps(old_features['res5'], 
                                        features['res5'], 
                                        filename=os.path.join(feat_vis_dir, os.path.basename(batched_inputs[0]['file_name'])),
                                        save_original_only=True # NOTE: need to modify this later
                                        )

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

    # NOTE: skip this part for now
    # def _process_images(self, images, batched_inputs, label, warp_aug_lzu, vp_dict, warp_debug, warp_image_norm):
    #     if warp_aug_lzu:
    #         features = process_and_update_features(batched_inputs, images, warp_aug_lzu, 
    #                                             vp_dict, self.grid_net, self.backbone, 
    #                                             warp_debug, warp_image_norm, warp_aug=self.warp_aug)
    #     else:
    #         features = self.backbone(images.tensor)

    #     features = grad_reverse(features[self.dis_type])
    #     D_img_out = self.D_img(features)
    #     loss_D_img = F.binary_cross_entropy_with_logits(D_img_out, torch.FloatTensor(D_img_out.data.size()).fill_(label).to(self.device))
    #     return loss_D_img

    # incorporate AT training in this code (DONE)
    def forward(
        self, batched_inputs, branch="supervised", given_proposals=None, val_mode=False, proposal_index = None,
        warp_aug_lzu=False, vp_dict=None, warp_debug=False, warp_image_norm=False
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
        if self.AT and self.D_img == None:
            self.build_discriminator()
        
        if (not self.training) and (not val_mode):  # only conduct when testing mode
            return self.inference(batched_inputs = batched_inputs)

        source_label = 0
        target_label = 1

        # NOTE: skip this part for now
        # if branch == "domain":

        #     # Assuming source_label and target_label are defined somewhere or passed as arguments
        #     images_s, images_t = self.preprocess_image_train(batched_inputs)

        #     loss_D_img_s = self._process_images(images_s, batched_inputs, source_label, warp_aug_lzu, vp_dict, warp_debug, warp_image_norm)
        #     loss_D_img_t = self._process_images(images_t, batched_inputs, target_label, warp_aug_lzu, vp_dict, warp_debug, warp_image_norm)

        #     losses = {}
        #     losses["loss_D_img_s"] = loss_D_img_s
        #     losses["loss_D_img_t"] = loss_D_img_t
        #     return losses, [], [], None


        images = self.preprocess_image(batched_inputs)

        # print("before batched_inputs", batched_inputs[0]["instances"])

        # print(f"batched_inputs len is {len(batched_inputs)}, [0].keys() is {batched_inputs[0].keys()}") 
            # 12, dict_keys(['file_name', 'height', 'width', 'image_id', 'transform', 'instances', 'image'])
        # print("images.tensor shape", images.tensor.shape) # [10, 3, 600, 1067]
        # print("batched_inputs[0]['instances'].gt_boxes.tensor.shape is", \
        #       batched_inputs[0]['instances'].gt_boxes.tensor.shape) # [N, 4]: x1, y1, x2, y2
        # .to(self.device)

        # print("warp_aug_lzu is", warp_aug_lzu)
        # NOTE: add zoom-unzoom here
        if warp_aug_lzu:
            # print("process_and_update_features")
            features, images = process_and_update_features(batched_inputs, images, warp_aug_lzu, 
                                                            vp_dict, self.grid_net, self.backbone, 
                                                            warp_debug, warp_image_norm, warp_aug=self.warp_aug)
        
            # NOTE: for debug only here
            # print("images.tensor shape", images.tensor.shape)
            # print("images.tensor mean", images.tensor.mean())
            # print("images.tensor min", images.tensor.min())
            # print("images.tensor max", images.tensor.max())

            # old_features = self.backbone(images.tensor)
            # # print("saving feature maps")
            # save_compared_feature_maps(old_features['res5'], 
            #                         features['res5'], 
            #                         filename=os.path.join("feat_vis", os.path.basename(batched_inputs[0]['file_name']))
            #                         )
            # print("after batched_inputs", batched_inputs[0]["instances"])

        else:
            features = self.backbone(images.tensor)
            
        # first_key = next(iter(features))
        # first_value = features[first_key]
        # print(f"len {len(features)} all_keys {features.keys()} first_key {first_key}, first_value {first_value.shape}")
            # 1, dict_keys(['res5']), res5, torch.Size([BS, 2048, 38, 67])

        # NOTE: move gt_instances to here, since batched_inputs were updated in process_and_update_features
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        # print("images.tensor shape", images.tensor.shape)
        # print("features['res5'] shape", features['res5'].shape)
        # print("gt_instances", gt_instances)

        # TODO: remove the usage of if else here. This needs to be re-organized
        if branch == "supervised":

            if self.AT:
                # print("features key is", [key for key in features.keys()])
                features_s = grad_reverse(features[self.dis_type])
                D_img_out_s = self.D_img(features_s)
                loss_D_img_s = F.binary_cross_entropy_with_logits(D_img_out_s, torch.FloatTensor(D_img_out_s.data.size()).fill_(source_label).to(self.device))

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

            if self.AT:
                losses["loss_D_img_s"] = loss_D_img_s*0.001

            return losses, [], [], None

        elif branch == "supervised_target":

            # features_t = grad_reverse(features_t[self.dis_type])
            # D_img_out_t = self.D_img(features_t)
            # loss_D_img_t = F.binary_cross_entropy_with_logits(D_img_out_t, torch.FloatTensor(D_img_out_t.data.size()).fill_(target_label).to(self.device))

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
            # losses["loss_D_img_t"] = loss_D_img_t*0.001
            # losses["loss_D_img_s"] = loss_D_img_s*0.001
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

    # def visualize_training(self, batched_inputs, proposals, branch=""):
    #     """
    #     This function different from the original one:
    #     - it adds "branch" to the `vis_name`.

    #     A function used to visualize images and proposals. It shows ground truth
    #     bounding boxes on the original image and up to 20 predicted object
    #     proposals on the original image. Users can implement different
    #     visualization functions for different models.

    #     Args:
    #         batched_inputs (list): a list that contains input to the model.
    #         proposals (list): a list that contains predicted proposals. Both
    #             batched_inputs and proposals should have the same length.
    #     """
    #     from detectron2.utils.visualizer import Visualizer

    #     storage = get_event_storage()
    #     max_vis_prop = 20

    #     for input, prop in zip(batched_inputs, proposals):
    #         img = input["image"]
    #         img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
    #         v_gt = Visualizer(img, None)
    #         v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
    #         anno_img = v_gt.get_image()
    #         box_size = min(len(prop.proposal_boxes), max_vis_prop)
    #         v_pred = Visualizer(img, None)
    #         v_pred = v_pred.overlay_instances(
    #             boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
    #         )
    #         prop_img = v_pred.get_image()
    #         vis_img = np.concatenate((anno_img, prop_img), axis=1)
    #         vis_img = vis_img.transpose(2, 0, 1)
    #         vis_name = (
    #             "Left: GT bounding boxes "
    #             + branch
    #             + ";  Right: Predicted proposals "
    #             + branch
    #         )
    #         storage.put_image(vis_name, vis_img)
    #         break  # only visualize one image in a batch


class TwoStagePseudoLabGeneralizedRCNN(GeneralizedRCNN):
    pass



def save_compared_feature_maps(original_features, processed_features, filename='feature_comparison.png', save_original_only=False):
    """
    Save a comparison between the original and processed feature maps, or just the original features.

    Args:
    - original_features: the original features as a PyTorch tensor of shape [batch, channels, height, width].
    - processed_features: the processed features as a PyTorch tensor of the same shape as original_features.
    - filename: the filename to save the image to.
    - save_original_only: if True, only save the original features; otherwise, save the comparison.
    """
    # Compute the average across the channels for the original features
    avg_original = original_features.mean(dim=1, keepdim=True)

    if save_original_only:
        # Normalize the original feature maps for better visualization
        avg_original = (avg_original - avg_original.min()) / (avg_original.max() - avg_original.min())
        # Save the original feature maps as an image
        vutils.save_image(avg_original, filename)
    else:
        # Compute the average across the channels for the processed features
        avg_processed = processed_features.mean(dim=1, keepdim=True)
        # Concatenate the average feature maps along width
        comparison = torch.cat((avg_original, avg_processed), dim=-1)
        # Normalize the concatenated feature maps for better visualization
        comparison = (comparison - comparison.min()) / (comparison.max() - comparison.min())
        # Save the concatenated feature maps as an image
        vutils.save_image(comparison, filename)

    # print(f'Saved feature map comparison image to {filename}.')


# NOTE: cannot work for now => Skip
# def visualize_grad_cam(backbone, images):
#     from pytorch_grad_cam import GradCAM
#     from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
#     from pytorch_grad_cam.utils.image import show_cam_on_image
#     from torchvision.transforms.functional import to_pil_image

#     # downscale images to half to avoid OOM
#     images = F.interpolate(images, scale_factor=0.25, mode='bilinear', align_corners=False)

#     # if not images.requires_grad:
#     #     images.requires_grad = True

#     # Set the model to evaluation mode
#     backbone.eval()

#     # Ensure target_layers is set correctly
#     target_layers = [backbone.res5[-1]]

#     # Check if images tensor is already in a batch or needs unsqueezing
#     if len(images.shape) == 3:
#         images = images.unsqueeze(0)

#     # Normalize the images tensor if your model requires normalization
#     # Adjust the mean and std to match your model's training
#     # images = normalize(images, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

#     # Create a GradCAM object
#     cam = GradCAM(model=backbone, target_layers=target_layers, use_cuda=True)

#     # Generate the class activation maps
#     grayscale_cam = cam(input_tensor=images, targets=None) # None targets for highest scoring class

#     # Extract the first image's CAM and image for visualization
#     grayscale_cam = grayscale_cam[0, :]
#     rgb_img = to_pil_image(images[0].cpu())

#     print("image.shape", rgb_img.size)

#     # Convert PIL image to numpy array for visualization
#     rgb_image = np.array(rgb_img)
#     rgb_image = rgb_image.astype(np.float32) / 255

#     # Overlay the CAM on the image
#     visualization = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)

#     # Display or save the visualization
#     plt.imshow(visualization)
#     plt.savefig("grad_cam_baseline.png")

#     exit()