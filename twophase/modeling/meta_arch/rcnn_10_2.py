
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

# import sys
# sys.path.append('/root/autodl-tmp/Methods/Night-Object-Detection')

from twophase.data.transforms.fovea import process_and_update_features

import torchvision.utils as vutils
import matplotlib.pyplot as plt
import time

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

        # TODO: define grid_net here (instead of in train.py)

        self.AT = AT
        self.dis_type = dis_type
        self.D_img = None
    def build_discriminator(self):
        self.D_img = FCDiscriminator_img(self.backbone._out_feature_channels[self.dis_type]).to(self.device) # Need to know the channel

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
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
        warp_aug_lzu = False,
        vp_dict = None,
        grid_net = None,

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

        # NOTE: add zoom-unzoom here
        # if warp_aug_lzu:
        #     print("Hello! Running model inference with warp_aug_lzu!")
        #     features = process_and_update_features(batched_inputs, images, warp_aug_lzu, 
        #                                             vp_dict, grid_net, self.backbone)
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

    def _process_images(self, images, batched_inputs, label, warp_aug_lzu, vp_dict, grid_net, warp_debug, warp_image_norm):
        if warp_aug_lzu:
            features = process_and_update_features(batched_inputs, images, warp_aug_lzu, 
                                                vp_dict, grid_net, self.backbone, 
                                                warp_debug, warp_image_norm)
        else:
            features = self.backbone(images.tensor)

        features = grad_reverse(features[self.dis_type])
        D_img_out = self.D_img(features)
        loss_D_img = F.binary_cross_entropy_with_logits(D_img_out, torch.FloatTensor(D_img_out.data.size()).fill_(label).to(self.device))
        return loss_D_img

    # incorporate AT training in this code (DONE)
    def forward(
        self, batched_inputs, branch="supervised", given_proposals=None, val_mode=False, proposal_index = None,
        warp_aug_lzu=False, vp_dict=None, grid_net=None, warp_debug=False, warp_image_norm=False
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
            # NOTE: seems like not using warp-unwarp during inference time, which still make sense
            return self.inference(batched_inputs = batched_inputs,
                                    warp_aug_lzu = warp_aug_lzu,
                                    vp_dict = vp_dict,
                                    grid_net = grid_net)

        source_label = 0
        target_label = 1

        if branch == "domain":

            # Assuming source_label and target_label are defined somewhere or passed as arguments
            images_s, images_t = self.preprocess_image_train(batched_inputs)

            loss_D_img_s = self._process_images(images_s, batched_inputs, source_label, warp_aug_lzu, vp_dict, grid_net, warp_debug, warp_image_norm)
            loss_D_img_t = self._process_images(images_t, batched_inputs, target_label, warp_aug_lzu, vp_dict, grid_net, warp_debug, warp_image_norm)

            losses = {}
            losses["loss_D_img_s"] = loss_D_img_s
            losses["loss_D_img_t"] = loss_D_img_t
            return losses, [], [], None

        images = self.preprocess_image(batched_inputs)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        # print(f"batched_inputs len is {len(batched_inputs)}, [0].keys() is {batched_inputs[0].keys()}") 
            # 12, dict_keys(['file_name', 'height', 'width', 'image_id', 'transform', 'instances', 'image'])
        # print("images.tensor shape", images.tensor.shape) # [10, 3, 600, 1067]
        # print("batched_inputs[0]['instances'].gt_boxes.tensor.shape is", \
        #       batched_inputs[0]['instances'].gt_boxes.tensor.shape) # [N, 4]: x1, y1, x2, y2
        # .to(self.device)

        # NOTE: add zoom-unzoom here
        if warp_aug_lzu:
            features = process_and_update_features(batched_inputs, images, warp_aug_lzu, 
                                                   vp_dict, grid_net, self.backbone, 
                                                   warp_debug, warp_image_norm)
        else:
            features = self.backbone(images.tensor)
        # first_key = next(iter(features))
        # first_value = features[first_key]
        # print(f"len {len(features)} all_keys {features.keys()} first_key {first_key}, first_value {first_value.shape}")
            # 1, dict_keys(['res5']), res5, torch.Size([BS, 2048, 38, 67])

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

# @META_ARCH_REGISTRY.register()
# class TwoStagePseudoLabGeneralizedRCNN(GeneralizedRCNN):
#     def forward(
#         self, batched_inputs, branch="supervised", given_proposals=None, val_mode=False
#     ):
#         if (not self.training) and (not val_mode):
#             return self.inference(batched_inputs)

#         images = self.preprocess_image(batched_inputs)

#         if "instances" in batched_inputs[0]:
#             gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
#         else:
#             gt_instances = None

#         features = self.backbone(images.tensor)

#         if branch == "supervised":
#             # Region proposal network
#             proposals_rpn, proposal_losses = self.proposal_generator(
#                 images, features, gt_instances
#             )

#             # # roi_head lower branch
#             _, detector_losses = self.roi_heads(
#                 images, features, proposals_rpn, gt_instances, branch=branch
#             )

#             losses = {}
#             losses.update(detector_losses)
#             losses.update(proposal_losses)
#             return losses, [], [], None

#         elif branch == "unsup_data_weak":
#             # Region proposal network
#             proposals_rpn, _ = self.proposal_generator(
#                 images, features, None, compute_loss=False
#             )

#             # roi_head lower branch (keep this for further production)  # notice that we do not use any target in ROI head to do inference !
#             proposals_roih, ROI_predictions = self.roi_heads(
#                 images,
#                 features,
#                 proposals_rpn,
#                 targets=None,
#                 compute_loss=False,
#                 branch=branch,
#             )

#             return {}, proposals_rpn, proposals_roih, ROI_predictions

#         elif branch == "val_loss":

#             # Region proposal network
#             proposals_rpn, proposal_losses = self.proposal_generator(
#                 images, features, gt_instances, compute_val_loss=True
#             )

#             # roi_head lower branch
#             _, detector_losses = self.roi_heads(
#                 images,
#                 features,
#                 proposals_rpn,
#                 gt_instances,
#                 branch=branch,
#                 compute_val_loss=True,
#             )

#             losses = {}
#             losses.update(detector_losses)
#             losses.update(proposal_losses)
#             return losses, [], [], None


