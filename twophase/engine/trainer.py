import os
import time
import logging
import json
import torch
from torch.nn.parallel import DistributedDataParallel
from fvcore.nn.precise_bn import get_bn_modules
import numpy as np
from collections import OrderedDict
import torchvision.transforms as T
import sys

import torchvision
from torchvision.utils import save_image
import time
import copy

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import DefaultTrainer, SimpleTrainer, TrainerBase
from detectron2.engine.train_loop import AMPTrainer
from detectron2.utils.events import EventStorage
from detectron2.evaluation import verify_results, DatasetEvaluators
# from detectron2.evaluation import COCOEvaluator, verify_results, DatasetEvaluators

from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.engine import hooks
from detectron2.structures.boxes import Boxes
from detectron2.structures.instances import Instances
from detectron2.utils.env import TORCH_VERSION
from detectron2.data import MetadataCatalog

from twophase.data.build import (
    build_detection_semisup_train_loader,
    build_detection_test_loader,
    build_detection_semisup_train_loader_two_crops,
)
from twophase.data.dataset_mapper import DatasetMapperTwoCropSeparate
from twophase.engine.hooks import LossEvalHook
from twophase.modeling.meta_arch.ts_ensemble import EnsembleTSModel
from twophase.checkpoint.detection_checkpoint import DetectionTSCheckpointer
from twophase.solver.build import build_lr_scheduler
from twophase.evaluation import PascalVOCDetectionEvaluator, COCOEvaluator
from twophase.modeling.custom_losses import ConsistencyLosses
from twophase.data.transforms.night_aug import NightAug
import copy

from twophase.modeling.masking import Masking

# Adaptive Teacher Trainer
class TwoPCTrainer(DefaultTrainer):
    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        Use the custom checkpointer, which loads other backbone models
        with matching heuristics.
        """
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())
        data_loader = self.build_train_loader(cfg)

        # create an student model
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)

        # create an teacher model
        model_teacher = self.build_model(cfg)
        self.model_teacher = model_teacher

        # For training, wrap with DDP. But don't need this for inference.
        if comm.get_world_size() > 1:
            model = DistributedDataParallel(
                model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
            )

        TrainerBase.__init__(self)
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, data_loader, optimizer
        )
        self.scheduler = self.build_lr_scheduler(cfg, optimizer)


        ensem_ts_model = EnsembleTSModel(model_teacher, model)

        self.checkpointer = DetectionTSCheckpointer(
            ensem_ts_model,
            cfg.OUTPUT_DIR,
            optimizer=optimizer,
            scheduler=self.scheduler,
        )
        self.start_iter = 0
        self.stu_scale = None
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.scale_list = np.array(cfg.SEMISUPNET.SCALE_LIST)
        self.scale_checkpoints = np.array(cfg.SEMISUPNET.SCALE_STEPS)
        self.cfg = cfg
        self.ext_data = []
        self.img_vals = {}
        self.consistency_losses = ConsistencyLosses()
        self.night_aug = NightAug()

        self.register_hooks(self.build_hooks()) 

        self.masking = Masking(
            block_size=cfg.MASKING_BLOCK_SIZE,
            ratio=cfg.MASKING_RATIO
            )        

    def resume_or_load(self, resume=True):
        """
        If `resume==True` and `cfg.OUTPUT_DIR` contains the last checkpoint (defined by
        a `last_checkpoint` file), resume from the file. Resuming means loading all
        available states (eg. optimizer and scheduler) and update iteration counter
        from the checkpoint. ``cfg.MODEL.WEIGHTS`` will not be used.
        Otherwise, this is considered as an independent training. The method will load model
        weights from the file `cfg.MODEL.WEIGHTS` (but will not load other states) and start
        from iteration 0.
        Args:
            resume (bool): whether to do resume or not
        """
        checkpoint = self.checkpointer.resume_or_load(
            self.cfg.MODEL.WEIGHTS, resume=resume
        )
        if resume: # and self.checkpointer.has_checkpoint():
            self.start_iter = checkpoint.get("iteration", -1) + 1
            # The checkpoint stores the training iteration that just finished, thus we start
            # at the next iteration (or iter zero if there's no checkpoint).
        if isinstance(self.model, DistributedDataParallel):
            # broadcast loaded data/model from the first rank, because other
            # machines may not have access to the checkpoint file
            if TORCH_VERSION >= (1, 7):
                self.model._sync_params_and_buffers()
            self.start_iter = comm.all_gather(self.start_iter)[0]

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

        if evaluator_type == "coco":
            evaluator_list.append(COCOEvaluator(
                dataset_name, output_dir=output_folder))
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        elif evaluator_type == "pascal_voc_water":
            return PascalVOCDetectionEvaluator(dataset_name, target_classnames=["bicycle", "bird", "car", "cat", "dog", "person"])
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]

        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DatasetMapperTwoCropSeparate(cfg, True)
        return build_detection_semisup_train_loader_two_crops(cfg, mapper)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        return build_lr_scheduler(cfg, optimizer)

    def train(self):
        self.train_loop(self.start_iter, self.max_iter)
        if hasattr(self, "_last_eval_results") and comm.is_main_process():
            verify_results(self.cfg, self._last_eval_results)
            return self._last_eval_results

    def before_train_json(self):
        if self.cfg.VANISHING_POINT is not None:
            with open(self.cfg.VANISHING_POINT, 'r') as f:
                self.vanishing_point = json.load(f)
            self.vanishing_point = {os.path.basename(k): v for k, v in self.vanishing_point.items()}
        else:
            self.vanishing_point = None

    def train_loop(self, start_iter: int, max_iter: int):
        logger = logging.getLogger(__name__)
        logger.info("Starting training from iteration {}".format(start_iter))

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter

        with EventStorage(start_iter) as self.storage:
            try:
                self.before_train()
                self.before_train_json()

                for self.iter in range(start_iter, max_iter):
                    self.before_step()
                    self.run_step_full_semisup()
                    self.after_step()
            except Exception:
                logger.exception("Exception during training:")
                raise
            finally:
                self.after_train()

    # =====================================================
    # ================== Pseduo-labeling ==================
    # =====================================================
    def threshold_bbox(self, proposal_bbox_inst, thres=0.7, proposal_type="roih"):
        if proposal_type == "rpn":
            # print("proposal_bbox_inst is", proposal_bbox_inst)
            # print("proposal_bbox_inst.proposal_boxes.tensor is", proposal_bbox_inst.proposal_boxes.tensor)
            # print("proposal_bbox_inst.proposal_boxes.tensor.shape is", proposal_bbox_inst.proposal_boxes.tensor.shape)
            # print("proposal_bbox_inst.objectness_logits is", len(proposal_bbox_inst.objectness_logits)) # 2000
            valid_map = proposal_bbox_inst.objectness_logits > thres
            # print("valid_map True is", torch.sum(valid_map))

            # create instances containing boxes and gt_classes
            image_shape = proposal_bbox_inst.image_size
            new_proposal_inst = Instances(image_shape)

            # create box
            new_bbox_loc = proposal_bbox_inst.proposal_boxes.tensor[valid_map, :]
            new_boxes = Boxes(new_bbox_loc)

            # add boxes to instances
            new_proposal_inst.gt_boxes = new_boxes
            new_proposal_inst.objectness_logits = proposal_bbox_inst.objectness_logits[
                valid_map
            ]
        elif proposal_type == "roih":
            # print("proposal_bbox_inst.scores is", len(proposal_bbox_inst.scores)) # 100
            valid_map = proposal_bbox_inst.scores > thres
            # print("valid_map True is", torch.sum(valid_map))

            # create instances containing boxes and gt_classes
            image_shape = proposal_bbox_inst.image_size
            new_proposal_inst = Instances(image_shape)

            # create box
            new_bbox_loc = proposal_bbox_inst.pred_boxes.tensor[valid_map, :]
            new_boxes = Boxes(new_bbox_loc)

            # add boxes to instances
            new_proposal_inst.gt_boxes = new_boxes
            new_proposal_inst.gt_classes = proposal_bbox_inst.pred_classes[valid_map]
            new_proposal_inst.scores = proposal_bbox_inst.scores[valid_map]

        return new_proposal_inst

    def process_pseudo_label(
        self, proposals_rpn_unsup_k, cur_threshold, proposal_type, psedo_label_method=""
    ):
        list_instances = []
        num_proposal_output = 0.0
        for proposal_bbox_inst in proposals_rpn_unsup_k: # proposals_rpn_unsup_k len: batch size (per gpu)
            # thresholding
            if psedo_label_method == "thresholding":
                proposal_bbox_inst = self.threshold_bbox(
                    proposal_bbox_inst, thres=cur_threshold, proposal_type=proposal_type
                )
            else:
                raise ValueError("Unkown pseudo label boxes methods")
            num_proposal_output += len(proposal_bbox_inst)
            list_instances.append(proposal_bbox_inst)
        num_proposal_output = num_proposal_output / len(proposals_rpn_unsup_k)
        return list_instances, num_proposal_output


    def process_pseudo_label_depth(
        self, proposals_rpn_unsup_k, cur_threshold, proposal_type, psedo_label_method="", depth_images=None,
    ):
        """
        This function processes pseudo labels by incorporating depth information. 
        It is used for object detection proposals in unsupervised learning.
        Args:
            proposals_rpn_unsup_k: List of proposal instances
            cur_threshold: Current threshold value for bounding box selection
            proposal_type: Type of proposal ('rpn' or 'roih')
            psedo_label_method: Method to create pseudo labels (currently only "thresholding" is supported)
            depth_images: List of depth images corresponding to the proposals
        Returns:
            list_instances: List of processed proposal instances
            num_proposal_output: Average number of proposals per image
        """
        list_instances = []
        num_proposal_output = 0.0
        for proposal_bbox_inst, depth_image_dict in zip(proposals_rpn_unsup_k, depth_images):
            # get the image tensor from the depth image dictionary
            depth_image = depth_image_dict['image'] # [3, 600, 1067]

            # reweighting logic goes here before thresholding
            if proposal_type == 'rpn':
                proposal_boxes = proposal_bbox_inst.proposal_boxes.tensor
            elif proposal_type == 'roih':
                proposal_boxes = proposal_bbox_inst.pred_boxes.tensor
            
            bboxes = proposal_boxes.long() # [2000, 4]
            x1, y1, x2, y2 = bboxes.T

            # ensure bbox and depth_image are on the same device
            device = bboxes.get_device()
            depth_image = depth_image.to(device)

            # ensure bbox coordinates are within image boundary
            x1 = torch.clamp(x1, 0, depth_image.shape[-1] - 1) # [2000,]
            y1 = torch.clamp(y1, 0, depth_image.shape[-2] - 1) # [2000,]
            x2 = torch.clamp(x2, 0, depth_image.shape[-1] - 1) # [2000,]
            y2 = torch.clamp(y2, 0, depth_image.shape[-2] - 1) # [2000,]

            # Initialize a list to hold the statistics for each bounding box
            stats = []

            # Loop over each bounding box
            for i in range(bboxes.shape[0]):
                # Extract the section of the depth image within the bounding box
                values = depth_image[:, y1[i]:y2[i], x1[i]:x2[i]]
                
                # Normalize the values to [0, 1], scale and shift to [0.9, 1.1]
                # and compute the mean
                stat = 0.9 + 0.2 * values.float().mean() / 255.0
                
                # Append the statistic to the list
                stats.append(stat)

            # Stack the list of statistics into a single tensor (NOTE: empty stack => skip!)
            if stats:
                stats = torch.stack(stats)

                # reweight the confidence score
                if proposal_type == 'rpn':
                    proposal_bbox_inst.objectness_logits *= stats
                elif proposal_type == 'roih':
                    proposal_bbox_inst.scores *= stats

            # thresholding
            if psedo_label_method == "thresholding":
                proposal_bbox_inst = self.threshold_bbox(
                    proposal_bbox_inst, thres=cur_threshold, proposal_type=proposal_type
                )
            else:
                raise ValueError("Unknown pseudo label boxes methods")
            
            num_proposal_output += len(proposal_bbox_inst)
            list_instances.append(proposal_bbox_inst)
        num_proposal_output = num_proposal_output / len(proposals_rpn_unsup_k)
        return list_instances, num_proposal_output


    def remove_label(self, label_data):
        for label_datum in label_data:
            if "instances" in label_datum.keys():
                del label_datum["instances"]
        return label_data

    def add_label(self, unlabled_data, label):
        for unlabel_datum, lab_inst in zip(unlabled_data, label):
            unlabel_datum["instances"] = lab_inst
        return unlabled_data
    
    def get_label(self, label_data):
        label_list = []
        for label_datum in label_data:
            if "instances" in label_datum.keys():
                label_list.append(copy.deepcopy(label_datum["instances"]))
        
        return label_list

    def process_data(self, unlabel_data, unlabel_dep_data=None, record_dict=None, prefix=None):

        #  3. Generate the easy pseudo-label using teacher model (Phase-1)
        with torch.no_grad():
            (
                _,
                proposals_rpn_unsup,
                proposals_roih_unsup,
                _,
            ) = self.model_teacher(unlabel_data, branch="unsup_data_weak")
            
        cur_threshold = self.cfg.SEMISUPNET.BBOX_THRESHOLD
        # make a RPN_thres and a ROI_thres

        joint_proposal_dict = {}
        joint_proposal_dict["proposals_rpn"] = proposals_rpn_unsup

        #Process pseudo labels and thresholding
        pseudo_label_function = self.process_pseudo_label_depth if self.cfg.USE_DEPTH else self.process_pseudo_label
        extra_arg = [unlabel_dep_data] if self.cfg.USE_DEPTH else []

        (
            pesudo_proposals_rpn_unsup,
            nun_pseudo_bbox_rpn,
        ) = pseudo_label_function(
            proposals_rpn_unsup, cur_threshold, "rpn", "thresholding", *extra_arg
        )
        joint_proposal_dict["proposals_pseudo_rpn"] = pesudo_proposals_rpn_unsup

        # Pseudo_labeling for ROI head (bbox location/objectness)
        pesudo_proposals_roih_unsup, _ = pseudo_label_function(
            proposals_roih_unsup, cur_threshold, "roih", "thresholding", *extra_arg
        )

        joint_proposal_dict["proposals_pseudo_roih"] = pesudo_proposals_roih_unsup

        # 5. Add pseudo-label to unlabeled data
        unlabel_data = self.add_label(
            unlabel_data, joint_proposal_dict["proposals_pseudo_roih"]
        )

        #6. Scale student inputs (pseudo-labels and image)
        if self.cfg.STUDENT_SCALE:
            scale_mask=np.where(self.iter<self.scale_checkpoints)[0]
            if len(scale_mask)>0:
                nstu_scale = self.scale_list[scale_mask[0]]
            else:
                nstu_scale = 1.0
            self.stu_scale = np.random.normal(nstu_scale,0.15)
            if self.stu_scale < 0.4:
                self.stu_scale = 0.4
            elif self.stu_scale > 1.0:
                self.stu_scale = 1.0
            scaled_unlabel_data = [x.copy() for x in unlabel_data]
            img_s = scaled_unlabel_data[0]['image'].shape[1:]
            self.scale_t = T.Resize((int(img_s[0]*self.stu_scale), int(img_s[1]*self.stu_scale)))
            for item in scaled_unlabel_data:
                item['image'] = item['image'].cuda()
                item['image']=self.scale_t(item['image'])
                item['instances'].gt_boxes.scale(self.stu_scale,self.stu_scale)
                if nstu_scale < 1.0:
                    gt_mask = item['instances'].gt_boxes.area()>16 #16*16
                else:
                    gt_mask = item['instances'].gt_boxes.area()>16 #8*8
                gt_boxes = item['instances'].gt_boxes[gt_mask]
                gt_classes = item['instances'].gt_classes[gt_mask]
                scores = item['instances'].scores[gt_mask]
                item['instances'] = Instances(item['image'].shape[1:],gt_boxes=gt_boxes, gt_classes = gt_classes, scores=scores)
            
        else:
            # if student scaling is not used
            scaled_unlabel_data = [x.copy() for x in unlabel_data] 

        #7. Input scaled inputs into student
        (pseudo_losses, 
        proposals_into_roih, 
        rpn_stu,
        roi_stu,
        pred_idx)= self.model(
            scaled_unlabel_data, branch="consistency_target"
        )
        new_pseudo_losses = {}
        for key in pseudo_losses.keys():
            new_pseudo_losses[prefix + key + "_pseudo"] = pseudo_losses[
                key
            ]
        record_dict.update(new_pseudo_losses)

        #8. Upscale student RPN proposals for teacher
        if self.cfg.STUDENT_SCALE:
                stu_resized_proposals = []
                for k,proposals in enumerate(proposals_into_roih):
                    stu_resized_proposals.append(Instances(scaled_unlabel_data[0]['image'].shape[1:],
                                            proposal_boxes = proposals.proposal_boxes.clone(),
                                            objectness_logits = proposals.objectness_logits,
                                            gt_classes = proposals.gt_classes,
                                            gt_boxes = proposals.gt_boxes))
                    stu_resized_proposals[k].proposal_boxes.scale(1/self.stu_scale,1/self.stu_scale)
                proposals_into_roih=stu_resized_proposals
        
        #9. Generate matched pseudo-labels from teacher (Phase-2)
        with torch.no_grad():
            (_,
            _,
            roi_teach,
            _
            )= self.model_teacher(
                unlabel_data, 
                branch="unsup_data_consistency", 
                given_proposals=proposals_into_roih, 
                proposal_index=pred_idx
            )
        
        # 10. Compute consistency loss
        # NOTE: add one-stage choice (not working for now)
        if self.cfg.ONE_STAGE:
            cons_loss = self.consistency_losses.losses(roi_stu,proposals_roih_unsup,prefix=prefix)
        else:
            cons_loss = self.consistency_losses.losses(roi_stu,roi_teach,prefix=prefix)
        # print("cons_loss is", cons_loss)
        record_dict.update(cons_loss)

        return record_dict, roi_stu, roi_teach


    # =====================================================
    # =================== Training Flow ===================
    # =====================================================

    def run_step_full_semisup(self):
        self._trainer.iter = self.iter
        assert self.model.training, "[UBTeacherTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        data = next(self._trainer._data_loader_iter)
        # data_q and data_k from different augmentations (q:strong, k:weak)
        # label_strong, label_weak, unlabed_strong, unlabled_weak

        # burn-in stage (supervised training with labeled data)
        if not self.cfg.USE_TGT_DEBUG:
            BURN_UP_STEP = self.cfg.SEMISUPNET.BURN_UP_STEP
            MID_ITER = self.cfg.SOLVER.MID_ITER
        else:
            BURN_UP_STEP = 0
            MID_ITER = 1

        # NOTE: add cur learning here
        if self.cfg.DATASETS.CUR_LEARN:

            # print("self.cfg.DATASETS.CUR_LEARN is", self.cfg.DATASETS.CUR_LEARN)
            label_data, unlabel_data, unlabel_dep_data, \
               unlabel_data_mid, unlabel_data_last = data
            
            # determine stages
            mid_stage = (BURN_UP_STEP <= self.iter < MID_ITER)
            last_stage = (self.iter >= MID_ITER)
            # print(f"mid_stage = {mid_stage}, last_stage = {last_stage}")
            
            # mid stage => adapt from day (src) to dawn/dusk (tgt)
            if mid_stage:
                # label_data = label_data
                unlabel_data = unlabel_data_mid
            
            # last stage => adapt from dawn/dusk  (src) to night  (tgt)
            elif last_stage:
                # label_data = unlabel_data_mid
                unlabel_data = unlabel_data_last

        else:
            label_data, unlabel_data, unlabel_dep_data = data


        data_time = time.perf_counter() - start

        # Add NightAug images into supervised batch
        if self.cfg.NIGHTAUG:
            # print("self.cfg.KEY_POINT", self.cfg.KEY_POINT)
            label_data_aug = self.night_aug.aug([x.copy() for x in label_data], 
                                                motion_blur=self.cfg.MOTION_BLUR,
                                                motion_blur_vet=self.cfg.MOTION_BLUR_VET,
                                                motion_blur_rand=self.cfg.MOTION_BLUR_RAND,
                                                light_render=self.cfg.LIGHT_RENDER,
                                                light_high=self.cfg.LIGHT_HIGH,
                                                key_point = self.cfg.KEY_POINT,
                                                vanishing_point= self.vanishing_point,
                                                path_blur_cons=self.cfg.PATH_BLUR_CONS, 
                                                path_blur_var=self.cfg.PATH_BLUR_VAR,
                                                reflect_render=self.cfg.REFLECT_RENDER,
                                                two_pc_aug=self.cfg.TWO_PC_AUG,
                                                aug_prob=self.cfg.AUG_PROB,
                                                hot_tail=self.cfg.HOT_TAIL,
                                                path_blur_new=self.cfg.PATH_BLUR,
                                                T_z_values=self.cfg.T_z_values,
                                                zeta_values=self.cfg.zeta_values,
                                                use_debug=self.cfg.USE_DEBUG,
                                                )
            label_data.extend(label_data_aug)

            if self.cfg.USE_DEBUG:
                print("saving self.cfg.USE_DEBUG")
                save_normalized_images(label_data, label_data_aug, 'debug_image/night_aug')
                sys.exit(1)

        
        # NOTE: add masking for src images
        if self.cfg.USE_MASK_SRC:
            label_mask_data = copy.deepcopy(label_data)
            for i in range(len(label_data)):
                label_mask_data[i]['image'] = self.masking(label_data[i]['image'].to('cuda')) # speed up with GPU

            # ########## For visualization with debug purposes ##########
            if self.cfg.USE_SRC_DEBUG:
                print("saving USE_SRC_DEBUG")
                save_normalized_images(label_data, label_mask_data, 'debug_image/src_mask')
                sys.exit(1)

        if self.iter < BURN_UP_STEP:

            record_dict, _, _, _ = self.model(
                label_data, branch="supervised")

            # NOTE: skip this part since no teacher-student in source domain
            # record_dict, _, roi_stu_src_1, _ = self.model(
            #     label_data, branch="supervised")

            # record_dict, _, roi_stu_src_2, _ = self.model(
            #     label_mask_data, branch='supervised'
            # )

            # if self.cfg.MASKING_CONS_SRC:
            #     mask_stu_cons_loss = self.consistency_losses.losses(roi_stu_src_1, roi_stu_src_2, use_match=True, prefix='mask_stu_src')
            #     record_dict.update(mask_stu_cons_loss)
            
            # weight losses
            loss_dict = {}
            for key in record_dict.keys():
                if key[:4] == "loss":
                    loss_dict[key] = record_dict[key] * 1
            losses = sum(loss_dict.values())

        # Student-teacher stage
        else:
            if self.iter == self.cfg.SEMISUPNET.BURN_UP_STEP:
                # update copy the the whole model
                self._update_teacher_model(keep_rate=0.00)

            elif (
                self.iter - self.cfg.SEMISUPNET.BURN_UP_STEP
            ) % self.cfg.SEMISUPNET.TEACHER_UPDATE_ITER == 0:
                self._update_teacher_model(
                    keep_rate=self.cfg.SEMISUPNET.EMA_KEEP_RATE)

            record_dict = {}

            # 1. Input labeled data into student model
            record_all_label_data, _, _, _ = self.model(
                label_data, branch="supervised"
            )
            record_dict.update(record_all_label_data)

            #  2. Remove unlabeled data labels
            gt_unlabel = self.get_label(unlabel_data)
            unlabel_data = self.remove_label(unlabel_data)

            _ = self.get_label(unlabel_dep_data)
            unlabel_dep_data = self.remove_label(unlabel_dep_data)

            # NOTE: process unlabel data
            record_dict, roi_stu_1, roi_teach_1 = self.process_data(unlabel_data,
                                                                    unlabel_dep_data,
                                                                    record_dict,
                                                                    prefix='')

            # NOTE: 2.5. Mask and process unlabeled data
            # start_time = time.time()
            if self.cfg.USE_MASK:

                unlabel_mask_data = copy.deepcopy(unlabel_data)
                for i in range(len(unlabel_data)):

                    if self.cfg.MASKING_SPA:
                        unlabel_mask_data[i]['image'] = self.masking(unlabel_data[i]['image'].to('cuda'),
                                                                    unlabel_dep_data[i]['image'].to('cuda'))
                    else:
                        unlabel_mask_data[i]['image'] = self.masking(unlabel_data[i]['image'].to('cuda')) # speed up with GPU


                _ = self.get_label(unlabel_mask_data)
                unlabel_mask_data = self.remove_label(unlabel_mask_data)

                record_dict, roi_stu_2, roi_teach_2 = self.process_data(unlabel_mask_data,
                                                                        unlabel_dep_data,
                                                                        record_dict,
                                                                        prefix='mask')
                
                # NOTE: add mask consistency here
                if self.cfg.MASKING_CONS:
                    
                    # cal cons loss
                    mask_stu_cons_loss = self.consistency_losses.losses(roi_stu_1, 
                                                                        roi_stu_2, 
                                                                        use_match=True, 
                                                                        prefix='mask_stu',
                                                                        wei=self.cfg.MASKING_CONS_WEI)
                    mask_teach_cons_loss = self.consistency_losses.losses(roi_teach_1, 
                                                                        roi_teach_2, 
                                                                        use_match=True, 
                                                                        prefix='mask_teach',
                                                                        wei=self.cfg.MASKING_CONS_WEI)

                    # reweight cons loss
                    # print("self.cfg.MASKING_CONS_WEI is", self.cfg.MASKING_CONS_WEI)
                    # print("mask_stu_cons_loss is", mask_stu_cons_loss)
                    # print("mask_teach_cons_loss is", mask_teach_cons_loss)

                    record_dict.update(mask_stu_cons_loss)
                    record_dict.update(mask_teach_cons_loss)


                # ########## For visualization with debug purposes ##########
                if self.cfg.USE_TGT_DEBUG:
                    print("saving USE_TGT_DEBUG")
                    save_normalized_images(unlabel_data, unlabel_mask_data, 'debug_image/tgt_mask')
                    sys.exit(1)

            # weight losses
            loss_dict = {}
            # print("record_dict.keys() is", record_dict.keys())
            for key in record_dict.keys():
                if key.startswith("loss"):
                    if key == "loss_rpn_loc_pseudo": 
                        loss_dict[key] = record_dict[key] * 0
                    elif key.endswith('loss_cls_pseudo'):
                        loss_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.UNSUP_LOSS_WEIGHT
                    elif key.endswith('loss_rpn_cls_pseudo'):
                        loss_dict[key] = record_dict[key] 
                    else: 
                        loss_dict[key] = record_dict[key] * 1

            losses = sum(loss_dict.values())

        metrics_dict = record_dict
        metrics_dict["data_time"] = data_time
        if self.iter >= self.cfg.SEMISUPNET.BURN_UP_STEP and self.cfg.STUDENT_SCALE:
            metrics_dict["scale"] = self.stu_scale
        self._write_metrics(metrics_dict)

        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()



    def _write_metrics(self, metrics_dict: dict):
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }

        # gather metrics among all workers for logging
        # This assumes we do DDP-style training, which is currently the only
        # supported method in detectron2.
        all_metrics_dict = comm.gather(metrics_dict)
        # all_hg_dict = comm.gather(hg_dict)

        if comm.is_main_process():
            if "data_time" in all_metrics_dict[0]:
                # data_time among workers can have high variance. The actual latency
                # caused by data_time is the maximum among workers.
                data_time = np.max([x.pop("data_time")
                                   for x in all_metrics_dict])
                self.storage.put_scalar("data_time", data_time)

            # average the rest metrics
            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict])
                for k in all_metrics_dict[0].keys()
            }

            # append the list
            loss_dict = {}
            for key in metrics_dict.keys():
                if key[:4] == "loss":
                    loss_dict[key] = metrics_dict[key]

            total_losses_reduced = sum(loss for loss in loss_dict.values())

            self.storage.put_scalar("total_loss", total_losses_reduced)
            if len(metrics_dict) > 1:
                self.storage.put_scalars(**metrics_dict)

    @torch.no_grad()
    def _update_teacher_model(self, keep_rate=0.9996):
        if comm.get_world_size() > 1:
            student_model_dict = {
                key[7:]: value for key, value in self.model.state_dict().items()
            }
        else:
            student_model_dict = self.model.state_dict()

        new_teacher_dict = OrderedDict()
        for key, value in self.model_teacher.state_dict().items():
            if key in student_model_dict.keys():
                new_teacher_dict[key] = (
                    student_model_dict[key] *
                    (1 - keep_rate) + value * keep_rate
                )
            else:
                raise Exception("{} is not found in student model".format(key))

        self.model_teacher.load_state_dict(new_teacher_dict)

    @torch.no_grad()
    def _copy_main_model(self):
        # initialize all parameters
        if comm.get_world_size() > 1:
            rename_model_dict = {
                key[7:]: value for key, value in self.model.state_dict().items()
            }
            self.model_teacher.load_state_dict(rename_model_dict)
        else:
            self.model_teacher.load_state_dict(self.model.state_dict())

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name)

    def build_hooks(self):
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(self.optimizer, self.scheduler),
            hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                cfg.TEST.EVAL_PERIOD,
                self.model,
                # Build a new data loader to not affect training
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if comm.is_main_process():
            ret.append(
                hooks.PeriodicCheckpointer(
                    self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD
                )
            )

        def test_and_save_results_student():
            self._last_eval_results_student = self.test(self.cfg, self.model)
            _last_eval_results_student = {
                k + "_student": self._last_eval_results_student[k]
                for k in self._last_eval_results_student.keys()
            }
            return _last_eval_results_student

        def test_and_save_results_teacher():
            self._last_eval_results_teacher = self.test(
                self.cfg, self.model_teacher)
            return self._last_eval_results_teacher

        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD,
                   test_and_save_results_student))
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD,
                   test_and_save_results_teacher))

        if comm.is_main_process():
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        return ret


# # NOTE: add baseline trainer based on AT
# # Supervised-only Trainer
class BaselineTrainer(DefaultTrainer):
    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        Use the custom checkpointer, which loads other backbone models
        with matching heuristics.
        """
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg)

        if comm.get_world_size() > 1:
            model = DistributedDataParallel(
                model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
            )

        TrainerBase.__init__(self)
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, data_loader, optimizer
        )
        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        self.checkpointer = DetectionCheckpointer(
            model,
            cfg.OUTPUT_DIR,
            optimizer=optimizer,
            scheduler=self.scheduler,
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.register_hooks(self.build_hooks())

    # NOTE: add scheduler here
    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        return build_lr_scheduler(cfg, optimizer)

    def resume_or_load(self, resume=True):
        """
        If `resume==True` and `cfg.OUTPUT_DIR` contains the last checkpoint (defined by
        a `last_checkpoint` file), resume from the file. Resuming means loading all
        available states (eg. optimizer and scheduler) and update iteration counter
        from the checkpoint. ``cfg.MODEL.WEIGHTS`` will not be used.
        Otherwise, this is considered as an independent training. The method will load model
        weights from the file `cfg.MODEL.WEIGHTS` (but will not load other states) and start
        from iteration 0.
        Args:
            resume (bool): whether to do resume or not
        """
        checkpoint = self.checkpointer.resume_or_load(
            self.cfg.MODEL.WEIGHTS, resume=resume
        )
        if resume and self.checkpointer.has_checkpoint():
            self.start_iter = checkpoint.get("iteration", -1) + 1
            # The checkpoint stores the training iteration that just finished, thus we start
            # at the next iteration (or iter zero if there's no checkpoint).
        if isinstance(self.model, DistributedDataParallel):
            # broadcast loaded data/model from the first rank, because other
            # machines may not have access to the checkpoint file
            if TORCH_VERSION >= (1, 7):
                self.model._sync_params_and_buffers()
            self.start_iter = comm.all_gather(self.start_iter)[0]

    def train_loop(self, start_iter: int, max_iter: int):
        """
        Args:
            start_iter, max_iter (int): See docs above
        """
        logger = logging.getLogger(__name__)
        logger.info("Starting training from iteration {}".format(start_iter))

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter

        with EventStorage(start_iter) as self.storage:
            try:
                self.before_train()
                for self.iter in range(start_iter, max_iter):
                    self.before_step()
                    self.run_step()
                    self.after_step()
            except Exception:
                logger.exception("Exception during training:")
                raise
            finally:
                self.after_train()

    def run_step(self):
        self._trainer.iter = self.iter

        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()

        data = next(self._trainer._data_loader_iter)
        data_time = time.perf_counter() - start

        record_dict, _, _, _ = self.model(data, branch="supervised")

        num_gt_bbox = 0.0
        for element in data:
            num_gt_bbox += len(element["instances"])
        num_gt_bbox = num_gt_bbox / len(data)
        record_dict["bbox_num/gt_bboxes"] = num_gt_bbox

        loss_dict = {}
        for key in record_dict.keys():
            if key[:4] == "loss" and key[-3:] != "val":
                loss_dict[key] = record_dict[key]

        losses = sum(loss_dict.values())

        metrics_dict = record_dict
        metrics_dict["data_time"] = data_time
        self._write_metrics(metrics_dict)

        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

        if evaluator_type == "coco":
            evaluator_list.append(COCOEvaluator(
                dataset_name, output_dir=output_folder))
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        elif evaluator_type == "pascal_voc_water":
            return PascalVOCDetectionEvaluator(dataset_name, target_classnames=["bicycle", "bird", "car", "cat", "dog", "person"])
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]

        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_semisup_train_loader(cfg, mapper=None)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """
        Returns:
            iterable
        """
        return build_detection_test_loader(cfg, dataset_name)

    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.

        Returns:
            list[HookBase]:
        """
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(self.optimizer, self.scheduler),
            hooks.PreciseBN(
                cfg.TEST.EVAL_PERIOD,
                self.model,
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        if comm.is_main_process():
            ret.append(
                hooks.PeriodicCheckpointer(
                    self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD
                )
            )

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            return self._last_eval_results

        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))

        if comm.is_main_process():
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        return ret

    def _write_metrics(self, metrics_dict: dict):
        """
        Args:
            metrics_dict (dict): dict of scalar metrics
        """
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        # gather metrics among all workers for logging
        # This assumes we do DDP-style training, which is currently the only
        # supported method in detectron2.
        all_metrics_dict = comm.gather(metrics_dict)

        if comm.is_main_process():
            if "data_time" in all_metrics_dict[0]:
                data_time = np.max([x.pop("data_time")
                                   for x in all_metrics_dict])
                self.storage.put_scalar("data_time", data_time)

            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict])
                for k in all_metrics_dict[0].keys()
            }

            loss_dict = {}
            for key in metrics_dict.keys():
                if key[:4] == "loss":
                    loss_dict[key] = metrics_dict[key]

            total_losses_reduced = sum(loss for loss in loss_dict.values())

            self.storage.put_scalar("total_loss", total_losses_reduced)
            if len(metrics_dict) > 1:
                self.storage.put_scalars(**metrics_dict)


# def save_tensors_as_images(tensors):
#     for i, tensor in enumerate(tensors):
#         # Ensure the tensor is in CPU
#         tensor = tensor.cpu()

#         # Normalize the tensor to [0, 1] range
#         tensor = tensor / 255.0
        
#         save_image(tensor, f'image_{i}.png')


# def adjust_threshold(depth, default_threshold=0.9, depth_scale=255.0, min_threshold=0.7):
#     # Normalize the depth to [0, 1]
#     normalized_depth = depth / depth_scale

#     # Adjust the threshold to be lower for larger depth
#     adjusted_threshold = default_threshold - (default_threshold - min_threshold) * normalized_depth

#     return adjusted_threshold


def save_normalized_images(data, data_aug, out_dir):
    """
    This function takes in two lists of image data, normalizes and saves them.

    Inputs:
    - data, data_aug: Lists containing dictionaries with 'image' key holding the image tensors.
    - out_dir: output directory for images or image folders

    Outputs:
    - None. This function saves the images in the specified output directory.
    """
    # Ensure the input data lists are of the same length
    assert len(data) >= len(data_aug), f"length invalid! len(data) = {len(data)}, len(data_aug) = {len(data_aug)}"

    # assert (data[0]['image'].device) == (data_aug[0]['image'].device), print(f"device mismatch! \
    #                                          data.device = {data[0]['image'].device}, \
    #                                          data_aug.device = {data_aug[0]['image'].device}")

    # Initialize a counter for the number of images saved
    image_counter = 0

    for i in range(len(data)):
        # If we have already saved 6 images, break the loop
        if image_counter >= 5:
            break

        image = data[i]['image'].cuda()
        aug_image = data_aug[i]['image'].cuda()

        # Normalize the images
        image = (image - image.min()) / (image.max() - image.min())
        aug_image = (aug_image - aug_image.min()) / (aug_image.max() - aug_image.min())

        # Switch from BGR to RGB
        image = torch.flip(image, [0])
        aug_image = torch.flip(aug_image, [0])

        # Concatenate images along width dimension
        combined_image = torch.cat((image, aug_image), 2)  # assumes image size is (C, H, W)

        os.makedirs(out_dir, exist_ok=True)
        
        # Save the combined image
        save_image(combined_image, f'{out_dir}/img_{i}.png')

        # Increment the counter
        image_counter += 1