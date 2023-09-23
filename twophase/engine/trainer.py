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
from twophase.data.transforms.grid_generator import CuboidGlobalKDEGrid, FixedKDEGrid, PlainKDEGrid
import copy


# NOTE: set basline teacher => no use for now
class BaselineTrainer(DefaultTrainer):
    pass

# Base Trainer for Domain Adaptation
class DATrainer(DefaultTrainer):
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
            # TODO: try if this works
            # if TORCH_VERSION >= (1, 7):
            #     self.model._sync_params_and_buffers()
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

    def _write_metrics(self, metrics_dict: dict): # NOTE: OK, keep this
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
    def _update_teacher_model(self, keep_rate=0.9996):  # NOTE: OK, keep this
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
    def _copy_main_model(self):  # NOTE: OK, keep this
        # initialize all parameters
        if comm.get_world_size() > 1:
            rename_model_dict = {
                key[7:]: value for key, value in self.model.state_dict().items()
            }
            self.model_teacher.load_state_dict(rename_model_dict)
        else:
            self.model_teacher.load_state_dict(self.model.state_dict())

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):  # NOTE: OK, keep this
        return build_detection_test_loader(cfg, dataset_name)

    def build_hooks(self):  # NOTE: OK, keep this
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

# Adaptive Teacher Trainer
class TwoPCTrainer(DATrainer):

    def process_data_AT(self, label_data_q, label_data_k, unlabel_data_q, unlabel_data_k, record_dict):
        ######################## For probe #################################
        # import pdb; pdb. set_trace() 
        gt_unlabel_k = self.get_label(unlabel_data_k)
        # gt_unlabel_q = self.get_label_test(unlabel_data_q)
        
        #  0. remove unlabeled data labels
        unlabel_data_q = self.remove_label(unlabel_data_q)
        unlabel_data_k = self.remove_label(unlabel_data_k)

        #  1. generate the pseudo-label using teacher model
        with torch.no_grad():
            (
                _,
                proposals_rpn_unsup_k,
                proposals_roih_unsup_k,
                _,
            ) = self.model_teacher(unlabel_data_k, branch="unsup_data_weak")

        ######################## For probe #################################
        # import pdb; pdb. set_trace() 

        # probe_metrics = ['compute_fp_gtoutlier', 'compute_num_box']
        # probe_metrics = ['compute_num_box']  
        # analysis_pred, _ = self.probe.compute_num_box(gt_unlabel_k,proposals_roih_unsup_k,'pred')
        # record_dict.update(analysis_pred)
        ######################## For probe END #################################

        #  2. Pseudo-labeling
        cur_threshold = self.cfg.SEMISUPNET.BBOX_THRESHOLD

        joint_proposal_dict = {}
        joint_proposal_dict["proposals_rpn"] = proposals_rpn_unsup_k
        # Process pseudo labels and thresholding
        (
            pesudo_proposals_rpn_unsup_k,
            nun_pseudo_bbox_rpn,
        ) = self.process_pseudo_label(
            proposals_rpn_unsup_k, cur_threshold, "rpn", "thresholding"
        )
        # analysis_pred, _ = self.probe.compute_num_box(gt_unlabel_k,pesudo_proposals_rpn_unsup_k,'pred',True)
        # record_dict.update(analysis_pred)

        joint_proposal_dict["proposals_pseudo_rpn"] = pesudo_proposals_rpn_unsup_k
        # Pseudo_labeling for ROI head (bbox location/objectness)
        pesudo_proposals_roih_unsup_k, _ = self.process_pseudo_label(
            proposals_roih_unsup_k, cur_threshold, "roih", "thresholding"
        )
        joint_proposal_dict["proposals_pseudo_roih"] = pesudo_proposals_roih_unsup_k

        # 3. add pseudo-label to unlabeled data

        unlabel_data_q = self.add_label(
            unlabel_data_q, joint_proposal_dict["proposals_pseudo_roih"]
        )
        unlabel_data_k = self.add_label(
            unlabel_data_k, joint_proposal_dict["proposals_pseudo_roih"]
        )

        all_label_data = label_data_q + label_data_k
        all_unlabel_data = unlabel_data_q

        # 4. input both strongly and weakly augmented labeled data into student model
        record_all_label_data, _, _, _ = self.model(
            all_label_data, branch="supervised"
        )
        record_dict.update(record_all_label_data)

        # 5. input strongly augmented unlabeled data into model
        record_all_unlabel_data, _, _, _ = self.model(
            all_unlabel_data, branch="supervised_target"
        )
        new_record_all_unlabel_data = {}
        for key in record_all_unlabel_data.keys():
            new_record_all_unlabel_data[key + "_pseudo"] = record_all_unlabel_data[
                key
            ]
        record_dict.update(new_record_all_unlabel_data)

        # 6. input weakly labeled data (source) and weakly unlabeled data (target) to student model
        # give sign to the target data

        for i_index in range(len(unlabel_data_k)):
            # unlabel_data_item = {}
            for k, v in unlabel_data_k[i_index].items():
                # label_data_k[i_index][k + "_unlabeled"] = v
                label_data_k[i_index][k + "_unlabeled"] = v
            # unlabel_data_k[i_index] = unlabel_data_item

        all_domain_data = label_data_k
        # all_domain_data = label_data_k + unlabel_data_k
        record_all_domain_data, _, _, _ = self.model(all_domain_data, branch="domain")
        record_dict.update(record_all_domain_data)

        return record_dict

    def process_data(self, label_data, unlabel_data, record_dict, prefix=''):

        # 1. Input labeled data into the student model
        record_all_label_data, _, _, _ = self.model(
            label_data, branch="supervised", 
            # warp_aug_lzu=self.cfg.WARP_AUG_LZU,
            # vp_dict=self.vanishing_point,
            # grid_net=self.grid_net,
            # warp_debug=self.cfg.WARP_DEBUG,
            # warp_image_norm=self.cfg.WARP_IMAGE_NORM
        )
        record_dict.update(record_all_label_data)

        # 2. Remove unlabeled data labels
        gt_unlabel = self.get_label(unlabel_data)
        unlabel_data = self.remove_label(unlabel_data)

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
        pseudo_label_function = self.process_pseudo_label
        extra_arg = []

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

        # NOTE: use cons_loss only if True
        if self.cfg.CONSISTENCY:
            record_dict.update(cons_loss)

        return record_dict


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

        # TODO: add different data reading schemes for Adaptive Teacher
        if self.cfg.AT:
            # print("trainer.py using AT!!!")
            # print("Number of values in data", len(data))
            label_data_q, label_data_k, unlabel_data_q, unlabel_data_k = data # strong, weak, strong, weak
        else:
            # print("Number of values in data", len(data))
            label_data, unlabel_data = data

        data_time = time.perf_counter() - start

        # NOTE: build grid_net here
        data_to_use = label_data if 'label_data' in locals() else label_data_q
        my_shape = data_to_use[0]['image'].shape[1:]

        if self.cfg.WARP_AUG_LZU:
            if self.cfg.WARP_FOVEA:
                saliency_file = 'dataset_saliency.pkl'
                self.grid_net = FixedKDEGrid(saliency_file,
                                            separable=True, 
                                            anti_crop=True, 
                                            input_shape=my_shape, 
                                            output_shape=my_shape)
            elif self.cfg.WARP_FOVEA_INST:
                self.grid_net = PlainKDEGrid(separable=True, 
                                                anti_crop=True, 
                                                input_shape=my_shape, 
                                                output_shape=my_shape)
            else:
                self.grid_net = CuboidGlobalKDEGrid(separable=True, 
                                                anti_crop=True, 
                                                input_shape=my_shape, 
                                                output_shape=my_shape)
        else:
            self.grid_net = None
                      
        # Add NightAug images into supervised batch
        # NOTE: do not use nightaug for label_data_mid!!!!
        label_data_aug = None
        label_data_mid = None

        if self.cfg.NIGHTAUG:
            # print("self.cfg.KEY_POINT", self.cfg.KEY_POINT)
            # print("self.cfg.TWO_PC_AUG is", self.cfg.TWO_PC_AUG)
            label_data_aug = self.night_aug.aug([x.copy() for x in data_to_use], 
                                                vanishing_point= self.vanishing_point,
                                                two_pc_aug=self.cfg.TWO_PC_AUG,
                                                aug_prob=self.cfg.AUG_PROB,
                                                path_blur_new=self.cfg.PATH_BLUR,
                                                T_z_values=self.cfg.T_z_values,
                                                zeta_values=self.cfg.zeta_values,
                                                use_debug=self.cfg.USE_DEBUG,
                                                )
            data_to_use.extend(label_data_aug)

        if self.cfg.USE_DEBUG:
            print("saving self.cfg.USE_DEBUG")
            
            if label_data_aug is not None:
                print('saving night_aug images')
                save_normalized_images(data_to_use, label_data_aug, 'debug_image/night_aug')
            elif label_data_mid is not None:
                save_normalized_images(data_to_use, label_data_mid, 'debug_image/gen_night_aug')
            else:
                save_normalized_images(data_to_use, data_to_use, 'debug_image/no_night_aug')
            sys.exit(1)


        if self.iter < self.cfg.SEMISUPNET.BURN_UP_STEP:
            common_args = {
                "branch": "supervised",
                "warp_aug_lzu": self.cfg.WARP_AUG_LZU,
                "vp_dict": self.vanishing_point,
                "grid_net": self.grid_net,
                "warp_debug": self.cfg.WARP_DEBUG,
                "warp_image_norm": self.cfg.WARP_IMAGE_NORM,
            }

            if self.cfg.AT:
                # print("Before: label_data_q len", len(label_data_q))
                label_data_q.extend(label_data_k)
                # print("After: label_data_q len", len(label_data_q))
                record_dict, _, _, _ = self.model(label_data_q, **common_args)
            else:
                # print("After: label_data len", len(label_data))
                record_dict, _, _, _ = self.model(label_data, **common_args)            

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

            # NOTE: process unlabel data
            if self.cfg.AT:
                record_dict = self.process_data_AT(label_data_q, label_data_k, unlabel_data_q, unlabel_data_k, record_dict)
            else:
                record_dict = self.process_data(label_data, unlabel_data, record_dict, prefix='')

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
                    # NOTE: add AT disriminator loss
                    elif key == "loss_D_img_s" or key == "loss_D_img_t":
                        loss_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.DIS_LOSS_WEIGHT
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
    data_len = min(len(data), len(data_aug))
    # assert len(data) >= len(data_aug), f"length invalid! len(data) = {len(data)}, len(data_aug) = {len(data_aug)}"

    # assert (data[0]['image'].device) == (data_aug[0]['image'].device), print(f"device mismatch! \
    #                                          data.device = {data[0]['image'].device}, \
    #                                          data_aug.device = {data_aug[0]['image'].device}")

    # Initialize a counter for the number of images saved
    image_counter = 0

    for i in range(data_len):

        # If we have already saved 6 images, break the loop
        if image_counter >= 6:
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
        # print(f"image shape {image.shape} aug_image shape {aug_image.shape}")
        combined_image = torch.cat((image, aug_image), 2)  # assumes image size is (C, H, W)

        os.makedirs(out_dir, exist_ok=True)
        
        # Save the combined image
        save_image(combined_image, f'{out_dir}/img_{i}.png')

        # Increment the counter
        image_counter += 1