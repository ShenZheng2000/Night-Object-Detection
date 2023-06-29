import torch
from torch.nn import KLDivLoss
from detectron2.structures import pairwise_iou, Boxes
from scipy.optimize import linear_sum_assignment

# class ConsistencyLosses:
#     def __init__(self):
#         self.kldivloss = KLDivLoss(reduction="none", log_target=False)

#     def losses(self,student_roi,teacher_roi, prefix=None):
#         loss = {}
#         class_scores_student = []
#         class_scores_teacher = []
#         for s_roi, t_roi in zip (student_roi, teacher_roi):
#             # print("s_roi.pred_boxes is", s_roi.pred_boxes.tensor.shape)
#             # print("t_roi.pred_boxes is", t_roi.pred_boxes.shape)
#             # print(f"s full_scores {s_roi.full_scores.shape}, t full_scores {t_roi.full_scores.shape}")
#             class_scores_student.append(s_roi.full_scores) #[:,:-1])
#             class_scores_teacher.append(t_roi.full_scores) #[:,:-1])
#         class_scores_student=torch.cat(class_scores_student,axis=0)
#         class_scores_teacher=torch.cat(class_scores_teacher,axis=0)

#         # Weighted KL Divergence
#         weights = class_scores_teacher.max(axis=1).values
#         kl_loss = self.kldivloss(torch.log(class_scores_student),class_scores_teacher)
#         kl_loss = kl_loss.mean(axis=1)*weights
#         kl_loss = torch.mean(kl_loss)

#         if prefix is None:
#             loss['loss_cls_pseudo'] = kl_loss
#         else:
#             loss[f'{prefix}_loss_cls_pseudo'] = kl_loss

#         return loss


def convert_to_boxes(obj):
    if isinstance(obj, Boxes):
        return obj
    else:
        return Boxes(obj)


class ConsistencyLosses:
    def __init__(self):
        self.kldivloss = KLDivLoss(reduction="none", log_target=False)

    def losses(self, student_roi, teacher_roi, use_match=False, prefix=None):

        loss = {}
        class_scores_student = []
        class_scores_teacher = []

        for s_roi, t_roi in zip(student_roi, teacher_roi):

            if use_match:
                s_boxes = convert_to_boxes(s_roi.pred_boxes)
                t_boxes = convert_to_boxes(t_roi.pred_boxes)

                # Compute pairwise IoU
                ious = pairwise_iou(s_boxes, t_boxes)

                # Apply Hungarian algorithm
                cost_matrix = -ious.cpu().detach().numpy() # Convert IoU to 'cost' for Hungarian algorithm
                row_indices, col_indices = linear_sum_assignment(cost_matrix)

                s_roi = s_roi[row_indices]
                t_roi = t_roi[col_indices]

            class_scores_student.append(s_roi.full_scores)
            class_scores_teacher.append(t_roi.full_scores)

        class_scores_student = torch.cat(class_scores_student, axis=0)
        class_scores_teacher = torch.cat(class_scores_teacher, axis=0)

        # Weighted KL Divergence
        weights = class_scores_teacher.max(axis=1).values
        kl_loss = self.kldivloss(torch.log(class_scores_student), class_scores_teacher)
        kl_loss = kl_loss.mean(axis=1) * weights
        kl_loss = torch.mean(kl_loss)

        if prefix is None:
            loss['loss_cls_pseudo'] = kl_loss
        else:
            loss[f'{prefix}_loss_cls_pseudo'] = kl_loss

        return loss
