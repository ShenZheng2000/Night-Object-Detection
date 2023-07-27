import torch
import torch.nn.functional as F
from .grid_generator import CuboidGlobalKDEGrid
from .invert_grid import invert_grid
import os
from PIL import Image
import torchvision.transforms as transforms
import json
import time
from pycocotools.coco import COCO
from copy import deepcopy
    

def unwarp_bboxes(bboxes, grid, output_shape):
    """Unwarps a tensor of bboxes of shape (n, 4) or (n, 5) according to the grid \
    of shape (h, w, 2) used to warp the corresponding image and the \
    output_shape (H, W, ...)."""
    bboxes = bboxes.clone()
    # image map of unwarped (x,y) coordinates
    img = grid.permute(2, 0, 1).unsqueeze(0)

    warped_height, warped_width = grid.shape[0:2]

    xgrid = 2 * (bboxes[:, 0:4:2] / float(warped_width)) - 1
    ygrid = 2 * (bboxes[:, 1:4:2] / float(warped_height)) - 1
    grid = torch.stack((xgrid, ygrid), dim=2).unsqueeze(0)

    # warped_bboxes has shape (2, num_bboxes, 2)
    warped_bboxes = F.grid_sample(
        img, grid, align_corners=True, padding_mode="border").squeeze(0)

    bboxes[:, 0:4:2] = (warped_bboxes[0] + 1) / 2 * output_shape[1]
    bboxes[:, 1:4:2] = (warped_bboxes[1] + 1) / 2 * output_shape[0]

    return bboxes


def warp_bboxes(bboxes, grid, separable=True):
    size = grid.shape
    inverse_grid_shape = torch.Size((size[0], size[3], size[1], size[2]))
    print("bboxes is", bboxes.shape)
    inverse_grid = invert_grid(grid, inverse_grid_shape, separable) # [1, 2, 720, 1280]
    bboxes = unwarp_bboxes(bboxes, inverse_grid.squeeze(0), size[1:3]) #output_shape[1:3]=[720, 1280]
    return bboxes


def simple_test(grid_net, imgs, vanishing_point):
    """Test function without test time augmentation.
    Args:
        grid_net (CuboidGlobalKDEGrid): An instance of CuboidGlobalKDEGrid.
        imgs (list[torch.Tensor]): List of multiple images
        img_metas (list[dict]): List of image information.
    Returns:
        list[list[np.ndarray]]: BBox results of each image and classes.
            The outer list corresponds to each image. The inner list
            corresponds to each class.
    """
    if len(imgs.shape) != 4:
        imgs = imgs.unsqueeze(0)

    imgs = torch.stack(tuple(imgs), dim=0)

    grid = grid_net(imgs, vanishing_point)

    warped_imgs = F.grid_sample(imgs, grid, align_corners=True)

    return grid, warped_imgs



def make_warp_aug(img, ins, vanishing_point):
    # read image
    img = img.float()
    device = img.device
    my_shape = img.shape[1:]

    # read bboxes
    bboxes = ins.gt_boxes.tensor
    bboxes = bboxes.to(device)

    # Create an instance of CuboidGlobalKDEGrid
    grid_net = CuboidGlobalKDEGrid(separable=True, 
                                    anti_crop=True, 
                                    input_shape=my_shape, 
                                    output_shape=my_shape)
    
    # warp image
    imgs = img.unsqueeze(0) 
    grid, warped_imgs = simple_test(grid_net, imgs, vanishing_point)

    # # NOTE: hardcode clamp border values => no use for now
    # bboxes = torch.clamp(bboxes, min=1, max=1066)

    # warp bboxes
    print("bboxes is", bboxes)
    warped_bboxes = warp_bboxes(bboxes, grid, separable=True)

    # update ins
    ins.gt_boxes.tensor = warped_bboxes

    return warped_imgs, ins


# # NOTE: for one-way inference only. Do not use during training (NOTE: not working for now)
# def set_up(input_dir, vanishing_points_file):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Load vanishing points from the json file
#     with open(vanishing_points_file, 'r') as f:
#         vanishing_points = json.load(f)
    
#     # Modify vanishing_points to use basenames as keys
#     vanishing_points = {os.path.basename(k): v for k, v in vanishing_points.items()}

#     # Load one image to get the shape
#     one_filename = next(os.path.join(input_dir, f) for f in os.listdir(input_dir) 
#                         if os.path.isfile(os.path.join(input_dir, f)) and f.lower().endswith(('.png', '.jpg')))
#     one_img = Image.open(one_filename)
#     one_img_tensor = transforms.ToTensor()(one_img).float().to(device)
#     my_shape = one_img_tensor.shape[1:]
    
#     # Create an instance of CuboidGlobalKDEGrid
#     grid_net = CuboidGlobalKDEGrid(separable=True, 
#                                     anti_crop=True, 
#                                     input_shape=my_shape, 
#                                     output_shape=my_shape).to(device)
    
#     return device, vanishing_points, grid_net


# def make_warp_aug_debug(input_dir, output_dir, vanishing_points_file, coco_json_file, coco_json_out):
#     device, vanishing_points, grid_net = set_up(input_dir, vanishing_points_file)
    
#     # Load the original COCO JSON
#     coco = COCO(coco_json_file)
    
#     # Create a copy of the original COCO dataset that we will modify
#     new_coco = deepcopy(coco.dataset)

#     # Make an empty list to hold the new annotations
#     new_annotations = []

#     # Iterate over each image
#     for img_id, img_info in coco.imgs.items():
#         filename = img_info['file_name']

#         # Check if the image exists in the input directory
#         img_path = os.path.join(input_dir, filename)
#         if not os.path.isfile(img_path):
#             continue

#         # Skip this image if it doesn't have a vanishing point
#         if filename not in vanishing_points:
#             continue

#         # Load the image and convert to tensor
#         img = Image.open(img_path)
#         img_tensor = transforms.ToTensor()(img).float().to(device)

#         # Get the vanishing point for this image
#         vanishing_point = vanishing_points[filename]
#         vanishing_point = torch.tensor(vanishing_point).float().to(device)

#         # Warp image
#         imgs = img_tensor.unsqueeze(0)
#         grid, warped_imgs = simple_test(grid_net, imgs, vanishing_point)

#         # Get all annotations for this image
#         ann_ids = coco.getAnnIds(imgIds=img_id)
#         anns = coco.loadAnns(ann_ids)

#         # Extract all bounding boxes for the current image and convert to (x1, y1, x2, y2)
#         bboxes = [torch.tensor([ann['bbox'][0], ann['bbox'][1], ann['bbox'][0] + ann['bbox'][2], ann['bbox'][1] + ann['bbox'][3]]) for ann in anns]

#         # Concatenate all bounding boxes into a single tensor
#         bboxes = torch.stack(bboxes).to(device)

#         print(f"unwarped_bboxes {bboxes}")

#         # Warp all bounding boxes together
#         warped_bboxes = warp_bboxes(bboxes, grid, separable=True)

#         print(f"warped_bboxes {warped_bboxes}")

#         # Update the 'bbox' field for each annotation
#         for i, ann in enumerate(anns):
#             warped_bbox = warped_bboxes[i]
#             ann['bbox'] = warped_bbox.tolist()
#             new_annotations.append(ann)

#         # Save the warped image
#         warped_img_pil = transforms.ToPILImage()(warped_imgs.squeeze(0).cpu())
#         output_path = os.path.join(output_dir, filename)
#         warped_img_pil.save(output_path)

#     # Replace the annotations in the new COCO dataset
#     new_coco['annotations'] = new_annotations

#     # Save the new COCO dataset to a JSON file
#     with open(coco_json_out, 'w') as f:
#         json.dump(new_coco, f)

#     # TODO: visualization gt images

# if __name__ == '__main__':
#     # TODO: add image reading and saving code to warp all training images!
#     input_dir = "/home/aghosh/Projects/2PCNet/Datasets/debug_10"
#     output_dir = "/home/aghosh/Projects/2PCNet/Datasets/debug_10_warp"
#     os.makedirs(output_dir, exist_ok=True)
#     vanishing_points_file = "/home/aghosh/Projects/2PCNet/Datasets/VP/train_day.json"
#     coco_json_file = '/home/aghosh/Projects/2PCNet/Datasets/bdd100k/coco_labels/train_day.json'
#     coco_json_out = 'debug_10.json'

#     start_time = time.time()
#     make_warp_aug_debug(input_dir, output_dir, vanishing_points_file, coco_json_file, coco_json_out)
#     end_time = time.time()

#     print(f"time elapsed = {(end_time - start_time):.3f}") # about 15 minutes
