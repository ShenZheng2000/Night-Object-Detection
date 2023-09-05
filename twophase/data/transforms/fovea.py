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
from .path_blur import is_out_of_bounds, get_vanising_points
from detectron2.data.transforms import ResizeTransform, HFlipTransform, NoOpTransform
import sys
import torchvision.utils as vutils
    

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
    if len(imgs.shape) == 3:
        imgs = imgs.unsqueeze(0)

    imgs = torch.stack(tuple(imgs), dim=0)
    # print("imgs shape", imgs.shape)

    grid = grid_net(imgs, vanishing_point)
    # print("grid shape", grid.shape)

    warped_imgs = F.grid_sample(imgs, grid, align_corners=True)

    return grid, warped_imgs



def make_warp_aug(img, ins, vanishing_point, grid_net, use_ins=True):

    # read image
    img = img.float()
    device = img.device
    my_shape = img.shape[-2:]
    imgs = img.unsqueeze(0) 

    if use_ins:

        # read bboxes
        bboxes = ins.gt_boxes.tensor
        bboxes = bboxes.to(device)

        # # Create an instance of CuboidGlobalKDEGrid
        # grid_net = CuboidGlobalKDEGrid(separable=True, 
        #                                 anti_crop=True, 
        #                                 input_shape=my_shape, 
        #                                 output_shape=my_shape)

        # warp image
        grid, warped_imgs = simple_test(grid_net, imgs, vanishing_point)

        # warp bboxes
        warped_bboxes = warp_bboxes(bboxes, grid, separable=True)

        # # NOTE: hardcode for debug only. Delete later
        # warped_bboxes = unwarp_bboxes(warped_bboxes, grid.squeeze(0), [600, 1067])

        # update ins
        ins.gt_boxes.tensor = warped_bboxes

        return warped_imgs, ins, grid
    
    else:
        # warp image
        grid, warped_imgs = simple_test(grid_net, imgs, vanishing_point)

        return warped_imgs, ins, grid
    


def apply_warp_aug(img, ins, vanishing_point, warp_aug=False, 
                    warp_aug_lzu=False, grid_net=None, keep_size=True):
    # print(f"img is {img.shape}") # [3, 600, 1067]
    grid = None

    img_height, img_width = img.shape[-2:]

    if is_out_of_bounds(vanishing_point, img_width, img_height):
        return img, ins, grid
    elif warp_aug:
        img, ins, grid = make_warp_aug(img, ins, vanishing_point, grid_net, use_ins=True)
    elif warp_aug_lzu:
        img, ins, grid = make_warp_aug(img, ins, vanishing_point, grid_net, use_ins=False)

    # reshape 4d to 3d
    if (len(img.shape) == 4) and keep_size:
        img = img.squeeze(0) 

    return img, ins, grid


# debug single images (DONE)
# read vp and file_name in rcnn.py (DONE)
# use rcnn.py to call this file (DONE)
# visualize warpped images (DONE)
# comment out night_aug's unwarp code (DONE)
# address both oob cases => skip images or edit original json files (DONE)
# move all to GPU (DONE)
# vectorize the code (TODO later)
# clean into one function here (TODO later)
# start training
def apply_unwarp(warped_x, grid, keep_size=True):
    if (len(warped_x.shape) == 3) and keep_size:
        warped_x = warped_x.unsqueeze(0)

    # print(f'warped_x is {warped_x.shape} grid is {grid.shape}') # [1, 3, 600, 1067], [1, 600, 1067, 2]

    # Compute inverse_grid
    inverse_grid = invert_grid(grid, warped_x.shape, separable=True)[0:1]

    # Expand inverse_grid to match batch size
    B = warped_x.shape[0]
    inverse_grid = inverse_grid.expand(B, -1, -1, -1)

    # Perform unzoom
    unwarped_x = F.grid_sample(
        warped_x, inverse_grid, mode='bilinear',
        align_corners=True, padding_mode='zeros'
    )
    # print("unwarped_x shape", unwarped_x.shape) # [1, 3, 600, 1067]

    if (len(unwarped_x.shape) == 4) and keep_size:
        unwarped_x = unwarped_x.squeeze(0)

    # print("unwarped_x shape", unwarped_x.shape) # [3, 600, 1067]

    # print(f"unwarped_x min {unwarped_x.min()} max {unwarped_x.max()}") # [0, 255]

    return unwarped_x



def extract_ratio_and_flip(transform_list):
    for transform in transform_list:
        if isinstance(transform, ResizeTransform):
            ratio = transform.new_h / transform.h
        elif isinstance(transform, (HFlipTransform, NoOpTransform)):
            flip = transform
    return ratio, flip



def process_and_update_features(batched_inputs, images, warp_aug_lzu, vp_dict, grid_net, backbone):
    features = None
    if warp_aug_lzu:
        # Preprocessing
        vanishing_points = [
            get_vanising_points(
                sample['file_name'], 
                vp_dict, 
                *extract_ratio_and_flip(sample['transform'])
            ) for sample in batched_inputs
        ]

        # Apply warping
        warped_images, _, grids = zip(*[
            apply_warp_aug(image, None, vp, False, warp_aug_lzu, grid_net) 
            for image, vp in zip(images.tensor, vanishing_points)
        ])
        warped_images = torch.stack(warped_images)

        # NOTE: debug visualization
        # for i, img in enumerate(warped_images):
        #     # Save the image
        #     vutils.save_image(img, f'warped_image_{i}.jpg', normalize=True)        

        # sys.exit(1)

        # Call the backbone
        features = backbone(warped_images)

        # Apply unwarping
        feature_key = next(iter(features))
        unwarped_features = torch.stack([
            apply_unwarp(feature, grid)
            for feature, grid in zip(features[feature_key], grids)
        ])

        # Replace the original features with unwarped ones
        features[feature_key] = unwarped_features

    return features






# def apply_unwarp(warped_x, grid):
#     print(f'warped_x is {warped_x.shape} grid is {grid.shape}' ) # [1, 3, 600, 1067], [1, 600, 1067, 2]

#     # NOTE: unsqueeze to 5d, maybe change later
#     if len(warped_x.shape) == 3:
#         warped_x = warped_x.unsqueeze(0).unsqueeze(0)
    
#     elif len(warped_x.shape) == 4:
#         warped_x = warped_x.unsqueeze(0)
    
#     # init all these for now
#     inverse_grids = None

#     # Unzoom
#     # x = []
    
#     # precompute and cache inverses
#     if inverse_grids is None:
#         inverse_grids = []
#         for i in range(len(warped_x)):
#             input_shape = warped_x[i].shape
#             inverse_grid = invert_grid(grid, input_shape,
#                                         separable=True)[0:1]
#             print("inverse_grid shape", inverse_grid.shape)
#             inverse_grids.append(inverse_grid)
#     # perform unzoom
#     for i in range(len(warped_x)):
#         B = len(warped_x[i])
#         inverse_grid = inverse_grids[i].expand(B, -1, -1, -1)
#         unwarped_x = F.grid_sample(
#             warped_x[i], inverse_grid, mode='bilinear',
#             align_corners=True, padding_mode='zeros'
#         )
#         print("unwarped_x shape", unwarped_x.shape) # [1, 3, 600, 1067]
#         # x.append(unwarped_x)

#     # return tuple(x)
#     return unwarped_x



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
