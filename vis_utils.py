from PIL import Image
import torchvision.transforms as transforms
from twophase.data.transforms.grid_generator import PlainKDEGrid, MixKDEGrid, CuboidGlobalKDEGrid, FixedKDEGrid
from twophase.data.transforms.fovea import make_warp_aug, apply_unwarp
import torch
import os
from torchvision.utils import save_image
import cv2
import sys
from pycocotools.coco import COCO
import json
import glob
from torchvision.transforms.functional import to_pil_image, to_tensor
import numpy as np
import matplotlib.pyplot as plt




def clip_bounding_boxes(tensor, img_width, img_height):
    """
    Clip bounding boxes in a tensor to ensure they are within specified image bounds.
    
    Args:
        tensor (torch.Tensor): A tensor of shape (N, 4) representing bounding boxes as (x1, y1, x2, y2).
        img_width (int): The width of the image.
        img_height (int): The height of the image.

    Returns:
        torch.Tensor: The clipped bounding boxes tensor.
    """
    # Clone the input tensor to avoid modifying it in-place
    clipped_tensor = tensor.clone()
    
    # Clip x1, y1, x2, y2 to ensure they are within specified bounds
    clipped_tensor[:, 0].clamp_(min=0, max=img_width)  # Clip x1
    clipped_tensor[:, 1].clamp_(min=0, max=img_height)  # Clip y1
    clipped_tensor[:, 2].clamp_(min=0, max=img_width)  # Clip x2
    clipped_tensor[:, 3].clamp_(min=0, max=img_height)  # Clip y2

    return clipped_tensor


# Function to calculate the areas of bounding boxes
def calculate_bbox_areas(bboxes):
    # Calculate areas from bounding boxes in x1, y1, x2, y2 format
    areas = []
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        area = (x2 - x1) * (y2 - y1)
        areas.append(area)
    return areas

def calculate_bbox_hw(bboxes):
    # Calculate areas from bounding boxes in x1, y1, x2, y2 format
    hws = []
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        hws.append([width, height])
    return hws

def save_data_to_txt(data, filename):
    np.savetxt(filename, data, fmt='%f')


def convert_format(bboxes):
    # Splitting the tensor columns
    x1, y1, width, height = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    
    # Calculating x2 and y2
    x2 = x1 + width
    y2 = y1 + height
    
    # Stacking them into the new format
    new_bboxes = torch.stack([x1, y1, x2, y2], dim=1)
    
    return new_bboxes



# def get_ground_truth_bboxes_coco(image_file_name, coco, category_name=None):
#     # # Get the image info for the specified image file name
#     # img_info = coco.imgs_by_name[image_file_name]

#     # Extract the basename of the image file
#     image_basename = os.path.basename(image_file_name)

#     # Search for the image info using the basename in the dictionary keys
#     img_info = next((info for name, info in coco.imgs_by_name.items() if os.path.basename(name) == image_basename), None)

#     # Check if we found the image
#     if img_info is None:
#         raise ValueError("Image not found in the dataset.")

#     # Get the annotation IDs associated with the image
#     ann_ids = coco.getAnnIds(imgIds=img_info['id'])

#     # Load the annotations
#     annotations = coco.loadAnns(ann_ids)

#     # Extract the ground truth bounding boxes
#     gt_bboxes = [ann['bbox'] for ann in annotations]

#     return gt_bboxes

def get_ground_truth_bboxes_coco(image_file_name, coco, category_name=None):
    # Extract the basename of the image file
    image_basename = os.path.basename(image_file_name)

    # Search for the image info using the basename in the dictionary keys
    img_info = next((info for name, info in coco.imgs_by_name.items() if os.path.basename(name) == image_basename), None)

    # Check if we found the image
    if img_info is None:
        raise ValueError("Image not found in the dataset.")

    # Get the annotation IDs associated with the image
    ann_ids = coco.getAnnIds(imgIds=img_info['id'])

    # Load the annotations
    annotations = coco.loadAnns(ann_ids)

    if category_name is not None:
        # Get the category IDs for the specified category
        cat_ids = coco.getCatIds(catNms=[category_name])
        # Extract the ground truth bounding boxes for the specified category
        gt_bboxes = [ann['bbox'] for ann in annotations if ann['category_id'] in cat_ids]
    else:
        # Extract the ground truth bounding boxes for all categories
        gt_bboxes = [ann['bbox'] for ann in annotations]

    return gt_bboxes

def get_ground_truth_bboxes_simple(img_path, bbox_dict, root_path):
    # Get the relative path from the root_path to use as key in the dictionary
    relative_path = os.path.relpath(img_path, root_path)
    return bbox_dict.get(relative_path, [])


def before_train_json(json_file_path): 
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    # Construct a new dictionary with the basenames as keys
    transformed_data = {os.path.basename(key): value for key, value in data.items()}

    return transformed_data


def draw_bboxes_on_image(image_tensor, bboxes_tensor, thickness=5):
    # Convert tensor image to PIL Image for drawing
    image_pil = to_pil_image(image_tensor.cpu().detach())

    # Convert to NumPy array for OpenCV to use
    image_np = np.array(image_pil)

    # Convert RGB to BGR for OpenCV
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Define BGR colors based on provided RGB values
    # RGB to BGR conversion: (R, G, B) -> (B, G, R)
    red_bgr = (28, 26, 228)  # RGB: (228, 26, 28)
    green_bgr = (74, 175, 77)  # RGB: (77, 175, 74)
    blue_bgr = (184, 126, 55)  # RGB: (55, 126, 184)

    # Draw each bounding box
    for box in bboxes_tensor:
        start_point = (int(box[0]), int(box[1]))  # (x1, y1)
        end_point = (int(box[2]), int(box[3]))  # (x2, y2)

        # Calculate the area of the bounding box
        box_area = (end_point[0] - start_point[0]) * (end_point[1] - start_point[1])

        # Decide the color based on the area size
        if box_area < 32**2:
            color = red_bgr  # Blue for small boxes
        elif 32**2 <= box_area <= 96**2:
            color = green_bgr  # Green for medium boxes
        else:
            color = blue_bgr  # Red for large boxes

        # Draw the rectangle on the image
        image_np = cv2.rectangle(image_np, start_point, end_point, color, thickness)

    # Convert back to RGB from BGR
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

    # Convert back to PIL image
    image_pil = to_pil_image(image_np)

    return image_pil


def load_vp_data(vp_base):
    return before_train_json(vp_base)

# NOTE: hardcode to change now, might be some issues for tpp. So please use validation set for visualization!!!!!!!!!!!!!
def get_v_pts_for_images(img_paths, vp_dict):
    v_pts_list = []
    for img_path in img_paths:
        img_filename = os.path.basename(img_path)
        if img_filename in vp_dict:
            v_pts = torch.tensor(vp_dict[img_filename]).cuda()
        else:
            print(f"img_filename {img_filename} not found in vp_dict! Using empty tensor instead.")
            v_pts = torch.tensor([])  # Append an empty tensor
        v_pts_list.append(v_pts)
    return v_pts_list

