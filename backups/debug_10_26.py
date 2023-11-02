# one vp in-bound and one vp out-bound
# img_path = "/home/aghosh/Projects/2PCNet/Datasets/bdd100k/images/100k/train_day/5250cae5-14dc826a.jpg"
# v_pts = torch.tensor([-235.50865750631317, 340.43002132438045])

from PIL import Image
import torchvision.transforms as transforms
from twophase.data.transforms.grid_generator import PlainKDEGrid
from twophase.data.transforms.fovea import make_warp_aug
import torch
import os
from torchvision.utils import save_image
import cv2
import sys
from pycocotools.coco import COCO
import json

def convert_format(bboxes):
    # Splitting the tensor columns
    x1, y1, width, height = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    
    # Calculating x2 and y2
    x2 = x1 + width
    y2 = y1 + height
    
    # Stacking them into the new format
    new_bboxes = torch.stack([x1, y1, x2, y2], dim=1)
    
    return new_bboxes


def vis_image_exp(img_path, v_pts, gt_bboxes):
    output_dir = 'vis_images_exp'
    os.makedirs(output_dir, exist_ok=True)

    # Load the image
    img = cv2.imread(img_path)

    # Convert the image to RGB format
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Draw bounding boxes on the image
    for bbox in gt_bboxes:
        x, y, w, h = bbox
        x, y, w, h = int(x), int(y), int(w), int(h)
        cv2.rectangle(img_rgb, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Draw a red rectangle

    # Save the image with bounding boxes
    output_image_path = os.path.join(output_dir, "image_with_bboxes.jpg")
    cv2.imwrite(output_image_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))


def get_ground_truth_bboxes(image_file_name, coco_file_path):
    # Load the COCO format JSON file
    coco = COCO(coco_file_path)

    # Get the image info for the specified image file name
    img_info = next(item for item in coco.dataset['images'] if item["file_name"] == image_file_name)

    # Check if we found the image
    if img_info is None:
        raise ValueError("Image not found in the dataset.")

    # Get the annotation IDs associated with the image
    ann_ids = coco.getAnnIds(imgIds=img_info['id'])

    # Load the annotations
    annotations = coco.loadAnns(ann_ids)

    # Extract the ground truth bounding boxes
    gt_bboxes = [ann['bbox'] for ann in annotations]

    return gt_bboxes


def before_train_json(json_file_path): 
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    # Construct a new dictionary with the basenames as keys
    transformed_data = {os.path.basename(key): value for key, value in data.items()}

    return transformed_data

def warp_image_exp(img_path, v_pts, gt_bboxes, warp_fovea_inst_scale=False, save_flag=False):

    # Define your saving directory here:
    output_dir = "warped_images_exp"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # NOTE: convert the format to x1, y1, x2, y2
    converted_bboxes = convert_format(gt_bboxes)
    converted_bboxes = converted_bboxes.clone() # NOTE: must use this to avoid inplace operation

    # Load the image as a torch tensor
    img = Image.open(img_path).convert("RGB")
    transform = transforms.Compose([transforms.ToTensor()])
    img_tensor = transform(img).cuda()

    # Unsqueeze to correct dimension
    img_tensor = img_tensor.unsqueeze(0)

    # Loop through grid types
    grid_type = 'PlainKDEGrid'

    # generate image basename
    base_name = os.path.splitext(img_path)[0]

    # Create an output directory
    output_dir = os.path.join("warped_images_exp", grid_type, base_name)
    os.makedirs(output_dir, exist_ok=True)

    # bandwidth_scales = [8, 16, 32, 64, 128, 256, 512]
    # bandwidth_scales = [16, 64, 256]
    bandwidth_scales = [64]
    # bandwidth_scales = [None, 64]

    for bs in bandwidth_scales:

        # Create the grid
        # TODO: change depth_saliency = True for debug
        grid_net = PlainKDEGrid(bandwidth_scale=bs, warp_fovea_inst_scale=warp_fovea_inst_scale).cuda()

        # Warp the image
        # print(f"bs = {bs}")

        try:
            warped_img, _, _ = make_warp_aug(img_tensor, converted_bboxes, v_pts, grid_net, use_ins=False)
        except:
            print("Error in make_warp_aug")
            print("v_pts is", v_pts)
            print("basename is", base_name)

        if save_flag:

            warped_img_name = os.path.join(output_dir, f"bs_{bs}_{warp_fovea_inst_scale}.jpg")

            # Save the warped image
            save_image(warped_img[0], warped_img_name)  # Assuming warped_img is [1, C, H, W], take the 0-th index

            print(f"Saved warped image to: {warped_img_name}")


if __name__ == "__main__":

    # img_path = "/home/aghosh/Projects/2PCNet/Datasets/bdd100k/images/100k/train_debug/0a2a1a40-44173767.jpg"
    img_path = "/home/aghosh/Projects/2PCNet/Datasets/bdd100k/images/100k/train_day/14a2bb54-12317429.jpg"
    img_base = os.path.basename(img_path)
    
    # read coco labels
    # coco_base = "/home/aghosh/Projects/2PCNet/Datasets/bdd100k/coco_labels/train_day.json"
    coco_base = "home/aghosh/Projects/2PCNet/Datasets/cityscapes_seg2det.json"

    # read vps
    # vp_base = "/home/aghosh/Projects/2PCNet/Datasets/VP/backups/bdd100k_all_vp.json"
    vp_base = "/home/aghosh/Projects/2PCNet/Datasets/VP/ind_vps/cityscapes_all_vp.json"
    vp_dict = before_train_json(vp_base)

    # get vps
    v_pts = vp_dict[img_base]
    v_pts = torch.tensor(v_pts).cuda()

    # get gt bboxes
    gt_bboxes = get_ground_truth_bboxes(img_base, coco_base) # TODO: use a different function for non-coco datasets
    gt_bboxes = torch.tensor(gt_bboxes).cuda()

    # print("v_pts is", v_pts)
    # print("gt_bboxes is", gt_bboxes)

    # depth_saliencys = [False, True]
    depth_saliencys = [True]

    for ds in depth_saliencys:
        warp_image_exp(img_path, 
                       v_pts, 
                       gt_bboxes, 
                       warp_fovea_inst_scale=ds, 
                       save_flag=False)

    # vis_image_exp(img_path, v_pts, gt_bboxes)
