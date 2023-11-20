# one vp in-bound and one vp out-bound
# img_path = "/home/aghosh/Projects/2PCNet/Datasets/bdd100k/images/100k/train_day/5250cae5-14dc826a.jpg"
# v_pts = torch.tensor([-235.50865750631317, 340.43002132438045])

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

from debug_10_27 import load_vp_data, get_v_pts_for_images, get_gt_bboxes_for_images, convert_format, draw_bboxes_on_image

# NOTE: use this to get coco file for different weather: /home/aghosh/Projects/2PCNet/Scripts/bdd/filter_file.py


# general parameters
is_coco_style = True  # NOTE: You can set this to False if not in COCO style
save_flag = True  # True or False
use_ins = True # NOTE: be careful about setting this as True

bandwidth = int(sys.argv[1])
amplitude = float(sys.argv[2])

# NOTE: use this for debug, switch to 
# tod = '_day'
dataset = 'bdd100k' # ['bdd100k', 'cityscapes', 'acdc', 'dark_zurich']
split = 'val' # ['train', 'val']

# NOTE:this is for cityscapes
if dataset == 'cityscapes':
    root_path = f"/home/aghosh/Projects/2PCNet/Datasets/{dataset}/leftImg8bit/{split}"
    coco_base = f"/home/aghosh/Projects/2PCNet/Datasets/{dataset}/gt_detection/instancesonly_filtered_gtFine_{split}_poly.json"
    vp_base = f"/home/aghosh/Projects/2PCNet/Datasets/VP/{dataset}_all_vp.json"

elif dataset == 'bdd100k':
    root_path = f"/home/aghosh/Projects/2PCNet/Datasets/{dataset}/images/100k/{split}"
    coco_base = f"/home/aghosh/Projects/2PCNet/Datasets/{dataset}_ori/labels/det_20/det_{split}_coco.json"
    vp_base = f"/home/aghosh/Projects/2PCNet/Datasets/VP/{dataset}_all_vp.json"

elif dataset == 'acdc':
    root_path = f"/home/aghosh/Projects/2PCNet/Datasets/{dataset}/rgb_anon/{split}"
    coco_base = f"/home/aghosh/Projects/2PCNet/Datasets/{dataset}/gt_detection/{split}.json"
    vp_base = "/home/aghosh/Projects/2PCNet/Datasets/VP/cs_dz_acdc_synthia_all_vp.json"


def process_images(img_paths, v_pts_list, gt_bboxes_list, grid_net):
    for index in range(len(img_paths)):
        img_path = img_paths[index]
        v_pts = v_pts_list[index]
        gt_bboxes = gt_bboxes_list[index]

        warp_image_exp(img_path, v_pts, gt_bboxes, grid_net)

        if index % 100 == 0:
            print(f"Done with {index} images")


def warp_image_exp(img_path, v_pts, gt_bboxes, grid_net):

    # Always point to the main saving directory
    main_output_dir = "warped_images_exp"
    if not os.path.exists(main_output_dir):
        os.makedirs(main_output_dir)

    # Check if bboxes is empty and return None
    if gt_bboxes.nelement() == 0:
        print("Empty gt_bboxes detected. Exiting function.")
        return None        

    # Convert the bbox format
    if is_coco_style:
        converted_bboxes = convert_format(gt_bboxes).clone()
    else:
        converted_bboxes = gt_bboxes.clone()

    # get image base name without extension
    base_path = os.path.splitext(os.path.basename(img_path))[0]

    # Load the image as a torch tensor
    img = Image.open(img_path).convert("RGB")
    transform = transforms.Compose([transforms.ToTensor()])
    img_tensor = transform(img).cuda().unsqueeze(0)  # Load, convert, and add batch dimension

    # Constants
    grid_type = f"{grid_net.__class__.__name__}_{bandwidth}_{amplitude}"
    # print("grid_type is", grid_type)
    
    output_dir = os.path.join(main_output_dir, grid_type)

    os.makedirs(output_dir, exist_ok=True)
    # print(f"output_dir is {output_dir}")

    warped_img, ins, grid = make_warp_aug(img_tensor, converted_bboxes, v_pts, grid_net, use_ins=use_ins, file_name=base_path)
        
    if save_flag:
        # Draw bounding boxes on the warped image
        image_with_bboxes = draw_bboxes_on_image(warped_img[0], ins)

        # Save the resulting image
        warped_img_name = os.path.join(output_dir, f"{base_path}.jpg") # NOTE: skip bs for now
        image_with_bboxes.save(warped_img_name)  



def main():

    # img_paths = glob.glob(f'{root_path}/**/*.[jJ][pP][gG]', recursive=True) + glob.glob(f'{root_path}/**/*.[pP][nN][gG]', recursive=True)

    # TODO: hardcode one image for now!!!!!!!!!!!
    img_paths = [
        '/home/aghosh/Projects/2PCNet/Datasets/bdd100k/images/100k/val/b1e1a7b8-b397c445.jpg',
    ]
    
    vp_dict = load_vp_data(vp_base)
    v_pts_list = get_v_pts_for_images(img_paths, vp_dict)

    if is_coco_style:
        coco = COCO(coco_base)
        coco.imgs_by_name = {img['file_name']: img for img in coco.dataset['images']}
        gt_bboxes_list = get_gt_bboxes_for_images(img_paths, coco, root_path)
    else:
        with open(coco_base, 'r') as f:
            bbox_dict = json.load(f)
        gt_bboxes_list = get_gt_bboxes_for_images(img_paths, bbox_dict, root_path)


    # Retrieve the class from globals() and instantiate it
    # print("bandwidth is", bandwidth)
    grid_net = PlainKDEGrid(
                                bandwidth_scale=bandwidth,
                                amplitude_scale=amplitude,
                                ).cuda()


    # Use the instantiated grid_net in your process_images function
    process_images(img_paths, 
                    v_pts_list, 
                    gt_bboxes_list, 
                    grid_net
                    )


if __name__ == "__main__":
    main()