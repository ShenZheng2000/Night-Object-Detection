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

from vis_utils import *

# NOTE: use this to get coco file for different weather: /home/aghosh/Projects/2PCNet/Scripts/bdd/filter_file.py

# general parameters
is_coco_style = True  # NOTE: You can set this to False if not in COCO style
save_flag = True  # True or False
draw_bboxes = False
use_ins = True # NOTE: be careful about setting this as True
use_hw = False

# TODO: hardcode this for now. Might need to change how we save txt later
category = None
# category = 'car'
# category = 'traffic light'
# category = 'traffic sign'

bandwidth = 64  
amplitude = 1.0

# NOTE: use this for debug, switch to 
# tod = '_day'
dataset = 'bdd100k' # ['bdd100k', 'cityscapes', 'acdc']
# split = 'train' # ['train', 'val']

# NOTE:this is for cityscapes
if dataset == 'cityscapes':
    split = 'train'
    img_width = 2048
    img_height = 1024
    root_path = f"/home/aghosh/Projects/2PCNet/Datasets/{dataset}/leftImg8bit/{split}"
    coco_base = f"/home/aghosh/Projects/2PCNet/Datasets/{dataset}/gt_detection/instancesonly_filtered_gtFine_{split}_poly.json"
    vp_base = f"/home/aghosh/Projects/2PCNet/Datasets/VP/{dataset}_all_vp.json"

elif dataset == 'bdd100k':
    split = 'val'
    img_width = 1280
    img_height = 720
    root_path = f"/home/aghosh/Projects/2PCNet/Datasets/{dataset}/images/100k/{split}"
    coco_base = f"/home/aghosh/Projects/2PCNet/Datasets/{dataset}_ori/labels/det_20/det_{split}_coco.json"
    vp_base = f"/home/aghosh/Projects/2PCNet/Datasets/VP/{dataset}_all_vp.json"

elif dataset == 'acdc':
    split = 'train'
    img_width = 2048
    img_height = 1024
    root_path = f"/home/aghosh/Projects/2PCNet/Datasets/{dataset}/rgb_anon/{split}"
    coco_base = f"/home/aghosh/Projects/2PCNet/Datasets/{dataset}/gt_detection/{split}.json"
    vp_base = "/home/aghosh/Projects/2PCNet/Datasets/VP/cs_dz_acdc_synthia_all_vp.json"


def process_images(img_paths, v_pts_list, gt_bboxes_list, grid_net):
    hist_data_orig = []  # To collect areas of original bounding boxes
    hist_data_warped = []  # To collect areas of warped bounding boxes

    for index in range(len(img_paths)):
        img_path = img_paths[index]
        v_pts = v_pts_list[index]
        gt_bboxes = gt_bboxes_list[index]

        warp_image_exp(img_path, v_pts, gt_bboxes, grid_net, hist_data_orig, hist_data_warped)

        if index % 100 == 0:
            print(f"Done with {index} images")

    # # After all images have been processed, plot the histogram
    if grid_net == 'NO':
        output_path = 'NO'
    else:
        output_path = grid_net.__class__.__name__

    # Save the data to txt files
    if use_hw:
        save_data_to_txt(hist_data_orig, f"hists/hist_data_{dataset}_orig_hw_{str(category)}.txt")
        save_data_to_txt(hist_data_warped, f"hists/hist_data_{dataset}_{output_path}_hw_{str(category)}.txt")
    else:
        save_data_to_txt(hist_data_orig, f"hists/hist_data_{dataset}_orig_{str(category)}.txt")
        save_data_to_txt(hist_data_warped, f"hists/hist_data_{dataset}_{output_path}_{str(category)}.txt")

        # print mean and std/mean before warping
        mean_orig = np.mean(hist_data_orig)
        coef_orig = np.std(hist_data_orig) / mean_orig
        print(f"hist_data_orig; mean = {mean_orig:.2f}, coef of variation = {coef_orig:.2f}")

        # print mean and std/mean after warping
        mean_warped = np.mean(hist_data_warped)
        coef_warped = np.std(hist_data_warped) / mean_warped
        print(f"hist_data_warped; mean = {mean_warped:.2f}, coef of variation = {coef_warped:.2f}")

    # plot_histogram(hist_data_orig, 'Original BBoxes', num_bins=50, title='Origin', output_path='Origin.png')
    # plot_histogram(hist_data_warped, 'Warped BBoxes', num_bins=50, title=output_path, output_path=f'{output_path}.png')


def warp_image_exp(img_path, v_pts, gt_bboxes, grid_net, hist_data_orig, hist_data_warped):

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
    if grid_net == 'NO':
        grid_type = 'NO'
    else:
        grid_type = grid_net.__class__.__name__
    # print("grid_type is", grid_type)
    
    output_dir = os.path.join(main_output_dir, grid_type)

    os.makedirs(output_dir, exist_ok=True)
    # print(f"output_dir is {output_dir}"); exit()

    
    if grid_net == 'NO':
        warped_img, ins, grid = img_tensor, converted_bboxes, None
    else:
        warped_img, ins, grid = make_warp_aug(img_tensor, converted_bboxes, v_pts, grid_net, use_ins=use_ins, file_name=base_path)

        # TODO: hardcode now: if any element in ins greater than 1280, print filename, and exit(
        # clip (x1, y1, x2, y2) so that 0 < x1, x2 < width, 0 < y1, y2 < height
        ins = clip_bounding_boxes(ins, img_width, img_height)
        # unwarp_x = apply_unwarp(warped_img, grid, keep_size=True)
        
    if save_flag:

        # Save the resulting image
        warped_img_name = os.path.join(output_dir, f"{base_path}.jpg") # NOTE: skip bs for now

        if draw_bboxes:
            # Draw bounding boxes on the warped image
            image_with_bboxes = draw_bboxes_on_image(warped_img[0], ins)
            image_with_bboxes.save(warped_img_name)
        else:
            # save torch tensor as image
            save_image(warped_img, warped_img_name)

    if use_hw == True:
        orig_hws = calculate_bbox_hw(converted_bboxes.cpu().numpy())
        warped_hws = calculate_bbox_hw(ins.cpu().numpy())
        hist_data_orig.extend(orig_hws)
        hist_data_warped.extend(warped_hws)

    else:
        orig_areas = calculate_bbox_areas(converted_bboxes.cpu().numpy())
        warped_areas = calculate_bbox_areas(ins.cpu().numpy())
        hist_data_orig.extend(orig_areas)
        hist_data_warped.extend(warped_areas)

    # # NOTE: code for 
    # # Calculate original bbox areas and add to the hist_data_orig list
    # orig_areas = calculate_bbox_areas(converted_bboxes.cpu().numpy())
    # hist_data_orig.extend(orig_areas)            

    # # Calculate warped bbox areas and add to the hist_data_warped list
    # warped_areas = calculate_bbox_areas(ins.cpu().numpy())
    # hist_data_warped.extend(warped_areas)

    # print(f"orig_areas = {orig_areas}")
    # print(f"warped_areas = {warped_areas}")            


def get_gt_bboxes_for_images(img_paths, data_source, root_path):
    if is_coco_style:
        return [torch.tensor(get_ground_truth_bboxes_coco(
            os.path.basename(img_path), data_source, category_name=category
            )
        ).cuda() for img_path in img_paths]
    else:
        return [torch.tensor(get_ground_truth_bboxes_simple(img_path, data_source, root_path)).cuda() for img_path in img_paths]


def main():
    # root_path = "/home/aghosh/Projects/2PCNet/Datasets/cityscapes/leftImg8bit/train"
    # coco_base = "/home/aghosh/Projects/2PCNet/Datasets/cityscapes_seg2det.json"
    # vp_base = "/home/aghosh/Projects/2PCNet/Datasets/VP/ind_vps/cityscapes_all_vp.json" # NOTE: this one is in [x1, y1, x2, y2] format

    img_paths = glob.glob(f'{root_path}/**/*.[jJ][pP][gG]', recursive=True) + glob.glob(f'{root_path}/**/*.[pP][nN][gG]', recursive=True)
    
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


    grid_net_names = [
        # 'FixedKDEGrid',
        # 'CuboidGlobalKDEGrid',
        'PlainKDEGrid',
        # 'NO'
        # 'MixKDEGrid'
    ]

    ################### Important hyperparameters ###################

    for grid_net_name in grid_net_names:

        # Retrieve the class from globals() and instantiate it
        if grid_net_name == 'PlainKDEGrid':
            grid_net = PlainKDEGrid(
                                    bandwidth=bandwidth,
                                    amplitude_scale=amplitude,
                                    ).cuda()

        elif grid_net_name == 'NO':
            grid_net = 'NO'
        else:
            grid_net_class = globals()[grid_net_name]
            grid_net = grid_net_class().cuda()  # Instantiate and move to GPU

        # Use the instantiated grid_net in your process_images function
        process_images(img_paths, 
                        v_pts_list, 
                        gt_bboxes_list, 
                        grid_net
                        )


if __name__ == "__main__":
    main()