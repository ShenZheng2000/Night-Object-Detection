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
import time

from vis_utils import *

def setup_directories(main_output_dir):
    if not os.path.exists(main_output_dir):
        os.makedirs(main_output_dir)

def process_images(img_paths, v_pts_list, gt_bboxes_list, grid_net, config):
    hist_data_orig = []
    hist_data_warped = []

    for index, (img_path, v_pts, gt_bboxes) in enumerate(zip(img_paths, v_pts_list, gt_bboxes_list)):
        warp_image_exp(img_path, v_pts, gt_bboxes, grid_net, hist_data_orig, hist_data_warped, config)

        if index % 100 == 0:
            print(f"Processed {index} images")
    # NOTE: comment this noew to save time. 
    # save_hist_data(hist_data_orig, hist_data_warped, grid_net, config) 

import matplotlib.pyplot as plt

def warp_image_exp(img_path, v_pts, gt_bboxes, grid_net, hist_data_orig, hist_data_warped, config):
    main_output_dir = config['main_output_dir']
    setup_directories(main_output_dir)

    if gt_bboxes.nelement() == 0:
        print("Empty gt_bboxes detected. Skipping image.")
        return

    converted_bboxes = convert_format(gt_bboxes).clone() if config['is_coco_style'] else gt_bboxes.clone()
    base_path = os.path.splitext(os.path.basename(img_path))[0]
    img_tensor = load_image_tensor(img_path)
    
    # output_dir = os.path.join(main_output_dir, grid_net if grid_net == 'NO' else grid_net.__class__.__name__)

    # Modify this line to include bandwidth in the output directory name
    if grid_net == 'NO':
        output_dir = os.path.join(main_output_dir, grid_net)
    else:
        output_dir = os.path.join(main_output_dir, f"{grid_net.__class__.__name__}_bs_{config['bandwidth']}")

    os.makedirs(output_dir, exist_ok=True)

    if grid_net == 'NO':
        warped_img, ins, grid = img_tensor, converted_bboxes, None
    else:
        warped_img, ins, grid = make_warp_aug(img_tensor, converted_bboxes, v_pts, grid_net, use_ins=config['use_ins'], file_name=base_path)
        ins = clip_bounding_boxes(ins, config['img_width'], config['img_height'])

        # TODO: save the original image
        ori_img_name = os.path.join(output_dir, f"{base_path}_ori.jpg")
        save_image(img_tensor, ori_img_name)

        # TODO: do unwarp, and save the images!
        unwarp_img = apply_unwarp(warped_img, grid)
        # print("unwarp_img.shape", unwarp_img.shape) # [3, 720, 1280]
        unwarp_img_name = os.path.join(output_dir, f"{base_path}_unwarp.jpg")
        save_image(unwarp_img, unwarp_img_name)

        # TODO: save (ori - unwarp) image => Maybe using a binary mask?
        ori_img = img_tensor.squeeze(0)
        # print("ori_img shape", ori_img.shape)  [1, 3, 720, 1280]
        diff_img = ori_img - unwarp_img
        diff_img = torch.sqrt(torch.sum(diff_img ** 2, dim=0, keepdim=True))
        
        # diff_img = diff_img * 255

        diff_img_numpy = diff_img.permute(1, 2, 0).cpu().numpy()


        ## log error
        # diff_img_numpy = np.log2(diff_img_numpy + 1) #/ np.log2(1.2)

        fig, ax = plt.subplots(figsize=(16, 12), dpi=300)

        # Display the image with the reversed colormap
        im = ax.imshow(diff_img_numpy, cmap='jet', vmin=0, vmax=1)

        # Remove axis labels
        ax.axis('off')

        # Add a colorbar with a size that matches the image height
        cbar = fig.colorbar(im, ax=ax, fraction=0.024, pad=0.01)
        cbar.ax.tick_params(labelsize=16)

        # Save the plot with a high resolution
        plt.savefig(os.path.join(output_dir, f"{base_path}_diff.jpg"), bbox_inches='tight', pad_inches=0.05)
        plt.close()

        # diff_img_name = os.path.join(output_dir, f"{base_path}_diff.jpg")
        # save_image(diff_img, diff_img_name)

    if config['save_flag']:
        save_warped_image(warped_img, ins, base_path, output_dir, config)

    update_hist_data(converted_bboxes, ins, hist_data_orig, hist_data_warped, config)

def load_image_tensor(img_path):
    img = Image.open(img_path).convert("RGB")
    transform = transforms.Compose([transforms.ToTensor()])
    return transform(img).cuda().unsqueeze(0)

def save_warped_image(warped_img, ins, base_path, output_dir, config):
    warped_img_name = os.path.join(output_dir, f"{base_path}_warped.jpg")
    if config['draw_bboxes']:
        image_with_bboxes = draw_bboxes_on_image(warped_img[0], ins)
        image_with_bboxes.save(warped_img_name)
    else:
        save_image(warped_img, warped_img_name)

def update_hist_data(converted_bboxes, ins, hist_data_orig, hist_data_warped, config):
    if config['use_hw']:
        hist_data_orig.extend(calculate_bbox_hw(converted_bboxes.cpu().numpy()))
        hist_data_warped.extend(calculate_bbox_hw(ins.cpu().numpy()))
    else:
        hist_data_orig.extend(calculate_bbox_areas(converted_bboxes.cpu().numpy()))
        hist_data_warped.extend(calculate_bbox_areas(ins.cpu().numpy()))

def save_hist_data(hist_data_orig, hist_data_warped, grid_net, config):
    output_path = 'NO' if grid_net == 'NO' else grid_net.__class__.__name__
    save_data_to_txt(hist_data_orig, f"hists/hist_data_{config['dataset']}_orig_{str(config['category'])}.txt")
    save_data_to_txt(hist_data_warped, f"hists/hist_data_{config['dataset']}_{output_path}_{str(config['category'])}.txt")
    print_hist_stats(hist_data_orig, 'orig')
    print_hist_stats(hist_data_warped, 'warped')

def print_hist_stats(data, label):
    mean = np.mean(data)
    coef = np.std(data) / mean
    print(f"hist_data_{label}: mean = {mean:.2f}, coef of variation = {coef:.2f}")

def get_gt_bboxes_for_images(img_paths, data_source, root_path, config):
    if config['is_coco_style']:
        gt_bboxes_list = []
        coco_filenames = {os.path.basename(img['file_name']) for img in data_source.dataset['images']}
        
        for img_path in img_paths:
            img_basename = os.path.basename(img_path)
            if img_basename not in coco_filenames:
                print(f"Image {img_basename} is not found in COCO dataset, skipping.")
                continue

            try:
                gt_bboxes = get_ground_truth_bboxes_coco(img_basename, data_source, category_name=config['category'])
                gt_bboxes_list.append(torch.tensor(gt_bboxes).cuda())
            except ValueError as e:
                print(f"Error for image {img_basename}: {e}")
                print(f"Image path: {img_path}")
                continue
        return gt_bboxes_list
    else:
        return [torch.tensor(get_ground_truth_bboxes_simple(img_path, data_source, root_path)).cuda() for img_path in img_paths]

def instantiate_grid_net(grid_net_name, config):
    if grid_net_name == 'PlainKDEGrid':
        return PlainKDEGrid(bandwidth_scale=config['bandwidth'], amplitude_scale=config['amplitude']).cuda()
    elif grid_net_name == 'NO':
        return 'NO'
    else:
        grid_net_class = globals()[grid_net_name]
        return grid_net_class().cuda()


# NOTE: saliency scale (s) = bandwidth / 4
    # bandwidth = 4, then s = 1.
    # bandwidth = 64, then s = 16.
    # bandwidth = 1024, then s = 256.

def main():
    config = {
        "is_coco_style": True,
        "save_flag": True,
        "draw_bboxes": False,
        "use_ins": True,
        "use_hw": False,
        "category": None,
        "bandwidths": [
                        4, 
                       64, 
                       1024
                       ],
        "amplitude": 1.0,
        "dataset": "bdd100k",
        "split": "train",
        "img_width": 1280,
        "img_height": 720,
        "root_path": "/home/aghosh/Projects/2PCNet/Datasets/bdd100k/images/100k/val",
        "coco_base": "/home/aghosh/Projects/2PCNet/Datasets/bdd100k_ori/labels/det_20/det_val_coco.json",
        "vp_base": "/home/aghosh/Projects/2PCNet/Datasets/VP/bdd100k_all_vp.json",
        "main_output_dir": "warped_images_exp",
        "grid_net_names": ["PlainKDEGrid", "NO"],
        "hardcoded_basename": "b1ceb32e-51852abe.jpg"
        # "hardcoded_basename": "b38b2b6a-cb374ce8.jpg"
        # "hardcoded_basename": None  # Set to None to process all images in the folder
    }

    if config["hardcoded_basename"]:
        img_paths = [os.path.join(config["root_path"], config["hardcoded_basename"])]
    else:
        img_paths = [os.path.join(config["root_path"], img) for img in os.listdir(config["root_path"]) if img.endswith(".jpg")]
    
    # img_paths = [os.path.join(config["root_path"], config["hardcoded_basename"])]
    vp_dict = load_vp_data(config['vp_base'])
    v_pts_list = get_v_pts_for_images(img_paths, vp_dict)

    if config['is_coco_style']:
        coco = COCO(config['coco_base'])
        coco.imgs_by_name = {img['file_name']: img for img in coco.dataset['images']}
        gt_bboxes_list = get_gt_bboxes_for_images(img_paths, coco, config['root_path'], config)
    else:
        with open(config['coco_base'], 'r') as f:
            bbox_dict = json.load(f)
        gt_bboxes_list = get_gt_bboxes_for_images(img_paths, bbox_dict, config['root_path'], config)

    grid_net_names = config['grid_net_names']
    
    for bandwidth in config['bandwidths']:
        config['bandwidth'] = bandwidth  # Set the current bandwidth in the config

        for grid_net_name in grid_net_names:
            grid_net = instantiate_grid_net(grid_net_name, config)
            process_images(img_paths, v_pts_list, gt_bboxes_list, grid_net, config)


if __name__ == "__main__":
    main()
