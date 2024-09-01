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

    save_hist_data(hist_data_orig, hist_data_warped, grid_net, config)

def warp_image_exp(img_path, v_pts, gt_bboxes, grid_net, hist_data_orig, hist_data_warped, config):
    main_output_dir = config['main_output_dir']
    setup_directories(main_output_dir)

    if gt_bboxes.nelement() == 0:
        print("Empty gt_bboxes detected. Skipping image.")
        return

    converted_bboxes = convert_format(gt_bboxes).clone() if config['is_coco_style'] else gt_bboxes.clone()
    base_path = os.path.splitext(os.path.basename(img_path))[0]
    img_tensor = load_image_tensor(img_path)
    
    output_dir = os.path.join(main_output_dir, grid_net if grid_net == 'NO' else grid_net.__class__.__name__)
    os.makedirs(output_dir, exist_ok=True)

    if grid_net == 'NO':
        warped_img, ins, grid = img_tensor, converted_bboxes, None
    else:
        warped_img, ins, grid = make_warp_aug(img_tensor, converted_bboxes, v_pts, grid_net, use_ins=config['use_ins'], file_name=base_path)
        ins = clip_bounding_boxes(ins, config['img_width'], config['img_height'])

    if config['save_flag']:
        save_warped_image(warped_img, ins, base_path, output_dir, config)

    update_hist_data(converted_bboxes, ins, hist_data_orig, hist_data_warped, config)

def load_image_tensor(img_path):
    img = Image.open(img_path).convert("RGB")
    transform = transforms.Compose([transforms.ToTensor()])
    return transform(img).cuda().unsqueeze(0)

def save_warped_image(warped_img, ins, base_path, output_dir, config):
    warped_img_name = os.path.join(output_dir, f"{base_path}.jpg")
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

def main():
    config = {
        "is_coco_style": True,
        "save_flag": True,
        "draw_bboxes": False,
        "use_ins": True,
        "use_hw": False,
        "category": None,
        "bandwidth": 64,
        "amplitude": 1.0,
        "dataset": "bdd100k",
        "split": "train",
        "img_width": 1280,
        "img_height": 720,
        "root_path": "/home/aghosh/Projects/2PCNet/Datasets/bdd100k/images/100k/train",
        "coco_base": "/home/aghosh/Projects/2PCNet/Datasets/bdd100k_ori/labels/det_20/det_train_coco.json",
        "vp_base": "/home/aghosh/Projects/2PCNet/Datasets/VP/bdd100k_all_vp.json",
        "main_output_dir": "warped_images_exp",
        "grid_net_names": ["PlainKDEGrid", "NO"],
        "hardcoded_basename": "0a5b3a25-81c3be4d.jpg" # NOTE: hardcode for now
    }
    
    img_paths = [os.path.join(config["root_path"], config["hardcoded_basename"])]
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
    for grid_net_name in grid_net_names:
        grid_net = instantiate_grid_net(grid_net_name, config)
        process_images(img_paths, v_pts_list, gt_bboxes_list, grid_net, config)

def instantiate_grid_net(grid_net_name, config):
    if grid_net_name == 'PlainKDEGrid':
        return PlainKDEGrid(bandwidth=config['bandwidth'], amplitude_scale=config['amplitude']).cuda()
    elif grid_net_name == 'NO':
        return 'NO'
    else:
        grid_net_class = globals()[grid_net_name]
        return grid_net_class().cuda()

if __name__ == "__main__":
    main()
