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

# NOTE: use this to get coco file for different weather: /home/aghosh/Projects/2PCNet/Scripts/bdd/filter_file.py

# general parameters
is_coco_style = True  # NOTE: You can set this to False if not in COCO style
save_flag = False  # True or False
use_ins = True # NOTE: be careful about setting this as True

# parameters for PlainKDEGrid only
warp_fovea_inst_scale = False
warp_fovea_inst_scale_l2 = False

assert not (warp_fovea_inst_scale and warp_fovea_inst_scale_l2), "warp_fovea_inst_scale and warp_fovea_inst_scale_l2 cannot be True at the same time."

# root_path = "/home/aghosh/Projects/2PCNet/Datasets/bdd100k/images/100k/train_day"
# coco_base = "/home/aghosh/Projects/2PCNet/Datasets/bdd100k/coco_labels/train_day.json"
# vp_base = "/home/aghosh/Projects/2PCNet/Datasets/VP/train_day.json"

# NOTE: use this for debug, switch to 
tod = 'night'
root_path = f"/home/aghosh/Projects/2PCNet/Datasets/bdd100k/images/100k/val_{tod}"
coco_base = f"/home/aghosh/Projects/2PCNet/Datasets/bdd100k/coco_labels/val_{tod}.json"
vp_base = "/home/aghosh/Projects/2PCNet/Datasets/VP/bdd100k_all_vp.json"

# Function to calculate the areas of bounding boxes
def calculate_bbox_areas(bboxes):
    # Calculate areas from bounding boxes in x1, y1, x2, y2 format
    areas = []
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        area = (x2 - x1) * (y2 - y1)
        areas.append(area)
    return areas

def save_data_to_txt(data, filename):
    np.savetxt(filename, data, fmt='%f')

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
    output_path = grid_net.__class__.__name__

    # Save the data to txt files
    save_data_to_txt(hist_data_orig, f"hists/hist_data_{tod}_orig.txt")

    if output_path == 'PlainKDEGrid':
        save_data_to_txt(hist_data_warped, f"hists/hist_data_{tod}_{output_path}_{warp_fovea_inst_scale}_{warp_fovea_inst_scale_l2}.txt")
    else:
        save_data_to_txt(hist_data_warped, f"hists/hist_data_{tod}_{output_path}.txt")

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
    grid_type = grid_net.__class__.__name__
    # print("grid_type is", grid_type)

    try:
        grid_fusion = grid_net.fusion_method
    except:
        grid_fusion = ''

    try:
        pyramid_layer = str(grid_net.pyramid_layer)
    except:
        pyramid_layer = ''
    
    # Adjusted this line to structure the directories
    if grid_type == 'PlainKDEGrid':
        output_dir = os.path.join(main_output_dir, grid_type, grid_fusion, pyramid_layer, f"l1_{warp_fovea_inst_scale}_l2_{warp_fovea_inst_scale_l2}")
    else:
        output_dir = os.path.join(main_output_dir, grid_type, grid_fusion, pyramid_layer)

    os.makedirs(output_dir, exist_ok=True)
    # print(f"output_dir is {output_dir}"); exit()

    bandwidth_scales = [64]  # can be modified based on your needs

    for bs in bandwidth_scales:
        try:
            warped_img, ins, grid = make_warp_aug(img_tensor, converted_bboxes, v_pts, grid_net, use_ins=use_ins, file_name=base_path)

            # unwarp_x = apply_unwarp(warped_img, grid, keep_size=True)
            
        except Exception as e:
            print(f"Error in make_warp_aug: {e}")
            print("v_pts is", v_pts)
            print("basename is", base_path)
            continue

        if save_flag:
            # Draw bounding boxes on the warped image
            image_with_bboxes = draw_bboxes_on_image(warped_img[0], ins)

            # Save the resulting image
            warped_img_name = os.path.join(output_dir, f"{base_path}.jpg") # NOTE: skip bs for now
            image_with_bboxes.save(warped_img_name)

        # NOTE: code for 
        # Calculate original bbox areas and add to the hist_data_orig list
        orig_areas = calculate_bbox_areas(converted_bboxes.cpu().numpy())
        hist_data_orig.extend(orig_areas)            

        # Calculate warped bbox areas and add to the hist_data_warped list
        warped_areas = calculate_bbox_areas(ins.cpu().numpy())
        hist_data_warped.extend(warped_areas)

        # print(f"orig_areas = {orig_areas}")
        # print(f"warped_areas = {warped_areas}")            


def convert_format(bboxes):
    # Splitting the tensor columns
    x1, y1, width, height = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    
    # Calculating x2 and y2
    x2 = x1 + width
    y2 = y1 + height
    
    # Stacking them into the new format
    new_bboxes = torch.stack([x1, y1, x2, y2], dim=1)
    
    return new_bboxes

def get_ground_truth_bboxes_coco(image_file_name, coco):
    # Get the image info for the specified image file name
    img_info = coco.imgs_by_name[image_file_name]

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


def draw_bboxes_on_image(image_tensor, bboxes_tensor, color=(255, 0, 0), thickness=5):
    # Convert tensor image to PIL Image for drawing
    image_pil = to_pil_image(image_tensor.cpu().detach())

    if use_ins:

        # Convert to NumPy array for OpenCV to use
        image_np = np.array(image_pil)

        # Convert RGB to BGR for OpenCV
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        # Draw each bounding box
        for box in bboxes_tensor:
            start_point = (int(box[0]), int(box[1]))  # (x1, y1)
            end_point = (int(box[2]), int(box[3]))  # (x2, y2)
            
            # Draw the rectangle on the image
            image_np = cv2.rectangle(image_np, start_point, end_point, color, thickness)

        # Convert back to RGB for OpenCV
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

        # Convert back to PIL image
        image_pil = to_pil_image(image_np)

    return image_pil


def load_vp_data(vp_base):
    return before_train_json(vp_base)

def get_v_pts_for_images(img_paths, vp_dict):
    return [torch.tensor(vp_dict[os.path.basename(img_path)]).cuda() for img_path in img_paths]

def get_gt_bboxes_for_images(img_paths, data_source, root_path):
    if is_coco_style:
        return [torch.tensor(get_ground_truth_bboxes_coco(os.path.basename(img_path), data_source)).cuda() for img_path in img_paths]
    else:
        return [torch.tensor(get_ground_truth_bboxes_simple(img_path, data_source, root_path)).cuda() for img_path in img_paths]


def main():
    # TODO: switch dataset to bdd100k (change images and coco labels at the same time)
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
        'FixedKDEGrid',
        'CuboidGlobalKDEGrid',
        'PlainKDEGrid',
    ]

    ################### Important hyperparameters ###################

    for grid_net_name in grid_net_names:
        # Check if the grid_net_name is 'PlainKDEGrid' and instantiate with specific parameters
        if grid_net_name == 'PlainKDEGrid':
            grid_net = PlainKDEGrid(warp_fovea_inst_scale=warp_fovea_inst_scale, 
                                    warp_fovea_inst_scale_l2=warp_fovea_inst_scale_l2).cuda()
        else:
            # Retrieve the class from globals() and instantiate it
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