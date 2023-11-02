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

# TODO: maybe the empty bboxes are causing the error?

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

def warp_image_exp(img_path, v_pts, gt_bboxes, grid_net, save_flag=False, warp_fovea_inst_scale=False):

    # Always point to the main saving directory
    main_output_dir = "warped_images_exp"
    if not os.path.exists(main_output_dir):
        os.makedirs(main_output_dir)

    # Check if bboxes is empty and return None
    if gt_bboxes.nelement() == 0:
        print("Empty gt_bboxes detected. Exiting function.")
        return None        

    # Convert the bbox format
    converted_bboxes = convert_format(gt_bboxes).clone()

    # get image base name without extension
    base_path = os.path.splitext(os.path.basename(img_path))[0]

    # Load the image as a torch tensor
    img = Image.open(img_path).convert("RGB")
    transform = transforms.Compose([transforms.ToTensor()])
    img_tensor = transform(img).cuda().unsqueeze(0)  # Load, convert, and add batch dimension

    # Constants
    grid_type = grid_net.__class__.__name__

    try:
        grid_fusion = grid_net.fusion_method
    except:
        grid_fusion = ''

    try:
        pyramid_layer = str(grid_net.pyramid_layer)
    except:
        pyramid_layer = ''
    
    # Adjusted this line to structure the directories
    output_dir = os.path.join(main_output_dir, grid_type, grid_fusion, pyramid_layer)
    os.makedirs(output_dir, exist_ok=True)

    bandwidth_scales = [64]  # can be modified based on your needs

    for bs in bandwidth_scales:
        try:
            warped_img, _, grid = make_warp_aug(img_tensor, converted_bboxes, v_pts, grid_net, use_ins=False, file_name=base_path)

            # unwarp_x = apply_unwarp(warped_img, grid, keep_size=True)
            
        except Exception as e:
            print(f"Error in make_warp_aug: {e}")
            print("v_pts is", v_pts)
            print("basename is", base_path)
            continue

        if save_flag:
            warped_img_name = os.path.join(output_dir, f"{base_path}_{bs}_{warp_fovea_inst_scale}.jpg")
            save_image(warped_img[0], warped_img_name)  # Assuming warped_img is [1, C, H, W], take the 0-th index
            # print(f"Saved warped image to: {warped_img_name}")


def load_vp_data(vp_base):
    return before_train_json(vp_base)

def get_v_pts_for_images(img_paths, vp_dict):
    return [torch.tensor(vp_dict[os.path.basename(img_path)]).cuda() for img_path in img_paths]

def get_gt_bboxes_for_images(img_paths, data_source, is_coco_style, root_path):
    if is_coco_style:
        return [torch.tensor(get_ground_truth_bboxes_coco(os.path.basename(img_path), data_source)).cuda() for img_path in img_paths]
    else:
        return [torch.tensor(get_ground_truth_bboxes_simple(img_path, data_source, root_path)).cuda() for img_path in img_paths]

def process_images(img_paths, v_pts_list, gt_bboxes_list, grid_net, save_flag=False, warp_fovea_inst_scale=False):
    for index in range(len(img_paths)):
        img_path = img_paths[index]
        v_pts = v_pts_list[index]
        gt_bboxes = gt_bboxes_list[index]
        warp_image_exp(img_path, v_pts, gt_bboxes, grid_net, save_flag, warp_fovea_inst_scale)

        if index % 100 == 0:
            print(f"Done with {index} images")


def main():
    root_path = "/home/aghosh/Projects/2PCNet/Datasets/cityscapes/leftImg8bit/train"
    img_paths = glob.glob(f'{root_path}/**/*.[jJ][pP][gG]', recursive=True) + glob.glob(f'{root_path}/**/*.[pP][nN][gG]', recursive=True)

    coco_base = "/home/aghosh/Projects/2PCNet/Datasets/cityscapes_seg2det.json"
    vp_base = "/home/aghosh/Projects/2PCNet/Datasets/VP/ind_vps/cityscapes_all_vp.json"
    
    vp_dict = load_vp_data(vp_base)
    v_pts_list = get_v_pts_for_images(img_paths, vp_dict)

    is_coco_style = False  # You can set this to False if not in COCO style

    if is_coco_style:
        coco = COCO(coco_base)
        coco.imgs_by_name = {img['file_name']: img for img in coco.dataset['images']}
        gt_bboxes_list = get_gt_bboxes_for_images(img_paths, coco, is_coco_style, root_path)
    else:
        with open(coco_base, 'r') as f:
            bbox_dict = json.load(f)
        gt_bboxes_list = get_gt_bboxes_for_images(img_paths, bbox_dict, is_coco_style, root_path)


    grid_net_names = [
        'FixedKDEGrid',
        'CuboidGlobalKDEGrid',
        'PlainKDEGrid',
    ]

    ################### Important hyperparameters ###################
    save_flag = False  # True or False
    warp_fovea_inst_scale = False  # True or False

    for grid_net_name in grid_net_names:
        # Retrieve the class from globals() and instantiate it
        grid_net_class = globals()[grid_net_name]
        grid_net = grid_net_class().cuda()  # Instantiate and move to GPU

        # Use the instantiated grid_net in your process_images function
        process_images(img_paths, 
                        v_pts_list, 
                        gt_bboxes_list, 
                        grid_net, 
                        save_flag=save_flag, 
                        warp_fovea_inst_scale=warp_fovea_inst_scale)


if __name__ == "__main__":
    main()