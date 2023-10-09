import torch
import torch.nn.functional as F
from .invert_grid import invert_grid
from .reblur import is_out_of_bounds, get_vanising_points
from detectron2.data.transforms import ResizeTransform, HFlipTransform, NoOpTransform
from torchvision import utils as vutils
import sys
import os
import cv2
import numpy as np

# TODO: address no-gt in test case
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


def simple_test(grid_net, imgs, vanishing_point, bboxes=None):
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

    grid = grid_net(imgs, vanishing_point, bboxes)
    # print("grid shape", grid.shape)

    warped_imgs = F.grid_sample(imgs, grid, align_corners=True)

    return grid, warped_imgs



def make_warp_aug(img, ins, vanishing_point, grid_net, use_ins=False):

    # read image
    img = img.float()
    device = img.device
    my_shape = img.shape[-2:]
    imgs = img.unsqueeze(0) 

    # read bboxes
    try:
        bboxes = ins.gt_boxes.tensor
        bboxes = bboxes.to(device)
    except:
        bboxes = None

    # print("img.shape", img.shape) # [3, 600, 1067]
    # print("bboxes shape", bboxes.shape) # [N, 4]: x1, y1, x2, y2
    # print("vanishing_point", vanishing_point)

    grid, warped_imgs = simple_test(grid_net, imgs, vanishing_point, bboxes)

    if use_ins:

        # warp bboxes
        warped_bboxes = warp_bboxes(bboxes, grid, separable=True)

        # # NOTE: hardcode for debug only. Delete later
        # warped_bboxes = unwarp_bboxes(warped_bboxes, grid.squeeze(0), [600, 1067])

        # update ins
        # print("before ins.gt_boxes.tensor", ins.gt_boxes.tensor)
        ins.gt_boxes.tensor = warped_bboxes
        # print("after ins.gt_boxes.tensor", ins.gt_boxes.tensor)

    return warped_imgs, ins, grid
    


def apply_warp_aug(img, ins, vanishing_point, warp_aug=False, 
                    warp_aug_lzu=False, grid_net=None, keep_size=True):
    # print(f"img is {img.shape}") # [3, 600, 1067]
    grid = None

    # img_height, img_width = img.shape[-2:]
    
    # NOTE: this for debug only
    # if is_out_of_bounds(vanishing_point, img_width, img_height):
    #     print("both vp coords OOB. Training Stops!!!!")
    #     sys.exit(1)
        # return img, ins, grid
    if warp_aug:
        img, ins, grid = make_warp_aug(img, ins, vanishing_point, grid_net, use_ins=True)
    elif warp_aug_lzu:
        img, ins, grid = make_warp_aug(img, ins, vanishing_point, grid_net, use_ins=False)

    # reshape 4d to 3d
    if (len(img.shape) == 4) and keep_size:
        img = img.squeeze(0) 

    return img, ins, grid



def apply_unwarp(warped_x, grid, keep_size=True):
    if (len(warped_x.shape) == 3) and keep_size:
        warped_x = warped_x.unsqueeze(0)

    # print(f'warped_x is {warped_x.shape} grid is {grid.shape}') # [1, 3, 600, 1067], [1, 600, 1067, 2]

    # Compute inverse_grid
    inverse_grid = invert_grid(grid, warped_x.shape, separable=True)[0:1]

    # Expand inverse_grid to match batch size
    B = warped_x.shape[0]
    inverse_grid = inverse_grid.expand(B, -1, -1, -1)

    # Perform unzoom (TODO: might consider using bicubic)
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



def process_and_update_features(batched_inputs, images, warp_aug_lzu, vp_dict, grid_net, backbone, 
                                warp_debug=False, warp_image_norm=False, warp_aug=False):
    '''
    '''
    # print(f"batched_inputs = {batched_inputs}")   # list: [...]
    # print(f"images = {images.tensor}")            # detectron2.structures.image_list.ImageList
    # print(f"warp_aug_lzu = {warp_aug_lzu}")       # bool: True/False
    # print(f"vp_dict = {vp_dict}")                 # dict: {'xxx.jpg': [vpw, vph], ...}
    # print(f"grid_net = {grid_net}")               # CuboidGlobalKDEGrid
    # print(f"backbone = {backbone}")               # ResNet
    # print(f"warp_debug = {warp_debug}")           # bool: True/False
    # print(f"warp_image_norm = {warp_image_norm}") # bool: True/False
    
    # Preprocessing
    img_height, img_width = images.tensor.shape[-2:]
    ori_height, ori_width = batched_inputs[0]['height'], batched_inputs[0]['width']
    # vanishing_points = [
    #     get_vanising_points(
    #         sample['file_name'], 
    #         vp_dict, 
    #         *extract_ratio_and_flip(sample['transform'])
    #     ) for sample in batched_inputs
    # ]
    vanishing_points = []
    for sample in batched_inputs:
        if 'transform' in sample:
            ratio, flip = extract_ratio_and_flip(sample['transform'])
        else:
            # Handle the absence of 'transform' key in test case
            ratio, flip = img_height/ori_height, False  # Or any default value suitable for your case
        # print("ratio is ", ratio)
            
        vp = get_vanising_points(sample['file_name'], vp_dict, ratio, flip)
        vanishing_points.append(vp)

    # # Correcting vanishing_points if they are out of bounds
    # if not warp_aug:
    #     corrected_vanishing_points = []
    #     for vp in vanishing_points:
    #         if is_out_of_bounds(vp, img_width, img_height):
    #             vp_x = min(max(1, vp[0]), img_width - 2)
    #             vp_y = min(max(1, vp[1]), img_height - 2)
    #             print(f"Vanishing point {vp} was out of bounds and has been clipped to {[vp_x, vp_y]}")
    #             corrected_vanishing_points.append([vp_x, vp_y])
    #         else:
    #             corrected_vanishing_points.append(vp)
        
    #     vanishing_points = corrected_vanishing_points  # Update the vanishing_points list

    # Mask to check which vanishing points are out of bounds
    out_of_bounds_mask = [is_out_of_bounds(vp, img_width, img_height) for vp in vanishing_points]

    # 1. Apply warping for the images with in-bounds vanishing points
    warped_images_list = []
    grids_list = []

    for (image, vp, sample, is_out) in zip(images.tensor, vanishing_points, batched_inputs, out_of_bounds_mask):
        if not is_out:  # If the vanishing point is in-bounds
            warped_image, _, grid = apply_warp_aug(image, sample.get('instances', None), vp, warp_aug, warp_aug_lzu, grid_net)
            warped_images_list.append(warped_image)
            grids_list.append(grid)
        else:  # If the vanishing point is out-of-bounds, simply append the original image and None for the grid
            warped_images_list.append(image)
            grids_list.append(None)

    warped_images = torch.stack(warped_images_list)

    # Apply warping
    # warped_images, _, grids = zip(*[
    #     apply_warp_aug(image, None, vp, False, warp_aug_lzu, grid_net) 
    #     for image, vp in zip(images.tensor, vanishing_points)
    # ])
    # # NOTE: gt bboxes already updated in apply_warp_aug => no need to return
    # warped_images, _, grids = zip(*[
    #     apply_warp_aug(image, sample.get('instances', None), vp, warp_aug, warp_aug_lzu, grid_net) 
    #     for image, vp, sample in zip(images.tensor, vanishing_points, batched_inputs)
    # ])
    # warped_images = torch.stack(warped_images)

    # Normalize warped images
    if warp_image_norm:
        warped_images = torch.stack([(img - img.min()) / (img.max() - img.min()) * 255 for img in warped_images])

    # debug images
    # print("len batched_inputs", len(batched_inputs)) # BS
    # print("warped_images", warped_images.shape) # [BS, C, H, W]
    concat_and_save_images(batched_inputs, warped_images, debug=warp_debug, warp_aug=warp_aug)

    # Call the backbone
    features = backbone(warped_images)

    # 2. Apply unwarping for the images with in-bounds vanishing points if warp_aug is False
    if not warp_aug:
        feature_key = next(iter(features))
        unwarped_features_list = []

        for feature, grid, is_out in zip(features[feature_key], grids_list, out_of_bounds_mask):
            if not is_out and grid is not None:  # Only apply unwarp if the vanishing point was in-bounds and we have a valid grid
                unwarped_features_list.append(apply_unwarp(feature, grid))
            else:
                unwarped_features_list.append(feature)

        # Replace the original features with unwarped ones
        features[feature_key] = torch.stack(unwarped_features_list)

    return features


def concat_and_save_images(batched_inputs, warped_images, debug=False, warp_aug=None):
    """
    Concatenate original and warped images side by side and save them.

    Inputs:
    - batched_inputs: List of dictionaries containing image information, including 'image' for the original image.
    - warped_images: List of warped image tensors.
    - debug: Boolean flag to determine whether to save the images or not.

    Outputs:
    - None. This function saves the concatenated images with bounding boxes if debug is True.
    """
    cnt = 0
    if debug:
        for (input_info, warped_img) in zip(batched_inputs, warped_images):
            original_img = input_info['image'].cuda()  # Access the original image
            warped_ins = input_info['instances'].gt_boxes.tensor

            # Convert tensors to numpy arrays
            original_img = original_img.cpu().numpy()
            warped_img = warped_img.cpu().numpy()

            # transpose to (H, W, C)
            original_img = original_img.transpose(1, 2, 0)
            warped_img = warped_img.transpose(1, 2, 0)

            # Switch from BGR to RGB
            original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            warped_img = cv2.cvtColor(warped_img, cv2.COLOR_BGR2RGB)

            # Normalize the images
            original_img = (original_img - original_img.min()) / (original_img.max() - original_img.min())
            warped_img = (warped_img - warped_img.min()) / (warped_img.max() - warped_img.min())

            # convert to uint8 and multiply by 255
            original_img = (original_img * 255).astype(np.uint8)
            warped_img = (warped_img * 255).astype(np.uint8)

            # Overlay bounding boxes on the warped image
            for box in warped_ins:
                x1, y1, x2, y2 = map(int, box)  # Convert to integers
                if warp_aug:
                    cv2.rectangle(warped_img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw bounding box in green
                else:
                    cv2.rectangle(original_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Concatenate images along the width dimension
            original_img = original_img.astype(warped_img.dtype)
            # print("original_img.shape", original_img.shape)
            # print("warped_img.shape", warped_img.shape)
            # print("original_img.dtype", original_img.dtype)
            # print("warped_img.dtype", warped_img.dtype)
            combined_image = cv2.hconcat([original_img, warped_img])

            # Save images to the output path
            file_name = os.path.basename(input_info['file_name'])  # Extract the original file name
            file_name_without_extension, file_extension = os.path.splitext(file_name)

            warp_out_dir = 'warped_images'
            os.makedirs(warp_out_dir, exist_ok=True)

            # Append cnt to the file name before the extension
            save_path = os.path.join(warp_out_dir, f"{cnt}_{file_name_without_extension}{file_extension}")

            # Convert the combined image back to BGR format for saving with OpenCV
            combined_image_bgr = cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR)

            # Save the combined image
            cv2.imwrite(save_path, combined_image_bgr)
            cnt += 1
        
        print(f"Saved {cnt} images!")

        sys.exit(1)