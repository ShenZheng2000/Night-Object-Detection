import torch
import torch.nn.functional as F
from .invert_grid import invert_grid
from .reblur import is_out_of_bounds, get_vanising_points, update_vp_ins
from torchvision import utils as vutils

from .grid_generator import CuboidGlobalKDEGrid, FixedKDEGrid, PlainKDEGrid, MixKDEGrid, MidKDEGrid, FixedKDEGrid_New

import sys
import os
import json
import torch


def read_seg_to_det(SEG_TO_DET):
    if SEG_TO_DET is not None:
        with open(SEG_TO_DET, 'r') as f:
            seg_to_det = json.load(f)
    else:
        seg_to_det = None
    return seg_to_det


# NOTE: read vp from here
def before_train_json(VP): 
    # print("VANISHING_POINT is", VP)
    if VP is not None:
        with open(VP, 'r') as f:
            vanishing_point = json.load(f)
        vanishing_point = {os.path.basename(k): v for k, v in vanishing_point.items()}
    else:
        vanishing_point = None
    # print("vanishing_point is", vanishing_point)
    return vanishing_point


def build_grid_net(warp_aug_lzu, warp_fovea, warp_fovea_inst, warp_fovea_mix, warp_middle, warp_scale,
                   warp_fovea_center=False, warp_fovea_inst_scale=False, warp_fovea_inst_scale_l2=False,
                   fusion_method='max', pyramid_layer=2, is_seg=False,
                   bandwidth_scale=64, amplitude_scale=1.0):
    if warp_aug_lzu:
        # NOTE: remove saliency for now, because it has been set by default
        # saliency_file = 'dataset_saliency.pkl'
        if warp_fovea:
            return FixedKDEGrid(warp_scale=warp_scale)
        elif warp_fovea_center:
            return FixedKDEGrid_New(warp_scale=warp_scale)
        elif warp_fovea_inst:
            return PlainKDEGrid(warp_scale=warp_scale, 
                                warp_fovea_inst_scale=warp_fovea_inst_scale,
                                warp_fovea_inst_scale_l2=warp_fovea_inst_scale_l2,
                                bandwidth_scale=bandwidth_scale,
                                amplitude_scale=amplitude_scale,
                                )
        elif warp_fovea_mix:
            return MixKDEGrid(warp_scale=warp_scale, 
                              fusion_method=fusion_method, 
                              pyramid_layer=pyramid_layer,
                              is_seg=is_seg)
        elif warp_middle:
            return MidKDEGrid(warp_scale)
        else:
            return CuboidGlobalKDEGrid(warp_scale)
    else:
        return None

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


def simple_test(grid_net, imgs, vanishing_point, bboxes=None, file_name=None, use_flip=False):
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

    # print("imgs.shape is", imgs.shape)
    # print("vanishing_point is", vanishing_point)
    # print("bboxes is", bboxes)
    if isinstance(grid_net, MixKDEGrid):
        grid = grid_net(imgs, vanishing_point, bboxes, file_name=file_name, use_flip=use_flip)
    else:
        grid = grid_net(imgs, vanishing_point, bboxes)
    # print("grid shape", grid.shape)

    warped_imgs = F.grid_sample(imgs, grid, align_corners=True)

    # print("imgs.shape", imgs.shape)

    return grid, warped_imgs



def make_warp_aug(img, ins, vanishing_point, grid_net, use_ins=False, file_name=None, use_flip=False):

    # read image
    img = img.float()
    device = img.device
    my_shape = img.shape[-2:]

    if len(img.shape) == 3:
        imgs = img.unsqueeze(0)
    else:
        imgs = img

    # read bboxes
    if isinstance(ins, torch.Tensor):
        bboxes = ins.to(device)
    elif isinstance(ins, list):
        bboxes = torch.tensor(ins).to(device)
    elif ins is None:
        bboxes = None
    else:
        bboxes = ins.gt_boxes.tensor
        bboxes = bboxes.to(device)

    # print("img.shape", img.shape) # [3, 600, 1067]
    # print("bboxes shape", bboxes.shape) # [N, 4]: x1, y1, x2, y2
    # print("vanishing_point", vanishing_point)

    grid, warped_imgs = simple_test(grid_net, imgs, vanishing_point, bboxes, file_name=file_name, use_flip=use_flip)

    if use_ins:

        # warp bboxes
        warped_bboxes = warp_bboxes(bboxes, grid, separable=True)

        # # NOTE: hardcode for debug only. Delete later
        # warped_bboxes = unwarp_bboxes(warped_bboxes, grid.squeeze(0), [600, 1067])

        # update ins
        # print("before ins.gt_boxes.tensor", ins.gt_boxes.tensor)
        # print("before; bboxes", bboxes)
        try:
            ins.gt_boxes.tensor = warped_bboxes
        except:
            ins = warped_bboxes
        # print("after; warped_bboxes", warped_bboxes)
        # print("after ins.gt_boxes.tensor", ins.gt_boxes.tensor)

    return warped_imgs, ins, grid
    


def apply_warp_aug(img, ins, vanishing_point, warp_aug=False, 
                    warp_aug_lzu=False, grid_net=None, keep_size=True, 
                    file_name=None, use_flip=False):
    # print(f"img is {img.shape}") # [3, 600, 1067]
    grid = None

    # print("start apply_warp_aug")

    img_height, img_width = img.shape[-2:]
    
    # NOTE: this for debug only
    if vanishing_point and is_out_of_bounds(vanishing_point, img_width, img_height):
        print("Warning: both vp coords OOB. !!!!")
        vanishing_point = (img_width // 2, img_height // 2)  # Set vanishing point to image center
        # sys.exit(1)
        # return img, ins, grid
    if warp_aug:
        img, ins, grid = make_warp_aug(img, ins, vanishing_point, grid_net, use_ins=True, file_name=file_name, use_flip=use_flip)
    elif warp_aug_lzu:
        # print("BEFORE, ins is", ins.gt_boxes.tensor)
        # print("img shape is", img.shape)
        img, ins, grid = make_warp_aug(img, ins, vanishing_point, grid_net, use_ins=False, file_name=file_name, use_flip=use_flip)

    # reshape 4d to 3d
    if (len(img.shape) == 4) and keep_size:
        img = img.squeeze(0) 

    return img, ins, grid


# NOTE: for unwarp vis debug only
def save_grid(input_tensor, save_path):
    """
    Convert a PyTorch tensor of shape [1, H, W, 2] to [1, 1, H, W] (grayscale) and save it.

    Args:
        input_tensor (torch.Tensor): Input tensor of shape [1, H, W, 2].
        save_path (str): File path to save the grayscale tensor.
    """
    # check image min and max value
    # print(f"input_tensor min {input_tensor.min()} max {input_tensor.max()}") # [-1, 1]

    # Transpose the tensor to change its shape from [1, H, W, 2] to [1, 2, H, W]
    transposed_tensor = input_tensor.permute(0, 3, 1, 2)

    # Extract the first component (grayscale) by selecting the first channel
    grayscale_tensor = transposed_tensor[:, 0:1, :, :]

    # Save the grayscale tensor using torch.save() or any other desired method
    vutils.save_image(grayscale_tensor, save_path, normalize=True)


def save_feature(image1, image2, save_path):
    # Calculate the mean along the second dimension for both images
    averaged_image1 = torch.mean(image1, dim=1, keepdim=True)
    averaged_image2 = torch.mean(image2, dim=1, keepdim=True)

    # Concatenate the averaged images side by side along the width dimension (dimension 3)
    concatenated_image = torch.cat((averaged_image1, averaged_image2), dim=3)

    # Save the concatenated image as an image using torchvision's save_image
    vutils.save_image(concatenated_image, save_path)

# # TODO: This is for vis debug only
# def apply_unwarp(warped_x, grid, keep_size=True):
#     if (len(warped_x.shape) == 3) and keep_size:
#         warped_x = warped_x.unsqueeze(0)

#     # Process for both keep_grid = True and False
#     for keep_grid in [True, False]:
#         grid_suffix = "keep_grid" if keep_grid else "no_keep_grid"

#         if keep_grid == True:
#             # First downsample grid to match warped_x, and directly inverse grid
#             grid_resized = grid.permute(0, 3, 1, 2)
#             grid_resized = F.interpolate(grid_resized, size=(warped_x.shape[-2], warped_x.shape[-1]), mode='bilinear', align_corners=True)
#             grid_resized = grid_resized.permute(0, 2, 3, 1)
#             inverse_grid = invert_grid(grid_resized, warped_x.shape, separable=True)[0:1]
#         elif keep_grid == False:
#             inverse_grid = invert_grid(grid, warped_x.shape, separable=True)[0:1]

#         # Save the inverse grid
#         save_grid(inverse_grid, f'feat_vis/inverse_grid_{grid_suffix}.png')

#         # Expand inverse_grid to match batch size
#         B = warped_x.shape[0]
#         inverse_grid = inverse_grid.expand(B, -1, -1, -1)

#         # Perform unzoom
#         unwarped_x = F.grid_sample(
#             warped_x, inverse_grid, mode='bilinear',
#             align_corners=True, padding_mode='zeros'
#         )

#         # Save the unwarped features
#         save_feature(warped_x, unwarped_x, f'feat_vis/warp_unwarp_feat_{grid_suffix}.png')
    
#     exit()

#     if (len(unwarped_x.shape) == 4) and keep_size:
#         unwarped_x = unwarped_x.squeeze(0)

#     return unwarped_x


# NOTE: This is the running version
def apply_unwarp(warped_x, grid, keep_size=True, keep_grid=False):
    if (len(warped_x.shape) == 3) and keep_size:
        warped_x = warped_x.unsqueeze(0)

    # print(f'warped_x is {warped_x.shape}; grid is {grid.shape}') # [1, 2048, 38, 67]; [1, 600, 1067, 2]
    # print(f'warped_x is {warped_x.dtype}; grid is {grid.dtype}') # torch.float32, torch.float32

    if keep_grid:
        # First downsample grid to match warped_x, and directly inverse grid
        grid_resized = grid.permute(0, 3, 1, 2)
        grid_resized = F.interpolate(grid_resized, size=(warped_x.shape[-2], warped_x.shape[-1]), mode='bilinear', align_corners=True)
        grid_resized = grid_resized.permute(0, 2, 3, 1)
        inverse_grid = invert_grid(grid_resized, warped_x.shape, separable=True)[0:1]
    else:
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
    from detectron2.data.transforms import ResizeTransform, HFlipTransform, NoOpTransform
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

    # NOTE: skip for for now, since all vps are valid
    # # Correcting vanishing_points if they are out of bounds
    # corrected_vanishing_points = []
    # for vp in vanishing_points:
    #     if is_out_of_bounds(vp, img_width, img_height):
    #         vp_x = min(max(1, vp[0]), img_width - 2)
    #         vp_y = min(max(1, vp[1]), img_height - 2)
    #         print(f"Vanishing point {vp} was out of bounds and has been clipped to {[vp_x, vp_y]}")
    #         corrected_vanishing_points.append([vp_x, vp_y])
    #     else:
    #         corrected_vanishing_points.append(vp)
    
    # vanishing_points = corrected_vanishing_points  # Update the vanishing_points list

    # Apply warping
    # warped_images, _, grids = zip(*[
    #     apply_warp_aug(image, None, vp, False, warp_aug_lzu, grid_net) 
    #     for image, vp in zip(images.tensor, vanishing_points)
    # ])
    # NOTE: gt bboxes already updated in apply_warp_aug => no need to return
    # NOTE: add file_name to avoid dup scaling for the same image

    warped_images, _, grids = zip(*[
        apply_warp_aug(image, 
                       sample.get('instances', None), 
                       vp, 
                       warp_aug, 
                       warp_aug_lzu, 
                       grid_net) 
        for image, vp, sample in zip(images.tensor, vanishing_points, batched_inputs)
    ])
    warped_images = torch.stack(warped_images)

    # NOTE: scale images and ins if necessary
    if grid_net.warp_scale != 1.0:
        
        # Import ImageList
        from detectron2.structures import ImageList

        # Scale images
        scaled_image_tensor = F.interpolate(images.tensor, scale_factor=grid_net.warp_scale, mode='bilinear', align_corners=False)
        images = ImageList(scaled_image_tensor, images.image_sizes) # TODO: hardcode size for now since images.image_sizes is not used later

        # Scale ins
        processed_files = set()

        for sample in batched_inputs:
            ins = sample.get('instances', None)

            file_name = sample.get('file_name')
            if file_name in processed_files:
                continue  # Skip this entry if it's been processed already
            else:
                processed_files.add(file_name)  # Add the file_name to the set

            # print("before", ins.gt_boxes.tensor)
            ins.gt_boxes.tensor *= grid_net.warp_scale
            # print("after", ins.gt_boxes.tensor)

            height, width = ins.image_size
            # print(f"height = {height}, width = {width}")
            ins._image_size = (height*grid_net.warp_scale, 
                            width*grid_net.warp_scale)
            # new_height, new_width = ins.image_size; print(f"new_height = {new_height}, new_width = {new_width}")
        
    # Normalize warped images
    if warp_image_norm:
        warped_images = torch.stack([(img - img.min()) / (img.max() - img.min()) * 255 for img in warped_images])

    # debug images
    # print("len batched_inputs", len(batched_inputs)) # BS
    # print("warped_images", warped_images.shape) # [BS, C, H, W]
    concat_and_save_images(batched_inputs, warped_images, debug=warp_debug)

    # Call the backbone
    features = backbone(warped_images)

    # print("Before features['res5']", features['res5'].shape) # [BS, C, H, W]

    # Apply unwarping
    if not warp_aug:
        feature_key = next(iter(features))
        unwarped_features = torch.stack([
            apply_unwarp(feature, grid)
            for feature, grid in zip(features[feature_key], grids)
        ])
        # Replace the original features with unwarped ones
        features[feature_key] = unwarped_features

    # print("After features['res5']", features['res5'].shape) # [BS, C, H, W]
    
    return features, images



# def add_bbox_to_meta(self, img_metas):
#     if img_metas is None:
#         return img_metas
    
#     if self.seg_to_det is None:
#         return img_metas
    
#     for meta in img_metas:
#         bboxes = self.seg_to_det.get(meta['ori_filename'], [])
#         meta['instances'] = bboxes
    
#     return img_metas


def process_mmseg(batched_inputs, images, warp_aug_lzu, vp_dict, grid_net, backbone, 
                warp_debug=False, warp_image_norm=False, warp_aug=False, seg_to_det=None, keep_grid=False):
    '''
    '''
    # print(f"batched_inputs = {batched_inputs}")   # list: [...]
    # print(f"images = {images.shape}")            # torch tensor
    # print(f"warp_aug_lzu = {warp_aug_lzu}")       # bool: True/False
    # print(f"vp_dict = {vp_dict}")                 # dict: {'xxx.jpg': [vpw, vph], ...}
    # print(f"grid_net = {grid_net}")               # CuboidGlobalKDEGrid
    # print(f"backbone = {backbone}")               # ResNet
    # print(f"warp_debug = {warp_debug}")           # bool: True/False
    # print(f"warp_image_norm = {warp_image_norm}") # bool: True/False
    
    # Preprocessing
    img_height, img_width = images.shape[-2:]
    # print("process_mmseg; images.shape", images.shape)
    # print(f"batched_inputs[0] is", batched_inputs[0]['filename'])
    # print(f"img_height = {img_height}, img_width = {img_width}") # 512, 1024

    # ori_height, ori_width = batched_inputs[0]['height'], batched_inputs[0]['width']
    # print("batched_inputs is", batched_inputs)
    ori_height, ori_width, _ = batched_inputs[0]['ori_shape']
    # print(f"ori_height = {ori_height}, ori_width = {ori_width}") # 1024, 2048

    vanishing_points = []
    for sample in batched_inputs:
        # Handle the absence of 'transform' key in test case
        ratio = img_height/ori_height  # Or any default value suitable for your case
        # print("ratio is ",  ratio) # 0.5
        # print("flip is", flip) # True/False

        # NOTE: disable vp_dict for now since it's not used
        update_vp_ins(sample, ratio=ratio, img_width=img_width, seg_to_det=seg_to_det)

        # vanishing_points.append(sample['vanishing_point'])
        # NOTE: Assign None to each vanishing point (highlighted change)
        vanishing_points.append(None)  # <--- This is the new line

        # print("vp ==", sample['vanishing_point'])

    # print("start apply_warp_aug")
    warped_images, _, grids = zip(*[
        apply_warp_aug(img = image, 
                       ins = sample.get('instances', None), 
                       vanishing_point = vp, 
                       warp_aug = warp_aug, 
                       warp_aug_lzu = warp_aug_lzu, 
                       grid_net = grid_net,
                       file_name = os.path.basename(sample['filename']),
                       use_flip = sample['flip']) 
        for image, vp, sample in zip(images, vanishing_points, batched_inputs)
    ])
    warped_images = torch.stack(warped_images)
    # print("end apply_warp_aug")

    # NOTE: scale images and ins if necessary
    images, batched_inputs = scale_images_and_metadata(images, batched_inputs, grid_net.warp_scale)

    # print("batched_inputs[0]['img_shape']", batched_inputs[0]['img_shape'])
    # print("batched_inputs[0]['pad_shape']", batched_inputs[0]['pad_shape'])
    # print("batched_inputs[0]['scale_factor']", batched_inputs[0]['scale_factor'])

    # print("warped_images shape", warped_images.shape) # [2, 3, 512, 1024]
    if warp_debug:
        print("using warp_debug!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        first_image = images[0]
        first_warped_image = warped_images[0]
        cat_image = torch.cat([first_image, first_warped_image], dim=2)
        vutils.save_image(cat_image, f"visuals/warped_images_{grid_net.__class__.__name__}.png", normalize=True)
        exit()
        
    # Normalize warped images
    if warp_image_norm:
        warped_images = torch.stack([(img - img.min()) / (img.max() - img.min()) * 255 for img in warped_images])

    # debug images
    # print("len batched_inputs", len(batched_inputs)) # BS
    # print("warped_images", warped_images.shape) # [BS, C, H, W]
    # concat_and_save_images(batched_inputs, warped_images, debug=warp_debug)

    # Call the backbone
    # print("start backbone")
    features = backbone(warped_images) # a list with many torch tensors
    # print("end backbone")

    # Apply unwarping to all feature levels
    # print("start unwarp")
    if not warp_aug:

        for idx, feature_level in enumerate(features):
            unwarped_list = []
                
            for batch_idx in range(feature_level.size(0)):  # Loop over the batch size
                feature = feature_level[batch_idx]
                grid = grids[batch_idx]
                # print(f"feature shape = {feature.shape}") # [64, 128, 256]
                # print(f"grid shape = {grid.shape}") # [1, 512, 1024, 2]
                try:
                    unwarped = apply_unwarp(feature, grid, keep_grid=keep_grid)
                    unwarped_list.append(unwarped)
                except:
                    # Handle the error and process original images
                    print("Error encountered during unwarping. Images in this batch are:")
                    for img_data in batched_inputs:
                        print(os.path.basename(img_data['filename']))
                    return handle_unwarp_exception(images, backbone)                

            # Stack the results along the batch dimension again
            features[idx] = torch.stack(unwarped_list)
            # print(f"idx = {idx}, features[idx] shape = {features[idx].shape}")
    # print("end unwarp")

    return features, images, batched_inputs

def scale_images_and_metadata(images, batched_inputs, warp_scale):
    """
    Scales images and updates 'img_shape', 'pad_shape', and 'scale_factor' in metadata.

    Args:
    - images (torch.Tensor): The batch of images to be scaled.
    - batched_inputs (list[dict]): List of metadata dictionaries for each image.
    - warp_scale (float): The scaling factor.

    Returns:
    - torch.Tensor: The scaled images.
    - list[dict]: The updated metadata dictionaries.
    """
    if warp_scale == 1.0:
        return images, batched_inputs  # No scaling needed if warp_scale is 1.0

    # Scale images using bilinear interpolation
    scaled_images = F.interpolate(images, scale_factor=warp_scale, mode='bilinear', align_corners=False)

    # Scale image metadata accordingly
    for meta in batched_inputs:
        meta['img_shape'] = tuple(int(dim * warp_scale) for dim in meta['img_shape'])
        meta['pad_shape'] = tuple(int(dim * warp_scale) for dim in meta['pad_shape'])
        meta['scale_factor'] = meta['scale_factor'] * warp_scale

    return scaled_images, batched_inputs


def handle_unwarp_exception(images, backbone):
    # If there's an error during unwarping, use original images to get features
    features = backbone(images)
    return features, images

def concat_and_save_images(batched_inputs, warped_images, debug=False):
    cnt = 0
    if debug:
        for (input_info, warped_img) in zip(batched_inputs, warped_images):
            original_img = input_info['image'].cuda()
            
            # Switch from BGR to RGB
            original_img = torch.flip(original_img, [0])
            warped_img = torch.flip(warped_img, [0])
            
            # Debugging: Print the sizes and check for mismatch
            print(f"Original Image Size: {original_img.shape}")
            print(f"Warped Image Size: {warped_img.shape}")
            
            if original_img.shape != warped_img.shape:
                print("Resizing original image to match warped image dimensions.")
                original_img = torch.nn.functional.interpolate(original_img.unsqueeze(0), 
                                                               size=(warped_img.shape[1], 
                                                                     warped_img.shape[2])).squeeze(0)
            
            # Normalize the images
            original_img = (original_img - original_img.min()) / (original_img.max() - original_img.min())
            warped_img = (warped_img - warped_img.min()) / (warped_img.max() - warped_img.min())
            
            combined_image = torch.cat((original_img, warped_img), dim=2)

            # Save images to output path
            file_name = os.path.basename(input_info['file_name'])
            file_name_without_extension, file_extension = os.path.splitext(file_name)

            warp_out_dir = 'warped_images'
            os.makedirs(warp_out_dir, exist_ok=True)
            
            save_path = os.path.join(warp_out_dir, f"{cnt}_{file_name_without_extension}{file_extension}")
            
            vutils.save_image(combined_image, save_path, normalize=True)
            cnt += 1

        print(f"Saved {cnt} images!")
        sys.exit(1)


if __name__ == "__main__":
    json_path = "/home/aghosh/Projects/2PCNet/Datasets/cityscapes_seg2det.json"
    seg_to_det = read_seg_to_det(json_path)

    for idx, (image_path, bboxes) in enumerate(seg_to_det.items()):
        print(f"Image Path: {image_path}")
        for j, bbox in enumerate(bboxes, 1):
            print(f"    Bounding Box {j}: Top-Left ({bbox[0]}, {bbox[1]}), Bottom-Right ({bbox[2]}, {bbox[3]})")
        print("\n")  # Separate different image data with a newline
        
        if idx == 2:  # Stop after 3 items
            break