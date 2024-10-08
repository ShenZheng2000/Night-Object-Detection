# from flow_utils import flow_to_numpy_rgb
# from reblur_package import FlowBlurrer
from .flow_utils import flow_to_numpy_rgb
import torch
import time
import random
import os
import copy

def vp_flow_generator(im, v_pts, T_z, zeta):
    '''
    im: image
    v_pts: vanishing point
    T_z: strength of the motion
    zeta: variability of the motion
    '''
    B, C, H, W = im.size()
    flow = torch.zeros(B, 2, H, W, device=im.device)
    
    # # Create coordinate grids
    # grid_x, grid_y = torch.meshgrid(torch.arange(H), torch.arange(W))
    # loc = torch.stack([grid_x, grid_y], dim=0).float().cuda()
    grid_x, grid_y = torch.meshgrid(torch.arange(H, device=im.device), torch.arange(W, device=im.device))
    loc = torch.stack([grid_x, grid_y], dim=0).float()
    
    # Expand v_pts to match the batch size
    v_pts = v_pts.unsqueeze(2).unsqueeze(3).expand(B, 2, H, W)
    
    # Calculate distances
    # d = torch.sqrt(torch.sum((loc.unsqueeze(0) - v_pts)**2, dim=1))
    d = torch.sqrt(torch.sum(torch.square(loc.unsqueeze(0) - v_pts), dim=1)) # TODO: e.g., d **0.5 to scale distance
    
    # # Calculate flow using the CVPR 17 model
    # flow[:, 0] = T_z * torch.pow(d, zeta) * (loc[0].unsqueeze(0) - v_pts[:, 0])
    # flow[:, 1] = T_z * torch.pow(d, zeta) * (loc[1].unsqueeze(0) - v_pts[:, 1])
    T_z_d_zeta = T_z * torch.pow(d, zeta)
    loc_v_pts_diff = loc.unsqueeze(0) - v_pts
    flow = T_z_d_zeta * loc_v_pts_diff
    
    return flow


def generate_flow_and_blur(im, v_pts, T_z, zeta, B, C, H, W):
    """
    Generate flow and blur image.
    """
    flow = vp_flow_generator(im, v_pts, T_z, zeta)
    np_rgb = flow_to_numpy_rgb(flow)
    from reblur_package import FlowBlurrer
    flow_blurrer = FlowBlurrer.create_with_implicit_mesh(B, C, H, W, 30)
    blur_image, mask = flow_blurrer(im, flow)

    return np_rgb, blur_image


def make_path_blur_new(im, v_pt, T_z, zeta):
    # start_time = time.time()

    if len(im.shape) < 4:
        im = im.unsqueeze(0)

    # Swap height and width for vp here
    v_pt = v_pt[::-1]

    # Create a zero tensor with the same size and device as im
    zero_im = torch.zeros_like(im)

    im = torch.cat([im, zero_im], dim=0)[:,:3,:,:]
    im = im / 255.
    
    # Duplicate the v_pt for the second image
    v_pts = torch.tensor([v_pt, v_pt]).cuda()
    B, C, H, W = im.size()

    # Iterate over parameter combinations
    np_rgb, blur_image = generate_flow_and_blur(im, v_pts, T_z, zeta, B, C, H, W)
    
    # get the first image
    blur_image = blur_image[0]

    # clamp to [0, 1]
    blur_image = torch.clamp(blur_image, 0, 1)

    # end_time = time.time()
    # print(f"spend time {end_time - start_time}")

    return blur_image


def apply_path_blur(img, vanishing_point, T_z_values=None, zeta_values=None):
    '''
    img: torch.tensor [C, H, W], (min:0, max:255)
    vanishing_point: [vw, vh]
    '''        
    T_z = random.uniform(float(T_z_values[0]), float(T_z_values[1]))
    zeta = random.uniform(float(zeta_values[0]), float(zeta_values[1]))
    img = make_path_blur_new(img, vanishing_point, T_z, zeta)
    img = img * 255.0

    # reshape 4d to 3d
    if len(img.shape) == 4:
        img = img.squeeze(0)
    
    return img


# NOTE: no use for now
# def get_vanising_points(image_path, vanishing_points, ratio=1.0, flip_transform=False, assign_width=None):

#     # Get flip and new_width information
#     try:
#         from detectron2.data.transforms import ResizeTransform, HFlipTransform, NoOpTransform
#         flip = isinstance(flip_transform, HFlipTransform)
#         new_width = flip_transform.width
#     except:
#         flip = flip_transform
#         new_width = assign_width

#     # if flip:
#     #     new_width = flip_transform.width

#     # Get the vanishing point for the current image
#     image_basename = os.path.basename(image_path)
#     vanishing_point = vanishing_points[image_basename]

#     # print("vanishing_point Before", vanishing_point)
#     # print("flip is", flip)
#     # print("new_width  is", new_width)

#     # Scale vanishing_point according to the ratio
#     vanishing_point = [n * ratio for n in vanishing_point]

#     # print("flip_transform is", flip_transform)
#     # print(f"vanishing_point Before {vanishing_point}")

#     if flip:
#         # Flip x-coordinates of vanishing_point
#         vanishing_point[0] = new_width - vanishing_point[0]

#     # print(f"vanishing_point After {vanishing_point}")

#     return vanishing_point


def get_vanising_points(image_path, vanishing_points, ratio=1.0, flip_transform=False):

    # Get flip and new_width information
    from detectron2.data.transforms import ResizeTransform, HFlipTransform, NoOpTransform
    flip = isinstance(flip_transform, HFlipTransform)
    if flip:
        new_width = flip_transform.width

    # Get the vanishing point for the current image
    image_basename = os.path.basename(image_path)

    # print all keys of vanishing_points
    vanishing_point = vanishing_points[image_basename]

    # Scale vanishing_point according to the ratio
    vanishing_point = [n * ratio for n in vanishing_point]

    # print("flip_transform is", flip_transform)
    # print(f"vanishing_point Before {vanishing_point}")

    if flip:
        # Flip x-coordinates of vanishing_point
        vanishing_point[0] = new_width - vanishing_point[0]

    # print(f"vanishing_point After {vanishing_point}")

    return vanishing_point

# NOTE: for mmseg, we use this function
# NOTE: no longer require vanishing_points
def new_update_vp_ins(sample, ratio=1.0, img_width=None, seg_to_det=None):
    sample_basename = os.path.basename(sample['filename'])

    # Initialize instances to None or existing value in sample
    instances = sample.get('instances', None)

    if seg_to_det is not None and 'ori_filename' in sample and sample_basename in seg_to_det:
        # Get a copy of the instances from seg_to_det
        instances = copy.deepcopy(seg_to_det[sample_basename])
        # print("Before, instances are", instances)
        
        # Scale instances according to the ratio
        for i in range(len(instances)):
            instances[i] = [n * ratio for n in instances[i]]
        # print("After, instances are", instances)

    if sample.get('flip', False) and img_width is not None and instances is not None:
        # print("Before flipping, instances are", instances)

        # Flip x-coordinates of instances
        for instance in instances:
            old_x1 = instance[0]
            old_x2 = instance[2]
            
            # Update x1 and x2 after flipping
            # print("img_width is", img_width)
            instance[0] = img_width - old_x2  # new x1
            instance[2] = img_width - old_x1  # new x2

        # print("After flipping, instances are", instances)

    return instances


def update_vp_ins(sample, ratio=1.0, img_width=None, seg_to_det=None):

    sample_basename = os.path.basename(sample['filename'])

    if seg_to_det is not None and 'ori_filename' in sample and sample_basename in seg_to_det:
        # Get the instances from seg_to_det
        instances = seg_to_det[sample_basename]
        # Scale instances according to the ratio
        # print("Before, instances are", instances)
        for i in range(len(instances)):
            instances[i] = [n * ratio for n in instances[i]]
        # print("After, instances are", instances)
        sample['instances'] = instances

    if sample.get('flip', False) and img_width is not None:
        # Flip x-coordinates of instances
        if 'instances' in sample:
            for instance in sample['instances']:
                # Store the old x1 and x2
                old_x1 = instance[0]
                old_x2 = instance[2]
                
                # Update x1 and x2 after flipping
                instance[0] = img_width - old_x2  # new x1
                instance[2] = img_width - old_x1  # new x2


# def update_vp_ins(sample, vanishing_points, ratio=1.0, img_width=None, 
#                               seg_to_det=None):

#     # Get the vanishing point for the current image
#     image_basename = os.path.basename(sample['filename'])
#     vanishing_point = vanishing_points[image_basename]

#     # Scale vanishing_point according to the ratio
#     vanishing_point = [n * ratio for n in vanishing_point]

#     sample_basename = os.path.basename(sample['filename'])

#     if seg_to_det is not None and 'ori_filename' in sample and sample_basename in seg_to_det:
#         # get the instances from seg_to_det
#         instances = seg_to_det[sample_basename]
#         # scale instances according to the ratio
#         for i in range(len(instances)):
#             instances[i] = [n * ratio for n in instances[i]]
#         sample['instances'] = instances

#     # print("flip_transform is", flip_transform)
#     # print(f"vanishing_point Before {vanishing_point}")

#     if sample.get('flip', False) and img_width is not None:
#         # Flip x-coordinates of vanishing_point
#         vanishing_point[0] = img_width - vanishing_point[0]

#         # Flip x-coordinates of instances
#         if 'instances' in sample:
#             # print("before, " + str(sample['instances']))
#             for instance in sample['instances']:
#                 # Store the old x1 and x2
#                 old_x1 = instance[0]
#                 old_x2 = instance[2]
                
#                 # Update x1 and x2 after flipping
#                 instance[0] = img_width - old_x2  # new x1
#                 instance[2] = img_width - old_x1  # new x2
#             # print("after, " + str(sample['instances']))

#     # Update the sample dictionary with the modified vanishing point
#     sample['vanishing_point'] = vanishing_point


def is_out_of_bounds(pt, img_width, img_height):
    '''
    pt: [pw, ph]
    img_width: Width
    img_height: Height
    '''
    if (pt[0] < 0 and pt[1] < 0) or \
        (pt[0] > img_width and pt[1] < 0) or \
            (pt[0] < 0 and pt[1] > img_height) or \
                (pt[0] > img_width and pt[1] > img_height):
        return True
    return False