# from flow_utils import flow_to_numpy_rgb
# from reblur_package import FlowBlurrer
from reblur_package import FlowBlurrer
from .flow_utils import flow_to_numpy_rgb
import torch
import time
import random
from detectron2.data.transforms import ResizeTransform, HFlipTransform, NoOpTransform
import os

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


def get_vanising_points(image_path, vanishing_points, ratio=1.0, flip_transform=False):

    # Get flip and new_width information
    flip = isinstance(flip_transform, HFlipTransform)
    if flip:
        new_width = flip_transform.width

    # Get the vanishing point for the current image
    image_basename = os.path.basename(image_path)
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