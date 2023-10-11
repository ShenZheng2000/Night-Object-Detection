import numpy as np

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

import kornia as K

import sys
from torchvision.utils import save_image
import cv2

# NOTE: add a base class for others to inherit
class BaseLayerGlobal(nn.Module):
    def __init__(self, 
                 min_theta=110, max_theta=120, min_alpha=0.2, max_alpha=0.4,
                 min_theta_top=None, max_theta_top=None, min_alpha_top=None, max_alpha_top=None, 
                 min_p=1, max_p=5, lambd=0.97):
        
        super(BaseLayerGlobal, self).__init__()
        
        min_theta = np.deg2rad(min_theta)
        max_theta = np.deg2rad(max_theta)
        
        self.theta_l = self.init_param(min_theta, max_theta)
        self.theta_r = self.init_param(min_theta, max_theta)
        self.alpha_1 = self.init_param(min_alpha, max_alpha)
        self.alpha_2 = self.init_param(min_alpha, max_alpha)

        self.p = self.init_param(min_p, max_p)
        self.p_top = self.init_param(min_p, max_p)

        self.alpha_top_1 = self.init_param(min_alpha, max_alpha)
        self.alpha_top_2 = self.init_param(min_alpha, max_alpha)

        self.lambd = nn.Parameter(torch.Tensor([1])*lambd, requires_grad=False)

        self.cached_maps = {}

    def init_param(self, value1, value2):
        return nn.Parameter(torch.Tensor([1]) * (value1 + value2) / 2, requires_grad=False)

    def update_cache_map(self, imgs):
        self.im_shape = imgs.shape[-2:]
        if self.im_shape not in self.cached_maps:
            self.cached_maps[self.im_shape] = self.compute_init_map(self.im_shape)
        self.init_map = self.cached_maps[self.im_shape]
        self.device = imgs.device

    def compute_init_map(self, im_shape):
        init_map = torch.zeros(im_shape)
        for r in range(init_map.shape[0]):
            init_map[r, :] = (init_map.shape[0]*1.0 - r) / init_map.shape[0]*1.0
        init_map = init_map.unsqueeze(0).unsqueeze(0)
        init_map = init_map - 1
        return init_map

    def parametric_homography(self, v_pts, thetas_l, thetas_r, alphas_1, alphas_2, bottom):
        h, w = self.im_shape
        B = v_pts.shape[0]
        p_l = torch.zeros(B, 2, device=self.device)
        p_l[:, 1] = v_pts[:, 1] + torch.mul(v_pts[:, 0], 1./torch.tan(thetas_l))   
        p_r = torch.zeros(B, 2, device=self.device)
        p_r[:, 0] += w - 1
        p_r[:, 1] = v_pts[:, 1] + torch.mul(w - 1 - v_pts[:, 0], 1./torch.tan(thetas_r))
        p1 = alphas_1[:, None]*p_l + (1 - alphas_1[:, None])*v_pts
        p2 = alphas_2[:, None]*p_r + (1 - alphas_2[:, None])*v_pts
        pt_src = torch.zeros(B, 4, 2, device=self.device)
        if bottom:
            pt_src[:, 0, :] = p1
            pt_src[:, 1, :] = p2
            pt_src[:, 2, :] = torch.tensor([w - 1, h - 1], device=self.device).repeat(B, 1)
            pt_src[:, 3, :] = torch.tensor([0, h - 1], device=self.device).repeat(B, 1)
        else:
            pt_src[:, 0, :] = torch.tensor([0, 0], device=self.device).repeat(B, 1)
            pt_src[:, 1, :] = torch.tensor([w - 1, 0], device=self.device).repeat(B, 1)
            pt_src[:, 2, :] = p2
            pt_src[:, 3, :] = p1
        pt_dst = torch.tensor([[
            [0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]
        ]], dtype=torch.float32, device=self.device).repeat(B, 1, 1)
        M = K.geometry.get_perspective_transform(pt_dst, pt_src)
        return pt_src, M
    
    def map_warp(self, B, v_pts, thetas_l, thetas_r, alphas_1, alphas_2, ps, bottom):
        points_src, M = self.parametric_homography(v_pts, thetas_l, thetas_r, alphas_1, alphas_2, bottom)
        init_map = self.init_map.to(self.device).repeat(B, 1, 1, 1)        
        init_map = torch.exp( torch.mul(ps, init_map) )
        map_warp: torch.tensor = K.geometry.warp_perspective(init_map.float(), M, 
                                    dsize=( round(self.im_shape[0]), round(self.im_shape[1]) ))
        return points_src, map_warp

    def process_warp(self, B, v_pts, thetas_l, thetas_r, alphas_1, alphas_2, p, bottom_flag):
        thetas_l_exp = thetas_l.expand(B).to(self.device)
        thetas_r_exp = thetas_r.expand(B).to(self.device)
        alphas_1_exp = alphas_1.expand(B).to(self.device)
        alphas_2_exp = alphas_2.expand(B).to(self.device)
        ps = p.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(self.device)
        ps_exp = ps.expand(B, 1, self.init_map.shape[2], self.init_map.shape[3])
        
        points, side = self.map_warp(B, v_pts, thetas_l_exp, thetas_r_exp, alphas_1_exp, alphas_2_exp, ps_exp, bottom=bottom_flag)
        return points, side 
    
    def visualize(self, v_pts, bottom, top, filename):
        v_pts_original = v_pts[0].cpu().numpy()
        saliency_bottom = bottom.squeeze(0).squeeze(0).cpu().detach().numpy()
        saliency_top = top.squeeze(0).squeeze(0).cpu().detach().numpy()
        saliency_final = saliency_bottom + saliency_top

        # Draw a circle at the vanishing point on the saliency map
        vanishing_point_color = (0, 0, 255)  # BGR color format (red)
        cv2.circle(saliency_final, 
                    (int(v_pts_original[0]), int(v_pts_original[1])), 
                    5, 
                    vanishing_point_color, 
                    -1)  # Draw a filled circle
        
        saliency_final = torch.tensor(saliency_final)[None, None, ...]
        save_image(saliency_final, f"{filename}")

        # print("bottom mean: ", bottom.mean())
        # print("top mean", top.mean())

    def forward(self, imgs, v_pts, vis_flag=False):
        raise NotImplementedError("This is a base class, forward method should be defined in the child class.")


class CuboidLayerGlobal(BaseLayerGlobal):
    
    # # NOTE NOTE: for debug only!!!
    # def __init__(self, 
    #              min_theta=0, max_theta=150,
    #              **kwargs):
        
    #     # Pass the required arguments to the base class.
    #     super(CuboidLayerGlobal, self).__init__(min_theta=min_theta, max_theta=max_theta,
    #                                             **kwargs)

    def forward(self, imgs, v_pts, vis_flag=False):
        # NOTE: use cache map to save computation
        # self.im_shape = imgs.shape[-2:]
        # if self.im_shape not in self.cached_maps:
        #     self.cached_maps[self.im_shape] = self.compute_init_map(self.im_shape)
        # self.init_map = self.cached_maps[self.im_shape]
        self.update_cache_map(imgs)  

        self.device = imgs.device
        B = imgs.shape[0]

        points_bt, bottom = self.process_warp(B, v_pts, self.theta_l, self.theta_r, self.alpha_1, self.alpha_2, self.p, bottom_flag=True)
        points_tp, top = self.process_warp(B, v_pts, self.theta_l, self.theta_r, self.alpha_top_1, self.alpha_top_2, self.p_top, bottom_flag=False)

        lambd = (1.0 - self.lambd).to(self.device)
        map_warp = bottom + lambd * top 

        if vis_flag:
            save_image(map_warp, f"cuboid_layer_global.png")
            # self.visualize(v_pts, bottom, top, filename='cuboid_layer_global.png')

        return map_warp


# NOTE: add a middle plane in between the top and bottom planes
class TripetLayerGlobal(BaseLayerGlobal):
    #  min_theta=110, max_theta=120, min_alpha=0.2, max_alpha=0.4,
    #  min_p=1, max_p=5, lambd=0.97, requires_grad=False,
    #  min_theta_top=70, max_theta_top=60,

    # DEFAULTS = {
    #     'min_theta': 0,
    #     'max_theta': 150, # NOTE: tune it later
    #     'min_theta_top': 0, 
    #     'max_theta_top': 240, # NOTE: tune it later
    #     'min_alpha_top': 0.2,
    #     'max_alpha_top': 0.4
    # }

    DEFAULTS = {
        'min_theta': 0,
        'max_theta': 170, # NOTE: tune it later
        'min_theta_top': 0, 
        'max_theta_top': 230, # NOTE: tune it later
        'min_alpha_top': 0.2,
        'max_alpha_top': 0.4
    }

    def __init__(self, **kwargs):
        # Use the provided kwargs, or fallback to the defaults
        min_theta = kwargs.get('min_theta', self.DEFAULTS['min_theta'])
        max_theta = kwargs.get('max_theta', self.DEFAULTS['max_theta'])
        min_theta_top = kwargs.get('min_theta_top', self.DEFAULTS['min_theta_top'])
        max_theta_top = kwargs.get('max_theta_top', self.DEFAULTS['max_theta_top'])
        min_alpha_top = kwargs.get('min_alpha_top', self.DEFAULTS['min_alpha_top'])
        max_alpha_top = kwargs.get('max_alpha_top', self.DEFAULTS['max_alpha_top'])

        # print("min_theta: ", min_theta)
        # print("max_theta: ", max_theta)
        # print("min_theta_top: ", min_theta_top)
        # print("max_theta_top: ", max_theta_top)
        # print("min_alpha_top: ", min_alpha_top)
        # print("max_alpha_top: ", max_alpha_top)

        # Pass the arguments to the base class.
        # I'm assuming the base class's __init__ method accepts these parameters. 
        # If it doesn't, you might need to adjust this part.
        super(TripetLayerGlobal, self).__init__(min_theta=min_theta, max_theta=max_theta,
                                                min_theta_top=min_theta_top, max_theta_top=max_theta_top,
                                                min_alpha_top=min_alpha_top, max_alpha_top=max_alpha_top)
        # NOTE: 
        min_theta_top = np.deg2rad(min_theta_top)
        max_theta_top = np.deg2rad(max_theta_top)

        self.theta_top_l = self.init_param(min_theta_top, max_theta_top)
        self.theta_top_r = self.init_param(min_theta_top, max_theta_top)
        self.alpha_top_1 = self.init_param(min_alpha_top, max_alpha_top)
        self.alpha_top_2 = self.init_param(min_alpha_top, max_alpha_top)

    def compute_mid_plane(self, B, points_bt, points_tp, bottom, w, h):
        ## I want to create an array of the top plane bottom points 
        ## and the bottom plane top points in a clockwise manner
        ## See paper for details of the values
        points_mid = torch.zeros(B, 4, 2, device=self.device)
        points_mid[:, 0, :] = points_tp[:, 3, :] # q1
        points_mid[:, 1, :] = points_tp[:, 2, :] # q2
        points_mid[:, 2, :] = points_bt[:, 1, :] # u2
        points_mid[:, 3, :] = points_bt[:, 0, :] # u1

        # print("T3", points_mid[:, 0, :])
        # print("T2", points_mid[:, 1, :])
        # print("B1", points_mid[:, 2, :])
        # print("B0", points_mid[:, 3, :])

        ## let's compute the homography of the mid plane
        pt_dst = torch.tensor([[
            [0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]
        ]], dtype=torch.float32, device=self.device).repeat(B, 1, 1)
        M_mid_plane = K.geometry.get_perspective_transform(pt_dst, points_mid)

        ## top view of the far away plane is the max of the bottom plane
        init_middle_map = torch.zeros(self.im_shape, device=self.device) + torch.max(bottom)
        # init_middle_map.to(self.device).repeat(B, 1, 1, 1)
        init_middle_map = init_middle_map.to(self.device).repeat(B, 1, 1, 1)
        ## project it to the image viewpoint
        mid_plane = K.geometry.warp_perspective(init_middle_map.float(), M_mid_plane, 
                                    dsize=( round(self.im_shape[0]), round(self.im_shape[1]) ))    
        return mid_plane

    def forward(self, imgs, v_pts, vis_flag=False):
        # NOTE: use cache map to save computation
        # self.im_shape = imgs.shape[-2:]
        # if self.im_shape not in self.cached_maps:
        #     self.cached_maps[self.im_shape] = self.compute_init_map(self.im_shape)
        # self.init_map = self.cached_maps[self.im_shape]
        self.update_cache_map(imgs)

        self.device = imgs.device
        B = imgs.shape[0]
        h, w = self.im_shape

        points_bt, bottom = self.process_warp(B, v_pts, self.theta_l, self.theta_r, self.alpha_1, self.alpha_2, self.p, bottom_flag=True)
        points_tp, top = self.process_warp(B, v_pts, self.theta_top_l, self.theta_top_r, self.alpha_top_1, self.alpha_top_2, self.p_top, bottom_flag=False)

        mid_plane = self.compute_mid_plane(B, points_bt, points_tp, bottom, w, h)

        # Create a mask where mid_plane has valid values
        mid_plane_mask = mid_plane > 0

        # Convert mid_plane_mask to float for arithmetic operations
        mid_plane_mask_float = mid_plane_mask.float()

        # Merge bottom and mid_plane according to the mask
        merged_plane = mid_plane_mask_float * mid_plane + (1.0 - mid_plane_mask_float) * bottom

        lambd = (1.0 - self.lambd).to(self.device)
        map_warp = merged_plane + lambd * top
        # print("map_warp min: ", map_warp.min())
        # print("map_warp max: ", map_warp.max())

        # NOTE: below debug only
        # print("v_pts: ", v_pts)
        # print("bottom: ", bottom.shape)
        # print("top: ", top.shape)

        if vis_flag:
            save_image(map_warp, f"triplet_layer_global.png")
            # self.visualize(v_pts, bottom, top, filename='triplet_layer_global.png')

        return map_warp
    
def has_learnable_parameters(module):
    return any(param.requires_grad for param in module.parameters())

if __name__ == '__main__':
    # 
    from PIL import Image
    import torchvision.transforms as transforms

    img_path = "/home/aghosh/Projects/2PCNet/Datasets/bdd100k/images/100k/train_debug/0a0a0b1a-7c39d841.jpg"
    v_pts = torch.tensor([560.4213953681794, 315.3206647347985])

    # img_path = "/home/aghosh/Projects/2PCNet/Datasets/bdd100k/images/100k/train_day/5250cae5-14dc826a.jpg"
    # v_pts = torch.tensor([-235.50865750631317, 340.43002132438045])

    # input_shape = (720, 1280)
    cuboid_layer = CuboidLayerGlobal()
    tripet_layer  = TripetLayerGlobal()

    # # Check if parameters in cuboid_layer and tripet_layer are learnable => DONE, none are learnable
    # if has_learnable_parameters(cuboid_layer):
    #     print("CuboidLayerGlobal has learnable parameters.")
    # else:
    #     print("CuboidLayerGlobal does not have any learnable parameters.")

    # if has_learnable_parameters(tripet_layer):
    #     print("TripetLayerGlobal has learnable parameters.")
    # else:
    #     print("TripetLayerGlobal does not have any learnable parameters.")

    # Load the image as a torch tensor
    img = Image.open(img_path).convert("RGB")
    transform = transforms.Compose([transforms.ToTensor()])
    img_tensor = transform(img)

    # Unsqueeze to correct dimension
    img_tensor = img_tensor.unsqueeze(0)
    v_pts = v_pts.unsqueeze(0)

    # print("img_tensor shape", img_tensor.shape) # 1, 3, 720, 1280
    # print("v_pts shape", v_pts.shape) # 1, 2

    # Perform operations on the image tensor
    saliency = cuboid_layer.forward(img_tensor, v_pts, vis_flag=True)
    saliency = tripet_layer.forward(img_tensor, v_pts, vis_flag=True)
