import numpy as np

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

import kornia as K

import sys
from torchvision.utils import save_image
import cv2

# # TODO: for debug now, remove later
# import logging
# logging.basicConfig(filename="parameter_updates.log", level=logging.INFO, format='%(message)s')

# class HomographySaliencyParamsNet(nn.Module):
#     def __init__(self, 
#             backbone='resnet18', 
#             scale_factor=0.2,
#             min_theta=110,
#             max_theta=120,
#             min_alpha=0.2,
#             max_alpha=0.4,
#             min_p=1, max_p=4
#             ):
#         self.min_theta = torch.deg2rad(torch.tensor([min_theta])).item()        
#         self.max_theta = torch.deg2rad(torch.tensor([max_theta])).item()        
#         self.min_alpha = min_alpha
#         self.max_alpha = max_alpha       
#         self.min_p = min_p
#         self.max_p = max_p
#         super(HomographySaliencyParamsNet, self).__init__()
#         self.scale_factor = scale_factor
#         if backbone == "resnet18":
#             self.backbone = torchvision.models.resnet18(pretrained=True)
#             self.backbone = torch.nn.Sequential(*(list(self.backbone.children())[:-1]))
#         elif backbone == "resnet50":
#             self.backbone = torchvision.models.resnet50(pretrained=True)
#             self.backbone = torch.nn.Sequential(*(list(self.backbone.children())[:-1]))
#         self.predictor = nn.Sequential(
#             nn.Linear(512, 512),
#             nn.LeakyReLU(),
#             nn.Linear(512, 4),
#             nn.LeakyReLU()
#         )
        
#     def forward(self, imgs):
#         device = imgs.device
#         self.backbone.to(device)
#         self.predictor.to(device)
#         scaled_imgs = F.interpolate(imgs, scale_factor=self.scale_factor)
#         feats = self.backbone(scaled_imgs).squeeze(-1).squeeze(-1)
#         outs = self.predictor(feats)
#         thetas_l = torch.clamp(outs[:, 0], min=self.min_theta, max=self.max_theta)
#         thetas_r = torch.clamp(outs[:, 1], min=self.min_theta, max=self.max_theta)
#         alphas = torch.clamp(outs[:, 2], min=self.min_alpha, max=self.max_alpha)
#         ps = torch.clamp(outs[:, 3], min=self.min_p, max=self.max_p)                
#         return thetas_l, thetas_r, alphas, ps

# class HomographyLayer(nn.Module):

#     def __init__(self, im_shape):
#         super(HomographyLayer, self).__init__()
#         self.im_shape = im_shape
#         self.init_map = torch.zeros(self.im_shape)
#         for r in range(self.init_map.shape[0]):
#             self.init_map[r, :] = (self.init_map.shape[0]*1.0 - r) / self.init_map.shape[0]*1.0
#         self.init_map = self.init_map.unsqueeze(0).unsqueeze(0)
#         self.init_map = self.init_map - 1

#     def parametric_homography(self, v_pts, thetas_l, thetas_r, alphas):
#         h, w = self.im_shape
#         B = v_pts.shape[0]
#         p_l = torch.zeros(B, 2, device=self.device)
#         p_l[:, 1] = v_pts[:, 1] + torch.mul(v_pts[:, 0], 1./torch.tan(thetas_l))   
#         p_r = torch.zeros(B, 2, device=self.device)
#         p_r[:, 0] += w - 1
#         p_r[:, 1] = v_pts[:, 1] + torch.mul(w - 1 - v_pts[:, 0], 1./torch.tan(thetas_r)) 
#         p1 = alphas[:, None]*p_l + (1 - alphas[:, None])*v_pts
#         p2 = alphas[:, None]*p_r + (1 - alphas[:, None])*v_pts
#         pt_src = torch.zeros(B, 4, 2, device=self.device)
#         pt_src[:, 0, :] = p1
#         pt_src[:, 1, :] = p2
#         pt_src[:, 2, :] = torch.tensor([w - 1, h - 1], device=self.device).repeat(B, 1)
#         pt_src[:, 3, :] = torch.tensor([0, h - 1], device=self.device).repeat(B, 1)
#         pt_dst = torch.tensor([[
#             [0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]
#         ]], dtype=torch.float32, device=self.device).repeat(B, 1, 1)
#         M = K.geometry.get_perspective_transform(pt_dst, pt_src)
#         return pt_src, M
    
#     def forward(self, imgs, v_pts, thetas_l, thetas_r, alphas, ps, return_homo=False):
#         self.device = imgs.device
#         B = imgs.shape[0]
#         points_src, M = self.parametric_homography(v_pts, thetas_l, thetas_r, alphas)
#         ps = ps.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
#         ps = ps.expand(B, 1, self.init_map.shape[2], self.init_map.shape[3])
#         init_map = self.init_map.to(self.device).repeat(imgs.shape[0], 1, 1, 1)        
#         init_map = torch.exp( torch.mul(ps, init_map) )
#         map_warp: torch.tensor = K.geometry.warp_perspective(init_map.float(), M, 
#                                     dsize=( round(self.im_shape[0]), round(self.im_shape[1]) ))
#         if return_homo == True:
#             return map_warp, (points_src, M)
#         return map_warp

# class HomographyLayerGlobal(nn.Module):

#     def __init__(self, im_shape,
#             min_theta=110,
#             max_theta=120,
#             min_alpha=0.2,
#             max_alpha=0.4,
#             min_p=1, max_p=5
#     ):
#         super(HomographyLayerGlobal, self).__init__()
#         self.im_shape = im_shape
#         self.init_map = torch.zeros(self.im_shape)
#         for r in range(self.init_map.shape[0]):
#             self.init_map[r, :] = (self.init_map.shape[0]*1.0 - r) / self.init_map.shape[0]*1.0
#         self.init_map = self.init_map.unsqueeze(0).unsqueeze(0)
#         self.init_map = self.init_map - 1
        
#         min_theta = np.deg2rad(min_theta)
#         max_theta = np.deg2rad(max_theta)

#         self.theta_l = nn.Parameter(torch.Tensor([1])*(min_theta + max_theta)/2, requires_grad=True)
#         self.theta_r = nn.Parameter(torch.Tensor([1])*(min_theta + max_theta)/2, requires_grad=True)
#         self.alpha = nn.Parameter(torch.Tensor([1])*(min_alpha + max_alpha)/2, requires_grad=True)
#         self.p = nn.Parameter(torch.Tensor([1])*(min_p + max_p)/2, requires_grad=True)

#     def parametric_homography(self, v_pts, thetas_l, thetas_r, alphas):
#         h, w = self.im_shape
#         B = v_pts.shape[0]
#         p_l = torch.zeros(B, 2, device=self.device)
#         p_l[:, 1] = v_pts[:, 1] + torch.mul(v_pts[:, 0], 1./torch.tan(thetas_l))   
#         p_r = torch.zeros(B, 2, device=self.device)
#         p_r[:, 0] += w - 1
#         p_r[:, 1] = v_pts[:, 1] + torch.mul(w - 1 - v_pts[:, 0], 1./torch.tan(thetas_r))
#         p1 = alphas[:, None]*p_l + (1 - alphas[:, None])*v_pts
#         p2 = alphas[:, None]*p_r + (1 - alphas[:, None])*v_pts
#         pt_src = torch.zeros(B, 4, 2, device=self.device)
#         pt_src[:, 0, :] = p1
#         pt_src[:, 1, :] = p2
#         pt_src[:, 2, :] = torch.tensor([w - 1, h - 1], device=self.device).repeat(B, 1)
#         pt_src[:, 3, :] = torch.tensor([0, h - 1], device=self.device).repeat(B, 1)
#         pt_dst = torch.tensor([[
#             [0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]
#         ]], dtype=torch.float32, device=self.device).repeat(B, 1, 1)
#         M = K.geometry.get_perspective_transform(pt_dst, pt_src)
#         return pt_src, M
    
#     def forward(self, imgs, v_pts):
#         self.device = imgs.device
#         B = imgs.shape[0]
#         thetas_l = self.theta_l.expand(B).to(self.device)
#         thetas_r = self.theta_r.expand(B).to(self.device)
#         alphas = self.alpha.expand(B).to(self.device)
#         points_src, M = self.parametric_homography(v_pts, thetas_l, thetas_r, alphas)
#         ps = self.p.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(self.device)
#         ps = ps.expand(B, 1, self.init_map.shape[2], self.init_map.shape[3])
#         init_map = self.init_map.to(self.device).repeat(imgs.shape[0], 1, 1, 1)        
#         init_map = torch.exp( torch.mul(ps, init_map) )
#         map_warp: torch.tensor = K.geometry.warp_perspective(init_map.float(), M, 
#                                     dsize=( round(self.im_shape[0]), round(self.im_shape[1]) ))
#         return map_warp

class CuboidLayerGlobal(nn.Module):
    def __init__(self, 
                #  im_shape,
            min_theta=110,
            max_theta=120,           
            min_alpha=0.2,
            max_alpha=0.4,            
            min_p=1, 
            max_p=5,
            lambd=0.97,
            requires_grad=False, # NOTE: hardcode here, change later
    ):
        super(CuboidLayerGlobal, self).__init__()
        # NOTE: Move to forward() to handle different shapes
        # self.im_shape = im_shape
        # self.init_map = torch.zeros(self.im_shape)
        # for r in range(self.init_map.shape[0]):
        #     self.init_map[r, :] = (self.init_map.shape[0]*1.0 - r) / self.init_map.shape[0]*1.0
        # self.init_map = self.init_map.unsqueeze(0).unsqueeze(0)
        # self.init_map = self.init_map - 1
        
        min_theta = np.deg2rad(min_theta)
        max_theta = np.deg2rad(max_theta)

        self.theta_l = nn.Parameter(torch.Tensor([1])*(min_theta + max_theta)/2)
        self.theta_r = nn.Parameter(torch.Tensor([1])*(min_theta + max_theta)/2)
        self.alpha_1 = nn.Parameter(torch.Tensor([1])*(min_alpha + max_alpha)/2)
        self.alpha_2 = nn.Parameter(torch.Tensor([1])*(min_alpha + max_alpha)/2)
        self.p = nn.Parameter(torch.Tensor([1])*(min_p + max_p)/2)

        self.lambd = nn.Parameter(torch.Tensor([1])*lambd)

        # self.theta_top_l = nn.Parameter(torch.Tensor([1])*(min_theta + max_theta)/2)
        # self.theta_top_r = nn.Parameter(torch.Tensor([1])*(min_theta + max_theta)/2)
        self.alpha_top_1 = nn.Parameter(torch.Tensor([1])*(min_alpha + max_alpha)/2)
        self.alpha_top_2 = nn.Parameter(torch.Tensor([1])*(min_alpha + max_alpha)/2)
        self.p_top = nn.Parameter(torch.Tensor([1])*(min_p + max_p)/2)

        for param in self.parameters():
            param.requires_grad = requires_grad        

        self.cached_maps = {}

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
        return map_warp

    def forward(self, imgs, v_pts):
        # NOTE: use cache map to save computation
        self.im_shape = imgs.shape[-2:]
        if self.im_shape not in self.cached_maps:
            self.cached_maps[self.im_shape] = self.compute_init_map(self.im_shape)
        self.init_map = self.cached_maps[self.im_shape]     
        # print("self.im_shape is", self.im_shape)
        # print("self.init_map.shape is", self.init_map.shape)

        # # NOTE: debug this parameters
        # logging.info(f"theta_l: {self.theta_l}, theta_l.grad: {self.theta_l.grad}, "
        #             f"theta_r: {self.theta_r}, theta_r.grad: {self.theta_r.grad}, "
        #             f"alpha_1: {self.alpha_1}, alpha_1.grad: {self.alpha_1.grad}, "
        #             f"alpha_2: {self.alpha_2}, alpha_2.grad: {self.alpha_2.grad}, "
        #             f"p: {self.p}, p.grad: {self.p.grad}, "
        #             f"lambd: {self.lambd}, lambd.grad: {self.lambd.grad}, "
        #             f"alpha_top_1: {self.alpha_top_1}, alpha_top_1.grad: {self.alpha_top_1.grad}, "
        #             f"alpha_top_2: {self.alpha_top_2}, alpha_top_2.grad: {self.alpha_top_2.grad}, "
        #             f"p_top: {self.p_top}, p_top.grad: {self.p_top.grad}")
                
        # if self.theta_l.grad is not None and torch.isnan(self.theta_l.grad).any():
        #     print("NaN gradient detected for theta_l")

        self.device = imgs.device
        B = imgs.shape[0]
        thetas_l = self.theta_l.expand(B).to(self.device)
        thetas_r = self.theta_r.expand(B).to(self.device)
        alphas_1 = self.alpha_1.expand(B).to(self.device)
        alphas_2 = self.alpha_2.expand(B).to(self.device)

        ps = self.p.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(self.device)
        ps = ps.expand(B, 1, self.init_map.shape[2], self.init_map.shape[3])        
        bottom = self.map_warp(B, v_pts, thetas_l, thetas_r, alphas_1, alphas_2, ps, bottom=True)

        thetas_top_l = self.theta_l.expand(B).to(self.device)
        thetas_top_r = self.theta_r.expand(B).to(self.device)
        alphas_top_1 = self.alpha_top_1.expand(B).to(self.device)
        alphas_top_2 = self.alpha_top_2.expand(B).to(self.device)
        ps_top = self.p_top.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(self.device)
        ps_top = ps_top.expand(B, 1, self.init_map.shape[2], self.init_map.shape[3])        
        top = self.map_warp(B, v_pts, thetas_top_l, thetas_top_r, alphas_top_1, alphas_top_2, ps_top, bottom=False)

        lambd = (1.0 - self.lambd).to(self.device)
        map_warp = bottom + lambd * top 

        # NOTE: for debug only
        # print("v_pts: ", v_pts)
        # print("bottom: ", bottom.shape)
        # print("top: ", top.shape)

        # v_pts_original = v_pts[0].cpu().numpy()
        # saliency_bottom = bottom.squeeze(0).squeeze(0).cpu().detach().numpy()
        # saliency_top = top.squeeze(0).squeeze(0).cpu().detach().numpy()
        # saliency_final = saliency_bottom + saliency_top

        # # Draw a circle at the vanishing point on the saliency map
        # vanishing_point_color = (0, 0, 255)  # BGR color format (red)
        # cv2.circle(saliency_final, 
        #             (int(v_pts_original[0]), int(v_pts_original[1])), 
        #             5, 
        #             vanishing_point_color, 
        #             -1)  # Draw a filled circle
        
        # saliency_final = torch.tensor(saliency_final)[None, None, ...]

        # save_image(saliency_final, "saliency_final.png")

        # print("bottom is nan", torch.isnan(bottom).any())
        # print("top is nan", torch.isnan(top).any())
        # print("map_warp is nan", torch.isnan(map_warp).any())

        return map_warp

# NOTE: add a middle plane in between the top and bottom planes => uncomment and train later
class TripetLayerGlobal(nn.Module):
    def __init__(self, im_shape,
            min_theta=110,
            max_theta=120,           
            min_alpha=0.2,
            max_alpha=0.4,
            ## four more initializations for top (below values should be <= 90)
            # TODO: chanhge it till looks good, and train the model with this
            min_theta_top=70,
            max_theta_top=60,
            min_alpha_top=0.2,
            max_alpha_top=0.4,
            ##
            min_p=1, 
            max_p=5,
            lambd=0.97
            requires_grad=False, # NOTE: hardcode here, change later
    ):
        super(TripetLayerGlobal, self).__init__()
        self.im_shape = im_shape
        self.init_map = torch.zeros(self.im_shape)
        for r in range(self.init_map.shape[0]):
            self.init_map[r, :] = (self.init_map.shape[0]*1.0 - r) / self.init_map.shape[0]*1.0
        self.init_map = self.init_map.unsqueeze(0).unsqueeze(0)
        self.init_map = self.init_map - 1
        
        min_theta = np.deg2rad(min_theta)
        max_theta = np.deg2rad(max_theta)

        # NOTE: 
        min_theta_top = np.deg2rad(min_theta_top)
        max_theta_top = np.deg2rad(max_theta_top)

        self.theta_l = self.init_param(min_theta, max_theta)
        self.theta_r = self.init_param(min_theta, max_theta)
        self.alpha_1 = self.init_param(min_alpha, max_alpha)
        self.alpha_2 = self.init_param(min_alpha, max_alpha)
        self.p = nn.Parameter(torch.Tensor([1])*(min_p + max_p)/2)

        self.lambd = nn.Parameter(torch.Tensor([1])*lambd)

        self.theta_top_l = self.init_param(min_theta_top, max_theta_top)
        self.theta_top_r = self.init_param(min_theta_top, max_theta_top)
        self.alpha_top_1 = self.init_param(min_alpha_top, max_alpha_top)
        self.alpha_top_2 = self.init_param(min_alpha_top, max_alpha_top)
        self.p_top = nn.Parameter(torch.Tensor([1])*(min_p + max_p)/2)

    def init_param(self, value1, value2):
        return nn.Parameter(torch.Tensor([1]) * (value1 + value2) / 2)

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

    def forward(self, imgs, v_pts):
        self.device = imgs.device
        B = imgs.shape[0]
        h, w = self.im_shape

        points_bt, bottom = self.process_warp(B, v_pts, self.theta_l, self.theta_r, self.alpha_1, self.alpha_2, self.p, True)
        points_tp, top = self.process_warp(B, v_pts, self.theta_top_l, self.theta_top_r, self.alpha_top_1, self.alpha_top_2, self.p_top, False)

        ## I want to create an array of the top plane bottom points 
        ## and the bottom plane top points in a clockwise manner
        ## See paper for details of the values
        points_mid = torch.zeros(B, 4, 2, device=self.device)
        points_mid[:, 0, :] = points_tp[:, 3, :] # q1
        points_mid[:, 1, :] = points_tp[:, 2, :] # q2
        points_mid[:, 2, :] = points_bt[:, 1, :] # u2
        points_mid[:, 3, :] = points_bt[:, 0, :] # u1

        ## let's compute the homography of the mid plane
        pt_dst = torch.tensor([[
            [0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]
        ]], dtype=torch.float32, device=self.device).repeat(B, 1, 1)
        M_mid_plane = K.geometry.get_perspective_transform(pt_dst, points_mid)

        ## top view of the far away plane is the max of the bottom plane
        init_middle_map = torch.zeros(self.im_shape) + torch.max(bottom)
        # init_middle_map.to(self.device).repeat(B, 1, 1, 1)
        init_middle_map = init_middle_map.to(self.device).repeat(B, 1, 1, 1)
        ## project it to the image viewpoint
        mid_plane = K.geometry.warp_perspective(init_middle_map.float(), M_mid_plane, 
                                    dsize=( round(self.im_shape[0]), round(self.im_shape[1]) ))        


        lambd = (1.0 - self.lambd).to(self.device)
        map_warp = bottom + mid_plane + lambd * top 

        # NOTE: for debug only
        # print("v_pts: ", v_pts)
        # print("bottom: ", bottom.shape)
        # print("top: ", top.shape)

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

        save_image(saliency_final, "saliency_final.png")

        return map_warp
    

if __name__ == '__main__':
    from PIL import Image
    import torchvision.transforms as transforms

    img_path = "/home/aghosh/Projects/2PCNet/Datasets/bdd100k/images/100k/train_debug/0a0a0b1a-7c39d841.jpg"
    v_pts = torch.tensor([560.4213953681794, 315.3206647347985])

    # input_shape = (720, 1280)
    homo = CuboidLayerGlobal()
    # homo = TripetLayerGlobal(input_shape)

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
    saliency = homo.forward(img_tensor, v_pts)
