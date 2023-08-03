import torch
import torchvision.transforms as T
from numpy import random as R
import torch.nn.functional as F
from torchvision.utils import save_image
from PIL import Image
import os
import sys
import math
from detectron2.structures import Boxes
from detectron2.data.transforms import ResizeTransform, HFlipTransform, NoOpTransform
import json
from .blur import motion_blur_adjustable
from .path_blur import make_path_blur, get_vanising_points, is_out_of_bounds
from .light import get_keypoints, generate_light
import time
from .grid_generator import CuboidGlobalKDEGrid
from .fovea import apply_warp_aug, apply_unwarp, extract_ratio_and_flip
# from .fixed_grid import FixedGrid
import random


class NightAug:
    def __init__(self):
        self.gaussian = T.GaussianBlur(11,(0.1,2.0))

    def mask_img(self,img,cln_img):
        while R.random()>0.4:
            x1 = R.randint(img.shape[1])
            x2 = R.randint(img.shape[1])
            y1 = R.randint(img.shape[2])
            y2 = R.randint(img.shape[2])
            img[:,x1:x2,y1:y2]=cln_img[:,x1:x2,y1:y2]
        return img

    def gaussian_heatmap(self,x):
        """
        It produces single gaussian at a random point
        """
        sig = torch.randint(low=1,high=150,size=(1,)).cuda()[0]
        image_size = x.shape[1:]
        center = (torch.randint(image_size[0],(1,))[0].cuda(), torch.randint(image_size[1],(1,))[0].cuda())
        x_axis = torch.linspace(0, image_size[0]-1, image_size[0]).cuda() - center[0]
        y_axis = torch.linspace(0, image_size[1]-1, image_size[1]).cuda() - center[1]
        xx, yy = torch.meshgrid(x_axis, y_axis)
        kernel = torch.exp(-0.5 * (torch.square(xx) + torch.square(yy)) / torch.square(sig))
        new_img = (x*(1-kernel) + 255*kernel).type(torch.uint8)
        return new_img

    def apply_motion_blur(self, img, motion_blur=False, motion_blur_vet=False, motion_blur_rand=False):
        if motion_blur:
            img = motion_blur_adjustable(img)

        elif motion_blur_vet:
            img = motion_blur_adjustable(img, direction=90)

        elif motion_blur_rand:
            img = motion_blur_adjustable(img, direction=R.random() * 360)

        return img


    def apply_path_blur(self, img, vanishing_point, path_blur_cons=False, 
                        path_blur_var=False,
                        path_blur_new=False, T_z_values=None, zeta_values=None):
        # start_time = time.time()
        # print(f"Before img min {img.min()} {img.max()}") # 0, 255
        # print("img shape", img.shape) # C, H, W

        # if vanishing_point is None:
        #     return img

        # vanishing_point = get_vanising_points(file_name, vanishing_point, self.ratio, flip)

        img_height, img_width = img.shape[1:]

        if path_blur_new:
            from .reblur import make_path_blur_new
            T_z = random.uniform(float(T_z_values[0]), float(T_z_values[1]))
            zeta = random.uniform(float(zeta_values[0]), float(zeta_values[1]))
            # print(f"T_z = {T_z}, zeta = {zeta}")
            img = make_path_blur_new(img, vanishing_point, T_z, zeta)
            img = img * 255.0
            # print(f"img shape {img.shape} min {img.min()} max {img.max()}")
        else:
            # Skip path blur if any element of vanishing_point is negative
            if is_out_of_bounds(vanishing_point, img_width, img_height):
                print("Warning: Vanishing point both coords outside. Skipping path blur.")
            else:
                if path_blur_cons:
                    img = make_path_blur(img, vanishing_point, change_size=False)
                    img = img * 255.0

                elif path_blur_var:
                    img = make_path_blur(img, vanishing_point, change_size=True)
                    img = img * 255.0

        # reshape 4d to 3d
        if len(img.shape) == 4:
            img = img.squeeze(0)
        
        return img



    def apply_light_render(self, img, ins, file_name, key_point, light_render, light_high, flip, reflect_render, hot_tail):

        if light_render:
            keypoints_list = get_keypoints(file_name, key_point, self.ratio, flip)
            for keypoints in keypoints_list:
                img = generate_light(image=img, 
                                    ins=ins, 
                                    keypoints=keypoints, 
                                    HIGH=light_high, 
                                    reflect_render=reflect_render,
                                    hot_tail=hot_tail)

        return img

    def aug(self,
                x,
                motion_blur=False,
                motion_blur_vet=False,
                motion_blur_rand=False,
                light_render=False,
                light_high=None,
                key_point=None,
                vanishing_point=None, 
                path_blur_cons=False,
                path_blur_var=False,
                reflect_render=False,
                two_pc_aug=True,
                aug_prob=0.5,
                hot_tail=False,
                path_blur_new=False,
                T_z_values=None,
                zeta_values=None,
                warp_aug=False,
                warp_aug_lzu=False,
                grid_net=None,
                use_debug=False):

        # NOTE: add debug mode here
        if use_debug:
            aug_prob = 0.0
                            
        for sample in x:

            # read filenames and transforms
            file_name = sample['file_name']
            transform_list = sample['transform']

            self.ratio, flip = extract_ratio_and_flip(transform_list)

            # self.ratio = None
            # flip = None

            # for transform in transform_list:
            #     if isinstance(transform, ResizeTransform):
            #         if self.ratio is None:
            #             self.ratio = transform.new_h / transform.h
            #     elif isinstance(transform, (HFlipTransform, NoOpTransform)):
            #         flip = transform

            # print("transform_list is", transform_list)

            # read images and instances
            img = sample['image'].cuda()
            ins = sample['instances']
            # print(f"file_name {file_name} Before ins {ins}")
            g_b_flag = True
            
            # print("ins.gt_boxes.tensor is", ins.gt_boxes.tensor) # x1, y1, x2, y2
            # print("ins.gt_boxes.tensor.shape is", ins.gt_boxes.tensor.shape) # [N, 4]
            # print("two_pc_aug is", two_pc_aug)
            # print("aug_prob is", aug_prob)

            if two_pc_aug:

                # Guassian Blur
                if R.random()>aug_prob:
                    img = self.gaussian(img)
                
                cln_img_zero = img.detach().clone()

                # Gamma
                if R.random()>aug_prob:
                    cln_img = img.detach().clone()
                    val = 1/(R.random()*0.8+0.2)
                    img = T.functional.adjust_gamma(img,val)
                    img= self.mask_img(img,cln_img)
                    g_b_flag = False
                
                # Brightness
                if R.random()>aug_prob or g_b_flag:
                    cln_img = img.detach().clone()
                    val = R.random()*0.8+0.2
                    img = T.functional.adjust_brightness(img,val)
                    img= self.mask_img(img,cln_img)

                # Contrast
                if R.random()>aug_prob:
                    cln_img = img.detach().clone()
                    val = R.random()*0.8+0.2
                    img = T.functional.adjust_contrast(img,val)
                    img= self.mask_img(img,cln_img)
                img= self.mask_img(img,cln_img_zero)

                prob = aug_prob
                while R.random()>prob:
                    img=self.gaussian_heatmap(img)
                    prob+=0.1

                #Noise
                if R.random()>aug_prob:
                    n = torch.clamp(torch.normal(0,R.randint(50),img.shape),min=0).cuda()
                    img = n + img
                    img = torch.clamp(img,max = 255).type(torch.uint8)

            # Apply motion blur
            if R.random()>aug_prob:
                img = self.apply_motion_blur(img, 
                                             motion_blur=motion_blur, 
                                             motion_blur_vet=motion_blur_vet,
                                             motion_blur_rand=motion_blur_rand)

            # Light Rendering
            if R.random()>aug_prob:
                # start_time = time.time()
                img = self.apply_light_render(img, ins, file_name, key_point, light_render, light_high, flip, reflect_render, hot_tail)
                # end_time = time.time()
                # print(f"apply_light_render elapsed {end_time - start_time}:.3f")

            # save_image(img / 255.0, 'before_blur.png')

            # NOTE: pre-compute vp here to save time
            if vanishing_point is not None:
                new_vanishing_point = get_vanising_points(file_name, vanishing_point, self.ratio, flip)

                if R.random()>aug_prob:
                    img = self.apply_path_blur(img, 
                                                new_vanishing_point, 
                                                path_blur_cons, 
                                                path_blur_var, 
                                                path_blur_new,
                                                T_z_values,
                                                zeta_values)
                    # save_image(img / 255.0, 'after_blur.png')


                if R.random()>aug_prob:
                    # NOTE: only warp_aug here. use xxx.lzu later in rcnn
                    if warp_aug:
                        img, ins, grid = apply_warp_aug(img, ins, new_vanishing_point, 
                                                    warp_aug, warp_aug_lzu, grid_net)
                    
            
            # NOTE: format should be same as previous
            # print(f"file_name {file_name} After ins {ins}")
            
            # send image to cpu
            sample['image'] = img.cpu()                

        return x

if __name__ == '__main__':
    pass