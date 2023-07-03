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
from .light import get_keypoints, generate_light
import time


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

    def apply_motion_blur(self, img, motion_blur=False, motion_blur_rand=False):
        if motion_blur:
            img = motion_blur_adjustable(img)

        if motion_blur_rand:
            img = motion_blur_adjustable(img, direction=R.random() * 360)

        return img


    def apply_light_render(self, img, ins, file_name, key_point, light_render, light_high, flip):

        if light_render:
            keypoints_list = get_keypoints(file_name, key_point, self.ratio, flip)
            for keypoints in keypoints_list:
                img = generate_light(image=img, ins=ins, keypoints=keypoints, HIGH=light_high)

        return img


    def aug(self,
                x,
                motion_blur=False,
                motion_blur_rand=False,
                light_render=False,
                light_high=None,
                key_point=None):
        for sample in x:

            # print("sample is", sample)

            # read filenames and transforms
            file_name = sample['file_name']
            transform_list = sample['transform']

            self.ratio = None
            flip = None

            for transform in transform_list:
                if isinstance(transform, ResizeTransform):
                    if self.ratio is None:
                        self.ratio = transform.new_h / transform.h
                elif isinstance(transform, (HFlipTransform, NoOpTransform)):
                    flip = transform

            # print("transform_list is", transform_list)

            # read images and instances
            img = sample['image'].cuda()
            ins = sample['instances']
            g_b_flag = True

            # TODO: comment this for debug only
            # Guassian Blur
            if R.random()>0.5:
                img = self.gaussian(img)
            
            cln_img_zero = img.detach().clone()

            # Gamma
            if R.random()>0.5:
                cln_img = img.detach().clone()
                val = 1/(R.random()*0.8+0.2)
                img = T.functional.adjust_gamma(img,val)
                img= self.mask_img(img,cln_img)
                g_b_flag = False
            
            # Brightness
            if R.random()>0.5 or g_b_flag:
                cln_img = img.detach().clone()
                val = R.random()*0.8+0.2
                img = T.functional.adjust_brightness(img,val)
                img= self.mask_img(img,cln_img)

            # Contrast
            if R.random()>0.5:
                cln_img = img.detach().clone()
                val = R.random()*0.8+0.2
                img = T.functional.adjust_contrast(img,val)
                img= self.mask_img(img,cln_img)
            img= self.mask_img(img,cln_img_zero)

            prob = 0.5
            while R.random()>prob:
                img=self.gaussian_heatmap(img)
                prob+=0.1

            #Noise
            if R.random()>0.5:
                n = torch.clamp(torch.normal(0,R.randint(50),img.shape),min=0).cuda()
                img = n + img
                img = torch.clamp(img,max = 255).type(torch.uint8)

            # Apply motion blur
            # if True:
            if R.random()>0.5:
                img = self.apply_motion_blur(img, motion_blur=motion_blur, motion_blur_rand=motion_blur_rand)

            # Light Rendering
            # if True:
            if R.random()>0.5:
                # start_time = time.time()
                img = self.apply_light_render(img, ins, file_name, key_point, light_render, light_high, flip)
                # end_time = time.time()
                # print(f"apply_light_render elapsed {end_time - start_time}:.3f")

            # send image to cpu
            sample['image'] = img.cpu()                

        return x

if __name__ == '__main__':
    pass