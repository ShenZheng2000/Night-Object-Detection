import torch
import torchvision.transforms as T
from numpy import random as R
from .reblur import get_vanising_points, apply_path_blur, is_out_of_bounds
from .fovea import extract_ratio_and_flip

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
    
    def apply_two_pc_aug(self, img, aug_prob):
        cln_img_zero = img.detach().clone()
        g_b_flag = True

        if R.random() > aug_prob:
            img = self.gaussian(img)

        # Gamma
        if R.random() > aug_prob:
            cln_img = img.detach().clone()
            val = 1 / (R.random() * 0.8 + 0.2)
            img = T.functional.adjust_gamma(img, val)
            img = self.mask_img(img, cln_img)
            g_b_flag = False

        # Brightness
        if R.random() > aug_prob or g_b_flag:
            cln_img = img.detach().clone()
            val = R.random() * 0.8 + 0.2
            img = T.functional.adjust_brightness(img, val)
            img = self.mask_img(img, cln_img)

        # Contrast
        if R.random() > aug_prob:
            cln_img = img.detach().clone()
            val = R.random() * 0.8 + 0.2
            img = T.functional.adjust_contrast(img, val)
            img = self.mask_img(img, cln_img)
        img = self.mask_img(img, cln_img_zero)

        prob = aug_prob
        while R.random() > prob:
            img = self.gaussian_heatmap(img)
            prob += 0.1

        # Noise
        if R.random() > aug_prob:
            n = torch.clamp(torch.normal(0, R.randint(50), img.shape), min=0).cuda()
            img = n + img
            img = torch.clamp(img, max=255).type(torch.uint8)

        return img


    def aug(self,
                x,
                vanishing_point=None, 
                two_pc_aug=True,
                aug_prob=0.5,
                path_blur_new=True,
                T_z_values=None,
                zeta_values=None,
                use_debug=False):

        # NOTE: add debug mode here
        if use_debug:
            aug_prob = 0.0
                            
        for sample in x:

            # read filenames and transforms
            file_name = sample['file_name']
            transform_list = sample['transform']

            self.ratio, flip = extract_ratio_and_flip(transform_list)

            # read images and instances
            img = sample['image'].cuda()
            # ins = sample['instances']
            # print(f"file_name {file_name} Before ins {ins}")
            
            # print("ins.gt_boxes.tensor is", ins.gt_boxes.tensor) # x1, y1, x2, y2
            # print("ins.gt_boxes.tensor.shape is", ins.gt_boxes.tensor.shape) # [N, 4]
            # print("two_pc_aug is", two_pc_aug)
            # print("aug_prob is", aug_prob)

            if two_pc_aug:
                self.apply_two_pc_aug(img, aug_prob)
                
            # save_image(img / 255.0, 'before_blur.png')

            # NOTE: pre-compute vp here to save time
            if vanishing_point is not None:
                new_vanishing_point = get_vanising_points(file_name, vanishing_point, self.ratio, flip)

                if R.random()>aug_prob:

                    img_height, img_width = img.shape[1:]
                    if is_out_of_bounds(new_vanishing_point, img_width, img_height):
                        print("Warning: Vanishing point both coords outside. Skipping path blur.")

                    # NOTE: use elif here to use path blur only for inbound cases
                    elif path_blur_new:
                        print("Using Path BLur!")
                        # print(f"Before img min {img.min()} shape {img.shape}") # 0, [3,600,1067]
                        # print(f"new_vanishing_point is {new_vanishing_point} and len {len(new_vanishing_point)}")
                        img = apply_path_blur(img, 
                                        new_vanishing_point, 
                                        T_z_values,
                                        zeta_values)
                        # print(f"Before img max {img.max()} shape {img.shape}") # 255, [3,600,1067]

                # NOTE: skip this for now because using it at feature-level
                # if R.random()>aug_prob:
                #     # NOTE: only warp_aug here. use xxx.lzu later in rcnn
                #     if warp_aug:
                #         img, ins, grid = apply_warp_aug(img, ins, new_vanishing_point, 
                #                                         warp_aug, warp_aug_lzu, grid_net)
            
            # NOTE: format should be same as previous
            # print(f"file_name {file_name} After ins {ins}")
            
            # send image to cpu
            sample['image'] = img.cpu()                

        return x

if __name__ == '__main__':
    pass