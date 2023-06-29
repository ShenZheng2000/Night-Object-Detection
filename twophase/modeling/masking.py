import random
import warnings

import kornia
import numpy as np
import torch
from einops import repeat
from torch import nn, Tensor
from torch.nn import functional as F

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Old function => for 4d tensor
# def resize(input,
#            size=None,
#            scale_factor=None,
#            mode='nearest',
#            align_corners=None,
#            warning=True):
#     if warning:
#         if size is not None and align_corners:
#             input_h, input_w = tuple(int(x) for x in input.shape[2:])
#             output_h, output_w = tuple(int(x) for x in size)
#             if output_h > input_h or output_w > output_h:
#                 if ((output_h > 1 and output_w > 1 and input_h > 1
#                      and input_w > 1) and (output_h - 1) % (input_h - 1)
#                         and (output_w - 1) % (input_w - 1)):
#                     warnings.warn(
#                         f'When align_corners={align_corners}, '
#                         'the output would more aligned if '
#                         f'input size {(input_h, input_w)} is `x+1` and '
#                         f'out size {(output_h, output_w)} is `nx+1`')
#     return F.interpolate(input, size, scale_factor, mode, align_corners)

# New function => for 3d tensor
def resize(input, size=None, scale_factor=None, mode='nearest', align_corners=None, warning=True):
    # CHANGE 1: Add an extra dimension to make input a 4D tensor
    input = input.unsqueeze(0)

    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')

    output = F.interpolate(input, size, scale_factor, mode, align_corners)

    # CHANGE 2: Remove the extra dimension to return a 3D tensor
    output = output.squeeze(0)

    return output



def strong_transform(param, data):
    # Adding an extra dimension
    data = data.unsqueeze(0)
    data = color_jitter(
        color_jitter=param['color_jitter'],
        s=param['color_jitter_s'],
        p=param['color_jitter_p'],
        mean=param['mean'],
        std=param['std'],
        data=data)
    data = data.float() # avoid reflection padding errors
    data = gaussian_blur(blur=param['blur'], data=data)

    # Removing the extra dimension
    data = data.squeeze(0)
    return data


def denorm(img, mean, std):
    return img.mul(std).add(mean)


def renorm(img, mean, std):
    return img.sub(mean).div(std)


def color_jitter(color_jitter, mean, std, data, s=.25, p=.2):
    # s is the strength of colorjitter
    if color_jitter > p:
        mean = torch.as_tensor(mean, device=data.device)
        mean = repeat(mean, 'C -> B C 1 1', B=data.shape[0], C=3)
        std = torch.as_tensor(std, device=data.device)
        std = repeat(std, 'C -> B C 1 1', B=data.shape[0], C=3)
        if isinstance(s, dict):
            seq = nn.Sequential(kornia.augmentation.ColorJitter(**s))
        else:
            seq = nn.Sequential(
                kornia.augmentation.ColorJitter(
                    brightness=s, contrast=s, saturation=s, hue=s))
        data = denorm(data, mean, std)
        data = seq(data)
        data = renorm(data, mean, std)
    return data


def gaussian_blur(blur, data):
    if blur > 0.5:
        sigma = np.random.uniform(0.15, 1.15)
        kernel_size_y = int(
            np.floor(
                np.ceil(0.1 * data.shape[2]) - 0.5 +
                np.ceil(0.1 * data.shape[2]) % 2))
        kernel_size_x = int(
            np.floor(
                np.ceil(0.1 * data.shape[3]) - 0.5 +
                np.ceil(0.1 * data.shape[3]) % 2))
        kernel_size = (kernel_size_y, kernel_size_x)
        seq = nn.Sequential(
            kornia.filters.GaussianBlur2d(
                kernel_size=kernel_size, sigma=(sigma, sigma)))
        data = seq(data)
    return data


# NOTE: this is for before 6/21/2023
# class Masking(nn.Module):
#     def __init__(self, block_size, ratio, color_jitter_s, color_jitter_p, blur, mean, std):
#         super(Masking, self).__init__()

#         self.block_size = block_size
#         self.ratio = ratio

#         # self.augmentation_params = None
#         # if (color_jitter_p > 0 and color_jitter_s > 0) or blur:
#         #     print('[Masking] Use color augmentation.')
#         #     self.augmentation_params = {
#         #         'color_jitter': random.uniform(0, 1),
#         #         'color_jitter_s': color_jitter_s,
#         #         'color_jitter_p': color_jitter_p,
#         #         'blur': random.uniform(0, 1) if blur else 0,
#         #         'mean': mean,
#         #         'std': std
#         #     }

    # @torch.no_grad()
    # def forward(self, img: Tensor):
    #     img = img.clone()

    #     # NOTE: enforce shape here
    #     if len(img.shape) == 3:
    #         _, H, W = img.shape

    #     # print("img.device", img.device)
    #     # print(f"Before Aug: img min {img.min()} max {img.max()}")

    #     # NOTE: skip strong_transform for now because it has bugs
    #     # if self.augmentation_params is not None:
    #     #     img = strong_transform(self.augmentation_params, data=img.clone())

    #     # mshape = B, 1, round(H / self.block_size), round(W / self.block_size)
    #     mshape = 1, round(H / self.block_size), round(W / self.block_size)
    #     input_mask = torch.rand(mshape, device=img.device)
    #     input_mask = (input_mask > self.ratio).float()
    #     input_mask = resize(input_mask, size=(H, W))

    #     # print(f"After Aug: img min {img.min()} max {img.max()}")
    #     # print(f"input_mask min {input_mask.min()} max {input_mask.max()}")

    #     # NOTE: create a gray image of the same size as input image
    #     gray_img = torch.full_like(img, 128)

    #     # NOTE:  combine the original image and the gray image using the mask
    #     masked_img = img * input_mask + gray_img * (1 - input_mask)

    #     return masked_img




# # NOTE: this is for 6/21/2023 night
# class Masking(nn.Module):
#     def __init__(self, block_size, ratio, color_jitter_s, color_jitter_p, blur, mean, std):
#         super(Masking, self).__init__()

#         self.block_size = block_size
#         self.ratio = ratio

#         # self.augmentation_params = None
#         # if (color_jitter_p > 0 and color_jitter_s > 0) or blur:
#         #     print('[Masking] Use color augmentation.')
#         #     self.augmentation_params = {
#         #         'color_jitter': random.uniform(0, 1),
#         #         'color_jitter_s': color_jitter_s,
#         #         'color_jitter_p': color_jitter_p,
#         #         'blur': random.uniform(0, 1) if blur else 0,
#         #         'mean': mean,
#         #         'std': std
#         #     }

#     @torch.no_grad()
#     def forward(self, img: Tensor, depth_map: Tensor = None):
#         img = img.clone()

#         if len(img.shape) == 3:
#             _, H, W = img.shape

#         # mshape = 1, round(H / self.block_size), round(W / self.block_size)

#         # NOTE: force int consist here
#         mshape = 1, H // self.block_size, W // self.block_size

#         initial_mask = torch.rand(mshape, device=img.device)

#         if depth_map is not None:
#             # print("using depth map here")
#             # Convert depth_map to grayscale and normalize to [0,1]
#             grayscale_depth_map = depth_map.float().mean(dim=0) / 255

#             # Compute average depth for each block
#             # print("grayscale_depth_map shape", grayscale_depth_map.shape)
#             avg_depth_map = F.avg_pool2d(grayscale_depth_map.unsqueeze(0), self.block_size)
#             # print("avg_depth_map shape", avg_depth_map.shape)
#             # print("initial_mask shape", initial_mask.shape)

#             # Construct the input_mask
#             input_mask = (initial_mask > (avg_depth_map * self.ratio)).float()

#         else:
#             input_mask = (initial_mask > self.ratio).float()
        
#         input_mask = resize(input_mask, size=(H, W))
#         gray_img = torch.full_like(img, 128)
#         masked_img = img * input_mask + gray_img * (1 - input_mask)

#         return masked_img




class Masking(nn.Module):
    def __init__(self, block_size, ratio):
        super(Masking, self).__init__()

        self.block_size = block_size
        self.ratio = ratio

    @torch.no_grad()
    def forward(self, img: Tensor, depth_map: Tensor = None):
        img = img.clone()

        if len(img.shape) == 3:
            _, H, W = img.shape

        mshape = 1, H // self.block_size, W // self.block_size 

        initial_mask = torch.rand(mshape, device=img.device)

        if depth_map is not None:

            if depth_map.max() > 1.0:
                depth_map = depth_map / 255

            grayscale_depth_map = depth_map.float().mean(dim=0)
            avg_depth_map = F.avg_pool2d(grayscale_depth_map.unsqueeze(0), self.block_size)

            resized_mask = torch.zeros((H, W), 
                                        dtype=initial_mask.dtype, 
                                        device=initial_mask.device)
            input_mask = (initial_mask > self.ratio).float()

            for i in range(0, H, self.block_size):
                for j in range(0, W, self.block_size):
                    # Ensure i, j are within array dimensions
                    i_ind = min(i // self.block_size, initial_mask.shape[1] - 1)
                    j_ind = min(j // self.block_size, initial_mask.shape[2] - 1)
                    block_value = input_mask[:, i_ind, j_ind]
                    block_depth = avg_depth_map[:, i_ind, j_ind]

                    # Adjust block size for border cases
                    curr_block_height = min(self.block_size, H - i)
                    curr_block_width = min(self.block_size, W - j)

                    if block_value.item() == 0:
                        region_depth = block_depth.item()
                        region = generate_region(region_depth, curr_block_width, curr_block_height)
                    else:
                        region = block_value.unsqueeze(1).repeat(curr_block_height, curr_block_width)
                    
                    resized_mask[i:i+curr_block_height, j:j+curr_block_width] = region
            input_mask = resized_mask.unsqueeze(0)

        else:
            input_mask = (initial_mask > self.ratio).float()
            
        input_mask = resize(input_mask, size=(H, W))
        gray_img = torch.full_like(img, 128)
        masked_img = img * input_mask + gray_img * (1 - input_mask)

        return masked_img



def generate_region(depth: float, width: int, height: int) -> torch.Tensor:
    # Generate a tensor of the specified size filled with 1's
    region = torch.ones(height, width)

    # Compute the number of 0's based on the depth
    num_zeros = int(round(depth * width * height))

    # Randomly select num_zeros positions and set them to 0
    indices = torch.randperm(width * height)[:num_zeros]
    region.view(-1)[indices] = 0

    return region



# NOTE: for debug run only 
if __name__ == '__main__':

    import torch
    from torchvision import transforms
    from PIL import Image
    import os
    import sys
    import numpy as np
    import random

    # Set random seeds
    seed = 2023
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    tod = sys.argv[1]
    dep = sys.argv[2] == 'depth'
        
    # init masking
    masking = Masking(block_size=32, ratio=0.3)

    # define a transformation to convert image to tensor
    to_tensor = transforms.ToTensor()
    to_image = transforms.ToPILImage()

    # read data with tod
    if tod == 'day':
        input_image_path = '/root/autodl-tmp/Datasets/bdd100k/images/100k/train/0a0c3694-487a156f.jpg'
    elif tod == 'night': 
        input_image_path = '/root/autodl-tmp/Datasets/bdd100k/images/100k/train/0a0cc110-7f2fd761.jpg'
        if dep:
            input_depth_path = '/root/autodl-tmp/Datasets/Depth/train__night_dawn_dusk/0a0cc110-7f2fd761.png'

    input_image = Image.open(input_image_path)
    input_image_tensor = to_tensor(input_image)

    if dep:
        input_depth_image = Image.open(input_depth_path).convert('L')  # Convert to grayscale
        input_depth_tensor = to_tensor(input_depth_image)

        # If the depth image is grayscale, it'll have only 1 channel, so repeat it to have the same number of channels as the input image
        input_depth_tensor = input_depth_tensor.repeat(input_image_tensor.shape[0], 1, 1)
        
        # add masking
        output_image_tensor = masking(input_image_tensor, input_depth_tensor)
    else:
        # add masking
        output_image_tensor = masking(input_image_tensor)

    # convert tensors back to images
    input_image = to_image(input_image_tensor)
    output_image = to_image(output_image_tensor)

    # save images
    os.makedirs(f'masking/{tod}_{dep}', exist_ok=True)
    input_image.save(f'masking/{tod}_{dep}/input_image.jpg')
    output_image.save(f'masking/{tod}_{dep}/output_image.jpg')