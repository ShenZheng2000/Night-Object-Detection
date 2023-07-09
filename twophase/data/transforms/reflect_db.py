# Purpose: render reflection based on a single light 
# TODO: later move this to light.py because light and reflection happens together
# TODO: utilize keypoints of wheels to extract reflection start point
# TODO: optimize code efficiency
# Later: consider reflection due to multiple light sources 
# Later: consider refleciton dues to road surface unevenness (i.e. not a flat mirror plane)

# elevation (no need for now): 
    # head/tail-light: 0.6-0.9m
    # camera: 1.2-1.5m

# Think: do headlight or taillight contribute to reflection => do both for now

# Decide reflection area
    # light source coords = (x1, y1), obtained from light keypoints
    # reflect coords = (x1, y2)
        # y2 selected randomly from wheel keypoints to image bottom

# Fill reflection area
    # similar to gaussian heatmap
    # area rect instead of square

import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from PIL import Image
import os
import time


def gaussian_heatmap_kernel(height, width, ch, cw, y1, y2, HIGH, wei, device, use_reflect=False):
    """
    This function generates a Gaussian heatmap (kernel) and a rectangular reflection of the light region in the kernel.

    Inputs:
    - height, width: The height and width of the image.
    - ch, cw: The center coordinates of the Gaussian distribution in the kernel.
    - y1, y2: The y coordinates defining the vertical span of the rectangular reflection region.
    - HIGH: The standard deviation of the Gaussian distribution in the kernel.
    - wei: A weight factor applied to HIGH to adjust the size of the light region.
    - device: The device on which to perform the calculations (e.g., 'cpu' or 'cuda').

    Outputs:
    - kernel: The generated Gaussian heatmap.
    - reflection: The rectangular reflection of the light region in the kernel.
    """
    HIGH = max(1, int(HIGH * wei))
    sig = torch.tensor(HIGH).to(device)
    center = (ch, cw)
    x_axis = torch.linspace(0, height-1, height).to(device) - center[0]
    y_axis = torch.linspace(0, width-1, width).to(device) - center[1]
    xx, yy = torch.meshgrid(x_axis, y_axis)
    kernel = torch.exp(-0.5 * (torch.square(xx) + torch.square(yy)) / torch.square(sig))

    if use_reflect:

        # Define the boundaries of the reflection
        x1 = max(0, cw - 3 * HIGH)
        x2 = min(width, cw + 3 * HIGH)

        # Compute the width and height of the rectangular region
        rect_width = x2 - x1
        rect_height = y2 - y1

        assert ch < y1 < y2 < height , print(f"rect_height invalid: ch = {ch}, y1 = {y1}, y2 = {y2}, height = {height}")

        # Crop the light region from the kernel
        light_region = kernel[ch-3*HIGH:ch+3*HIGH, cw-3*HIGH:cw+3*HIGH]
        
        # Resize the light region to fit the dimensions of the rectangular region
        light_region_resized = F.interpolate(light_region[None, None, ...], 
                                            size=(rect_height, rect_width), 
                                            mode='bilinear', 
                                            align_corners=False)[0, 0]
        
        # Create an empty tensor of the same size as the image
        reflection = torch.zeros((height, width), device=device)
        
        # Place the resized light region in the rectangular region in the image
        reflection[y1:y2, x1:x2] = light_region_resized

        return kernel, reflection

    else:
        return kernel


def main():
    device = 'cuda'

    # Load an image. Assume image is a numpy array
    image = Image.open('/root/autodl-tmp/Datasets/bdd100k/images/100k/train/000f8d37-d4c09a0f.jpg')

    # Convert to numpy and tensor, and add an extra dimension
    image_np = np.array(image)
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).float().to(device)

    # unsqueeze to 3d
    if len(image_tensor) != 3:
        image_tensor = image_tensor.squeeze(0)
        # print(f"image_tensor max {image_tensor.max()} {image_tensor.min()}")

    # Get the kernel and reflection from the function we defined before
    height, width = image_tensor.shape[1], image_tensor.shape[2]
    ch, cw, y1, y2, HIGH, wei = 450, 450, 500, 700, 50, 1.0  # Assume these are the parameters you want to use
    
    # start_time = time.time()
    kernel, reflection = gaussian_heatmap_kernel(height, width, ch, cw, y1, y2, HIGH, wei, device, use_reflect=True)
    # end_time = time.time()
    # print("gaussian_heatmap_kernel", (end_time - start_time))

    # Now we have one image and one kernel
    # Define a light color for visualization purpose
    light_color = torch.tensor([255., 255., 255.]).view(3, 1, 1).to(device)  # white color in RGB

    # Generate the lighted image by blending the original image with the kernel
    lighted_image = (image_tensor * (1 - kernel) + light_color * kernel).clamp(0, 255)

    # Similarly, add the reflection to the image
    lighted_reflected_image = (lighted_image * (1 - reflection) + light_color * reflection).clamp(0, 255)

    # Convert back to numpy array for visualization
    lighted_image_np = lighted_image.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    lighted_reflected_image_np = lighted_reflected_image.permute(1, 2, 0).cpu().numpy().astype(np.uint8)

    # Convert back to PIL Image and save
    original_img = Image.fromarray(image_np)
    lighted_img = Image.fromarray(lighted_image_np)
    lighted_reflected_img = Image.fromarray(lighted_reflected_image_np)

    out_dir = 'reflect_images'
    os.makedirs(out_dir, exist_ok=True)

    original_img.save(f'{out_dir}/original_img.jpg')
    lighted_img.save(f'{out_dir}/lighted_img.jpg')
    lighted_reflected_img.save(f'{out_dir}/lighted_reflected_img.jpg')


if __name__ == '__main__':
    main()