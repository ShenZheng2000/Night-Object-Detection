# Steps
    # (1) visualize (dis_to_vp VS object size) (DONE)
    # (2) object size  => blur size (DONE)
    # (3) object center to vp => blur direction (DONE)
    # (4) debug vis
    # (5) tune blur size
    # (6) put to 2pcnet for visualization
    # NOTE: be aware of flips and resize operations at (6), and perhaps flip blur dirs
    # (7) debug vis
    # (8) start training


import torch
import torchvision.transforms as T
from numpy import random as R
import torch.nn.functional as F
from PIL import Image
import os
import json
import math
from PIL import Image, ImageFilter
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import Polygon
import torch
from PIL import Image
from torchvision.transforms.functional import to_pil_image


def create_vp_masks():
    pass


def calculate_blur_sizes():
    pass


def create_motion_blur_kernel():
    pass


def make_path_blur(image, size, vp):
    # Ensure image is a 4D tensor (Batch Size, Channels, Height, Width)
    if len(image.shape) != 4:
        image = image.unsqueeze(0)

    # Convert image to float
    image = image.float()

    # Get the number of channels
    num_channels = image.shape[1]

    # Get the image dimensions
    _, _, height, width = image.shape

    # Create empty output image tensor
    output = torch.zeros_like(image)

    # Split the image into four triangles based on VP
    # This can be done by applying four different masks to the image
    # Each mask corresponds to one of the four triangular parts
    masks = create_vp_masks(height, width, vp) # TODO: replace this function

    # Apply no blur to the top and bottom triangles
    output += image * masks['top']
    output += image * masks['bottom']

    # Apply blur to the left and right triangles
    for side in ['left', 'right']:
        # Get the mask for this side
        mask = masks[side]

        # Apply the mask to get the side image
        side_image = image * mask

        # Calculate the blur size for each pixel in the side image
        # This should increase from 1 at the VP to 'size' at the border
        blur_sizes = calculate_blur_sizes(height, width, vp, size, side)

        # Apply the blur to the side image
        # The direction of the blur should be from the border to the VP
        # This means that the kernel should be flipped depending on the side
        for y in range(height):
            for x in range(width):
                # Calculate the blur kernel for this pixel
                blur_size = blur_sizes[y, x]
                if blur_size <= 2:
                    continue
                kernel = create_motion_blur_kernel(num_channels, blur_size, side)
                
                # Apply the kernel to this pixel
                output[:, :, y, x] += F.conv2d(side_image[:, :, y:y+blur_size, x:x+blur_size], kernel, padding=0, groups=num_channels)

    return output


def segment_image(img: torch.Tensor, pt: list, blur_directions: dict) -> dict:
    """
    Segment an image into triangles based on a point and the four corners.

    Parameters:
    img (torch.Tensor): The image tensor.
    pt (list): The point coordinates [x, y].
    blur_directions (dict): Dictionary specifying the blur directions for each triangle.
                            Example: {"top": "up", "right": "right", "bottom": "down", "left": "left"}

    Returns:
    dict: The segmented image triangles.
    """
    # Validate point
    if not (0 <= pt[0] < img.shape[1] and 0 <= pt[1] < img.shape[2]):
        raise ValueError('Point must be within the image boundaries.')

    # Convert the torch tensor to a PIL Image
    img_np = img.permute(1, 2, 0).numpy()
    img_pil = Image.fromarray(img_np)

    # Get image size
    img_width, img_height = img_pil.size

    # Define corner points
    corners = [(0, 0), (img_width, 0), (img_width, img_height), (0, img_height)]

    # Create triangles
    triangles = [Polygon([pt, corners[i], corners[(i+1)%4]]) for i in range(4)]

    # Segment image
    segmented = {}
    for i, triangle in enumerate(triangles):
        mask = Image.new('L', img_pil.size, 0)
        ImageDraw.Draw(mask).polygon(list(map(tuple, triangle.exterior.coords)), outline=1, fill=1)
        mask = np.array(mask)
        segmented_image = np.array(img_pil) * np.dstack([mask]*3)

        # Apply motion blur based on the blur direction
        if i == 0:  # Top triangle
            direction = blur_directions.get("top", "up")
            if direction == "up":
                segmented_image = Image.fromarray(segmented_image).transpose(Image.FLIP_TOP_BOTTOM)
        elif i == 1:  # Right triangle
            direction = blur_directions.get("right", "right")
            if direction == "left":
                segmented_image = Image.fromarray(segmented_image).transpose(Image.FLIP_LEFT_RIGHT)
        elif i == 2:  # Bottom triangle
            direction = blur_directions.get("bottom", "down")
            if direction == "down":
                segmented_image = Image.fromarray(segmented_image).transpose(Image.FLIP_TOP_BOTTOM)
        elif i == 3:  # Left triangle
            direction = blur_directions.get("left", "left")
            if direction == "right":
                segmented_image = Image.fromarray(segmented_image).transpose(Image.FLIP_LEFT_RIGHT)

        segmented_image_pil = Image.fromarray(segmented_image)
        segmented[i] = segmented_image_pil

    return segmented


if __name__ == '__main__':

    # Example usage
    image_path = "/root/autodl-tmp/Datasets/bdd100k/images/100k/train/0081da60-5fa22cc6.jpg"
    json_file_path = "/root/autodl-tmp/Datasets/VP/train_day.json"

    # Read the vanishing points from the JSON file
    with open(json_file_path, 'r') as file:
        vanishing_points = json.load(file)

    vanishing_points = {os.path.basename(k): v for k, v in vanishing_points.items()}

    # Get the vanishing point for the current image
    image_basename = os.path.basename(image_path)
    vanishing_point = vanishing_points[image_basename]

    # Load the RGB image from the file
    img = Image.open(image_path).convert('RGB')

    # Convert the RGB image to a torch tensor
    img_tensor = torch.tensor(np.array(img)).permute(2, 0, 1)

    # Call the segment_image() function
    segmented = segment_image(img_tensor, vanishing_point)

    # Save the segmented triangles
    for key, triangle in segmented.items():
        triangle_image = to_pil_image(triangle)
        triangle_image.save(f'segmented_{key}_triangle.png')