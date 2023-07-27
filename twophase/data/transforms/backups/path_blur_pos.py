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
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon
from torchvision.transforms.functional import to_pil_image
from PIL import Image, ImageDraw
import torchvision.transforms.functional as TF
import kornia
from detectron2.data.transforms import ResizeTransform, HFlipTransform, NoOpTransform
import time

# TODO: sometimes vp out of bound

def make_path_blur(img, vanishing_point, change_size=False, size=15):

    device = img.device

    if change_size == False:

        # Call the segment_image() function
        # print(f"img {img.shape} vanishing_point {vanishing_point}")

        start_time = time.time()

        segmented = segment_image(img, vanishing_point)

        end_time_1 = time.time()
        print(f"spent time1 = {end_time_1 - start_time}")

        # Apply motion blur to the segmented image
        combined_tensor = apply_blur_and_combine(segmented, size, device)

        end_time_2 = time.time()
        print(f"spent time2 = {end_time_2 - end_time_1}")

    elif change_size == True:
        # TODO: write code later
        pass

    return combined_tensor.unsqueeze(0)


def draw_arrow(image, start, end, color, thickness):
    """
    Draw an arrow on the image from start point to end point.

    Parameters:
    image (PIL.Image.Image): The image to draw on.
    start (tuple): The starting point of the arrow (x, y).
    end (tuple): The ending point of the arrow (x, y).
    color (tuple): The color of the arrow (R, G, B).
    thickness (int): The thickness of the arrow.

    Returns:
    None
    """
    draw = ImageDraw.Draw(image)
    xy = ((start[0], start[1]), (end[0], end[1]))  # Convert start and end points into a tuple of tuples
    draw.line(xy, fill=color, width=thickness)
    arrow_size = thickness * 3
    angle = 30  # Angle of arrowhead

    # Compute the angle between the line and the x-axis
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    line_angle = math.atan2(dy, dx)

    # Compute the coordinates of the arrowhead
    angle_rad = math.radians(angle)
    arrowhead1 = (end[0] - arrow_size * math.cos(line_angle + angle_rad),
                  end[1] - arrow_size * math.sin(line_angle + angle_rad))
    arrowhead2 = (end[0] - arrow_size * math.cos(line_angle - angle_rad),
                  end[1] - arrow_size * math.sin(line_angle - angle_rad))

    # Draw the arrowhead
    draw.line((end, arrowhead1), fill=color, width=thickness)
    draw.line((end, arrowhead2), fill=color, width=thickness)

    del draw


def debug_segment_image():
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
        triangle_image = triangle["image"]
        triangle_image.save(f'segmented_{key}_triangle.png')

        # Visualize the blur direction
        blur_direction = triangle["blur_direction"]
        if blur_direction is not None:
            draw_arrow(triangle_image, vanishing_point,
                    (int(vanishing_point[0] + 100 * math.cos(math.radians(blur_direction))),
                        int(vanishing_point[1] + 100 * math.sin(math.radians(blur_direction)))),
                    color=(255, 0, 0), thickness=10)
            triangle_image.save(f'segmented_{key}_triangle_with_blur_direction.png')


def segment_image(img: torch.Tensor, pt: list) -> dict:
    """
    Segment an image into triangles based on a point and the four corners.

    Parameters:
    img (torch.Tensor): The image tensor.
    pt (list): The point coordinates [x, y].

    Returns:
    dict: The segmented image triangles.
    """
    # Validate point
    # print("vp is", pt)
    # print("img.device is", img.device)
    
    # squeeze to 3d if needed
    if len(img.shape) == 4:
        img = img.squeeze(0)
    
    # print("img shape", img.shape)

    pt_w = pt[0]
    pt_h = pt[1]

    img_c, img_h, img_w = img.shape

    if not (0 <= pt_w <= img_w and 0 <= pt_h <= img_h):
        raise ValueError(f'Point must be within the image boundaries. \
                         Received point ({pt_w}, {pt_h}) for image size ({img_w}, {img_h}).')

    # Convert the torch tensor to a PIL Image
    img_np = img.permute(1, 2, 0).cpu().numpy()
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

        # Compute left and right border middle
        lbm = [0, img_height / 2]
        rbm = [img_width, img_height / 2]

        # Compute motion blur direction
        if i == 0 or i == 2:  # Top and bottom triangles have no motion blur
            blur_direction = None
        elif i == 1:  # Right triangle
            blur_direction = math.degrees(math.atan2(rbm[1] - pt[1], rbm[0] - pt[0]))
            # print(f"Right = {blur_direction}")
        elif i == 3:  # Left triangle
            blur_direction = math.degrees(math.atan2(lbm[1] - pt[1], lbm[0] - pt[0]))
            # print(f"Left = {blur_direction}")

        segmented_image_pil = Image.fromarray(segmented_image)
        segmented[i] = {"image": segmented_image_pil, "blur_direction": blur_direction}

    return segmented


def motion_blur(image, size, angle):
    """
    Synthesize motion blur on the image in the specified direction.

    Parameters:
    image (torch.Tensor): The input image tensor.
    size (int): The size of the motion blur kernel.
    angle (float): The angle of the motion blur direction in degrees.

    Returns:
    torch.Tensor: The motion blurred image tensor.
    """

    # Ensure image is a 4D tensor (Batch Size, Channels, Height, Width)
    if len(image.shape) != 4:
        image = image.unsqueeze(0)

    # Get the number of channels
    num_channels = image.shape[1]

    # Compute the motion blur kernel for the specified direction
    kernel = torch.zeros((1, 1, size, size), device=image.device)
    kernel[0, 0, size // 2, :] = 1.0  # Horizontal line

    # Convert angle tensor to scalar
    angle = torch.tensor([angle], device=image.device)

    # Rotate the kernel by the specified angle using kornia's rotation method
    rotated_kernel = kornia.geometry.rotate(kernel, angle)

    # Repeat the rotated kernel for each input channel
    kernel = rotated_kernel.repeat((num_channels, 1, 1, 1))

    # Normalize the kernel
    kernel /= size

    # Compute padding size
    padding = size // 2

    # Apply convolution with 'same' padding using PyTorch's function
    blurred_image = F.conv2d(image, kernel, padding=padding, stride=1, groups=num_channels)

    # Clamp pixel values to between 0 and 1
    blurred_image = blurred_image.clamp(0, 1)

    return blurred_image



# # New function => Fast
def apply_blur_and_combine(segmented, size, device):
    """
    Applies a motion blur to the segments of an image and then combines them.

    Parameters:
    segmented (dict): A dictionary containing the segmented images and blur directions.
    size (int): The size of the motion blur kernel.
    device (torch.device): The device (CPU or CUDA device) where to operate on.

    Returns:
    torch.Tensor: The combined image tensor.
    """
    # Apply motion blur to the segmented image
    combined_tensor = None
    
    for i, segment in segmented.items():
        img_pil = segment["image"]
        img_tensor = TF.to_tensor(img_pil).to(device)  # Convert the PIL Image to torch tensor and move to device

        if segment["blur_direction"] is not None:
            img_tensor = motion_blur(img_tensor, size, segment["blur_direction"])
            img_tensor = img_tensor.squeeze(0)
            
        if combined_tensor is None:
            combined_tensor = img_tensor
        else:
            combined_tensor += img_tensor

    # Clamp the values of the combined tensor within [0, 1]
    combined_tensor = combined_tensor.clamp(0, 1)
    
    return combined_tensor


# NOTE: old function => slow
# def apply_blur_and_combine(segmented, size, device):
#     """
#     Applies a motion blur to the segments of an image and then combines them.

#     Parameters:
#     segmented (dict): A dictionary containing the segmented images and blur directions.
#     size (int): The size of the motion blur kernel.
#     device (torch.device): The device (CPU or CUDA device) where to operate on.

#     Returns:
#     torch.Tensor: The combined image tensor.
#     """
#     # Apply motion blur to the segmented image
#     blurred_segments = {}
#     for i, segment in segmented.items():

#         img_pil = segment["image"]
#         img_tensor = TF.to_tensor(img_pil).to(device)  # Convert the PIL Image to torch tensor and move to device

#         if segment["blur_direction"] is not None:
#             blurred_img_tensor = motion_blur(img_tensor, size, segment["blur_direction"])
#             blurred_img_pil = TF.to_pil_image(blurred_img_tensor.squeeze(0).cpu())  # Move the tensor to cpu for PIL conversion
#             blurred_segments[i] = {
#                 "image": blurred_img_pil,
#                 "blur_direction": segment["blur_direction"]
#             }
#         else:
#             blurred_segments[i] = segment

#     # Initialize an empty tensor for combining the segments
#     combined_tensor = torch.zeros_like(TF.to_tensor(blurred_segments[0]['image']).to(device))  # Moved to device

#     # Combine the segments onto the tensor
#     for segment in blurred_segments.values():
#         img_tensor = TF.to_tensor(segment["image"]).to(device)  # Moved to device
#         combined_tensor += img_tensor

#     # Clamp the values of the combined tensor within [0, 1]
#     combined_tensor = combined_tensor.clamp(0, 1)
    
#     return combined_tensor


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


def debug_path_blur():
    # Example usage
    image_path = "/root/autodl-tmp/Datasets/bdd100k/images/100k/train/0081da60-5fa22cc6.jpg"
    json_file_path = "/root/autodl-tmp/Datasets/VP/train_day.json"

    # Load the RGB image from the file
    img = Image.open(image_path).convert('RGB')

    # Convert the RGB image to a torch tensor
    img_tensor = torch.tensor(np.array(img)).permute(2, 0, 1).cuda()

    # get vanishing point
    with open(json_file_path, 'r') as f:
        vanishing_point = json.load(f)
    vanishing_point = {os.path.basename(k): v for k, v in vanishing_point.items()}    

    vanishing_point = get_vanising_points(image_path, vanishing_point)

    combined_tensor = make_path_blur(img_tensor, vanishing_point, change_size=False)
    # print(f"shape {combined_tensor.shape} min {combined_tensor.min()} max {combined_tensor.max()}")

    # Convert the combined tensor to a PIL Image
    combined_image = TF.to_pil_image(combined_tensor.squeeze(0))

    # Save ori and  combined image
    img.save("original_image.jpg")
    combined_image.save("combined_image.jpg")


if __name__ == '__main__':
    debug_path_blur()
