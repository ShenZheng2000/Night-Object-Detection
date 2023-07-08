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
from shapely.geometry import Polygon, LineString, Point
from torchvision.transforms.functional import to_pil_image
from PIL import Image, ImageDraw
import torchvision.transforms.functional as TF
import kornia
from detectron2.data.transforms import ResizeTransform, HFlipTransform, NoOpTransform
import time


def make_path_blur(img, vanishing_point, change_size=False, size=15):

    device = img.device

    if change_size == False:

        # Call the segment_image() function
        # print(f"img {img.shape} vanishing_point {vanishing_point}")

        # start_time = time.time()

        segmented = segment_image(img, vanishing_point)

        # end_time_1 = time.time()
        # print(f"spent time1 = {end_time_1 - start_time}")

        # Apply motion blur to the segmented image
        combined_tensor = apply_blur_and_combine(segmented, size, device)

        # end_time_2 = time.time()
        # print(f"spent time2 = {end_time_2 - end_time_1}")

    elif change_size == True:
        # TODO: write code later
        pass

    return combined_tensor.unsqueeze(0)



def segment_image(img: torch.Tensor, pt: list) -> dict:
    """
    Segment an image into regions based on a point and the four corners.

    Parameters:
    img (torch.Tensor): The image tensor.
    pt (list): The point coordinates [x, y].

    Returns:
    dict: The segmented image regions.
    """

    # print("pt is", pt)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
    # squeeze to 3d if needed
    if len(img.shape) == 4:
        img = img.squeeze(0)

    pt_w = pt[0]
    pt_h = pt[1]

    img_c, img_h, img_w = img.shape

    # Convert the torch tensor to a PIL Image
    img_np = img.permute(1, 2, 0).cpu().numpy()
    img_pil = Image.fromarray(img_np)

    # Define corner points
    corners = [(0, 0), (img_w, 0), (img_w, img_h), (0, img_h)]
    side_top = [(0, 0), (img_w, 0)]
    side_right = [(img_w, 0), (img_w, img_h)]
    side_bottom = [(img_w, img_h), (0, img_h)]
    side_left = [(0, img_h), (0, 0)]

    # Define image border lines
    borders = [LineString(side_top),  # Top
               LineString(side_right),  # Right
               LineString(side_bottom),  # Bottom
               LineString(side_left)]  # Left

    # Create lines from point to corners and find intersections with image borders
    intersections = []
    for corner in corners:
        line = LineString([pt, corner])
        for border in borders:
            intersection = line.intersection(border)
            if not intersection.is_empty and intersection not in intersections:
                intersections.append(intersection)

    # Convert intersection Points to tuples and calculate their distances to the vanishing point
    intersections = [(intersection.x, intersection.y) for intersection in intersections]

    # Define the polygons based on the sorted intersection points
    polygons = [None, None, None, None]  # Placeholder for top, right, bottom, left polygons

    if pt_w > 0 and pt_w < img_w and pt_h > 0 and pt_h < img_h:  # point is in-bound
        for i in range(4):
            polygons[i] = Polygon([pt, intersections[i], intersections[(i+1)%4]])
    else:  # point is out-of-bound
        # Sort intersections by their distances to the vanishing point
        distances = [np.linalg.norm(np.array(pt) - np.array(intersection)) for intersection in intersections]
        sorted_intersections = [inter for _, inter in sorted(zip(distances, intersections))]

        trap = [sorted_intersections[0], sorted_intersections[1], sorted_intersections[-1], sorted_intersections[-2]]
        tri_1 = [sorted_intersections[1], sorted_intersections[-1], sorted_intersections[3]]
        tri_2 = [sorted_intersections[0], sorted_intersections[-2], sorted_intersections[2]]

        possible_polygons = [Polygon(trap), Polygon(tri_1), Polygon(tri_2)]

        # Assign polygons to the right position in the list (top, right, bottom, left)
        for poly in possible_polygons:
            # print("===========poly is", poly)
            for i, side in enumerate([side_top, side_right, side_bottom, side_left]):
                # print(f"Point(side[0]) = {Point(side[0])} Point(side[1]) = {Point(side[1])}", )
                if point_in_polygon(poly, Point(side[0])) and point_in_polygon(poly, Point(side[1])):
                    polygons[i] = poly
                    break

    # for i in range(len(polygons)):
        # print(f"polygons[{i}] is", polygons[i])

    # Segment image
    segmented = {}
    for i, polygon in enumerate(polygons):
        if polygon is None:  # Skip if polygon is missing
            continue

        mask = Image.new('L', img_pil.size, 0)
        ImageDraw.Draw(mask).polygon(list(map(tuple, polygon.exterior.coords)), outline=1, fill=1)
        mask = np.array(mask)
        segmented_image = np.array(img_pil) * np.dstack([mask]*3)

        # Compute left and right border middle
        lbm = [0, img_h / 2]
        rbm = [img_w, img_h / 2]

        # Compute motion blur direction
        if i == 0 or i == 2:  # Top and bottom polygons have no motion blur
            blur_direction = None
        elif i == 1:  # Right polygon
            blur_direction = math.degrees(math.atan2(rbm[1] - pt[1], rbm[0] - pt[0]))
        elif i == 3:  # Left polygon
            blur_direction = math.degrees(math.atan2(lbm[1] - pt[1], lbm[0] - pt[0]))

        segmented_image_pil = Image.fromarray(segmented_image)
        segmented[i] = {"image": segmented_image_pil, "blur_direction": blur_direction}

    return segmented



def point_in_polygon(poly, point):
    """Check if a point is in a polygon including the boundary."""
    for coord in list(poly.exterior.coords):
        if Point(coord).equals(point):
            return True
    return point.within(poly)


# def segment_image(img: torch.Tensor, pt: list) -> dict:
#     """
#     Segment an image into regions based on a point and the four corners.

#     Parameters:
#     img (torch.Tensor): The image tensor.
#     pt (list): The point coordinates [x, y].

#     Returns:
#     dict: The segmented image regions.
#     """

#     print("pt is", pt)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          

#     # squeeze to 3d if needed
#     if len(img.shape) == 4:
#         img = img.squeeze(0)

#     pt_w = pt[0]
#     pt_h = pt[1]

#     img_c, img_h, img_w = img.shape

#     # Convert the torch tensor to a PIL Image
#     img_np = img.permute(1, 2, 0).cpu().numpy()
#     img_pil = Image.fromarray(img_np)

#     # Define corner points
#     corners = [(0, 0), (img_w, 0), (img_w, img_h), (0, img_h)]

#     # Define image border lines
#     borders = [LineString([(0, 0), (img_w, 0)]),  # Top
#                LineString([(img_w, 0), (img_w, img_h)]),  # Right
#                LineString([(img_w, img_h), (0, img_h)]),  # Bottom
#                LineString([(0, img_h), (0, 0)])]  # Left

#     # Create lines from point to corners and find intersections with image borders
#     intersections = []
#     for corner in corners:
#         line = LineString([pt, corner])
#         for border in borders:
#             intersection = line.intersection(border)
#             if not intersection.is_empty and intersection not in intersections:
#                 intersections.append(intersection)

#     # Convert intersection Points to tuples and calculate their distances to the vanishing point
#     intersections = [(intersection.x, intersection.y) for intersection in intersections]

#     # Define the polygons based on the sorted intersection points
#     polygons = []
    
#     if pt_w > 0 and pt_w < img_w and pt_h > 0 and pt_h < img_h:  # point is in-bound
#         for i in range(4):
#             polygons.append(Polygon([pt, intersections[i], intersections[(i+1)%4]]))
#     else:  # point is out-of-bound

#         # Sort intersections by their distances to the vanishing point
#         distances = [np.linalg.norm(np.array(pt) - np.array(intersection)) for intersection in intersections]
#         sorted_intersections = [inter for _, inter in sorted(zip(distances, intersections))]

#         trap = [sorted_intersections[0], sorted_intersections[1], sorted_intersections[-1], sorted_intersections[-2]]
#         tri_1 = [sorted_intersections[1], sorted_intersections[-1], sorted_intersections[3]]
#         tri_2 = [sorted_intersections[0], sorted_intersections[-2], sorted_intersections[2]]

#         # Define trapezoid
#         polygons.append(Polygon(trap))
        
#         # Define remaining triangles
#         polygons.append(Polygon(tri_1))
#         polygons.append(Polygon(tri_2))

#     for i in range(len(polygons)):
#         print(f"polygons[{i}] is", polygons[i])

#     # Segment image
#     segmented = {}
#     for i, polygon in enumerate(polygons):
#         mask = Image.new('L', img_pil.size, 0)
#         ImageDraw.Draw(mask).polygon(list(map(tuple, polygon.exterior.coords)), outline=1, fill=1)
#         mask = np.array(mask)
#         segmented_image = np.array(img_pil) * np.dstack([mask]*3)

#         # Compute left and right border middle
#         lbm = [0, img_h / 2]
#         rbm = [img_w, img_h / 2]

#         # Compute motion blur direction
#         if i == 0 or i == 2:  # Top and bottom polygons have no motion blur
#             blur_direction = None
#         elif i == 1:  # Right polygon
#             blur_direction = math.degrees(math.atan2(rbm[1] - pt[1], rbm[0] - pt[0]))
#         elif i == 3:  # Left polygon
#             blur_direction = math.degrees(math.atan2(lbm[1] - pt[1], lbm[0] - pt[0]))

#         segmented_image_pil = Image.fromarray(segmented_image)
#         segmented[i] = {"image": segmented_image_pil, "blur_direction": blur_direction}

#     return segmented


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
    # image_path = "/root/autodl-tmp/Datasets/bdd100k/images/100k/train/0081da60-5fa22cc6.jpg"
    # image_path = "/root/autodl-tmp/Datasets/bdd100k/images/100k/train/4285eaaa-5c506f11.jpg"
    image_path = "/root/autodl-tmp/Datasets/bdd100k/images/100k/train/5eebd0b4-b90b8950.jpg"
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


def is_out_of_bounds(pt, img_width, img_height):
    if (pt[0] < 0 and pt[1] < 0) or \
        (pt[0] > img_width and pt[1] < 0) or \
            (pt[0] < 0 and pt[1] > img_height) or \
                (pt[0] > img_width and pt[1] > img_height):
        return True
    return False


if __name__ == '__main__':
    # pt = [-615.628451105227, -615.628451105227]
    # print(is_out_of_bounds(pt, 720, 1280))
    debug_path_blur()
