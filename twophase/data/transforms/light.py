import torch
import os
import math
from detectron2.structures import Boxes
from detectron2.data.transforms import ResizeTransform, HFlipTransform, NoOpTransform
import json
import time
import random
import torch.nn.functional as F
import matplotlib
import numpy as np

# TODO: do not render reflection on other objects (skip for now)
# TODO: think about overfitting in path blur training (decrease prob or introduce early stop)


def convert_bbox_format(bbox):
    x_min, y_min, x_max, y_max = bbox
    return [x_min, y_min, x_max - x_min, y_max - y_min]  # [x1, y1, w, h]


# NOTE: gt_classes in here from 0-9 (instead of 1-10), so car is 2 instead of 3. 
def get_class_2_bboxes(ins, device):
    gt_classes = ins.gt_classes
    gt_boxes = ins.gt_boxes

    # move gt_classes to cuda
    gt_classes = gt_classes.to(device)

    # move gt_boxes to cuda
    gt_boxes_tensor = gt_boxes.tensor
    gt_boxes_tensor = gt_boxes_tensor.to(device)
    gt_boxes = Boxes(gt_boxes_tensor)

    # print(f"gt_classes {gt_classes} gt_boxes {gt_boxes}")
    # print(f"gt_classes device {gt_classes.device} gt_boxes device {gt_boxes.device}")

    # class_2_boxes = [box for i, box in enumerate(gt_boxes) if gt_classes[i] == 2]
    class_2_boxes = [box for box, class_id in zip(gt_boxes, gt_classes) if class_id == 2]
    # print(f"class_2_boxes {class_2_boxes}")

    return class_2_boxes



# for keypoint
# (1) specify path for keypoint folders in config (DONE)
# (2) use image basename to locate keypoint (DONE)
# (3) change keypoint format from long-list to smaller ones (DONE)
# (4) use keypoint in generate_ligt (DONE)
# (5) check visualization
    # NOTE: there is some resize and flip operations!!! (DONE)
    # NOTE: there is some color distortions (DONE)

def get_keypoints(file_name, key_point, ratio, flip_transform):
    # Create full path to the keypoints JSON file
    # print("file_name is", file_name)
    base_name = os.path.basename(file_name)
    kp_json = os.path.join(key_point, base_name + '.predictions.json')

    # Load the JSON file
    with open(kp_json, 'r') as f:
        data = json.load(f)

    # Get flip and new_width information
    flip = isinstance(flip_transform, HFlipTransform)
    if flip:
        new_width = flip_transform.width

    # Iteratively yield keypoints for each object in data
    for i in range(len(data)):
        keypoints = data[i]['keypoints']

        # Convert keypoints to a torch Tensor and reshape
        keypoints = torch.tensor(keypoints).view(-1, 3)

        # Scale keypoints according to the ratio
        keypoints[:, :2] *= ratio

        if flip:
            # Flip x-coordinates of keypoints
            keypoints[:, 0] = new_width - keypoints[:, 0]

        # Move keypoints Tensor to the GPU
        keypoints = keypoints.to('cuda')

        yield keypoints


def generate_head_color():
    white = torch.tensor([255, 255, 255]).view(3, 1, 1)
    return white


def generate_rear_color():
    # Define color gradient
    # cmap = matplotlib.cm.get_cmap('hot')
    # cmap = matplotlib.cm.get_cmap('afmhot')
    cmap = matplotlib.cm.get_cmap('gist_heat')

    lut = torch.tensor([cmap(i)[:3] for i in range(cmap.N)]).float()

    # Map gradient values to range [0, 255]
    lut = lut * 255

    # Swap RGB to BGR
    # print("lut.shape", lut.shape)
    lut = torch.flip(lut, [1])

    return lut


# NOTE: replace with reflection-aware version
def gaussian_heatmap_kernel(height, width, ch, cw, wheel_y1, wheel_y2, HIGH, wei, device, reflect_render=False):
    """
    This function generates a Gaussian heatmap (kernel) and a rectangular reflection of the light region in the kernel.

    Inputs:
    - height, width: The height and width of the image.
    - ch, cw: The center coordinates of the Gaussian distribution in the kernel.
    - wheel_y1, wheel_y2: The y coordinates defining the vertical span of the rectangular reflection region.
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

    if reflect_render:
        # Define the boundaries of the light
        std = 3 * HIGH
        x1 = max(0, int(cw-std))
        x2 = min(width, int(cw+std))
        y1 = max(0, int(ch-std))
        y2 = min(height, int(ch+std))

        # NOTE: shift wheel_y1 to reach the ground
        wheel_y1 += 1.5 * HIGH
        wheel_y1 = min(height, wheel_y1)

        # NOTE: hardcode reflecion length
        if wheel_y2 is None:
            wheel_y2 = wheel_y1 + 4 * (y2 - y1)
            wheel_y2 = min(height, wheel_y2)
            # print("wheel_y2 is", wheel_y2)

        # Compute the width and height of the rectangular region
        rect_width = int(x2) - int(x1)
        rect_height = int(wheel_y2) - int(wheel_y1)

        assert ch <= wheel_y1 <= wheel_y2 <= height , \
            print(f"rect_height invalid: ch = {ch}, wheel_y1 = {wheel_y1}, wheel_y2 = {wheel_y2}, height = {height}")

        # Crop the lower half of the light region from the kernel
        light_region_half = kernel[y1 + ((y2-y1)//2):y2, x1:x2]
        
        # Resize the lower half of the light region to fit the dimensions of the rectangular region
        light_region_resized = F.interpolate(light_region_half[None, None, ...], 
                                                size=(rect_height, rect_width), 
                                                mode='bilinear', 
                                                align_corners=False)[0, 0]
            
        # Create an empty tensor of the same size as the image
        reflection = torch.zeros((height, width), device=device)
        
        # Place the resized lower half of the light region in the rectangular region in the image
        reflection[int(wheel_y1):int(wheel_y2), int(x1):int(x2)] = light_region_resized

        return kernel, reflection

    else:
        return kernel, None



# NOTE: old code without vec. 
def generate_light(image, ins, keypoints, HIGH, reflect_render=False):
    device = image.device

    assert image.max() <= 255 and image.min() >= 0, "Image should have intensity values in the range [0, 255]"

    if len(image.shape) != 3 or image.shape[0] != 3:
        raise ValueError("Image should be CxHxW and have 3 color channels")
    
    kernels_head, kernels_rear = [], []
    reflects_head, reflects_rear = [], []
    img_h, img_w = image.shape[1], image.shape[2]
    img_area = img_h * img_w  # Total area of the image

    color_white = generate_head_color().to(device)
    color_hot = generate_rear_color().to(device)
    # print("color_hot[0]", color_hot[0])    # The color that corresponds to the lowest value (0)
    # print("color_hot[255]", color_hot[255])  # The color that corresponds to the highest value (1)

    # NOTE: keypoints for wheels
    indices_of_light = [4, 3, 13, 14] # front_light_left, front_light_right, rear_light_left, rear_light_right
    indices_of_wheel = [8, 20, 9, 19] # front_wheel_left, front_wheel_right, rear_wheel_left, rear_wheel_right

    # read bboxes from sample
    car_bboxes = get_class_2_bboxes(ins, device)

    # convert [x1, y1, x2, y2] to [x1, y1, w, h]
    car_bboxes = [convert_bbox_format(bbox) for bbox in car_bboxes]

    # Create a mapping from lights to wheels
    light_wheel_mapping = {light_idx: wheel_idx for light_idx, wheel_idx in zip(indices_of_light, indices_of_wheel)}

    # use indices_of_light and indices_of_wheel
    for idx in indices_of_light:
        x, y, c = keypoints[idx-1]
        if c > 0:
            weight = 0.0  # Default weight if keypoint is not in any bounding box
            # Check if the keypoint is inside a car bounding box
            for bbox in car_bboxes:
                x1, y1, w, h = bbox
                if x1 <= x <= x1+w and y1 <= y <= y1+h:
                    # If keypoint is inside the bounding box, calculate the weight
                    bbox_area = w * h
                    weight = math.sqrt(bbox_area / img_area)
                    break

            wheel_x, wheel_y1, wheel_c = keypoints[light_wheel_mapping[idx]-1]
            wheel_y2 = None

            kernel, reflect = gaussian_heatmap_kernel(img_h, 
                                                      img_w, 
                                                      y,
                                                      x,  
                                                      wheel_y1,
                                                      wheel_y2,
                                                      HIGH=HIGH, 
                                                      wei=weight, 
                                                      device=device, 
                                                      reflect_render=reflect_render and wheel_c > 0)

            # separate headlights and rear lights
            if idx in [4, 3]:  # front_light_left, front_light_right
                kernels_head.append(kernel)
                if reflect is not None:
                    reflects_head.append(reflect)
            else:  # rear_light_left, rear_light_right
                kernels_rear.append(kernel)
                if reflect is not None:
                    reflects_rear.append(reflect)

    if not (kernels_head or kernels_rear):  # if no keypoints passed the confidence score check, return the original image
        return image

    # print(f"color_white = {color_white.shape}, color_hot = {color_hot.shape}")

    # # Aggregate all kernels and make sure values are in range [0, 1]
    # if kernels_head:
    #     aggregate_kernel_head = torch.clamp(torch.sum(torch.stack(kernels_head), 0), 0, 1)
    #     image = (image * (1 - aggregate_kernel_head) + color_white * aggregate_kernel_head).type(torch.uint8)

    # if kernels_rear:
    #     aggregate_kernel_rear = torch.clamp(torch.sum(torch.stack(kernels_rear), 0), 0, 1)
    #     # print("aggregate_kernel_rear shape is", aggregate_kernel_rear.shape)
    #     # print("aggregate_kernel_rear.min()", aggregate_kernel_rear.min())
    #     # print("aggregate_kernel_rear.max()", aggregate_kernel_rear.max())

    #     # Rescale the aggregate_kernel_rear values to match the range of indices in lut
    #     aggregate_kernel_rear_idx = (aggregate_kernel_rear * (color_hot.shape[0] - 1)).long()
    #     # print("aggregate_kernel_rear_idx shape", aggregate_kernel_rear_idx.shape)
    #     # print("aggregate_kernel_rear_idx.min()", aggregate_kernel_rear_idx.min())
    #     # print("aggregate_kernel_rear_idx.max()", aggregate_kernel_rear_idx.max())

    #     # Apply color map
    #     color_rear = color_hot[aggregate_kernel_rear_idx].permute(2, 0, 1)
    #     # print("Color_rear shape after permute: ", color_rear.shape)
    #     # print("color_rear[:, 0, 0]", color_rear[:, 0, 0])  # The color applied to the pixel at (0, 0)

    #     image = (1 - aggregate_kernel_rear.unsqueeze(0).unsqueeze(1)) * image + \
    #         aggregate_kernel_rear.unsqueeze(0).unsqueeze(1) * color_rear.to(device=image.device)


    # # NOTE: add light reflected image here 
    # if reflect_render:

    #     if reflects_head:
    #         aggregate_reflect_head = torch.clamp(torch.sum(torch.stack(reflects_head), 0), 0, 1)
    #         image = (image * (1 - aggregate_reflect_head) + color_white * aggregate_reflect_head).type(torch.uint8)
        
    #     if reflects_rear:
    #         aggregate_reflect_rear = torch.clamp(torch.sum(torch.stack(reflects_rear), 0), 0, 1)
    #         aggregate_reflect_rear_idx = (aggregate_reflect_rear * (color_hot.shape[0] - 1)).long()

    #         color_rear_reflect = color_hot[aggregate_reflect_rear_idx].permute(2, 0, 1)
    #         # print("color_rear_reflect shape after permute: ", color_rear_reflect.shape)
    #         image = (1 - aggregate_reflect_rear.unsqueeze(0).unsqueeze(1)) * image + \
    #                     aggregate_reflect_rear.unsqueeze(0).unsqueeze(1) * color_rear_reflect.to(device=image.device)


    image = apply_light_effect("headlight", kernels_head, image, color_white)
    image = apply_light_effect("taillight", kernels_rear, image, color_hot)

    if reflect_render:
        image = apply_light_effect("headlight", reflects_head, image, color_white)
        image = apply_light_effect("taillight", reflects_rear, image, color_hot)


    if len(image.shape) == 4:
        image = image.squeeze(0)

    return image.type(torch.uint8)


def apply_light_effect(light_type, kernels, image, color, color_hot=None):
    if kernels:
        aggregate_kernel = torch.clamp(torch.sum(torch.stack(kernels), 0), 0, 1)
        if light_type == "headlight":
            image = (image * (1 - aggregate_kernel) + color * aggregate_kernel).type(torch.uint8)
        elif light_type == "taillight":
            aggregate_kernel_idx = (aggregate_kernel * (color_hot.shape[0] - 1)).long()
            color = color_hot[aggregate_kernel_idx].permute(2, 0, 1)
            image = (1 - aggregate_kernel.unsqueeze(0).unsqueeze(1)) * image \
                        + aggregate_kernel.unsqueeze(0).unsqueeze(1) * color.to(device=image.device)
    return image