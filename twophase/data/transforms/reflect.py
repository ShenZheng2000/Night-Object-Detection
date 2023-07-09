import torch
import os
import math
from detectron2.structures import Boxes
from detectron2.data.transforms import ResizeTransform, HFlipTransform, NoOpTransform
import json
import time


def generate_light_color():
    white = torch.tensor([255, 255, 255]).view(3, 1, 1)
    return white


# NOTE: replace with reflection-aware version
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
        return kernel, None


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


# NOTE: old code without vec. 
def generate_light(image, ins, keypoints, HIGH, use_reflect=False):
    device = image.device

    assert image.max() <= 255 and image.min() >= 0, "Image should have intensity values in the range [0, 255]"

    if len(image.shape) != 3 or image.shape[0] != 3:
        raise ValueError("Image should be CxHxW and have 3 color channels")

    kernels = []
    reflects = []
    img_area = image.shape[1] * image.shape[2]  # Total area of the image

    # NOTE: keypoints for wheels
    indices_of_light = [4, 3, 13, 14] # front_light_left, front_light_right, rear_light_left, rear_light_right
    indices_of_wheel = [8, 20, 9, 19] # front_wheel_left, front_wheel_right, rear_wheel_left, rear_wheel_right

    # read bboxes from sample
    # start_time = time.time()
    car_bboxes = get_class_2_bboxes(ins, device)

    # end_time_0_5 = time.time()
    # print(f"get_class_2_bboxes {end_time_0_5 - start_time}")

    # convert [x1, y1, x2, y2] to [x1, y1, w, h]
    car_bboxes = [convert_bbox_format(bbox) for bbox in car_bboxes]

    # Create a mapping from lights to wheels
    light_wheel_mapping = {light_idx: wheel_idx for light_idx, wheel_idx in zip(indices_of_light, indices_of_wheel)}

    # print("image device is", image.device)
    # print("keypoints device", keypoints.device)
    # end_time_1 = time.time()
    # print(f"convert_bbox_format {end_time_1 - end_time_0_5}")

    # NOTE: use indices_of_light and indices_of_wheel
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

            wheel_x, wheel_y, wheel_c = keypoints[light_wheel_mapping[idx]-1]
            
            # Randomize y2
            y1 = wheel_y
            y2 = random.randint(y1, image.shape[1])

            kernel, reflect = gaussian_heatmap_kernel(image.shape[1], 
                                                      image.shape[2], 
                                                      y,
                                                      x,  
                                                      y1,
                                                      y2,
                                                      HIGH=HIGH, 
                                                      wei=weight, 
                                                      device=device, 
                                                      use_reflect=use_reflect and wheel_c > 0) # NOTE: y->h, x->w
            kernels.append(kernel)
            reflects.append(reflect)

    # end_time_2 = time.time()
    # print(f"gaussian_heatmap_kernel {end_time_2 - end_time_1}")

    if not kernels:  # if no keypoints passed the confidence score check, return the original image
        return image

    # Aggregate all kernels and make sure values are in range [0, 1]
    aggregate_kernel = torch.clamp(torch.sum(torch.stack(kernels), 0), 0, 1)
    # print("aggregate_kernel min", aggregate_kernel.min())
    # print("aggregate_kernel max", aggregate_kernel.max())

    color = generate_light_color().to(device) # No need to scale
    # print(f"image min {image.min()} max {image.max()}")
    new_img = (image * (1-aggregate_kernel) + color * aggregate_kernel).type(torch.uint8)

    # NOTE: add light reflected image here 
    if use_reflect:
        aggregate_reflect = torch.clamp(torch.sum(torch.stack(reflects), 0), 0, 1)
        new_img = (new_img * (1-aggregate_reflect) + color * aggregate_reflect).type(torch.uint8)

    # end_time_3 = time.time()
    # print(f"aggregation {end_time_3 - end_time_2}")

    return new_img
