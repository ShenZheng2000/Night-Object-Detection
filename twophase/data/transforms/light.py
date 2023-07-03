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


def gaussian_heatmap_kernel(height, width, ch, cw, HIGH, wei, device):

    # HIGH = compute_high(height, width, ch, cw, vp)
    HIGH = max(1, int(HIGH * wei))
    # print("HIGH is", HIGH)

    # NOTE: change scale to fixed for now. 
    # sig = torch.randint(low=1, high=HIGH, size=(1,)).cuda()[0]
    # center = (torch.tensor(ch).cuda(), torch.tensor(cw).cuda())
    sig = torch.tensor(HIGH).to(device)
    center = (ch, cw)
    x_axis = torch.linspace(0, height-1, height).to(device) - center[0]
    y_axis = torch.linspace(0, width-1, width).to(device) - center[1]
    xx, yy = torch.meshgrid(x_axis, y_axis)
    kernel = torch.exp(-0.5 * (torch.square(xx) + torch.square(yy)) / torch.square(sig))

    return kernel



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
def generate_light(image, ins, keypoints, HIGH):
    device = image.device

    assert image.max() <= 255 and image.min() >= 0, "Image should have intensity values in the range [0, 255]"

    if len(image.shape) != 3 or image.shape[0] != 3:
        raise ValueError("Image should be CxHxW and have 3 color channels")

    kernels = []
    img_area = image.shape[1] * image.shape[2]  # Total area of the image

    indices_to_draw = [3, 4, 13, 14] # headlight and tailights

    # read bboxes from sample
    # start_time = time.time()
    car_bboxes = get_class_2_bboxes(ins, device)

    # end_time_0_5 = time.time()
    # print(f"get_class_2_bboxes {end_time_0_5 - start_time}")

    # convert [x1, y1, x2, y2] to [x1, y1, w, h]
    car_bboxes = [convert_bbox_format(bbox) for bbox in car_bboxes]

    # print("image device is", image.device)
    # print("keypoints device", keypoints.device)
    # end_time_1 = time.time()
    # print(f"convert_bbox_format {end_time_1 - end_time_0_5}")

    for idx in indices_to_draw:
        x, y, c = keypoints[idx-1]
        if c > 0:
            weight = 0.0  # Default weight if keypoint is not in any bounding box
            # Check if the keypoint is inside a car bounding box
            for bbox in car_bboxes:
                # print("bbox is", bbox)
                x1, y1, w, h = bbox
                if x1 <= x <= x1+w and y1 <= y <= y1+h:
                    # If keypoint is inside the bounding box, calculate the weight
                    bbox_area = w * h
                    weight = math.sqrt(bbox_area / img_area)
                    # print("Old Code: weight is", weight)
                    break

            kernel = gaussian_heatmap_kernel(image.shape[1], image.shape[2], y, x, HIGH=HIGH, wei=weight, device=device) # NOTE: y->h, x->w
            kernels.append(kernel)

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

    # end_time_3 = time.time()
    # print(f"aggregation {end_time_3 - end_time_2}")

    return new_img


# NOTE: vect. codes => lots of issue, debug later
# def generate_light(image, ins, keypoints, HIGH):
#     assert image.max() <= 255 and image.min() >= 0
#     if len(image.shape) != 3 or image.shape[0] != 3:
#         raise ValueError("Image should be CxHxW and have 3 color channels")

#     # Calculate the total area of the image
#     img_area = image.shape[1] * image.shape[2]

#     # Extract bounding boxes for the cars in the image
#     car_bboxes = get_class_2_bboxes(ins)
#     car_bboxes = [convert_bbox_format(bbox) for bbox in car_bboxes]
#     car_bboxes = torch.tensor(car_bboxes, device='cuda')

#     # Select indices of keypoints to draw
#     indices_to_draw = torch.tensor([3, 4, 13, 14], device='cuda') - 1  
#     keypoints_to_draw = keypoints[indices_to_draw]

#     # Filter out keypoints with low confidence
#     keypoints_to_draw = keypoints_to_draw[keypoints_to_draw[:, 2] > 0]

#     if keypoints_to_draw.size(0) == 0:
#         return image

#     # Calculate start and end of each bounding box
#     bbox_start, bbox_size = car_bboxes[:, :2], car_bboxes[:, 2:]
#     bbox_end = bbox_start + bbox_size

#     # Check if each keypoint is within a bounding box
#     bbox_start = bbox_start.unsqueeze(1)
#     bbox_end = bbox_end.unsqueeze(1)
#     keypoints = keypoints_to_draw[:, :2].unsqueeze(0)
#     is_inside_bbox = (bbox_start <= keypoints) & (keypoints <= bbox_end)

#     # Compute weight for each keypoint based on the bounding box area
#     bbox_areas = torch.prod(bbox_size, dim=1)
#     weights = torch.sqrt(bbox_areas / img_area)

#     # Determine if a keypoint is inside any bounding box
#     is_inside_any_bbox = is_inside_bbox.sum(dim=0) > 0  

#     kernels = []
#     for idx, (x, y, c) in enumerate(keypoints_to_draw):
#         # Ensure the weight is a scalar
#         inside_any_bbox = is_inside_any_bbox[idx].any()
#         weight = weights[inside_any_bbox.nonzero(as_tuple=True)[0]] if inside_any_bbox else 0.0
#         print("New Code weight is", weight.item())
#         kernel = gaussian_heatmap_kernel(image.shape[1], image.shape[2], y, x, HIGH=HIGH, wei=weight.item())
#         kernels.append(kernel)

#     if not kernels:
#         return image

#     # Combine all kernels and ensure values are in range [0, 1]
#     aggregate_kernel = torch.clamp(torch.sum(torch.stack(kernels), 0), 0, 1)

#     # Generate new image by blending original image and colored light
#     color = generate_light_color()
#     new_img = (image * (1 - aggregate_kernel) + color * aggregate_kernel).type(torch.uint8)

#     return new_img




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