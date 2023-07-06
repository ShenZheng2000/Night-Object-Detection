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


def motion_blur(image, size, direction):
    # tiny size => set to float and uns. to 4d
    if size <= 2:
        return image.float().unsqueeze(0)

    # Ensure image is a 4D tensor (Batch Size, Channels, Height, Width)
    if len(image.shape) != 4:
        image = image.unsqueeze(0)

    # Convert image to float
    image = image.float()

    # Get the number of channels
    num_channels = image.shape[1]

    if direction == 'horizontal':
        # Create the motion blur kernel for horizontal blur
        kernel = torch.zeros((num_channels, 1, size, size)).to(image.device)
        kernel[:, :, size // 2, :] = 1.0

    elif direction == 'vertical':
        # Create the motion blur kernel for vertical blur
        kernel = torch.zeros((num_channels, 1, size, size)).to(image.device)
        kernel[:, :, :, size // 2] = 1.0

    kernel /= size

    # Calculate padding
    padding = size // 2

    # Apply convolution with 'same' padding
    image = F.conv2d(image, kernel, padding=padding, stride=1, groups=num_channels)

    return image


def motion_blur_adjustable(image, size=15, direction=0):
    # direction is in degrees (0 for horizontal, 90 for vertical)
    # print("applying motion_blur_adjustable")
    # print("image shape", image.shape)
    # print(f"image {image.shape} min {image.min()} max {image.max()} ")
    # print("direction is", direction)

    # First, calculate the proportion of horizontal and vertical blur
    direction_rad = torch.deg2rad(torch.tensor([direction], dtype=torch.float32, device=image.device))
    horiz_prop = torch.abs(torch.cos(direction_rad))
    vert_prop = torch.abs(torch.sin(direction_rad))

    # Then, apply horizontal and vertical motion blur separately
    image_horiz_blurred = motion_blur(image, int(size * horiz_prop), 'horizontal')
    image_vert_blurred = motion_blur(image, int(size * vert_prop), 'vertical')

    # Check if sizes are the same
    if image_horiz_blurred.shape != image.shape:
        # Resize image_horiz_blurred to match the image shape
        image_horiz_blurred = F.interpolate(image_horiz_blurred, size=image.shape[-2:], mode='bilinear', align_corners=False)
        image_horiz_blurred = image_horiz_blurred.squeeze(0)

    if image_vert_blurred.shape != image.shape:
        # Resize image_vert_blurred to match the image shape
        image_vert_blurred = F.interpolate(image_vert_blurred, size=image.shape[-2:], mode='bilinear', align_corners=False)
        image_vert_blurred = image_vert_blurred.squeeze(0)

    # print(f"image_vert_blurred {image_vert_blurred.shape} min {image_vert_blurred.min()} max {image_vert_blurred.max()} ")
    # print(f"image_horiz_blurred {image_horiz_blurred.shape} min {image_horiz_blurred.min()} max {image_horiz_blurred.max()} ")

    # NOTE: update 6/25/2023:
    # After computing horiz_prop and vert_prop: scale them
    total = horiz_prop + vert_prop
    horiz_prop /= total
    vert_prop /= total

    # Finally, combine the two blurred images
    # print(f"horiz_prop is {horiz_prop} vert_prop {vert_prop}")
    blurred = horiz_prop * image_horiz_blurred + vert_prop * image_vert_blurred
    # print(f"blurred {blurred.shape} min {blurred.min()} max {blurred.max()} ")

    return blurred




# NOTE: currently not working (7/3/2023)
# def make_path_blur():
#     # Create a directory to save blurred images
#     os.makedirs('images_path_blur', exist_ok=True)

#     vp_json = '/root/autodl-tmp/Datasets/VP/train_day.json'
#     gt_json = '/root/autodl-tmp/Datasets/bdd100k/coco_labels/train_day.json'
#     img_dir = '/root/autodl-tmp/Datasets/bdd100k/images/100k/train'

#     BLUR_SIZE = 360 # TODO: tune this later

#     # Load the COCO JSON file
#     with open(gt_json) as f:
#         data = json.load(f)

#     # Load the Vanishing Point JSON file
#     with open(vp_json) as f:
#         vp_data = json.load(f)

#     # Convert vp_data to only contain basenames for keys for easier lookup
#     vp_data = {os.path.basename(k): v for k, v in vp_data.items()}

#     # Convert list of images to dictionary for easy lookup
#     images_dict = {image['id']: image for image in data['images']}

#     to_tensor = transforms.ToTensor()
#     to_pil = transforms.ToPILImage()

#     image_count = 0 # counter for processed images

#     # Iterate over each image
#     for image in data['images']:

#         # Get the filename and remove the directory to get the basename
#         filename = os.path.basename(image['file_name'])

#         # NOTE: hardcode to specify filename for debug
#         if filename != '001bad4e-2fa8f3b6.jpg':
#             continue

#         # Get the vanishing point for this image
#         vp = vp_data.get(filename)

#         # Get the annotations for this image
#         annotations = [ann for ann in data['annotations'] if ann['image_id'] == image['id']]

#         # Load the image
#         img = Image.open(os.path.join(img_dir, filename))

#         img_tensor = to_tensor(img)

#         print(f"============filename={filename}============>")

#         # Iterate over each annotation for current image
#         for annotation in annotations:

#             # Calculate the area using the bounding box
#             bbox = annotation['bbox']
#             x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
#             area = w * h  # width * height

#             # Calculate the center of the bounding box
#             center = (x + w/2, y + h/2)

#             # Calculate the reweighted blur size
#             image_area = image['width'] * image['height']
#             blur_size = BLUR_SIZE * math.sqrt(area / image_area)

#             # Calculate the blur direction
#             dx, dy = vp[0] - center[0], vp[1] - center[1]
#             blur_direction = math.degrees(math.atan2(dy, dx))  # angle in degrees

#             # Apply the motion blur
#             # TODO: you cannot use one object's status to decide the blur for all objects in image
#             bit = motion_blur_adjustable(img_tensor, size=int(blur_size), direction=blur_direction)

#             # NOTE: hor motion blur for debug 
#             bit_hor = motion_blur_adjustable(img_tensor, size=15, direction=0)

#             # NOTE: rand motion blur for debug
#             rand_hor = motion_blur_adjustable(img_tensor, size=15, direction=R.random() * 360)

#             # Save the original and blurred images
#             img.save(f'images_path_blur/original_{filename}')
#             to_pil(bit).save(f'images_path_blur/blurred_{filename}')
#             to_pil(bit_hor).save(f'images_path_blur/blurred_hor_{filename}')
#             to_pil(rand_hor).save(f'images_path_blur/blurred_rand_{filename}')

#             print(f"blur_size={blur_size:.3f} blur_direction={blur_direction:.3f}")

#             # break # only process one annotation per image for demo

#         image_count += 1
#         if image_count >= 1: # limit to 1 image pairs
#             break




# NOTE: NO USE for now
# def debug_motion_blur():

#     # Initialize NightAug object
#     naug = NightAug()

#     # Define transformation to convert image to tensor
#     to_tensor = T.ToTensor()
#     to_image = T.ToPILImage()

#     # Read your image
#     input_image_path = '/root/autodl-tmp/Datasets/bdd100k/images/100k/train/0a0c3694-487a156f.jpg'
#     input_image = Image.open(input_image_path)
#     input_image_tensor = to_tensor(input_image)

#     # Rescale the tensor values to 0-255
#     input_image_tensor = input_image_tensor * 255

#     # Prepare the data as required by the NightAug.aug method
#     data = [{'image': input_image_tensor.clone()}]

#     # Apply NightAug with motion_blur=True
#     augmented_data_motion_blur = naug.aug(data, motion_blur=True)

#     # Get the output tensor
#     output_image_tensor_motion_blur = augmented_data_motion_blur[0]['image']

#     # Prepare the data again for the second augmentation
#     data = [{'image': input_image_tensor.clone()}]

#     # Apply NightAug with motion_blur_rand=True
#     augmented_data_motion_blur_rand = naug.aug(data, motion_blur_rand=True)

#     # Get the output tensor
#     output_image_tensor_motion_blur_rand = augmented_data_motion_blur_rand[0]['image']

#     # Convert tensors back to images
#     # Here we assume the output image tensor is also in the range 0-255, if not, you might need to rescale it back to 0-1 before converting to image.
#     input_image = to_image(input_image_tensor / 255)
#     output_image_motion_blur = to_image(output_image_tensor_motion_blur / 255)
#     output_image_motion_blur_rand = to_image(output_image_tensor_motion_blur_rand / 255)

#     # Save images
#     os.makedirs('aug_images', exist_ok=True)
#     input_image.save('aug_images/input_image.jpg')
#     output_image_motion_blur.save('aug_images/output_image_motion_blur.jpg')
#     output_image_motion_blur_rand.save('aug_images/output_image_motion_blur_rand.jpg')



if __name__ == '__main__':
    pass
