import torch
import torchvision.transforms as T
from numpy import random as R
import torch.nn.functional as F
from PIL import Image
import os


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
