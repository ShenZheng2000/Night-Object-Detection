import torch
import torchvision.transforms as T
from numpy import random as R
import torch.nn.functional as F
from torchvision.utils import save_image

class NightAug:
    def __init__(self):
        self.gaussian = T.GaussianBlur(11,(0.1,2.0))

    def mask_img(self,img,cln_img):
        while R.random()>0.4:
            x1 = R.randint(img.shape[1])
            x2 = R.randint(img.shape[1])
            y1 = R.randint(img.shape[2])
            y2 = R.randint(img.shape[2])
            img[:,x1:x2,y1:y2]=cln_img[:,x1:x2,y1:y2]
        return img

    def gaussian_heatmap(self,x):
        """
        It produces single gaussian at a random point
        """
        sig = torch.randint(low=1,high=150,size=(1,)).cuda()[0]
        image_size = x.shape[1:]
        center = (torch.randint(image_size[0],(1,))[0].cuda(), torch.randint(image_size[1],(1,))[0].cuda())
        x_axis = torch.linspace(0, image_size[0]-1, image_size[0]).cuda() - center[0]
        y_axis = torch.linspace(0, image_size[1]-1, image_size[1]).cuda() - center[1]
        xx, yy = torch.meshgrid(x_axis, y_axis)
        kernel = torch.exp(-0.5 * (torch.square(xx) + torch.square(yy)) / torch.square(sig))
        new_img = (x*(1-kernel) + 255*kernel).type(torch.uint8)
        return new_img

    def apply_motion_blur(self, img, motion_blur=False, motion_blur_rand=False):
        if motion_blur:
            img = motion_blur_adjustable(img)

        if motion_blur_rand:
            img = motion_blur_adjustable(img, direction=R.random() * 360)

        return img

    def aug(self,x,motion_blur=False,motion_blur_rand=False):
        for sample in x:
            img = sample['image'].cuda()
            g_b_flag = True

            # TODO: comment this for debug only
            # Guassian Blur
            if R.random()>0.5:
                img = self.gaussian(img)
            
            cln_img_zero = img.detach().clone()

            # Gamma
            if R.random()>0.5:
                cln_img = img.detach().clone()
                val = 1/(R.random()*0.8+0.2)
                img = T.functional.adjust_gamma(img,val)
                img= self.mask_img(img,cln_img)
                g_b_flag = False
            
            # Brightness
            if R.random()>0.5 or g_b_flag:
                cln_img = img.detach().clone()
                val = R.random()*0.8+0.2
                img = T.functional.adjust_brightness(img,val)
                img= self.mask_img(img,cln_img)

            # Contrast
            if R.random()>0.5:
                cln_img = img.detach().clone()
                val = R.random()*0.8+0.2
                img = T.functional.adjust_contrast(img,val)
                img= self.mask_img(img,cln_img)
            img= self.mask_img(img,cln_img_zero)

            prob = 0.5
            while R.random()>prob:
                img=self.gaussian_heatmap(img)
                prob+=0.1

            #Noise
            if R.random()>0.5:
                n = torch.clamp(torch.normal(0,R.randint(50),img.shape),min=0).cuda()
                img = n + img
                img = torch.clamp(img,max = 255).type(torch.uint8)

            # Apply motion blur
            # TODO: comment for debug only
            # if True:
            if R.random()>0.5:
                img = self.apply_motion_blur(img, motion_blur=motion_blur, motion_blur_rand=motion_blur_rand)
            sample['image'] = img.cpu()
        return x




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


if __name__ == '__main__':
    import torch
    from torchvision import transforms
    from PIL import Image
    import os

    # Initialize NightAug object
    naug = NightAug()


    # Define transformation to convert image to tensor
    to_tensor = transforms.ToTensor()
    to_image = transforms.ToPILImage()

    # Read your image
    input_image_path = '/root/autodl-tmp/Datasets/bdd100k/images/100k/train/0a0c3694-487a156f.jpg'
    input_image = Image.open(input_image_path)
    input_image_tensor = to_tensor(input_image)

    # Rescale the tensor values to 0-255
    input_image_tensor = input_image_tensor * 255

    # Prepare the data as required by the NightAug.aug method
    data = [{'image': input_image_tensor.clone()}]

    # Apply NightAug with motion_blur=True
    augmented_data_motion_blur = naug.aug(data, motion_blur=True)

    # Get the output tensor
    output_image_tensor_motion_blur = augmented_data_motion_blur[0]['image']

    # Prepare the data again for the second augmentation
    data = [{'image': input_image_tensor.clone()}]

    # Apply NightAug with motion_blur_rand=True
    augmented_data_motion_blur_rand = naug.aug(data, motion_blur_rand=True)

    # Get the output tensor
    output_image_tensor_motion_blur_rand = augmented_data_motion_blur_rand[0]['image']

    # Convert tensors back to images
    # Here we assume the output image tensor is also in the range 0-255, if not, you might need to rescale it back to 0-1 before converting to image.
    input_image = to_image(input_image_tensor / 255)
    output_image_motion_blur = to_image(output_image_tensor_motion_blur / 255)
    output_image_motion_blur_rand = to_image(output_image_tensor_motion_blur_rand / 255)

    # Save images
    os.makedirs('aug_images', exist_ok=True)
    input_image.save('aug_images/input_image.jpg')
    output_image_motion_blur.save('aug_images/output_image_motion_blur.jpg')
    output_image_motion_blur_rand.save('aug_images/output_image_motion_blur_rand.jpg')

