from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
 
model = resnet50(pretrained=True)

target_layers = [model.layer4[-1]]
# print("target_layers is", target_layers)

# Load an image from a file path
image_path = '/home/aghosh/Projects/2PCNet/Datasets/bdd100k/images/100k/val/b1c9c847-3bda4659.jpg'  # Replace with the actual path to your image
image = Image.open(image_path)

# Define the transformation to convert PIL image to a PyTorch tensor
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert to a PyTorch tensor
])

# Apply the transformation to the image
input_tensor = transform(image)

# Add a batch dimension (unsqueeze) to create a 4D tensor with shape (1, 3, Height, Width)
input_tensor = input_tensor.unsqueeze(0)

# Construct the CAM object once, and then re-use it on many images:
# cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
 
targets = None # 选择最大置信度的类进行热力图展示。本例281 指的就是标签id [ClassifierOutputTarget(281)]
 
# You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
 
# In this example grayscale_cam has only one image in the batch:
grayscale_cam = grayscale_cam[0, :]

rgb_image = np.array(image)
rgb_image = rgb_image.astype(np.float32) / 255
 
visualization = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)

plt.imshow(visualization)
# plt.savefig("grad_cam.png")
plt.savefig("grad_cam_plus_plus.png")