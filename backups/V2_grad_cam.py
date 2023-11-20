from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50
# TODO: import resnet50 from detectron2
import detectron2.modeling.backbone.resnet as resnet
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import torch
from detectron2.layers import ShapeSpec

import detectron2
from twophase import add_teacher_config

# read model from configs
cfg_path = "/home/aghosh/Projects/2PCNet/Methods/Night-Object-Detection/configs/bdd100k_10_18_baseline.yaml"
cfg = detectron2.config.get_cfg()
add_teacher_config(cfg)
cfg.merge_from_file(cfg_path)
print(cfg.MODEL.RESNETS)

input_shape = ShapeSpec(channels=3)
model = resnet.build_resnet_backbone(cfg, input_shape)
print(model)
exit()

# model = resnet50(pretrained=False)

# Load the entire checkpoint
checkpoint = torch.load("/longdata/anurag_storage/2PCNet/outputs_11_14_det_ckpts/bdd100k_10_18_baseline/model_final.pth")

# Extract the model weights
model_weights = checkpoint['model']

# Filter and remap the weights (example: selecting only backbone weights)
backbone_weights = {k.replace('modelTeacher.backbone.', ''): v 
                    for k, v in model_weights.items() 
                    if k.startswith('modelTeacher.backbone.')}

# Load the filtered weights into your model
model.load_state_dict(backbone_weights, strict=False)

# target_layers = [model.layer4[-1]]
target_layers = [model.res5[-1]]
# print("target_layers is", target_layers)

# Load an image from a file path
image_path = '/home/aghosh/Projects/2PCNet/Datasets/bdd100k/images/100k/val/b1c9c847-3bda4659.jpg'  # Replace with the actual path to your image
image = Image.open(image_path)

# TODO: resize the image to half
# image = image.resize((int(image.size[0]/2), int(image.size[1]/2)))

# Define the transformation to convert PIL image to a PyTorch tensor
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert to a PyTorch tensor
])

# Apply the transformation to the image
input_tensor = transform(image)

# Add a batch dimension (unsqueeze) to create a 4D tensor with shape (1, 3, Height, Width)
input_tensor = input_tensor.unsqueeze(0)
print("input_tensor", input_tensor.shape)
# exit()

# Construct the CAM object once, and then re-use it on many images:
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
plt.savefig("grad_cam_new.png")
