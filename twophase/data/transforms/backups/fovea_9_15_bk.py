import torch
import torch.nn.functional as F
from .invert_grid import invert_grid
from .reblur import is_out_of_bounds, get_vanising_points
from detectron2.data.transforms import ResizeTransform, HFlipTransform, NoOpTransform
from torchvision import utils as vutils
import sys
import os

def unwarp_bboxes(bboxes, grid, output_shape):
    """Unwarps a tensor of bboxes of shape (n, 4) or (n, 5) according to the grid \
    of shape (h, w, 2) used to warp the corresponding image and the \
    output_shape (H, W, ...)."""
    bboxes = bboxes.clone()
    # image map of unwarped (x,y) coordinates
    img = grid.permute(2, 0, 1).unsqueeze(0)

    warped_height, warped_width = grid.shape[0:2]

    xgrid = 2 * (bboxes[:, 0:4:2] / float(warped_width)) - 1
    ygrid = 2 * (bboxes[:, 1:4:2] / float(warped_height)) - 1
    grid = torch.stack((xgrid, ygrid), dim=2).unsqueeze(0)

    # warped_bboxes has shape (2, num_bboxes, 2)
    warped_bboxes = F.grid_sample(
        img, grid, align_corners=True, padding_mode="border").squeeze(0)

    bboxes[:, 0:4:2] = (warped_bboxes[0] + 1) / 2 * output_shape[1]
    bboxes[:, 1:4:2] = (warped_bboxes[1] + 1) / 2 * output_shape[0]

    return bboxes


def warp_bboxes(bboxes, grid, separable=True):

    size = grid.shape
    
    inverse_grid_shape = torch.Size((size[0], size[3], size[1], size[2]))
    inverse_grid = invert_grid(grid, inverse_grid_shape, separable) # [1, 2, 720, 1280]

    bboxes = unwarp_bboxes(bboxes, inverse_grid.squeeze(0), size[1:3]) #output_shape[1:3]=[720, 1280]

    return bboxes


def simple_test(grid_net, imgs, vanishing_point):
    """Test function without test time augmentation.
    Args:
        grid_net (CuboidGlobalKDEGrid): An instance of CuboidGlobalKDEGrid.
        imgs (list[torch.Tensor]): List of multiple images
        img_metas (list[dict]): List of image information.
    Returns:
        list[list[np.ndarray]]: BBox results of each image and classes.
            The outer list corresponds to each image. The inner list
            corresponds to each class.
    """
    if len(imgs.shape) == 3:
        imgs = imgs.unsqueeze(0)

    imgs = torch.stack(tuple(imgs), dim=0)
    # print("imgs shape", imgs.shape)
    print("grid_net is", grid_net)
    grid = grid_net(imgs, vanishing_point)
    # print("grid shape", grid.shape)

    warped_imgs = F.grid_sample(imgs, grid, align_corners=True)

    return grid, warped_imgs



def make_warp_aug(img, ins, vanishing_point, grid_net, use_ins=False):

    # read image
    img = img.float()
    device = img.device
    my_shape = img.shape[-2:]
    imgs = img.unsqueeze(0) 

    if use_ins:

        # read bboxes
        bboxes = ins.gt_boxes.tensor
        bboxes = bboxes.to(device)

        # warp image
        grid, warped_imgs = simple_test(grid_net, imgs, vanishing_point)

        # warp bboxes
        warped_bboxes = warp_bboxes(bboxes, grid, separable=True)

        # # NOTE: hardcode for debug only. Delete later
        # warped_bboxes = unwarp_bboxes(warped_bboxes, grid.squeeze(0), [600, 1067])

        # update ins
        ins.gt_boxes.tensor = warped_bboxes

        return warped_imgs, ins, grid
    
    else:
        # warp image
        grid, warped_imgs = simple_test(grid_net, imgs, vanishing_point)

        return warped_imgs, ins, grid
    


def apply_warp_aug(img, ins, vanishing_point, warp_aug=False, 
                    grid_net=None, keep_size=True):
    # print(f"img is {img.shape}") # [3, 600, 1067]
    grid = None

    img_height, img_width = img.shape[-2:]
    
    # NOTE: this for debug only
    if is_out_of_bounds(vanishing_point, img_width, img_height):
        print(f"both vp coords OOB. Training Stops!!!! with vp = {vanishing_point}")
        sys.exit(1)
    img, ins, grid = make_warp_aug(img, ins, vanishing_point, grid_net, use_ins=False)

    # reshape 4d to 3d
    if (len(img.shape) == 4) and keep_size:
        img = img.squeeze(0) 

    return img, ins, grid



def apply_unwarp(warped_x, grid, keep_size=True):
    if (len(warped_x.shape) == 3) and keep_size:
        warped_x = warped_x.unsqueeze(0)

    # print(f'warped_x is {warped_x.shape} grid is {grid.shape}') # [1, 3, 600, 1067], [1, 600, 1067, 2]

    # Compute inverse_grid
    inverse_grid = invert_grid(grid, warped_x.shape, separable=True)[0:1]

    # Expand inverse_grid to match batch size
    B = warped_x.shape[0]
    inverse_grid = inverse_grid.expand(B, -1, -1, -1)

    # Perform unzoom
    unwarped_x = F.grid_sample(
        warped_x, inverse_grid, mode='bilinear',
        align_corners=True, padding_mode='zeros'
    )
    # print("unwarped_x shape", unwarped_x.shape) # [1, 3, 600, 1067]

    if (len(unwarped_x.shape) == 4) and keep_size:
        unwarped_x = unwarped_x.squeeze(0)

    # print("unwarped_x shape", unwarped_x.shape) # [3, 600, 1067]

    # print(f"unwarped_x min {unwarped_x.min()} max {unwarped_x.max()}") # [0, 255]

    return unwarped_x



def extract_ratio_and_flip(transform_list):
    ratio = 25/24  # default value (750/720) # TODO: stop hardcode and change later
    flip = NoOpTransform()  # default value

    if transform_list is None:
        return ratio, flip

    for transform in transform_list:
        if isinstance(transform, ResizeTransform):
            ratio = transform.new_h / transform.h
        elif isinstance(transform, (HFlipTransform, NoOpTransform)):
            flip = transform
            
    return ratio, flip



def process_and_update_features(batched_inputs, images, warp_aug_lzu, vp_dict, 
                                grid_net, backbone, warp_debug, warp_image_norm):
    '''
    '''
    # print(f"batched_inputs = {batched_inputs}")   # list: [...]
    # print(f"images = {images.tensor}")            # detectron2.structures.image_list.ImageList
    # print(f"warp_aug_lzu = {warp_aug_lzu}")       # bool: True/False
    # print(f"vp_dict = {vp_dict}")                 # dict: {'xxx.jpg': [vpw, vph], ...}
    print(f"grid_net = {grid_net}")               # CuboidGlobalKDEGrid
    # print(f"backbone = {backbone}")               # ResNet

    features = None
    
    # Preprocessing
    vanishing_points = [
        get_vanising_points(
            sample['file_name'], 
            vp_dict, 
            *extract_ratio_and_flip(sample['transform'])
        ) for sample in batched_inputs
    ]

    # Apply warping
    warped_images, _, grids = zip(*[
        apply_warp_aug(image, None, vp, False, warp_aug_lzu, grid_net) 
        for image, vp in zip(images.tensor, vanishing_points)
    ])
    warped_images = torch.stack(warped_images)

    # Normalize warped images
    if warp_image_norm:
        warped_images = torch.stack([(img - img.min()) / (img.max() - img.min()) * 255 for img in warped_images])

    # debug images
    concat_and_save_images(batched_inputs, warped_images, debug=warp_debug)

    # Call the backbone
    features = backbone(warped_images)

    # Apply unwarping
    feature_key = next(iter(features))
    unwarped_features = torch.stack([
        apply_unwarp(feature, grid)
        for feature, grid in zip(features[feature_key], grids)
    ])

    # Replace the original features with unwarped ones
    features[feature_key] = unwarped_features

    return features


def concat_and_save_images(batched_inputs, warped_images, debug=False):
    """
    Concatenate original and warped images side by side and save them.

    Inputs:
    - batched_inputs: List of dictionaries containing image information, including 'image' for original image.
    - warped_images: List of warped image tensors.
    - out_dir: Output directory for saving the images.
    - debug: Boolean flag to determine whether to save the images or not.

    Outputs:
    - None. This function saves the concatenated images if debug is True.
    """
    if debug:
        for (input_info, warped_img) in zip(batched_inputs, warped_images):
            original_img = input_info['image'].cuda()  # Access the original image
            # print("warped_img device", warped_img.device)

            # Switch from BGR to RGB
            original_img = torch.flip(original_img, [0])
            warped_img = torch.flip(warped_img, [0])

            # check image size
            print("original_img shape", original_img.shape)
            print("warped_img shape", warped_img.shape)

            # check min and max
            # TODO: debug this: original_image is from 0 to 255, but warpped images is from -100+ to 100+. 
            print(f"original_img. Min: {torch.min(original_img).item()}, Max: {torch.max(original_img).item()}")
            print(f"warped_img. Min: {torch.min(warped_img).item()}, Max: {torch.max(warped_img).item()}")

            # Normalize the images
            original_img = (original_img - original_img.min()) / (original_img.max() - original_img.min())
            warped_img = (warped_img - warped_img.min()) / (warped_img.max() - warped_img.min())

            # Concatenate images along width dimension
            combined_image = torch.cat((original_img, warped_img), dim=2)

            # Save images to output path
            file_name = os.path.basename(input_info['file_name'])  # Extract the original file name

            warp_out_dir = 'warped_images'
            os.makedirs(warp_out_dir, exist_ok=True)
            save_path = os.path.join(warp_out_dir, file_name)  # Create the save path

            vutils.save_image(combined_image, save_path, normalize=True)

        sys.exit(1)