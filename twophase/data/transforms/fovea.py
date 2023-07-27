import torch
import torch.nn.functional as F
from .grid_generator import CuboidGlobalKDEGrid
from .invert_grid import invert_grid


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
    if len(imgs.shape) != 4:
        imgs = imgs.unsqueeze(0)

    imgs = torch.stack(tuple(imgs), dim=0)

    grid = grid_net(imgs, vanishing_point)

    warped_imgs = F.grid_sample(imgs, grid, align_corners=True)

    return grid, warped_imgs



def make_warp_aug(img, ins, vanishing_point):
    # read image
    img = img.float()
    device = img.device
    my_shape = img.shape[1:]

    # read bboxes
    bboxes = ins.gt_boxes.tensor
    bboxes = bboxes.to(device)

    # Create an instance of CuboidGlobalKDEGrid
    grid_net = CuboidGlobalKDEGrid(separable=True, 
                                    anti_crop=True, 
                                    input_shape=my_shape, 
                                    output_shape=my_shape)
    
    # warp image
    imgs = img.unsqueeze(0) 
    grid, warped_imgs = simple_test(grid_net, imgs, vanishing_point)

    # warp bboxes
    warped_bboxes = warp_bboxes(bboxes, grid, separable=True)

    return warped_imgs, warped_bboxes


if __name__ == '__main__':
    make_warp_aug()