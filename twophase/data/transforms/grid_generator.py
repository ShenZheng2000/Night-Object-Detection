import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from torchvision.utils import save_image
import sys
import cv2
# from mmcv.utils import Registry
# from vis import vis_batched_imgs


# from mmcv.cnn import MODELS as MMCV_MODELS
# GRID_GENERATORS = Registry('models', parent=MMCV_MODELS)

import os, json
from .homography_layers import CuboidLayerGlobal, TripetLayerGlobal

# def build_grid_generator(cfg):
#     """Build grid generator."""
#     return GRID_GENERATORS.build(cfg)

def make1DGaussian(size, fwhm=3, center=None):
    """ Make a 1D gaussian kernel.

    size is the length of the kernel,
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """
    x = np.arange(0, size, 1, dtype=float)

    if center is None:
        center = size // 2

    return np.exp(-4*np.log(2) * (x-center)**2 / fwhm**2)


def make2DGaussian(size, fwhm=3, center=None):
    """ Make a square gaussian kernel.

    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)


class RecasensSaliencyToGridMixin(object):
    """Grid generator based on 'Learning to Zoom: a Saliency-Based Sampling \
    Layer for Neural Networks' [https://arxiv.org/pdf/1809.03355.pdf]."""

    # NOTE: add this to update output_shape
    def update_output_shape(self, output_shape):
        """Update attributes that are dependent on the output shape."""
        self.output_shape = output_shape

    def saliency_to_grid(self, imgs, saliency, device):
        """Convert saliency map to grid."""
        saliency = saliency.to(device)
        if self.separable:
            x_saliency = saliency.sum(dim=2)
            y_saliency = saliency.sum(dim=3)
            return self.separable_saliency_to_grid(imgs, None, x_saliency, y_saliency, device)
        else:
            return self.nonseparable_saliency_to_grid(imgs, None, saliency, device)

    def __init__(self, output_shape=None, grid_shape=(31, 51), separable=True,
                 attraction_fwhm=13, anti_crop=True, **kwargs):
        super(RecasensSaliencyToGridMixin, self).__init__()
        # self.output_shape = output_shape
        # self.output_height, self.output_width = output_shape
        self.grid_shape = grid_shape
        self.padding_size = min(self.grid_shape)-1
        self.total_shape = tuple(
            dim+2*self.padding_size
            for dim in self.grid_shape
        )
        self.padding_mode = 'reflect' if anti_crop else 'replicate'
        self.separable = separable

        # Initialize attributes dependent on output_shape
        self.update_output_shape(output_shape)

        if self.separable:
            self.filter = make1DGaussian(
                2*self.padding_size+1, fwhm=attraction_fwhm)
            self.filter = torch.FloatTensor(self.filter).unsqueeze(0) \
                                                        .unsqueeze(0).cuda()

            self.P_basis_x = torch.zeros(self.total_shape[1])
            for i in range(self.total_shape[1]):
                self.P_basis_x[i] = \
                    (i-self.padding_size)/(self.grid_shape[1]-1.0)
            self.P_basis_y = torch.zeros(self.total_shape[0])
            for i in range(self.total_shape[0]):
                self.P_basis_y[i] = \
                    (i-self.padding_size)/(self.grid_shape[0]-1.0)
        else:
            self.filter = make2DGaussian(
                2*self.padding_size+1, fwhm=attraction_fwhm)
            self.filter = torch.FloatTensor(self.filter) \
                               .unsqueeze(0).unsqueeze(0).cuda()

            self.P_basis = torch.zeros(2, *self.total_shape)
            for k in range(2):
                for i in range(self.total_shape[0]):
                    for j in range(self.total_shape[1]):
                        self.P_basis[k, i, j] = k*(i-self.padding_size)/(self.grid_shape[0]-1.0)+(1.0-k)*(j-self.padding_size)/(self.grid_shape[1]-1.0)  # noqa: E501

    def separable_saliency_to_grid(self, imgs, img_metas, x_saliency,
                                   y_saliency, device):
        assert self.separable
        x_saliency = F.pad(x_saliency, (self.padding_size, self.padding_size),
                           mode=self.padding_mode)
        y_saliency = F.pad(y_saliency, (self.padding_size, self.padding_size),
                           mode=self.padding_mode)

        N = imgs.shape[0]
        P_x = torch.zeros(1, 1, self.total_shape[1], device=device)
        P_x[0, 0, :] = self.P_basis_x
        P_x = P_x.expand(N, 1, self.total_shape[1])
        P_y = torch.zeros(1, 1, self.total_shape[0], device=device)
        P_y[0, 0, :] = self.P_basis_y
        P_y = P_y.expand(N, 1, self.total_shape[0])

        weights = F.conv1d(x_saliency, self.filter)
        weighted_offsets = torch.mul(P_x, x_saliency)
        weighted_offsets = F.conv1d(weighted_offsets, self.filter)
        xgrid = weighted_offsets/weights
        xgrid = torch.clamp(xgrid*2-1, min=-1, max=1)
        xgrid = xgrid.view(-1, 1, 1, self.grid_shape[1])
        xgrid = xgrid.expand(-1, 1, *self.grid_shape)

        weights = F.conv1d(y_saliency, self.filter)
        weighted_offsets = F.conv1d(torch.mul(P_y, y_saliency), self.filter)
        ygrid = weighted_offsets/weights
        ygrid = torch.clamp(ygrid*2-1, min=-1, max=1)
        ygrid = ygrid.view(-1, 1, self.grid_shape[0], 1)
        ygrid = ygrid.expand(-1, 1, *self.grid_shape)

        grid = torch.cat((xgrid, ygrid), 1)
        grid = F.interpolate(grid, size=self.output_shape, mode='bilinear',
                             align_corners=True)
        return grid.permute(0, 2, 3, 1)

    def nonseparable_saliency_to_grid(self, imgs, img_metas, saliency, device):
        assert not self.separable
        p = self.padding_size
        saliency = F.pad(saliency, (p, p, p, p), mode=self.padding_mode)

        N = imgs.shape[0]
        P = torch.zeros(1, 2, *self.total_shape, device=device)
        P[0, :, :, :] = self.P_basis
        P = P.expand(N, 2, *self.total_shape)

        saliency_cat = torch.cat((saliency, saliency), 1)
        weights = F.conv2d(saliency, self.filter)
        weighted_offsets = torch.mul(P, saliency_cat) \
                                .view(-1, 1, *self.total_shape)
        weighted_offsets = F.conv2d(weighted_offsets, self.filter) \
                            .view(-1, 2, *self.grid_shape)

        weighted_offsets_x = weighted_offsets[:, 0, :, :] \
            .contiguous().view(-1, 1, *self.grid_shape)
        xgrid = weighted_offsets_x/weights
        xgrid = torch.clamp(xgrid*2-1, min=-1, max=1)
        xgrid = xgrid.view(-1, 1, *self.grid_shape)

        weighted_offsets_y = weighted_offsets[:, 1, :, :] \
            .contiguous().view(-1, 1, *self.grid_shape)
        ygrid = weighted_offsets_y/weights
        ygrid = torch.clamp(ygrid*2-1, min=-1, max=1)
        ygrid = ygrid.view(-1, 1, *self.grid_shape)

        grid = torch.cat((xgrid, ygrid), 1)
        grid = F.interpolate(grid, size=self.output_shape, mode='bilinear',
                             align_corners=True)
        return grid.permute(0, 2, 3, 1)
    


# @GRID_GENERATORS.register_module()
class FixedKDEGrid(nn.Module, RecasensSaliencyToGridMixin):
    """ Grid generator that uses a fixed saliency map -- KDE SD.
        
        If the vanishing point is fixed, this class can be instead 
        used to load saliency during inference.
    """

    def __init__(self, saliency_file, **kwargs):
        super(FixedKDEGrid, self).__init__()
        RecasensSaliencyToGridMixin.__init__(self, **kwargs)
        self.saliency = pickle.load(open(saliency_file, 'rb'))
        # print("saliency is", self.saliency.shape) # [1, 1, 31, 51]

    def forward(self, imgs, v_pts, gt_bboxes # NOTE: vp no use here
                # img_metas, **kwargs
                ):
        
        # Check if imgs is in BCHW format
        assert len(imgs.shape) == 4, "Expected imgs to be in BCHW format"
        
        # Extract the shape of the input images
        self.update_output_shape(imgs.shape[2:4])

        # vis_options = kwargs.get('vis_options', {})
        device = imgs.device
        grid = self.saliency_to_grid(imgs, self.saliency, device)

        return grid
    

class BaseKDEGrid(nn.Module, RecasensSaliencyToGridMixin):
    """
    Base grid generator that uses a two-plane based saliency map 
    which has a fixed parameter set we learn.
    """

    def __init__(self, homo_layer, **kwargs):
        
        # Call parent class constructors
        super(BaseKDEGrid, self).__init__()
        RecasensSaliencyToGridMixin.__init__(self, **kwargs)

        self.homo = homo_layer

    def get_saliency(self, imgs, v_pts):
        device = imgs.device
        v_pts = torch.tensor(v_pts, device=device)
        v_pts = v_pts.unsqueeze(0)

        self.saliency = self.homo.forward(imgs, v_pts)

        self.saliency = F.interpolate(self.saliency, (31, 51))

        return self.saliency        

    def forward(self, imgs, v_pts, gt_bboxes):
        # Check if imgs is in BCHW format
        assert len(imgs.shape) == 4, f"Expected imgs to be in BCHW format. Now imgs.shape = {imgs.shape}"
        
        # Extract the shape of the input images
        self.update_output_shape(imgs.shape[2:4])

        device = imgs.device

        self.saliency = self.get_saliency(imgs, v_pts)

        grid = self.saliency_to_grid(imgs, self.saliency, device)

        return grid


class CuboidGlobalKDEGrid(BaseKDEGrid):
    def __init__(self, **kwargs):
        super(CuboidGlobalKDEGrid, self).__init__(homo_layer=CuboidLayerGlobal(), **kwargs)

class MidKDEGrid(BaseKDEGrid):
    def __init__(self, 
                 min_theta=TripetLayerGlobal.DEFAULTS['min_theta'], 
                 max_theta=TripetLayerGlobal.DEFAULTS['max_theta'],
                 min_theta_top=TripetLayerGlobal.DEFAULTS['min_theta_top'], 
                 max_theta_top=TripetLayerGlobal.DEFAULTS['max_theta_top'],
                 min_alpha_top=TripetLayerGlobal.DEFAULTS['min_alpha_top'], 
                 max_alpha_top=TripetLayerGlobal.DEFAULTS['max_alpha_top'],
                 **kwargs):
        super(MidKDEGrid, self).__init__(homo_layer=TripetLayerGlobal(min_theta=min_theta, 
                                                                      max_theta=max_theta,
                                                                      min_theta_top=min_theta_top, 
                                                                      max_theta_top=max_theta_top,
                                                                      min_alpha_top=min_alpha_top, 
                                                                      max_alpha_top=max_alpha_top), 
                                         **kwargs)


# NOTE: write this as separate class to reduce code duplication
class SaliencyMixin:
    # NOTE: single bboxes always symmetric, but multiple bboxes are not
    # TODO: combine saliency map wisely (instead of simple adding them together)
    def bbox2sal(self, batch_bboxes, img_shape, jitter=None, symmetry=False):
        device = batch_bboxes[0].device
        h_out, w_out = self.grid_shape
        sals = []

        # Assuming all bboxes in batch_bboxes are for a single image with shape img_shape
        h, w = img_shape[-2:]  # img_shape should be a tuple (h, w)

        if len(batch_bboxes) == 0:  # zero detections case
            sal = torch.ones(h_out, w_out, device=device).unsqueeze(0)
            sal /= sal.sum()
            sals.append(sal)
            return torch.stack(sals)  # Return early if no bboxes

        bboxes = batch_bboxes  # All bboxes are for the same image
        bboxes[:, 2:] -= bboxes[:, :2]  # ltrb -> ltwh
        cxy = bboxes[:, :2] + 0.5 * bboxes[:, 2:]
        # print("bboxes shape", bboxes.shape) # # [N, 4]
        # print("cxy.shape", cxy.shape) # [N, 2]

        if jitter is not None:
            cxy += 2 * jitter * (torch.randn(cxy.shape, device=device) - 0.5)

        widths = (bboxes[:, 2] * self.bandwidth_scale).unsqueeze(1)
        heights = (bboxes[:, 3] * self.bandwidth_scale).unsqueeze(1)

        X, Y = torch.meshgrid(
            torch.linspace(0, w, w_out, dtype=torch.float, device=device),
            torch.linspace(0, h, h_out, dtype=torch.float, device=device),
        )

        grids = torch.stack((X.flatten(), Y.flatten()), dim=1).t()
        # print("grids.shape", grids.shape) # [2, 31*51]
        m, n = cxy.shape[0], grids.shape[1]

        norm1 = (cxy[:, 0:1] ** 2 / widths + cxy[:, 1:2] ** 2 / heights).expand(m, n)
        norm2 = grids[0:1, :] ** 2 / widths + grids[1:2, :] ** 2 / heights
        norms = norm1 + norm2
        # print("norms.shape", norms.shape) # [N, 31*51]

        cxy_norm = cxy
        cxy_norm[:, 0:1] /= widths
        cxy_norm[:, 1:2] /= heights

        distances = norms - 2 * cxy_norm.mm(grids)
        # print("distances shape", distances.shape) # [N, 31*51]

        sal = (-0.5 * distances).exp()
        sal = self.amplitude_scale * (sal / (0.00001 + sal.sum(dim=1, keepdim=True)))
        sal += 1 / ((2 * self.padding_size + 1) ** 2)
        sal = sal.sum(dim=0)
        sal /= sal.sum()
        sal = sal.reshape(w_out, h_out).t().unsqueeze(0)
        sals.append(sal)
        # print("sal is", sal.shape)

        if symmetry:
            print("Using symmetry")
            sal = self.make_symmetric_around_max(sal)        

        return torch.stack(sals)
    
    # NOTE: use this to create symmetric saliency maps (NOTE: no use for now, since vis becomes worse)
    @staticmethod
    def make_symmetric_around_max(saliency_map):
        # Find the column (axis) with the max value in the entire saliency_map
        _, _, max_x = torch.where(saliency_map == saliency_map.max())
        max_x = max_x[0].item()

        # Define the range to iterate over for both left and right
        left_indices = torch.arange(max_x, -1, -1)
        right_indices = torch.arange(max_x, saliency_map.size(2))

        # Ensure the lengths of left_indices and right_indices are the same
        length = min(len(left_indices), len(right_indices))
        left_indices = left_indices[:length]
        right_indices = right_indices[:length]

        # Use broadcasting to find the max values between left and right for all rows
        max_values = torch.max(saliency_map[:, :, left_indices], saliency_map[:, :, right_indices])

        # Assign the max values to both sides of the matrix
        saliency_map[:, :, left_indices] = max_values
        saliency_map[:, :, right_indices] = max_values

        return saliency_map

class PlainKDEGrid(nn.Module, RecasensSaliencyToGridMixin, SaliencyMixin):
    """Image adaptive grid generator with fixed hyperparameters -- KDE SI"""

    def __init__(
        self,
        attraction_fwhm=4,
        bandwidth_scale=64,
        amplitude_scale=1,
        **kwargs
    ):
        super(PlainKDEGrid, self).__init__()
        RecasensSaliencyToGridMixin.__init__(self, **kwargs)
        self.attraction_fwhm = attraction_fwhm
        self.bandwidth_scale = bandwidth_scale
        self.amplitude_scale = amplitude_scale

    def forward(self, imgs, v_pts, gt_bboxes, jitter=False):
        # Check if imgs is in BCHW format
        assert len(imgs.shape) == 4, "Expected imgs to be in BCHW format"
        
        # Extract the shape of the input images
        self.update_output_shape(imgs.shape[2:4])

        img_shape = imgs.shape
        
        if isinstance(gt_bboxes, torch.Tensor):
            batch_bboxes = gt_bboxes
        else:
            if len(gt_bboxes[0].shape) == 3:
                batch_bboxes = gt_bboxes[0].clone()  # noqa: E501, removing the augmentation dimension
            else:
                batch_bboxes = [bboxes.clone() for bboxes in gt_bboxes]
        device = batch_bboxes[0].device
        saliency = self.bbox2sal(batch_bboxes, img_shape, jitter)

        # # ################# For debug only #################
        # v_pts = torch.tensor(v_pts, device=device)
        # v_pts = v_pts.unsqueeze(0)

        # # # Extract the vanishing point and saliency map
        # v_pts_original = v_pts[0].cpu().numpy()  # Assuming v_pts is like [[x, y]]
        # # print("saliency shape", saliency.shape) # [1, 1, 31, 51]
        # saliency_np = saliency.squeeze(0).squeeze(0).cpu().detach().numpy()
        # # NOTE: multiply 100 for visualization purposes
        # saliency_np = saliency_np * 100
        # # print("PlainKDEGrid saliency mean", saliency_np.mean())
        # # print("PlainKDEGrid saliency max", saliency_np.max())
        # # print("PlainKDEGrid saliency nonzero", np.count_nonzero(saliency_np > 0))
        # # print("saliency_np shape", saliency_np.shape) # [31, 51]

        # # Convert saliency map to 3 channel image
        # saliency_np_with_vp = cv2.cvtColor(saliency_np, cv2.COLOR_GRAY2BGR)

        # # Draw a circle at the vanishing point on the saliency map
        # vanishing_point_color = (0, 0, 255)  # BGR color format (red)
        # cv2.circle(saliency_np_with_vp, 
        #             (int(v_pts_original[0]), int(v_pts_original[1])), 
        #             5, 
        #             vanishing_point_color, 
        #             -1)  # Draw a filled circle

        # # Save the modified saliency map
        # # print("saliency_np_with_vp shape", saliency_np_with_vp.shape) # [31, 51, 3]
        # save_image(torch.from_numpy(saliency_np_with_vp).permute(2, 0, 1).unsqueeze(0), "saliency_PlainKDEGrid.png")
        # ################# For debug only #################

        grid = self.saliency_to_grid(imgs, saliency, device)

        return grid
    

class MixKDEGrid(BaseKDEGrid, SaliencyMixin):

    def __init__(self, 
                 attraction_fwhm = 4,
                 bandwidth_scale = 64,
                 amplitude_scale = 1,
                 homo_layer: str = 'cuboid', 
                 **kwargs):
        
        # Select the appropriate layer based on the homo_layer argument
        if homo_layer == 'cuboid':
            layer_instance = CuboidLayerGlobal()
        elif homo_layer == 'tripet':
            layer_instance = TripetLayerGlobal()
        else:
            raise ValueError(f"Unknown homo_layer: {homo_layer}. Supported values are 'cuboid' or 'tripet'.")

        # Initialize the base class with the chosen layer
        super(MixKDEGrid, self).__init__(layer_instance, **kwargs)

        # Set the additional attributes for the class
        self.attraction_fwhm = attraction_fwhm
        self.bandwidth_scale = bandwidth_scale
        self.amplitude_scale = amplitude_scale
        
        # Define learnable parameters (TODO: tune these parameter later, or make it learnable)
        self.alpha = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        self.beta = nn.Parameter(torch.tensor(0.5), requires_grad=True)

    def compute_bbox_saliency(self, imgs, gt_bboxes, jitter):
        # print("gt_bboxes is", gt_bboxes)
        img_shape = imgs.shape
        
        if isinstance(gt_bboxes, torch.Tensor):
            batch_bboxes = gt_bboxes
        else:
            if len(gt_bboxes[0].shape) == 3:
                batch_bboxes = gt_bboxes[0].clone()  # noqa: E501, removing the augmentation dimension
            else:
                batch_bboxes = [bboxes.clone() for bboxes in gt_bboxes]
        saliency = self.bbox2sal(batch_bboxes, img_shape, jitter)
        return saliency

    def forward(self, imgs, v_pts, gt_bboxes, jitter=False):
        device = imgs.device

        # NOTE: hardcode this line here to get image shape
        self.update_output_shape(imgs.shape[2:4])
        
        # Image-level saliency
        img_saliency = super().get_saliency(imgs, v_pts)
        
        # Bbox-level saliency
        bbox_saliency = self.compute_bbox_saliency(imgs, gt_bboxes, jitter)
        assert bbox_saliency.shape == img_saliency.shape, f"saliency {bbox_saliency.shape} mismatches img_saliency {img_saliency.shape}"
        
        # Mix saliencies
        # print(f"alpha = {self.alpha}; beta = {self.beta}")
        # print("bbox_saliency mean", bbox_saliency.mean())
        # print("img_saliency mean", img_saliency.mean())
        mixed_saliency = self.alpha * bbox_saliency + self.beta * img_saliency
        
        return self.saliency_to_grid(imgs, mixed_saliency, device)