import warnings
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

try:
    from kornia.geometry.linalg import transform_points
    from kornia.utils import create_meshgrid
    from kornia.testing import check_is_tensor
except:
    pass

import numpy as np
from .homography_torch import pinverse_using_numpy

# def _torch_inverse_cast_old(input: torch.Tensor) -> torch.Tensor:
#     """Helper function to make torch.inverse work with other than fp32/64.

#     The function torch.inverse is only implemented for fp32/64 which makes
#     impossible to be used by fp16 or others. What this function does, is cast
#     input data type to fp32, apply torch.inverse, and cast back to the input dtype.
#     """
#     assert isinstance(input, torch.Tensor), f"Input must be torch.Tensor. Got: {type(input)}."
#     dtype: torch.dtype = input.dtype
#     if dtype not in (torch.float32, torch.float64):
#         dtype = torch.float32
#     # torch.cuda.empty_cache()

#     # return torch.linalg.pinv(input.to(dtype)).to(input.dtype)
#     return torch.pinverse(input.to(dtype)).to(input.dtype)
#     # return torch.inverse(input.to(dtype)).to(input.dtype)

def _torch_inverse_cast(A: torch.Tensor, use_old = False) -> torch.Tensor:

    if use_old:
        # assert isinstance(input, torch.Tensor), f"Input must be torch.Tensor. Got: {type(input)}."
        dtype: torch.dtype = A.dtype
        if dtype not in (torch.float32, torch.float64):
            dtype = torch.float32
        # torch.cuda.empty_cache()

        # return torch.linalg.pinv(input.to(dtype)).to(input.dtype)
        return torch.pinverse(A.to(dtype)).to(A.dtype)
        # return torch.inverse(input.to(dtype)).to(input.dtype)

    else:
        # NOTE: use numpy for pinverse now
        # print("use numpy for pinverse now")
        A_inv = pinverse_using_numpy(A)

        return A_inv

        # NOTE: This still not working
        # print("====> Using new code")
        # # Step 1: Compute the SVD
        # U, S, V = torch.linalg.svd(A, full_matrices=False)
        
        # # Step 2: Invert the non-zero singular values
        # S_inv = torch.where(S > eps, 1.0 / S, torch.zeros_like(S))
        
        # # Use torch.diag_embed for batched diagonal matrices
        # S_inv_diag = torch.diag_embed(S_inv)
        
        # # Step 3: Compute the pseudoinverse
        # A_pinv = V.transpose(-2, -1) @ S_inv_diag @ U.transpose(-2, -1)
        
        # return A_pinv

def normal_transform_pixel(
    height: int,
    width: int,
    eps: float = 1e-14,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    r"""Compute the normalization matrix from image size in pixels to [-1, 1].

    Args:
        height image height.
        width: image width.
        eps: epsilon to prevent divide-by-zero errors

    Returns:
        normalized transform with shape :math:`(1, 3, 3)`.
    """
    tr_mat = torch.tensor([[1.0, 0.0, -1.0], [0.0, 1.0, -1.0], [0.0, 0.0, 1.0]], device=device, dtype=dtype)  # 3x3

    # prevent divide by zero bugs
    width_denom: float = eps if width == 1 else width - 1.0
    height_denom: float = eps if height == 1 else height - 1.0

    tr_mat[0, 0] = tr_mat[0, 0] * 2.0 / width_denom
    tr_mat[1, 1] = tr_mat[1, 1] * 2.0 / height_denom

    return tr_mat.unsqueeze(0)  # 1x3x3

def normalize_homography(
    dst_pix_trans_src_pix: torch.Tensor, dsize_src: Tuple[int, int], dsize_dst: Tuple[int, int], use_old = False
) -> torch.Tensor:
    r"""Normalize a given homography in pixels to [-1, 1].

    Args:
        dst_pix_trans_src_pix: homography/ies from source to destination to be
          normalized. :math:`(B, 3, 3)`
        dsize_src: size of the source image (height, width).
        dsize_dst: size of the destination image (height, width).

    Returns:
        the normalized homography of shape :math:`(B, 3, 3)`.
    """
    check_is_tensor(dst_pix_trans_src_pix)

    if not (len(dst_pix_trans_src_pix.shape) == 3 or dst_pix_trans_src_pix.shape[-2:] == (3, 3)):
        raise ValueError(
            "Input dst_pix_trans_src_pix must be a Bx3x3 tensor. Got {}".format(dst_pix_trans_src_pix.shape)
        )

    # source and destination sizes
    src_h, src_w = dsize_src
    dst_h, dst_w = dsize_dst

    # compute the transformation pixel/norm for src/dst
    src_norm_trans_src_pix: torch.Tensor = normal_transform_pixel(src_h, src_w).to(dst_pix_trans_src_pix)

    src_pix_trans_src_norm = _torch_inverse_cast(src_norm_trans_src_pix, use_old=use_old)
    dst_norm_trans_dst_pix: torch.Tensor = normal_transform_pixel(dst_h, dst_w).to(dst_pix_trans_src_pix)

    # compute chain transformations
    dst_norm_trans_src_norm: torch.Tensor = dst_norm_trans_dst_pix @ (dst_pix_trans_src_pix @ src_pix_trans_src_norm)
    return dst_norm_trans_src_norm

def warp_perspective(
    src: torch.Tensor,
    M: torch.Tensor,
    dsize: Tuple[int, int],
    mode: str = 'bilinear',
    padding_mode: str = 'zeros',
    align_corners: Optional[bool] = None,
    use_old = False,
) -> torch.Tensor:
    r"""Applies a perspective transformation to an image.

    .. image:: https://kornia-tutorials.readthedocs.io/en/latest/_images/warp_perspective_10_2.png

    The function warp_perspective transforms the source image using
    the specified matrix:

    .. math::
        \text{dst} (x, y) = \text{src} \left(
        \frac{M^{-1}_{11} x + M^{-1}_{12} y + M^{-1}_{13}}{M^{-1}_{31} x + M^{-1}_{32} y + M^{-1}_{33}} ,
        \frac{M^{-1}_{21} x + M^{-1}_{22} y + M^{-1}_{23}}{M^{-1}_{31} x + M^{-1}_{32} y + M^{-1}_{33}}
        \right )

    Args:
        src: input image with shape :math:`(B, C, H, W)`.
        M: transformation matrix with shape :math:`(B, 3, 3)`.
        dsize: size of the output image (height, width).
        mode: interpolation mode to calculate output values ``'bilinear'`` | ``'nearest'``.
        padding_mode: padding mode for outside grid values ``'zeros'`` | ``'border'`` | ``'reflection'``.
        align_corners(bool, optional): interpolation flag.

    Returns:
        the warped input image :math:`(B, C, H, W)`.

    Example:
       >>> img = torch.rand(1, 4, 5, 6)
       >>> H = torch.eye(3)[None]
       >>> out = warp_perspective(img, H, (4, 2), align_corners=True)
       >>> print(out.shape)
       torch.Size([1, 4, 4, 2])

    .. note::
        This function is often used in conjuntion with :func:`get_perspective_transform`.

    .. note::
        See a working example `here <https://kornia-tutorials.readthedocs.io/en/
        latest/warp_perspective.html>`_.
    """
    if not isinstance(src, torch.Tensor):
        raise TypeError("Input src type is not a torch.Tensor. Got {}".format(type(src)))

    if not isinstance(M, torch.Tensor):
        raise TypeError("Input M type is not a torch.Tensor. Got {}".format(type(M)))

    if not len(src.shape) == 4:
        raise ValueError("Input src must be a BxCxHxW tensor. Got {}".format(src.shape))

    if not (len(M.shape) == 3 and M.shape[-2:] == (3, 3)):
        raise ValueError("Input M must be a Bx3x3 tensor. Got {}".format(M.shape))

    # TODO: remove the statement below in kornia v0.6
    if align_corners is None:
        message: str = (
            "The align_corners default value has been changed. By default now is set True "
            "in order to match cv2.warpPerspective. In case you want to keep your previous "
            "behaviour set it to False. This warning will disappear in kornia > v0.6."
        )
        warnings.warn(message)
        # set default value for align corners
        align_corners = True

    B, C, H, W = src.size()
    h_out, w_out = dsize

    # we normalize the 3x3 transformation matrix and convert to 3x4
    dst_norm_trans_src_norm: torch.Tensor = normalize_homography(M, (H, W), (h_out, w_out), use_old=use_old)  # Bx3x3

    src_norm_trans_dst_norm = _torch_inverse_cast(dst_norm_trans_src_norm, use_old=use_old)  # Bx3x3

    # this piece of code substitutes F.affine_grid since it does not support 3x3
    grid = (
        create_meshgrid(h_out, w_out, normalized_coordinates=True, device=src.device).to(src.dtype).repeat(B, 1, 1, 1)
    )
    grid = transform_points(src_norm_trans_dst_norm[:, None, None], grid)

    return F.grid_sample(src, grid, align_corners=align_corners, mode=mode, padding_mode=padding_mode)


if __name__ == '__main__':
    # # NOTE: for debug the entire code
    img = torch.rand(1, 4, 5, 6)
    H = torch.eye(3)[None]
    out_new= warp_perspective(img, H, (4, 2), align_corners=True, use_old=False)
    out_old = warp_perspective(img, H, (4, 2), align_corners=True, use_old=True)

    # print(out_old.shape); print(out_new.shape) # [1, 4, 4, 2]
    # print(out_old); print(out_new)

    print(torch.allclose(out_old, out_new, atol=1e-6))

    # # Random tensor as input
    # A = torch.randn(4, 5).to(torch.float32)

    # # Using pseudo_inverse function
    # A_torch_inv_cast = _torch_inverse_cast(A, use_old=False)

    # # Using _torch_inverse_cast function (assuming torch.pinverse is working in your environment)
    # A_torch_inv_cast_old = _torch_inverse_cast(A, use_old=True)

    # print("Original Matrix:\n", A)
    # print("\nPseudo-inverse using pseudo_inverse:\n", A_torch_inv_cast_old)
    # print("\nPseudo-inverse using _torch_inverse_cast:\n", A_torch_inv_cast)

    # # print(torch.allclose(A @ A_pseudo_inv @ A, A, atol=1e-6))
    # # print(torch.allclose(A @ A_torch_inv_cast @ A, A, atol=1e-6))