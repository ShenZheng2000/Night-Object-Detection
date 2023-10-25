import torch
import kornia as K

from typing import Optional
import warnings
import numpy as np

try:
    from kornia.utils import _extract_device_dtype
    from kornia.geometry.epipolar import normalize_points
except:
    pass

def find_homography_dlt_unnormalized(
    points1: torch.Tensor, points2: torch.Tensor, weights: Optional[torch.Tensor] = None
) -> torch.Tensor:
    r"""Computes the homography matrix using the DLT formulation without normalization.

    The linear system is solved by using the Weighted Least Squares Solution for the 4 Points algorithm.

    Args:
        points1: A set of points in the first image with a tensor shape :math:`(B, N, 2)`.
        points2: A set of points in the second image with a tensor shape :math:`(B, N, 2)`.
        weights: Tensor containing the weights per point correspondence with a shape of :math:`(B, N)`.

    Returns:
        the computed homography matrix with shape :math:`(B, 3, 3)`.
    """
    assert points1.shape == points2.shape, points1.shape
    assert len(points1.shape) >= 1 and points1.shape[-1] == 2, points1.shape
    assert points1.shape[1] >= 4, points1.shape

    # print("points1.dtype", points1.dtype)
    # print("points2.dtype", points2.dtype)

    device, dtype = _extract_device_dtype([points1, points2])

    # print(points1.shape)
    eps: float = 1e-8
    points1_norm, transform1 = normalize_points(points1)
    points2_norm, transform2 = normalize_points(points2)

    x1, y1 = torch.chunk(points1_norm, dim=-1, chunks=2)  # BxNx1
    x2, y2 = torch.chunk(points2_norm, dim=-1, chunks=2)  # BxNx1
    ones, zeros = torch.ones_like(x1), torch.zeros_like(x1)

    # DIAPO 11: https://www.uio.no/studier/emner/matnat/its/nedlagte-emner/UNIK4690/v16/forelesninger/lecture_4_3-estimating-homographies-from-feature-correspondences.pdf  # noqa: E501
    ax = torch.cat([zeros, zeros, zeros, -x1, -y1, -ones, y2 * x1, y2 * y1, y2], dim=-1)
    ay = torch.cat([x1, y1, ones, zeros, zeros, zeros, -x2 * x1, -x2 * y1, -x2], dim=-1)
    A = torch.cat((ax, ay), dim=-1).reshape(ax.shape[0], -1, ax.shape[-1])

    if weights is None:
        # All points are equally important
        A = A.transpose(-2, -1) @ A
    else:
        # We should use provided weights
        assert len(weights.shape) == 2 and weights.shape == points1.shape[:2], weights.shape
        w_diag = torch.diag_embed(weights.unsqueeze(dim=-1).repeat(1, 1, 2).reshape(weights.shape[0], -1))
        A = A.transpose(-2, -1) @ w_diag @ A

    try:
        # U, S, V = torch.svd(A)

        # TODO: try batch_svd_package
        from torch_batch_svd import svd
        U, S, V = svd(A)

        # use numpy to hard-code implement the SVD => not working
        # A_np = A.cpu().numpy()
        # U_np, S_np, V_np = np.linalg.svd(A_np)
        # U, S, V = [torch.tensor(x, device=A.device, dtype=A.dtype) for x in [U_np, S_np, V_np]]

    except:
        warnings.warn('SVD did not converge', RuntimeWarning)
        return torch.empty((points1_norm.size(0), 3, 3), device=device, dtype=dtype)

    H = V[..., -1].view(-1, 3, 3)
    # H = transform2.inverse() @ (H @ transform1)
    
    # NOTE: use numpy for now
    # H = transform2.pinverse() @ (H @ transform1)
    H = pinverse_using_numpy(transform2) @ (H @ transform1)

    H_norm = H / (H[..., -1:, -1:] + eps)

    return H_norm


# NOTE: mimic torch.inverse() => DONE
def pinverse_using_numpy(tensor):
    """Compute the pseudo-inverse of a tensor using numpy."""
    tensor_np = tensor.cpu().numpy()
    tensor_np_inv = np.linalg.pinv(tensor_np)
    tensor_inv = torch.from_numpy(tensor_np_inv).to(tensor.device).to(tensor.dtype)
    return tensor_inv


# # NOTE: mimic torch.svd() => not working for now
# def numpy_svd(A: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor):
#     batch_size, C, H, W = A.shape
    
#     # Convert the batch tensor to numpy and reshape to 2D matrices
#     A_np = A.cpu().numpy().reshape(batch_size * C, H, W)
    
#     # Compute the SVD for all matrices in the batch
#     U_np, S_np, Vh_np = np.linalg.svd(A_np)
    
#     # Reshape the results back to the original shape
#     U = torch.tensor(U_np.reshape(batch_size, C, H, W), device=A.device, dtype=A.dtype)
#     S = torch.tensor(S_np.reshape(batch_size, C, H), device=A.device, dtype=A.dtype)
#     Vh = torch.tensor(Vh_np.reshape(batch_size, C, W, W), device=A.device, dtype=A.dtype)
    
#     return U, S, Vh



if __name__ == "__main__":
    pt_dst = torch.tensor(
        [[[   0.,    0.],
         [1023.,    0.],
         [1023.,  511.],
         [   0.,  511.]]]
         )
    pt_src = torch.tensor(
        [[[ 427.1642,  141.8750],
         [ 734.0642,  169.4995],
         [1023.0000,  511.0000],
         [   0.0000,  511.0000]]]
         )
    
    # print("pt_dst.shape", pt_dst.shape)
    # print("pt_src.shape", pt_src.shape)

    H_custom = find_homography_dlt_unnormalized(pt_dst, pt_src)
    H = K.geometry.homography.find_homography_dlt(pt_dst, pt_src)

    print(torch.allclose(H_custom, H, atol=1e-6))

    print("H_custom", H_custom)
    print("H", H)



