from math import pi as PI

import numpy as np
import torch
from pytorch3d.transforms import euler_angles_to_matrix, so3_relative_angle
from scipy import ndimage
from torch import Tensor

MAX_FLOAT32 = torch.finfo(torch.float32).max

N_TESTS = 100
BATCH_SIZE = 1
N_IMAGES = N_TESTS * BATCH_SIZE
H = 256
W = 256


def euler_grid_to_unfolded_matrices_batch(euler_grid: Tensor, patch_size: int) -> Tensor:
    matrices_grid = euler_angles_to_matrix(euler_grid, 'ZXZ')
    matrices_grid_unfolded = matrices_grid.unfold(
        1, patch_size, 1).unfold(2, patch_size, 1)
    matrices_grid_unfolded = matrices_grid_unfolded.transpose(
        3, 5).transpose(4, 6)
    matrices_batch_unfolded = matrices_grid_unfolded.reshape(-1, 3, 3)
    return matrices_batch_unfolded


def geodesic_distance_exemplar(target_point: tuple[int, int],
                               source_image: np.ndarray,
                               original_mask: np.ndarray,
                               unfilled_mask: np.ndarray,
                               patch_size: int,
                               known_weight: float = 1,
                               restrict_search: int | None = None,
                               euclidean_penalty_weight: float = 0,
                               squared_euclidean_penalty: bool = False) -> np.ndarray:

    with torch.no_grad():

        source_image_tensor = torch.tensor(source_image).unsqueeze(0) * 2*PI
        source_image_tensor = source_image_tensor[:, patch_size//2 : -(patch_size//2),
                                                  patch_size//2 : -(patch_size//2)]

        half_patch_size = patch_size // 2

        source_point_mask = (~original_mask).astype(np.uint8)
        source_point_mask = ndimage.binary_erosion(
            source_point_mask, np.ones((patch_size, patch_size))
        )
        if restrict_search is not None:
            restrict_search_mask = np.zeros_like(source_point_mask)
            restrict_search_mask[max(target_point[0]-restrict_search, 0):target_point[0]+restrict_search+1,
                                 max(target_point[1]-restrict_search, 0):target_point[1]+restrict_search+1] = 1
            source_point_mask_restricted = source_point_mask * restrict_search_mask
            if source_point_mask_restricted.sum() == 0:
                print(
                    "Warning: Ignoring search restriction because it resulted in no source points.")
            else:
                source_point_mask = source_point_mask_restricted
        source_point_mask = source_point_mask[2*(patch_size//2) : -2*(patch_size//2),
                                              2*(patch_size//2) : -2*(patch_size//2)]
        source_point_mask = torch.tensor(source_point_mask).unsqueeze(0)

        target_patch_unknown_mask = patch(
            unfilled_mask, target_point, half_patch_size)
        target_patch_unknown_mask_unfolded = torch.tensor(target_patch_unknown_mask).tile(
            (BATCH_SIZE, H-2*half_patch_size, W-2*half_patch_size, 1, 1))
        target_patch = patch(source_image, target_point,
                             half_patch_size) * 2*PI
        target_patch_unfolded = torch.tensor(target_patch).tile(
            (BATCH_SIZE, H-2*half_patch_size, W-2*half_patch_size, 1, 1, 1))
        target_patch_euler_batch = target_patch_unfolded.reshape(-1, 3)
        target_patch_matrices_batch = euler_angles_to_matrix(
            target_patch_euler_batch, 'ZXZ')
        relative_angles = so3_relative_angle(
            target_patch_matrices_batch,
            euler_grid_to_unfolded_matrices_batch(source_image_tensor, patch_size))
        relative_angles_grid = relative_angles.reshape(
            BATCH_SIZE, H-2*(patch_size//2), W-2*(patch_size//2), patch_size, patch_size)
        sum_squared_errors_known = torch.sum(
            torch.square(relative_angles_grid * ~target_patch_unknown_mask_unfolded), dim=(3, 4))
        sum_squared_errors_unknown = torch.sum(
            torch.square(relative_angles_grid * target_patch_unknown_mask_unfolded), dim=(3, 4))
        weighted_sum_squared_errors = (known_weight*sum_squared_errors_known
                                        + (1-known_weight)*sum_squared_errors_unknown)
        euclidean_distances_y, euclidean_distances_x = torch.meshgrid(
            torch.arange(0, weighted_sum_squared_errors.shape[1]) - target_point[0] + 2*(patch_size//2),
            torch.arange(0, weighted_sum_squared_errors.shape[2]) - target_point[1] + 2*(patch_size//2),
            indexing='ij'
        )
        euclidean_penalty = (euclidean_distances_x**2
                             + euclidean_distances_y**2)
        if not squared_euclidean_penalty:
            euclidean_penalty = euclidean_penalty**0.5
        distances = weighted_sum_squared_errors + euclidean_penalty_weight * euclidean_penalty
        distances[source_point_mask == 0] = MAX_FLOAT32
        min_indices = distances.reshape(
            BATCH_SIZE, -1).min(dim=1).indices
        min_indices += torch.arange(
            0, BATCH_SIZE*(H-2*(patch_size//2))*(W-2*(patch_size//2)),
            (H-2*(patch_size//2))*(W-2*(patch_size//2)))
        source_image_unfolded = source_image_tensor.unfold(1, patch_size, 1).unfold(
            2, patch_size, 1).permute(0, 1, 2, 4, 5, 3).reshape(
            BATCH_SIZE*(H-2*(patch_size//2))*(W-2*(patch_size//2)),
            patch_size, patch_size, 3)
        min_patches = source_image_unfolded[
            min_indices, :, :, :].reshape(BATCH_SIZE, patch_size, patch_size, 3)
        source_patch = min_patches[0].detach().cpu().numpy() / (2*PI)

        return source_patch


def patch_slice(point: tuple, half_patch_size: int) -> tuple:
    return (slice(max(point[0] - half_patch_size, 0), point[0] + half_patch_size+1),
            slice(max(point[1] - half_patch_size, 0), point[1] + half_patch_size+1))


def patch(im: np.ndarray, point: tuple, half_patch_size: int):
    return im[patch_slice(point, half_patch_size)]
