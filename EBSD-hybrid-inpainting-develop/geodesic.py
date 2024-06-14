from math import pi as PI

import torch
from pytorch3d.transforms import euler_angles_to_matrix, so3_relative_angle
from torch import Tensor
from torch.nn.common_types import _ratio_3_t
from torch.nn.modules.utils import _triple


def euler_grid_to_batch(euler_grid: Tensor) -> Tensor:
    """
    Reshape batched grid of Euler angles (4D) to batch of Euler angles (2D).

    Written for compatibility with PyTorch3D's euler_angles_to_matrix.
    """
    return euler_grid.transpose(1, 3).reshape(-1, 3)


def euler_scale(euler_batch: Tensor, scale: _ratio_3_t) -> Tensor:
    """
    Multiply batch of Euler angles by a scaling factor.

    If scale is of type float, the same scaling factor is applied to all Euler
    angles. If scale is of type tuple[float, float, float], a different scaling
    factor is applied to each successive rotation.
    """
    scale_tensor = torch.tensor(_triple(scale),
                                device=euler_batch.device).unsqueeze(0)
    return euler_batch * scale_tensor


def geodesic_mse(ground_truth_euler_grid: Tensor,
                 predicted_euler_grid: Tensor,
                 known_mask: Tensor,
                 convention: str = 'ZXZ',
                 scale: _ratio_3_t = (2*PI, PI, 2*PI),
                 unknown_only: bool = True
                 ) -> Tensor:
    """
    Calculate the geodesic loss between two grids of Euler angles.

    The geodesic loss is calculated as the mean of the squared geodesic
    distances on SO(3) between the ground truth and predicted Euler angles. If
    unknown_only is True, only the geodesic distances in the missing region are
    considered, otherwise the geodesic distances between all corresponding
    orientations are considered. The geodesic distance on SO(3) between a pair
    of orientations is the same as the misorientation angle between the pair of
    orientations (disregarding crystal symmetry).
    """

    # Convert grids of Euler angles to batches of Euler angles.
    ground_truth_euler_batch = euler_grid_to_batch(ground_truth_euler_grid)
    predicted_euler_batch = euler_grid_to_batch(predicted_euler_grid)

    # Scale Euler angles to correct range.
    ground_truth_euler_batch = euler_scale(ground_truth_euler_batch, scale)
    predicted_euler_batch = euler_scale(predicted_euler_batch, scale)

    # Convert batches of Euler angles to batches of matrices.
    ground_truth_matrices = euler_angles_to_matrix(ground_truth_euler_batch,
                                                   convention)
    predicted_matrices = euler_angles_to_matrix(predicted_euler_batch,
                                                convention)

    # Calculate the geodesic distance on SO(3) (misorientation angle
    # disregarding symmetry) between each pair of ground truth and predicted
    # orientations.
    geodesic_distances = so3_relative_angle(ground_truth_matrices,
                                            predicted_matrices)

    if unknown_only:
        # Ignore geodesic distances for the known region.
        reshaped_mask = euler_grid_to_batch(known_mask)[:, 0]
        geodesic_distances[reshaped_mask == 1] = 0
        loss = geodesic_distances.square().sum() / (reshaped_mask == 0).sum()
    else:
        loss = geodesic_distances.square().mean()

    return loss


def geodesic_distances(ground_truth_euler_grid: Tensor,
                       predicted_euler_grid: Tensor,
                       known_mask: Tensor,
                       convention: str = 'ZXZ',
                       scale: _ratio_3_t = (2*PI, PI, 2*PI),
                       unknown_only: bool = True
                       ) -> Tensor:
    """
    Calculate the pointwise geodesic distance between two grids of Euler angles.

    If unknown_only is True, only the geodesic distances in the missing region are
    considered, otherwise the geodesic distances between all corresponding
    orientations are considered. The geodesic distance on SO(3) between a pair
    of orientations is the same as the misorientation angle between the pair of
    orientations (disregarding crystal symmetry).
    """

    # Convert grids of Euler angles to batches of Euler angles.
    ground_truth_euler_batch = euler_grid_to_batch(ground_truth_euler_grid)
    predicted_euler_batch = euler_grid_to_batch(predicted_euler_grid)

    # Scale Euler angles to correct range.
    ground_truth_euler_batch = euler_scale(ground_truth_euler_batch, scale)
    predicted_euler_batch = euler_scale(predicted_euler_batch, scale)

    # Convert batches of Euler angles to batches of matrices.
    ground_truth_matrices = euler_angles_to_matrix(ground_truth_euler_batch,
                                                   convention)
    predicted_matrices = euler_angles_to_matrix(predicted_euler_batch,
                                                convention)

    # Calculate the geodesic distance on SO(3) (misorientation angle
    # disregarding symmetry) between each pair of ground truth and predicted
    # orientations.
    geodesic_distances = so3_relative_angle(ground_truth_matrices,
                                            predicted_matrices)

    if unknown_only:
        # Ignore geodesic distances for the known region.
        reshaped_mask = euler_grid_to_batch(known_mask)[:, 0]
        geodesic_distances = geodesic_distances[reshaped_mask == 0]

    return geodesic_distances
