import argparse
import os
import time
from time import time  # DEBUG
from typing import Literal

import numpy as np
import torch
from numba import njit
from pytorch3d.transforms import euler_angles_to_matrix, so3_relative_angle
from scipy import ndimage
from skimage.io import imsave

from dataset import load_ctf_to_tensor
from geodesic_distance_exemplar import geodesic_distance_exemplar

EULER_CONVENTION = 'ZXZ'


def save_euler_angles_image(path, image):
    scaling_factor = 255 * np.array([[[1, 2, 1]]])
    scaled_image = image * scaling_factor
    imsave(path, scaled_image.astype(np.uint8))


""" get_boundary
Inputs:
    im: An np.array. The mask of image
    known_bdry: Bool = If true, returns the boundary of known pixels.
                       Else, returns boundary of unknown pixels.
Returns:
    A boolean np.array that is the shape of the inputted image
    This array is all false except on the boundary points, which are true
"""


def get_boundary(unfilled: np.ndarray, known_bdry: bool = True, iterations: int = 1, patch_size: int = 3) -> np.ndarray:
    if known_bdry:
        boundary = ndimage.binary_dilation(
            unfilled, np.full((3, 3), True)) & ~unfilled
        # Exclude edges from boundary.
        boundary[:patch_size//2, :] = 0
        boundary[-(patch_size//2):, :] = 0
        boundary[:, :patch_size//2] = 0
        boundary[:, -(patch_size//2):] = 0
        return boundary
    else:
        return ~(ndimage.binary_erosion(unfilled, np.full((3, 3), True), iterations=iterations) ^ ~unfilled)


"""patch_slice
Inputs:
    point: a tuple. the index of the center point that we want the patch of
    half_patch_size: and int. the size of the desired patch divided by 2. 
Returns:
    A tuple that contains all four corner indexes of the patch.
"""


def patch_slice(point: tuple, half_patch_size: int) -> tuple:
    return (slice(max(point[0] - half_patch_size, 0), point[0] + half_patch_size+1),
            slice(max(point[1] - half_patch_size, 0), point[1] + half_patch_size+1))


"""patch
Inputs:
    im: np.ndarray, this is the array that we will pull the patch from. 
        i.e. it could pull that patch from the image we are filling in
            or pull the patch from the working mask so we know what is valid.
    point: A tuple. This is the center point of the patch that we want. 
    half_patch_size: an int, this is the size of the patch that we want. 
Returns:
    The patch centered at the point in the image
"""


def patch(image: np.ndarray, point: tuple, half_patch_size: int):
    return image[patch_slice(point, half_patch_size)]


"""
Inputs: 
    prev_confidences: An np.array, which is the array of the confidences for
        all points in the image. Note: expects `prev_confidences` 
        to be 0 in the unfilled region.
    point: a tuple, this is the point that we want to get the confidence of. 
    half_patch_size: an int, this is the size of the patch we are using /2
    patch_area: an int, this is the area of the patch we are using
Returns
    The sum of all the confidences in the patch, which will then become 
        the confidence of the inputted point,
"""


def get_confidence(prev_confidences: np.ndarray, point: tuple,
                   half_patch_size: int, patch_area: int) -> float:
    return np.sum(patch(prev_confidences, point, half_patch_size)) / patch_area


"""get_confidences
Inputs:
    prev_confidences: np.array the confidence matrix from the previous iteration    
    boundary: a list of all points at the boundary
    half_patch_size: an int, this is the size of the patch we are using /2
    patch_area: an int, this is the area of the patch we are using
Does:
    Sets up inputs for the get_confidence method. 
Returns:
    A list of the confidences at all of the boundary points.
"""


def get_confidences(prev_confidences: np.ndarray, boundary: list,
                    half_patch_size: int, patch_area: int
                    ) -> list:
    arg1 = [prev_confidences]*len(boundary)
    arg2 = boundary
    arg3 = [half_patch_size]*len(boundary)
    arg4 = [patch_area]*len(boundary)
    return list(map(get_confidence, arg1, arg2, arg3, arg4))


"""calc_normal_matrix
Inputs:
    in_mask: the working mask of the image
Does:
    Uses a sobel matrix to find the boundary of the mask
    Calculates the strngth of isophotes throughout the mask
Outputs:
    A matrix that contains the normal at each point in the mask
"""


def calc_normal_matrix(in_mask: np.ndarray) -> np.ndarray:
    x_kernel = np.array([[-.25, 0, .25], [-.5, 0, .5], [-.25, 0, .25]])
    y_kernel = np.array([[-.25, -.5, -.25], [0, 0, 0], [.25, .5, .25]])

    x_normal = ndimage.convolve(in_mask.astype(float), x_kernel)
    y_normal = ndimage.convolve(in_mask.astype(float), y_kernel)
    normal = np.dstack((x_normal, y_normal))

    height, width = normal.shape[:2]

    magnitude = np.sqrt(y_normal**2 + x_normal**2) \
        .reshape(height, width, 1) \
        .repeat(2, axis=2)
    magnitude[magnitude == 0] = 1
    unit_normal = normal/magnitude
    return unit_normal


def x_gradient(euler_grid: torch.Tensor) -> torch.Tensor:
    matrix_grid = euler_angles_to_matrix(euler_grid, EULER_CONVENTION)
    left_matrix_grid = matrix_grid[:, :, :-2, :, :]
    center_matrix_grid = matrix_grid[:, :, 1:-1, :, :]
    right_matrix_grid = matrix_grid[:, :, 2:, :, :]
    left_matrix_batch = left_matrix_grid.reshape(-1, 3, 3)
    center_matrix_batch = center_matrix_grid.reshape(-1, 3, 3)
    right_matrix_batch = right_matrix_grid.reshape(-1, 3, 3)
    left_center_angles = so3_relative_angle(left_matrix_batch,
                                            center_matrix_batch)
    center_right_angles = so3_relative_angle(center_matrix_batch,
                                             right_matrix_batch)
    x_gradient_grid = torch.zeros(euler_grid.shape[:-1])
    x_gradient_grid[:, :, 1:-1] -= left_center_angles.reshape(
        (*euler_grid.shape[:-2], euler_grid.shape[-2]-2))
    x_gradient_grid[:, :, 1:-1] += center_right_angles.reshape(
        (*euler_grid.shape[:-2], euler_grid.shape[-2]-2))
    return x_gradient_grid


def y_gradient(euler_grid: torch.Tensor) -> torch.Tensor:
    return x_gradient(euler_grid.transpose(1, 2)).transpose(1, 2)


def calc_gradient(image: np.ndarray, unfilled_mask: np.ndarray,
                  boundary: tuple, half_patch_size: int,
                  patch_size: int,
                  using_ml_template: bool,
                  mode: Literal['geodesic', 'RGB'] = 'geodesic') -> np.ndarray:
    if mode == 'RGB':
        grayscale_image = image.sum(axis=2) / 3
        gradient = np.dstack(np.gradient(grayscale_image))
        if not using_ml_template:
            dilation_kernel = np.zeros((3, 3), np.uint8)
            dilation_kernel[1, :] = 1
            dilation_kernel[:, 1] = 1
            invalid_gradient_mask = ndimage.binary_dilation(unfilled_mask, dilation_kernel)
            gradient[invalid_gradient_mask == 1] = 0
    elif mode == 'geodesic':
        with torch.no_grad():
            image_tensor = torch.tensor(image).unsqueeze(0)
            y_grad = y_gradient(image_tensor).squeeze(0)
            x_grad = x_gradient(image_tensor).squeeze(0)
            gradient = np.dstack((y_grad.numpy(), x_grad.numpy()))
    else:
        print('invalid mode for calc_gradient')
        quit()

    height, width = image.shape[:2]
    boundary_list = list(zip(boundary[0], boundary[1]))

    gradient_magnitude = np.sqrt(gradient[:, :, 0]**2 + gradient[:, :, 1]**2)
    max_gradient = np.zeros([height, width, 2])

    for point in boundary_list:
        patch_y_gradient = patch(gradient[:, :, 0], point, half_patch_size)
        patch_x_gradient = patch(gradient[:, :, 1], point, half_patch_size)
        patch_gradient_val = patch(gradient_magnitude, point, half_patch_size)

        patch_max_pos = np.unravel_index(
            patch_gradient_val.argmax(), patch_gradient_val.shape)

        max_gradient[point[0], point[1], 0] = patch_y_gradient[patch_max_pos]
        max_gradient[point[0], point[1], 1] = patch_x_gradient[patch_max_pos]

    return max_gradient


"""
Inputs:
    im: np.ndarray, the current working image
    unfilled_mask: np.ndarray, the current working mask
    boundary: tuple, list of boundary points as a tuple, 
    half_patch_size: int, the desired size of the patch /2
    patch_size: int, the size of the patch
Does:
    Calls functions to calculate the norm of the mask and gradient of the image
        returns the data at these points
Returns:
        data: a list of all of the data values at the points on the boundaries
"""


def get_data(im: np.ndarray, unfilled_mask: np.ndarray, boundary: tuple,
             half_patch_size: int, patch_size: int, using_ml_template: bool
             ) -> list:
    normal = calc_normal_matrix(unfilled_mask)
    gradient = calc_gradient(
        im, unfilled_mask, boundary, half_patch_size, patch_size, 
        using_ml_template, mode='RGB')
    gradient_perp = np.dstack((gradient[:, :, 1], -gradient[:, :, 0]))
    data = np.abs(np.sum(gradient_perp*normal, axis=2))
    boundary_list = list(zip(boundary[0], boundary[1]))
    data_list = []
    for point in boundary_list:
        data_list.append(data[point])
    return data_list


"""get_priority_point
Inputs:
    boundary: list, the list of points along the boundary of the damaged region
    confidences: list, the confidences of the points along damaged region 
    data: list, the datat of the points along the damaged region
Returns:
    The point with the highest priority,
        Note: Priority for a point is confidence * data at that point
"""


def get_priority_point(boundary: list, confidences: list, data: list):
    priorities = np.array(confidences) * np.array(data)
    target_ind = np.argmax(priorities)
    return (boundary[target_ind], confidences[target_ind])


"""Patch Distance Functions
Inputs:
    source_patch_flat: np.ndarray, the patch we will compare the target to
    target_patch_flat: np.ndarray, the patch we are looking to fill in
    filled_patch_flat: np.ndarray, the indices of the valid pixels in the target patch
    known_weight: float, the weight applied to the known part of the patch, unknown part weighted (1-known_weight)
Returns:
    The distance between the target and source patches, as defined by the particular distance function
Note: 
    This has been optimized for speed using @njit.
    We always take the patch with the smallest distance to be the best.  So some of these
    distance functions are modified slightly from their original definition to ensure that
    maximum similarity occurs at the minimum values, rather than maximum (noted in function
    specific comments)
"""

# Sum Squared Error


@njit
def SSE(source_patch_flat: np.ndarray, target_patch_flat: np.ndarray,
        filled_patch_flat: np.ndarray, known_weight: float = 1) -> float:
    return known_weight*np.sum(
        (source_patch_flat - target_patch_flat)[filled_patch_flat]**2
    ) + (1-known_weight)*np.sum(
        (source_patch_flat - target_patch_flat)[~filled_patch_flat]**2
    )

# Mean Squared Error


@njit
def MSE(source_patch_flat: np.ndarray, target_patch_flat: np.ndarray,
        filled_patch_flat: np.ndarray, known_weight: float = 1) -> float:
    return np.mean(
        (source_patch_flat - target_patch_flat)[filled_patch_flat]**2
    )


# Cosine similarity is bounded on [-1,1], with maximum similarity occuring at 1
# So we multiply by -1 to make maximum similarity occur at the minimum value
@njit
def cosine_similarity(source_patch_flat: np.ndarray, target_patch_flat: np.ndarray,
                      filled_patch_flat: np.ndarray) -> float:
    return -np.sum(source_patch_flat[filled_patch_flat] * target_patch_flat[filled_patch_flat])/(
        np.sqrt(np.sum(source_patch_flat[filled_patch_flat]**2)) *
        np.sqrt(np.sum(target_patch_flat[filled_patch_flat]**2)) + 1e-8)


# Structural Similarity Index Measure
# L is the 'dynamic range' of the pixels (max pixel value - min pixel value)
# For a normal image this would be 255, but for EBSD data it should be 359
# c1 and c2 ensure the denominator is never 0
# More similar images have higher SSIM, so multiply whole thing by -1
@njit
def SSIM(source_patch_flat: np.ndarray, target_patch_flat: np.ndarray,
         filled_patch_flat: np.ndarray) -> float:
    L = 359
    k1 = .01
    k2 = .03
    c1 = (k1*L)**2
    c2 = (k2*L)**2
    mux = np.mean(source_patch_flat[filled_patch_flat])
    muy = np.mean(target_patch_flat[filled_patch_flat])

    # Can't just pass [source, target] to np.cov when using @njit
    # So explicitly construct it as an ndarray
    cov_mat = np.empty((2, np.count_nonzero(filled_patch_flat)))
    cov_mat[0] = source_patch_flat[filled_patch_flat]
    cov_mat[1] = target_patch_flat[filled_patch_flat]

    cov = np.cov(cov_mat)
    covx = cov[0, 0]
    covy = cov[1, 1]
    covxy = cov[1, 0]
    return -((2*mux*muy + c1)*(2*covxy+c2))/((mux**2+muy**2+c1)*(covx+covy+c2))


# KL divergence can be +/-, with the best match around 0, so this is actually absolute value of KL divergence
@njit
def KLdivergence(source_patch_flat: np.ndarray, target_patch_flat: np.ndarray,
                 filled_patch_flat: np.ndarray) -> float:
    return np.abs(np.sum(target_patch_flat[filled_patch_flat] * np.log(
        target_patch_flat[filled_patch_flat] / source_patch_flat[filled_patch_flat])))


@njit
def logcosh(source_patch_flat: np.ndarray, target_patch_flat: np.ndarray,
            filled_patch_flat: np.ndarray) -> float:
    return np.sum(np.log(np.cosh(source_patch_flat[filled_patch_flat] - target_patch_flat[filled_patch_flat])))


"""dist
Inputs:
    source_patch_flat: np.ndarray, the patch we will compare the target to
    target_patch_flat: np.ndarray, the patch we are looking to fill in
    filled_patch_flat: np.ndarray, the indices of the valid pixels in the target patch
Returns:
    The Sum of Squared Distances between all of the known pixels in the target and source
        divided by the number of known pixels
Note: 
    This has been optimized for speed using @njit
    This is our new calculation 
"""


@njit
def our_dist(source_patch_flat: np.ndarray, target_patch_flat: np.ndarray,
             filled_patch_flat: np.ndarray) -> float:
    return np.sum(
        (source_patch_flat - target_patch_flat)[filled_patch_flat]**2
    ) / np.count_nonzero(filled_patch_flat)

# def get_exemplar(image: np.ndarray,
#                  target_point: tuple[int, int],
#                  known_mask: np.ndarray,
#                  distance_function:\
#                     Callable[[np.ndarray, np.ndarray, np.ndarray], float]
#                  ) -> np.ndarray:


"""get_best_exmeplar
Inputs:
    target_point: tuple, the point we are looking to inpaint
    im: np.ndarray, the working image
    original_mask: np.ndarray, the mask that we started with initially
    working_mask: np.ndarray, the mask that indicates what is currently filled and what is not
    patch_size: int, the patch size used for filling
    compare_psz: int = 0, if the user wishes to use a larger patch size for comparison that is used here,
        defaults to 0, as this is not in the original Criminisi paper
    our_distance: bool = True, if we are using our formula for distance this is true, 
        made togglable so it can be turned off as described in original Criminisi paper
    restrict_search: bool = False if we are restricting the search region this is true, 
        made togglable so it can be "turned off" 
    
Does:
    Initializes Search region, calculates distance of every patch
Returns:    
    The patch from the image that has the lowest distance
"""


def get_best_exemplar(target_point: tuple, im: np.ndarray,
                      original_mask: np.ndarray, working_mask: np.ndarray,
                      patch_size: int, compare_psz: int = 0,
                      distance_metric=SSE, restrict_search: bool = False,
                      euclidean_penalty: float = 0, known_weight: float = 1) -> np.ndarray:

    half_patch_size = compare_psz // 2

    filled_patch_flat = \
        ~patch(working_mask, target_point, half_patch_size).ravel()

    source_point_mask = ~original_mask
    source_point_mask = ndimage.binary_erosion(
        source_point_mask, np.full((3, 3), True), half_patch_size
    )
    if restrict_search:
        search_restriction = 5

        row_start = max(target_point[0] - im.shape[0] //
                        search_restriction + half_patch_size, 0)
        row_end = min(target_point[0] + im.shape[0]//search_restriction - half_patch_size,
                      im.shape[0] - half_patch_size)

        col_start = max(target_point[1] - im.shape[1] //
                        search_restriction + half_patch_size, 0)
        col_end = min(target_point[1] + im.shape[1]//search_restriction - half_patch_size,
                      im.shape[1] - half_patch_size)

        if row_start >= row_end or col_start >= col_end:
            raise ValueError("Search Restricted too much!")

        source_point_mask[:row_start, :] = False
        source_point_mask[row_end:, :] = False
        source_point_mask[:, :col_start] = False
        source_point_mask[:, col_end:] = False

        search_region = (slice(row_start, row_end), slice(col_start, col_end))

    else:
        source_point_mask[:half_patch_size, :] = False
        source_point_mask[-half_patch_size:, :] = False
        source_point_mask[:, :half_patch_size] = False
        source_point_mask[:, -half_patch_size:] = False

        search_region = (slice(half_patch_size, -half_patch_size),
                         slice(half_patch_size, -half_patch_size))

    source_points = source_point_mask.nonzero()
    source_points = list(zip(source_points[0], source_points[1]))

    if im.ndim == 3:
        distances = np.empty(im.shape)
        for channel in range(im.shape[2]):
            target_patch_flat = \
                patch(im[..., channel], target_point, half_patch_size).ravel()

            distances[search_region[0], search_region[1], channel] = ndimage.generic_filter(
                im[search_region[0], search_region[1], channel], distance_metric, size=(
                    compare_psz, compare_psz),
                extra_arguments=(target_patch_flat,
                                 filled_patch_flat, known_weight)
            )
        distances = np.sum(distances, axis=2)
    elif im.ndim == 2:
        target_patch_flat = patch(im, target_point, half_patch_size).ravel()
        distances = ndimage.generic_filter(
            im, distance_metric, size=(compare_psz, compare_psz),
            extra_arguments=(target_patch_flat,
                             filled_patch_flat, known_weight)
        )

    # TODO: Different metrics output on different scales, so the
    # Euclidean distance is probably having different levels of effect
    # for different metrics.
        # But since its weighted equally within the same metric, its
        # probably fine?
    if euclidean_penalty != 0:
        for row in range(im.shape[0]):
            for col in range(im.shape[1]):
                if not source_point_mask[row, col]:
                    continue
                distances[row, col] += euclidean_penalty * np.sqrt(
                    (target_point[0] - row)**2 + (target_point[1] - col)**2
                ) / np.sqrt(im.shape[0]**2 + im.shape[1]**2)

    source_point = source_points[np.argmin(distances[source_point_mask])]
    # print("Source: " + str(source_point))
    # print("Source Metric:" + str(np.min(distances[source_point_mask])))
    return patch(im, source_point, patch_size//2)


"""fill_patch
Inputs:
    im: np.ndarray, the working image
    target_point: tuple, the point we are filling in
    source_patch: np.ndarray, the patch we are using for filling
    unfilled_mask: np.ndarray, the working mask indicating what is and isnt filled in
    confidences: np.ndarray, array indicating the confidences of all pixels in the image
    target_confidence: float,the confidence of the target pixel
    half_patch_size: int, the size of the patch used for filling/2
Does:
    Updates the unfilled image, working mask and confidences
Note:
    modifies some inputs in-place
"""


def fill_patch(im: np.ndarray, target_point: tuple, source_patch: np.ndarray,
               unfilled_mask: np.ndarray, confidences: np.ndarray,
               target_confidence: float, half_patch_size: int,
               onion: bool = False):

    target_patch_slice = patch_slice(target_point, half_patch_size)
    unfilled_patch = unfilled_mask[target_patch_slice]

    im[target_patch_slice][unfilled_patch] = source_patch[unfilled_patch]
    unfilled_mask[target_patch_slice][unfilled_patch] = False

    if not onion:
        confidences[target_patch_slice] = target_confidence


"""inpaint
(This is essentially the main function)
Inputs:
    im: np.ndarray, the original image we are looking to have repaired
    original_mask: np.ndarray, indiciates damaged and known regions
    patch_size: int = 3, the size of patch used throughout the process
    compare_increase: int =0, if we would like to use a larger patch during filling, the increase in 
        size is indicated here
    distance_metric: (np.ndarray, np.ndarray, np.ndarray)->float, the distance metric to use when comparing patches
    restrict_search: bool =False, if True use a restricted search region centered at the target patch
    euclidean_penalty: bool=True, if we would like to add the Euclidean distance from a canidate patch to the fill region to the distance metric
    onion: bool=False, if we wish to use onion layering rather than the Criminisi priority for fill order
Throws:
    ValueError: If the patch size is not odd
    ValueError: if the comparison size is not even
    ValueError: If the color values of the pixels are not between 0 and 1
Does:
    Facilitates the inpainting 
Returns:
    The final inpainted image

"""

# `original_mask` should be a boolean mask


def inpaint(image: np.ndarray, original_mask: np.ndarray,
            using_ml_template: bool, patch_size: int = 3,
            compare_increase: int = 0, restrict_search: bool = False,
            euclidean_penalty: float = 0, mode: str = 'geodesic',
            distance_metric=SSE, onion: bool = False,
            known_weight: float = 1) -> np.ndarray:

    if original_mask.ndim == 3:
        original_mask = original_mask[:, :, 0]

    if patch_size % 2 != 1:
        raise ValueError("`patch_size` must be odd.")
    if compare_increase % 2 != 0:
        raise ValueError("`compare_increase` must be even.")

    image = image.copy()

    image_padding = ((patch_size // 2,), (patch_size // 2,), (0,))
    image = np.pad(image, image_padding, mode='edge')

    half_patch_size = patch_size // 2
    patch_area = patch_size**2
    # Unfilled is the "working mask", indicating what is left to be filled in
    mask_padding = ((patch_size // 2,), (patch_size // 2,))
    original_mask = np.pad(original_mask, mask_padding, mode='edge')
    unfilled = original_mask.copy()

    # Confidences is an array indicating the confidence of each point
    confidences = (~unfilled).copy().astype(float)

    compare_psz = patch_size + compare_increase

    total_exemplar_time = 0  # DEBUG
    total_data_time = 0  # DEBUG
    total_confidence_time = 0  # DEBUG
    t_start = time()  # DEBUG
    iter_counter = 0  # DEBUG

    # patch_counter = 0

    unpadded_unfilled = unfilled[(patch_size // 2) : -(patch_size // 2),
                                 (patch_size // 2) : -(patch_size // 2)]
    # Main loop, continue filling until nothing left to fill
    while np.any(unpadded_unfilled):
        iter_counter += 1
        # print(f"\nStarting iteration: {iter_counter}")

        boundary_tuple = get_boundary(unfilled, patch_size=patch_size).nonzero()
        boundary_list = list(zip(boundary_tuple[0], boundary_tuple[1]))

        # Follow Criminisi fill order
        if not onion:
            t_confidence_start = time()  # DEBUG
            boundary_confidences = get_confidences(
                confidences, boundary_list, half_patch_size, patch_area
            )

            total_confidence_time += time() - t_confidence_start

            t_data_start = time()  # DEBUG
            boundary_data = get_data(
                image, unfilled, boundary_tuple, half_patch_size, patch_size,
                using_ml_template)
            total_data_time += time() - t_data_start  # DEBUG

            target_point, target_confidence = get_priority_point(
                boundary_list, boundary_confidences, boundary_data
            )

            t_exemplar_start = time()  # DEBUG
            if mode == 'RGB':
                source_patch = get_best_exemplar(
                    target_point, image, original_mask, unfilled, patch_size,
                    compare_psz=compare_psz, distance_metric=distance_metric,
                    restrict_search=False, euclidean_penalty=0,
                    known_weight=known_weight)
            else:
                source_patch = geodesic_distance_exemplar(target_point,
                                                          image,
                                                          original_mask,
                                                          unfilled, patch_size,
                                                          known_weight=known_weight,
                                                          euclidean_penalty_weight=euclidean_penalty,
                                                          squared_euclidean_penalty=False)
            total_exemplar_time += (time() - t_exemplar_start)  # DEBUG

            # target_patch_image = (patch(image, target_point, patch_size // 2) * 255).astype(np.uint8)
            # target_patch_image[:, :, 1] *= 2
            # source_patch_image = (source_patch * 255).astype(np.uint8)
            # source_patch_image[:, :, 1] *= 2
            # imsave(f'source_and_target_patches/{patch_counter}_target_patch.png', target_patch_image)
            # imsave(f'source_and_target_patches/{patch_counter}_source_patch.png', source_patch_image)
            
            # patch_counter += 1

            fill_patch(
                image, target_point, source_patch, unfilled, confidences,
                target_confidence, half_patch_size, onion=onion
            )

            unpadded_unfilled = unfilled[(patch_size // 2) : -(patch_size // 2),
                                         (patch_size // 2) : -(patch_size // 2)]

        # Fill 1 layer using onion layering
        else:
            while len(boundary_list) > 0:

                target_point = boundary_list.pop()
                target_confidence = -1  # dummy value, since onion dosen't use confidence

                # print("Target point: " + str(target_point))

                t_exemplar_start = time()  # DEBUG
                source_patch = get_best_exemplar(
                    target_point, image, original_mask, unfilled,  patch_size, compare_psz,
                    distance_metric=distance_metric, restrict_search=restrict_search,
                    euclidean_penalty=0, known_weight=known_weight
                )
                total_exemplar_time += (time() - t_exemplar_start)  # DEBUG

                fill_patch(
                    image, target_point, source_patch, unfilled, confidences,
                    target_confidence, half_patch_size, onion=onion
                )

            return image, unfilled

    total_time = time() - t_start  # DEBUG

    return image[(patch_size // 2) : -(patch_size // 2),
                 (patch_size // 2) : -(patch_size // 2)]


if __name__ == '__main__':

    # Parse arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('source_ctf', type=str)
    parser.add_argument('output_dir', type=str)
    parser.add_argument('damage_side_length', type=int)
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    # Set up output directory.
    if os.path.isdir(args.output_dir):
        if len(os.listdir(args.output_dir)) != 0:
            print('ERROR: output_dir exists and is not empty.')
            quit()
    else:
        os.mkdir(args.output_dir)

    # Load image.
    image = load_ctf_to_tensor(
        args.source_ctf).numpy().transpose(1, 2, 0) / 360

    # Create mask.
    x_damage_center = np.random.randint(
        low=args.damage_side_length, high=image.shape[1]-args.damage_side_length+1)
    x_damage = slice(x_damage_center-args.damage_side_length//2,
                     x_damage_center+args.damage_side_length//2)
    y_damage_center = np.random.randint(
        low=args.damage_side_length, high=image.shape[0]-args.damage_side_length+1)
    y_damage = slice(y_damage_center-args.damage_side_length//2,
                     y_damage_center+args.damage_side_length//2)
    damage_slices = (y_damage, x_damage)
    mask = np.zeros_like(image, np.bool_)
    mask[damage_slices] = True

    damaged_image = image.copy()
    damaged_image[mask] = 1

    # Save original and damaged images.
    save_euler_angles_image(os.path.join(args.output_dir, 'original.png'),
                            image)
    save_euler_angles_image(os.path.join(args.output_dir, 'damaged.png'),
                            damaged_image)

    for euclidean_penalty in [0, 0.01, 0.1, 1]:
        inpainted = inpaint(damaged_image, mask, False,
                            euclidean_penalty=euclidean_penalty)
        save_euler_angles_image(
            os.path.join(args.output_dir,
                         f'euclidean_penalty_{euclidean_penalty}.png'),
            inpainted)
