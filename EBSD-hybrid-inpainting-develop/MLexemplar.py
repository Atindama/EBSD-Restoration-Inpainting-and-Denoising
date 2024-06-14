import numpy as np
import torch

from criminisi_mod import MSE, SSE, get_boundary, inpaint
from model import Model


def image_to_tensor(image: np.ndarray) -> torch.Tensor:
    channels_first = image.copy().transpose((2, 0, 1))
    tensor = torch.tensor(channels_first).to(torch.float).unsqueeze(0)
    return tensor


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    channels_first = tensor.clone().detach().squeeze(0).cpu().numpy()
    image = channels_first.transpose((1, 2, 0))
    return image


def single_predict(image: np.ndarray, mask: np.ndarray, model: torch.nn.Module):
    '''
    Evaluates model for the given image/mask
    The image and mask should both be numpy arrays of form (X,Y,C)
    where C is the channel.  The mask should be 1 where the image is known
    and 0 where the image is unknown.
    '''

    # Scale channel 1.
    image = image.copy()
    image[:, :, 1] *= 2
    image[image > 1] = 1

    # Convert image and mask to channel-first torch.Tensor and add batch
    # dimension.
    image_tensor = image_to_tensor(image)
    mask_tensor = image_to_tensor(mask)

    # Apply the mask to the image.
    input_tensor = image_tensor * mask_tensor

    # Evaluate the model on the image and mask.
    predicted_tensor, _ = model(input_tensor, mask_tensor)

    # Convert the output back to an np.ndarray of shape (X,Y,C).
    predicted_image = tensor_to_image(predicted_tensor)

    # Create composite image, which is the original image where it is
    # known and the model output where it is unknown.
    composite_image = image*mask + predicted_image*np.abs(1-mask)

    # Scale channel 1.
    composite_image[:, :, 1] *= 0.5

    return composite_image


def ML_criminisi(image: np.ndarray, mask: np.ndarray, MLmodel,
                 using_ml_template: bool, patch_size: int = 3,
                 compare_increase: int = 0, restrict_search: bool = False,
                 euclidean_penalty: float = 0, distance_metric=SSE,
                 onion: bool = False, known_weight: float = .75,
                 mode: str = 'geodesic') -> np.ndarray:
    '''
    Uses Criminisi's algorithm to fill in the missing region supplemented by
    information from a machine learning model.

    ARGS:
        image             = np.ndarray in the form (X,Y,c) where c is the channel, and
                            in the RGB values are in the range [0,1]
        mask              = np.ndarray in the form of integers or Booleans; if ints,
                            1 where region is known, 0 where unknown; if bools, False
                            where known and True where unknown
        MLmodel           = a PyTorch that takes in two NumPy arrays (corresponding to
                            the image and a mask) of the same shape and produces two NumPy
                            arrays in the same shape as the inputs
        patch_size        = integer, the size of patch used throughout the process (optional,
                            defaults to 3)
        compare_increase  = integer, if you would like to use a larger patch during filling,
                            the increase in size is indicated here (optional, defaults to 0)
        distance_metric   = (np.ndarray, np.ndarray, np.ndarray)->float, the distance metric
                            to use when comparing patches (including machine learning-filled
                            region) (optional, defaults to SSE)
        restrict_search   = Boolean that, if True use a restricted search region centered at
                            the target patch (optional, defaults to False)
        euclidean_penalty = Boolean that, if True, indicates you would like to add the Euclidean
                            distance from a canidate patch to the fill region to the distance
                            metric
        onion             = Boolean indicated whether or not you wish to use onion layering
                            that than the Criminisi priority fill order (optional, defaults to
                            False)
        known_weight      = Float between 0 and 1 that determines the weight of the known region
                            when calculating the similarity between patches in Criminisi's algorithm.
                            Weight of unknown (ML-filled) region is 1-known_weight
    RETURNS:
        final inainted image as a NumPy Array in the form (X,Y,c), where c is the channel
    '''

    if using_ml_template == False and known_weight != 1:
        print('Error: using_ml_template is False but known_weight is'
              ' not 1.')
        quit()

    if mask.dtype == bool:
        bmask = mask[:, :, 0]
        imask = (~mask).astype(int)
    else:
        bmask = ~(mask.astype(bool))[:, :, 0]
        imask = mask

    if using_ml_template:
        working_im = single_predict(image, imask, MLmodel)
    else:
        working_im = image

    final_im = inpaint(working_im, bmask, using_ml_template,
                       patch_size=patch_size,
                       compare_increase=compare_increase,
                       restrict_search=restrict_search,
                       euclidean_penalty=euclidean_penalty,
                       distance_metric=distance_metric, onion=onion,
                       known_weight=known_weight, mode=mode)
    
    return final_im


def ML_Onion(image: np.ndarray, mask: np.ndarray, MLmodel,
             using_ml_template: bool, patch_size: int = 3,
             compare_increase: int = 0, restrict_search: bool = False,
             euclidean_penalty: bool = False, distance_metric=SSE,
             known_weight: float = 1, ML_bdry_width: int = 2) -> np.ndarray:
    '''
    Uses ML model to fill one layer of missing region, then Onion layering with Criminisi's algorithm
    to fill another layer, and repeats until the image is filled.

    ARGS:
        image             = np.ndarray in the form (X,Y,c) where c is the channel, and
                            in the RGB values are in the range [0,1]
        mask              = np.ndarray in the form of integers or Booleans; if ints,
                            1 where region is known, 0 where unknown; if bools, False
                            where known and True where unknown
        MLmodel           = a PyTorch that takes in two NumPy arrays (corresponding to
                            the image and a mask) of the same shape and produces two NumPy
                            arrays in the same shape as the inputs
        patch_size        = integer, the size of patch used throughout the process (optional,
                            defaults to 3)
        compare_increase  = integer, if you would like to use a larger patch during filling,
                            the increase in size is indicated here (optional, defaults to 0)
        distance_metric   = (np.ndarray, np.ndarray, np.ndarray)->float, the distance metric
                            to use when comparing patches (including machine learning-filled
                            region) (optional, defaults to SSE)
        restrict_search   = Boolean that, if True use a restricted search region centered at
                            the target patch (optional, defaults to False)
        euclidean_penalty = Boolean that, if True, indicates you would like to add the Euclidean
                            distance from a canidate patch to the fill region to the distance
                            metric
        onion             = Boolean indicated whether or not you wish to use onion layering
                            that than the Criminisi priority fill order (optional, defaults to
                            False)
        known_weight      = Float between 0 and 1 that determines the weight of the known region
                            when calculating the similarity between patches in Criminisi's algorithm.
                            Weight of unknown (ML-filled) region is 1-known_weight
        ML_bdry_width     = int, determines how many pixels thick the boundary which the ML model
                            fills is.
    RETURNS:
        final inainted image as a NumPy Array in the form (X,Y,c), where c is the channel
    '''

    if mask.dtype == bool:
        bmask = mask[:, :, 0]
        imask = (~mask).astype(int)
    else:
        bmask = ~(mask.astype(bool))[:, :, 0]
        imask = mask

    working_mask = bmask.copy()
    working_imask = imask.copy()

    while np.any(working_mask):
        bdry = get_boundary(working_mask, known_bdry=False,
                            iterations=ML_bdry_width, patch_size=patch_size)

        image = image + np.expand_dims(bdry, -1) * \
            single_predict(image, working_imask, MLmodel)

        working_mask[bdry] = False
        working_imask[bdry] = 1.

        if not np.any(working_mask):
            return image

        image, unfilled = inpaint(image, working_mask, using_ml_template,
                                  patch_size=patch_size,
                                  compare_increase=compare_increase,
                                  restrict_search=restrict_search,
                                  euclidean_penalty=euclidean_penalty,
                                  distance_metric=distance_metric,
                                  onion=True, known_weight=known_weight)

        # bdry = get_boundary(working_mask)
        working_mask = unfilled
        working_imask[~unfilled] = 1.

        # print('Loop Ran')

    return image


if __name__ == '__main__':
    testim = np.load('../../Dream3d_data_generator/val128x128/3.npy')
    known_mask = np.ones_like(testim)
    known_mask[25:35, 18:28, :] = 0

    model = Model()
    state_dict = torch.load(
        '../../ML/model/cody_models/model22best', map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()

    MLCrim_im, working_im = ML_criminisi(
        testim*known_mask, known_mask, model, True, euclidean_penalty=True)
    ML_Onion_im = ML_Onion(testim*known_mask, known_mask, model, True,
                           euclidean_penalty=False, distance_metric=MSE, ML_bdry_width=2)

    crim_im = inpaint(testim*known_mask, ~(known_mask.astype(bool))[:, :, 0], False)

    print('ML Error: {}'.format(np.sum((testim-working_im)**2)))
    print('Crim Error: {}'.format(np.sum((testim-crim_im)**2)))
    print('ML Crim Error: {}'.format(np.sum((testim-MLCrim_im)**2)))
    print('ML Onion Error: {}'.format(np.sum((testim-ML_Onion_im)**2)))

    import matplotlib.pyplot as plt
    plt.imsave('./orig.png', testim)
    plt.imsave('./crim.png', crim_im)
    plt.imsave('./mlcrim.png', MLCrim_im)
    plt.imsave('./ml.png', working_im)
    plt.imsave('./mlonion.png', ML_Onion_im)
    plt.imsave('dam.png', testim*known_mask)
