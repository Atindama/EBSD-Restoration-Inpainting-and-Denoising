# Conor's model imports
import argparse
import os
from pathlib import Path

import numpy as np
import skimage.io
import torch
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation

from criminisi_mod import inpaint
from dataset import load_ctf_to_tensor, normalize
from model_tanimutomo import PConvUNet # modified model architecture from Liu et al. paper
from train import tensor_to_image

# Tanimoutomo model imports ==================================
import argparse
from distutils.util import strtobool
import os

from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from modelT import PConvUNet as PConvUNetT # model architecture from Liu et al. paper
#=============================================================


"""
pretrained_pconv.pth is the model from a partial convulutions(the NVIDIA paper: filling irregular holes) implementation 
by Tanimutomo. The model is loaded using it's respective files in the src folder
"""
def to_image(tensor):
    working_image = tensor_to_image(tensor)
    working_image *= np.array([2 * np.pi, np.pi, 2 * np.pi])
    working_image = (Rotation.from_euler(EULER_CONVENTION, working_image.reshape(-1, 3))
                     .as_euler(EULER_CONVENTION).reshape(working_image.shape) % (2 * np.pi))
    final_image = working_image / (2 * np.pi)
    return final_image


def to_uint8_image(tensor):
    working_image = tensor_to_image(tensor)
    working_image *= np.array([2 * np.pi, np.pi, 2 * np.pi])
    working_image = (Rotation.from_euler(EULER_CONVENTION, working_image.reshape(-1, 3))
                     .as_euler(EULER_CONVENTION).reshape(working_image.shape) % (2 * np.pi))
    working_image = working_image / \
        np.array([2 * np.pi, np.pi, 2 * np.pi]) * 255
    final_image = working_image.astype(np.uint8)
    return final_image


from scipy.ndimage import convolve

def damage_image(im_clean, **kwargs):
    """Creates masked images for a given N dimensional array of images
       Parameters
       ----------
        im_to_damage : np.ndarray
            is an N dimensional numpy array of the image(s) you wish to mask. The dimensions must be the same as the im_clean
            You may use np.ones_like(im_clean) if you want a mask to be returned, otherwise, put im_clean if you want the same image to be damaged
        im_to_damage : np.ndarray
            is an N dimensional numpy array of the image(s) you wish to use as reference for damaging. The dimensions must be the same as im_to_damage
        *kwargs
            EdgeProportion: specifies the proportion of edges to be damaged. Default = 1 (100%)
            InteriorProportion: Specifies the proportion of the interior regions to be damaged. Default = 0.1 (10%)
            EdgeThickness: specifies number of pixels including the edge to be damaged. Must be odd. Default = 1
            Width: specifies width of pixels including the edge to be damaged. Must be odd. Default = 1
            eps: threshold for selecting what is considered an edge. Default is 1e-4
       Returns
       -------
       a masked image
       Note: if input image is noisy, detecting edges may be extremely difficult. Hence you may have to tune the eps parameter
    """
    # Parse options
    edge_proportion = kwargs.get('EdgeProportion', 1)
    interior_proportion = kwargs.get('InteriorProportion', 0.1)
    edge_thickness = kwargs.get('EdgeThickness', 1)
    width = kwargs.get('Width', 1)
    eps = kwargs.get('eps', 1e-4)
    
    # Check if the input images have the same size and dimensionality
    M, N, nchans = im_clean.shape
    im_to_damage = np.ones((M,N,nchans))
    # Get edges
    edges = np.abs(convolve(im_clean[:, :, 0], np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]))) > eps
    edges = add_border(edges, edge_thickness // 2)
    # Damage image
    randvals = np.random.rand(M, N)
    # Select edge pixels
    nanmask_edges = (randvals <= edge_proportion) & edges
    nanmask_edges = add_border(nanmask_edges, width // 2)
    # Select interior pixels
    nanmask_interior = (randvals <= interior_proportion) & ~edges
    # Get final mask
    nanmask = nanmask_edges | nanmask_interior
    nanmask = np.repeat(nanmask[:, :, np.newaxis], nchans, axis=2)
    # Destroy pixels
    im_to_damage[nanmask] = 0
    return im_to_damage

def add_border(im, border_thickness):
    for _ in range(border_thickness):
        im = np.logical_or(im, np.roll(im, 1, axis=0))
        im = np.logical_or(im, np.roll(im, -1, axis=0))
        im = np.logical_or(im, np.roll(im, 1, axis=1))
        im = np.logical_or(im, np.roll(im, -1, axis=1))
    return im


EULER_CONVENTION = 'ZXZ'


if __name__ == '__main__':
    # Parse arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('source_dir', type=str)
    parser.add_argument('output_dir', type=str)
    parser.add_argument('damage_side_length', type=int)
    parser.add_argument('pretrained_model', type=str)
    parser.add_argument('device', type=str)
    parser.add_argument('--n-examples', '-n', type=int, default=1)
    parser.add_argument('--patch-size', '-p', type=int, default=3)
    parser.add_argument('--missing-region-location', '-l', type=str, default='center')
    args = parser.parse_args()

    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)

    model = PConvUNet(False)
    model.load_state_dict(torch.load(
        args.pretrained_model, map_location=args.device)) # Loading our trained model
    model.to(args.device)
    model.eval()

    source_ctfs = Path(args.source_dir).glob('*noisy.ctf')
    source_ctfs = list(map(os.path.abspath, source_ctfs))[:args.n_examples]
    
    # clean_ctfs = ['/mnt/home/atindaea/synthetic_EBSD_data/val/8370_clean.ctf']
    # clean_ctfs = ['../synthetic_EBSD_data/real/3_clean.ctf']
    source_ctfs = ['../synthetic_EBSD_data/real/3_clean.ctf']

    # for i, clean_ctf in enumerate(clean_ctfs):
    #     original_clean = normalize(load_ctf_to_tensor(clean_ctf)).to(args.device)
    #     image = original_clean
    #     if isinstance(image, torch.Tensor):
    #         image = to_uint8_image(image)
    #     else:
    #         image = (image * 255).astype(np.uint8)
    #     cropped_image = image[:128, 128:, :]
        
    #     single_image_name = 'clean.png'
    #     single_image_path = os.path.join(args.output_dir, single_image_name)
    #     np.save(single_image_path[:-4], cropped_image/255)
    #     skimage.io.imsave(single_image_path, cropped_image)
    

     # Define the model ===================================================================
    print("Loading the Model...")
    modelT = PConvUNetT(finetune=False)
    
    og_pconv_model_path = './pretrained_pconv.pth'
    # modelT = PConvUNet(False)
    modelT.load_state_dict(torch.load(
        og_pconv_model_path, map_location=args.device)['model']) #trained model from Liu et al. paper
    modelT.to(args.device)
    modelT.eval()
    #========================================================================================

    print("Model loaded...")
    
    
    for i, source_ctf in enumerate(source_ctfs):
        images = []
        original = normalize(load_ctf_to_tensor(source_ctf)).to(args.device)
        print("the shape of original is...", original.dtype)

        # # 1. one missing block
        # mask = torch.ones_like(original)
        # if args.missing_region_location == 'center':
        #     mask[:, 64-args.damage_side_length//2:64+args.damage_side_length//2,
        #         192-args.damage_side_length//2:192+args.damage_side_length//2] = 0
        # elif args.missing_region_location == 'top':
        #     mask[:, 0:args.damage_side_length,
        #          192-args.damage_side_length//2:192+args.damage_side_length//2] = 0
        
        # 1. Mask from real data
        load_mask = torch.from_numpy(np.load('mask.npy')); load_mask = load_mask.permute(2,0,1) # additional mask made up of isolated pixels in the real data
        mask = torch.ones_like(original)
        mask[original<0.001] = 0
        mask[load_mask>0.99] = 0

        # # 2. three missing blocks
        # size = 20
        # mask = torch.ones_like(original)
        # # Create 3 missing blocks
        # mask[:, 64-size//2:64+size//2, 192-size//2:192+size//2] = 0 # center missing
        # mask[:, 0+5:size, 177-size//2:172+size//2] = 0 # top left missing
        # mask[:, 0+15:size+10, 217-size//2:212+size//2] = 0 # top rightt missing
        
        # # 3. zigzagged lines of missing data
        # size = 256
        # mask = torch.ones_like(original)
        # # Create a zigzagged diagonal line
        # mask[:, 70, :] = 0 # Set the entire row to zero
        # mask[:, :, 180] = 0 # Set the entire column to zero
        # for i in range(size): # Set the diagonal elements to zero with a zigzag pattern
        #     if i % 2 == 0: mask[:, i+4:i+6, i-2:i+2] = 0 # main diagonal
        #     else: mask[:, i+4:i+6, size-i-2:size-i+2] = 0 # other diagonal

        # # 4. missing region near grain boundaries
        # mask = damage_image(original.permute(1,2,0).cpu().numpy(), InteriorProportion = 0, EdgeProportion=.5, EdgeThickness=1, Width=1, eps=0.2)
        # mask = torch.from_numpy(mask).clone().permute(2,0,1)
        # mask = mask.to(args.device).float()
        # torch.set_default_tensor_type(torch.FloatTensor)
        # print("mask generated, and the dtype of mask is...", mask.dtype, mask.shape, original.shape)
    
        # # 5. Mask of randomly distributed values
        # zero_fraction = 0.2              # Proportion of zeros in the array
        # mask_3d = np.random.choice([0, 1], size=(256, 256), p=[zero_fraction, 1-zero_fraction])
        # mask = np.ones((256,256,3)); mask[:,:,0]=mask_3d; mask[:,:,1]=mask_3d; mask[:,:,2]=mask_3d
        # mask = torch.from_numpy(mask).clone().permute(2,0,1)
        # mask = mask.to(args.device).float()
        # print("mask generated, and the dtype of mask is...", mask.dtype, mask.shape, original.shape)
    

    
        damaged = original.clone()
        damaged[mask == 0] = 0.99
        # Loading Input, Mask and Model prediction==============================================
        raw_outT, _ = modelT(damaged.unsqueeze(0), mask.unsqueeze(0))
        raw_outT = raw_outT.squeeze(0)

        # Post process
        raw_outT = raw_outT.clamp(0.0, 1.0)
        outT = mask * damaged + (1 - mask) * raw_outT

        og_ml = original.clone()
        og_ml[mask == 0] = outT[mask == 0]
        #=====================================================================================

        predicted, _ = model(damaged.unsqueeze(0), mask.unsqueeze(0))
        predicted = predicted.squeeze(0)
        ml = original.clone()
        ml[mask == 0] = predicted[mask == 0]
        
        criminisi_standard = inpaint(to_image(damaged),
                                     tensor_to_image(1 - mask).astype(bool), False,
                                     euclidean_penalty=0,
                                     patch_size=args.patch_size, mode='RGB')
        criminisi_standard[:, :, 1] *= 2
        criminisi_SSDelta = inpaint(to_image(damaged),
                                      tensor_to_image(1 - mask).astype(bool), False,
                                      euclidean_penalty=0,
                                      patch_size=args.patch_size)
        criminisi_SSDelta[:, :, 1] *= 2
        criminisi_SSDelta_euclidean_1 = inpaint(to_image(damaged),
                                            tensor_to_image(1 - mask).astype(bool), False,
                                            euclidean_penalty=1,
                                            patch_size=args.patch_size)
        criminisi_SSDelta_euclidean_1[:, :, 1] *= 2
        # criminisi_SSDelta_euclidean_1_WxH = inpaint(to_image(damaged),
        #                                           tensor_to_image(1 - mask).astype(bool), False,
        #                                           euclidean_penalty=1/65536,
        #                                           patch_size=args.patch_size)
        # criminisi_SSDelta_euclidean_1_WxH[:, :, 1] *= 2
        hybrid_00 = inpaint(to_image(ml),
                            tensor_to_image(1 - mask).astype(bool), True,
                            euclidean_penalty=1, known_weight=0.05,
                            patch_size=args.patch_size)
        hybrid_00[:, :, 1] *= 2
        # hybrid_01 = inpaint(to_image(ml),
        #                     tensor_to_image(1 - mask).astype(bool), True,
        #                     euclidean_penalty=1, known_weight=0.1,
        #                     patch_size=args.patch_size)
        # hybrid_01[:, :, 1] *= 2
        # hybrid_05 = inpaint(to_image(ml),
        #                     tensor_to_image(1 - mask).astype(bool), True,
        #                     euclidean_penalty=1, known_weight=0.5,
        #                     patch_size=args.patch_size)
        # hybrid_05[:, :, 1] *= 2
        # hybrid_09 = inpaint(to_image(ml),
        #                     tensor_to_image(1 - mask).astype(bool), True,
        #                     euclidean_penalty=1, known_weight=0.9,
        #                     patch_size=args.patch_size)
        # hybrid_09[:, :, 1] *= 2
        # hybrid_10 = inpaint(to_image(ml),
        #                     tensor_to_image(1 - mask).astype(bool), True,
        #                     euclidean_penalty=1, known_weight=1,
        #                     patch_size=args.patch_size)
        # hybrid_10[:, :, 1] *= 2
        images = [original, damaged, og_ml, ml, hybrid_00, criminisi_standard,
                  criminisi_SSDelta, criminisi_SSDelta_euclidean_1]
        
        names = ['Original', 'Damaged', 'og_pconv', 'ml','hybrid_00','Criminisi (Standard)',
                 'Criminisi (SSDelta)', 'Criminisi (SSDelta, Euclidean (1))']
        

        for image, name in zip(images, names):
            if isinstance(image, torch.Tensor):
                image = to_uint8_image(image)
            else:
                image = (image * 255).astype(np.uint8)
            cropped_image = image[:128, 128:, :]
            
            single_image_name = f'{i}_' + name.replace('=', '').replace('/', '_') + '.png'
            single_image_path = os.path.join(args.output_dir, single_image_name)
            np.save(single_image_path[:-4], image/255) #cropped_image/255)
            skimage.io.imsave(single_image_path, image) #cropped_image)
