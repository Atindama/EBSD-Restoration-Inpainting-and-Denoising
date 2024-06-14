"""
multi_test.py
Cody Mattice and Huston Wilhite, Summer 2022
Conor Miller-Lynch, Summer-Fall 2023

This script will run the Hybrid Criminisi/ML algorithm with various weights
and the ML algorithm alone on many image samples and save the errors for
these reconstructions to a CSV file.
"""

import argparse
import glob
import math
import os
import shutil
from os.path import join

import numpy as np
import torch
from tqdm import tqdm

import criminisi_mod
import MLexemplar
from dataset import EBSDDataset
from geodesic import geodesic_distances, geodesic_mse
from model_tanimutomo import PConvUNet # modified model architecture from Liu et al. paper

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



parser = argparse.ArgumentParser()
parser.add_argument('model')
parser.add_argument('patch_size', type=int)
parser.add_argument('device')
parser.add_argument('output_dir')
args = parser.parse_args()
torch.set_default_device(args.device)

# Prepare output directory.
OUTPUT_DIR = args.output_dir
if not os.path.isdir(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)
if not os.path.isdir(join(OUTPUT_DIR, 'raw_errors')):
    os.mkdir(join(OUTPUT_DIR, 'raw_errors'))
with open(join(OUTPUT_DIR, 'args.txt'), 'x') as f:
    f.write(str(args)[10:-1])
shutil.copy(args.model, join(OUTPUT_DIR, 'model.pt'))
os.mkdir(join(OUTPUT_DIR, 'code'))
for python_file in glob.glob('*.py'):
    shutil.copy(python_file, join(OUTPUT_DIR, 'code', python_file))

# Prepare the ML model.
model = PConvUNet(False)
state_dict = torch.load(args.model,
                        map_location=args.device)
model.load_state_dict(state_dict) # Loading our trained model
model.eval()

# Define the model ===================================================================
print("Loading the Model...")
modelT = PConvUNetT(finetune=False)

og_pconv_model_path = './pretrained_pconv.pth' #trained model from Liu et al. paper
modelT.load_state_dict(torch.load(
    og_pconv_model_path, map_location=args.device)['model'])
modelT.eval()

#=====================================================================================
# Set inpainting options.
RESTRICT_SEARCH = False

# The list of weights to use for the ML/Criminisi hybrid inpainting method.
# These are the weights for the known region. The ML-filled region is weighted
# as 1-known_weights.
known_weights = list(np.arange(0., 1.05, .05))


# Define inpainting methods.
method_strs = (['og_pconv', 'ML', 'Criminisi_standard', 'Criminisi_SSDelta', 'Criminisi_SSDelta_Euclidean_1']+ [f'ML_Criminisi_w{w:.2f}' for w in known_weights])

# Initialize geodesic_errors.csv.
with open(join(OUTPUT_DIR, 'MSD.csv'), 'w+') as f:
    f.write(','.join(f'{method}' for method in method_strs) + '\n')
with open(join(OUTPUT_DIR, 'MAD.csv'), 'w+') as f:
    f.write(','.join(f'{method}' for method in method_strs) + '\n')

dataset = EBSDDataset('../synthetic_EBSD_data/val', args.device, val=True)

for i, (noisy, mask, clean) in enumerate(tqdm(iter(dataset), total=len(dataset))):

    noisy_ndarray = MLexemplar.tensor_to_image(noisy)
    noisy_ndarray[:, :, 1] /= 2
    mask_ndarray = MLexemplar.tensor_to_image(mask)
    clean_ndarray = MLexemplar.tensor_to_image(clean)
    clean_ndarray[:, :, 1] /= 2
    damaged_ndarray = noisy_ndarray * mask_ndarray

    

    inpainted_images = []
    MSD_errors = []
    MAD_errors = []

    # Loading Input, Mask and Model prediction==============================================
    og_pconv = MLexemplar.single_predict(
        damaged_ndarray, mask_ndarray, modelT)
    og_pconv = np.clip(og_pconv, 0, 1)
    inpainted_images.append(og_pconv)
    #=====================================================================================
    
    ML_inpainted = MLexemplar.single_predict(
        damaged_ndarray, mask_ndarray, model)
    inpainted_images.append(ML_inpainted)
    
    criminisi_standard = MLexemplar.ML_criminisi(damaged_ndarray, mask_ndarray,
                                                 model, False,
                                                 distance_metric=criminisi_mod.SSE,
                                                 restrict_search=RESTRICT_SEARCH,
                                                 euclidean_penalty=0,
                                                 known_weight=1,
                                                 patch_size=args.patch_size,
                                                 mode='RGB')
    inpainted_images.append(criminisi_standard)

    criminisi_SSDelta = MLexemplar.ML_criminisi(damaged_ndarray, mask_ndarray, model,
                                        False,
                                        distance_metric=criminisi_mod.SSE,
                                        restrict_search=RESTRICT_SEARCH,
                                        euclidean_penalty=0, known_weight=1,
                                        patch_size=args.patch_size,
                                        mode='geodesic')
    inpainted_images.append(criminisi_SSDelta)

    criminisi_SSDelta_euclidean_1 = MLexemplar.ML_criminisi(damaged_ndarray,
                                                      mask_ndarray, model,
                                                      False,
                                                      distance_metric=criminisi_mod.SSE,
                                                      restrict_search=RESTRICT_SEARCH,
                                                      euclidean_penalty=1,
                                                      known_weight=1,
                                                      patch_size=args.patch_size,
                                                      mode='geodesic')
    inpainted_images.append(criminisi_SSDelta_euclidean_1)

    # Run hybrid inpainting with each weight in known_weights.
    for known_weight in known_weights:
        ML_crim = MLexemplar.ML_criminisi(damaged_ndarray, mask_ndarray, model,
                                          True,
                                          distance_metric=criminisi_mod.MSE,
                                          restrict_search=RESTRICT_SEARCH,
                                          euclidean_penalty=1,
                                          known_weight=known_weight,
                                          patch_size=args.patch_size,
                                          mode='geodesic')
        inpainted_images.append(ML_crim)


    for inpainted_image, method_str in zip(inpainted_images, method_strs):
        with torch.no_grad():
            MSD_error = geodesic_mse(
                MLexemplar.image_to_tensor(clean_ndarray),
                MLexemplar.image_to_tensor(inpainted_image),
                MLexemplar.image_to_tensor(mask_ndarray),
                scale=2*math.pi).item()
        MSD_errors.append(str(MSD_error))
        raw_errors = geodesic_distances(
            MLexemplar.image_to_tensor(clean_ndarray),
            MLexemplar.image_to_tensor(inpainted_image),
            MLexemplar.image_to_tensor(mask_ndarray),
            scale=2*math.pi).detach().cpu().numpy()
        np.save(join(OUTPUT_DIR, 'raw_errors', f'{i}_{method_str}.npy'),
                raw_errors)
        MAD_error = raw_errors.mean()
        MAD_errors.append(str(MAD_error))

    # Write the error data to geodesic_errors.csv.
    with open(join(OUTPUT_DIR, 'MSD.csv'), 'a') as f:
        f.write(','.join(MSD_errors) + '\n')
    with open(join(OUTPUT_DIR, 'MAD.csv'), 'a') as f:
        f.write(','.join(MAD_errors) + '\n')


# sbatch create_examples.sh ../synthetic_EBSD_data/val im_oneblock 20 ./epoch_may3.pt cuda:0 --n-examples 1 --patch-size 3 --missing-region-location 'center'
# sbatch multi_test.sh ./epoch_996.pt 3 cuda:0 ./errors_CriminisiML/
