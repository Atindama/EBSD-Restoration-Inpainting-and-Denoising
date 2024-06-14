#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 13:36:53 2023

@author: emmanuel
"""


from sys import path
path.insert(0, "..") # hack to get module `pyEBSD` in scope
import pyEBSD as ebsd
from numpy import deg2rad,pi
import matplotlib.pyplot as plt
import numpy as np

"""Read noisy ctf file and clean ctf file and restore for comparison
   If clean file is not available, use noisy file as both.
   In that case larger l2 errors indicate better restoration.
"""
# clean, noisy, preprocessed,filename = ebsd.orient.denoising_pipeline_ctf('/home/emmanuel/Desktop/EBSD_thesis_codes/old_EBSDctfFiles/Synthetic_test_noisy.ctf', '/home/emmanuel/Desktop/EBSD_thesis_codes/EBSDctfFiles/Synthetic_test.ctf', preprocess=True, denoise=True, denoise_type='tvflow', postprocess=False, l2error=True, plots=True)


# """
#    To use a different weight function for the edge map other than the one generated from 
#    our literature, you may compute and input the array as
#    Below, we use the result of our tv flow as weights for weighted tv flow denoising.
# """
e = deg2rad(ebsd.fileio.read_ctf("/home/emmanuel/Desktop/EBSD_thesis_codes/old_EBSDctfFiles/Synthetic_test.ctf"))

# u = ebsd.tvflow.denoise(e, weighted=False, beta=0.05, on_quats=False, weight_array=ebsd.misc.weight_from_TV_solution( noisy ))
plt.figure();
plt.imshow(e/2/pi); plt.show()
ebsd.ipf.plotIPF(e); plt.show()



x = deg2rad(ebsd.fileio.read_ctf('./jide/HDEDT8H Specimen 1 Site 1 Map Data 1 B.ctf'))
nan_map = np.isnan(x)
plt.figure(figsize=(16,9)); plt.imshow(nan_map[:,:,:]/1); plt.axis('off'); plt.savefig('./jide/nan_mask.png'); plt.show()

y = x.copy()
y[nan_map]=0; 

plt.figure(figsize=(16,9)); plt.imshow(y); plt.axis('off'); plt.savefig('./jide/euler_data.png'); plt.show()

ebsd.ipf.plotIPF(y, './jide/HDEDT8H Specimen 1 Site 1 Map Data 1 B.ctf')
