#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 10:02:26 2024

@author: emmanuel
"""
from sys import path
path.insert(0, "..") # hack to get module `pyEBSD` in scope
import pyEBSD as ebsd
from numpy import deg2rad
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf



damaged = np.load('/home/emmanuel/Desktop/images_for_IMMI_paper/pconv_paper_996_epochs/im_real_mask/0_Damaged.npy')*np.pi
damaged1 = ebsd.ipf.saveIPF(damaged, original_ctf=None); x = plt.imread('/home/emmanuel/Desktop/images_for_IMMI_paper/pconv_paper_996_epochs/im_real_mask/0_Damaged.png')
for i in range(damaged.shape[0]):
    for j in range(damaged.shape[1]):
        if x[i,j,0]>0.98 and x[i,j,1]>0.98 and x[i,j,2]>0.98:
            damaged1[i,j,:]=1
plt.imsave('/home/emmanuel/Desktop/images_for_IMMI_paper/pconv_paper_996_epochs/im_real_mask/Real_damaged.png', damaged1[56:,:200,:])



# crim_stand = np.load('/home/emmanuel/Desktop/images_for_IMMI_paper/pconv_paper_996_epochs/im_zigzag/255_Criminisi (Standard).npy')*np.pi; e = crim_stand[:,:,:]

# e = ebsd.orient.clean_discontinuities(e)
# e = ebsd.orient.fill_isolated_with_median(e)
# u = ebsd.tvflow.denoise(e, weighted=True, beta=0.0005, on_quats=False)
# plt.figure();
# plt.imshow(u/np.pi);
# plt.show()
# plt.figure()
# plt.imshow(e/np.pi)
# plt.show()
# denoised = ebsd.ipf.saveIPF(u); plt.imsave('/home/emmanuel/Desktop/images_for_IMMI_paper/pconv_paper_996_epochs/im_zigzag/denoised_ipf_Criminisi_standard.png', denoised)


# crim_ssd = np.load('/home/emmanuel/Desktop/images_for_IMMI_paper/pconv_paper_996_epochs/im_zigzag/255_Criminisi (SSDelta).npy')*np.pi; e = crim_ssd[:,:,:]

# e = ebsd.orient.clean_discontinuities(e)
# e = ebsd.orient.fill_isolated_with_median(e)
# u = ebsd.tvflow.denoise(e, weighted=True, beta=0.0005, on_quats=False)
# plt.figure();
# plt.imshow(u/np.pi);
# plt.show()
# plt.figure()
# plt.imshow(e/np.pi)
# plt.show()
# denoised = ebsd.ipf.saveIPF(u); plt.imsave('/home/emmanuel/Desktop/images_for_IMMI_paper/pconv_paper_996_epochs/im_zigzag/denoised_ipf_Criminisi_SSDelta.png', denoised)


# crim_ssd_eu = np.load('/home/emmanuel/Desktop/images_for_IMMI_paper/pconv_paper_996_epochs/im_zigzag/255_Criminisi (Standard).npy')*np.pi; e = crim_ssd_eu[:,:,:]

# e = ebsd.orient.clean_discontinuities(e)
# e = ebsd.orient.fill_isolated_with_median(e)
# u = ebsd.tvflow.denoise(e, weighted=True, beta=0.0005, on_quats=False)
# plt.figure();
# plt.imshow(u/np.pi);
# plt.show()
# plt.figure()
# plt.imshow(e/np.pi)
# plt.show()
# denoised = ebsd.ipf.saveIPF(u); plt.imsave('/home/emmanuel/Desktop/images_for_IMMI_paper/pconv_paper_996_epochs/im_zigzag/denoised_ipf_Criminisi_SSDelta_EU.png', denoised)




hybrid = np.load('/home/emmanuel/Desktop/images_for_IMMI_paper/pconv_paper_996_epochs/im_real_mask/0_hybrid_00.npy')*np.pi; e = hybrid[:,:,:]
hybrid_img = ebsd.ipf.saveIPF(hybrid, original_ctf=None);
plt.imsave('/home/emmanuel/Desktop/images_for_IMMI_paper/pconv_paper_996_epochs/im_real_mask/Real_inpainted_hybrid.png', hybrid_img[56:,:200,:])


e = ebsd.orient.clean_discontinuities(e)
e = ebsd.orient.fill_isolated_with_median(e)
plt.imsave('/home/emmanuel/Desktop/images_for_IMMI_paper/pconv_paper_996_epochs/im_real_mask/hybrid_inpainting_isolated_pixels.png', ebsd.ipf.saveIPF(e)[56:,:200,:])


u = ebsd.tvflow.denoise(e, weighted=True, beta=0.00001, on_quats=False)
denoised = ebsd.ipf.saveIPF(u, original_ctf=None); plt.imsave('/home/emmanuel/Desktop/images_for_IMMI_paper/pconv_paper_996_epochs/im_real_mask/Real_denoised_ipf_hybrid.png', denoised[56:,:200,:])


############################################################################################################################################################################


# errors = []
# clean = np.load('/home/emmanuel/Desktop/images_for_IMMI_paper/pconv_paper_996_epochs/im_oneblock/clean.npy')*np.pi;

# noisy = np.load('/home/emmanuel/Desktop/images_for_IMMI_paper/pconv_paper_996_epochs/im_oneblock/0_Original.npy')*np.pi; e = noisy[:,:,:]
# errors.append(ebsd.orientation.mean_misorientation_ang(clean, noisy, symmetry_op = None))
# e = ebsd.orient.clean_discontinuities(e)
# e = ebsd.orient.fill_isolated_with_median(e)
# u = ebsd.tvflow.denoise(e, weighted=True, beta=0.0005, on_quats=False)
# errors.append(ebsd.orientation.mean_misorientation_ang(clean, u, symmetry_op = None))
# plt.figure(); plt.imshow(u/np.pi); plt.show()

# hybrid = np.load('/home/emmanuel/Desktop/images_for_IMMI_paper/pconv_paper_996_epochs/im_oneblock/0_hybrid_00.npy')*np.pi; e = hybrid[:,:,:]
# errors.append(ebsd.orientation.mean_misorientation_ang(clean, hybrid, symmetry_op = None))
# e = ebsd.orient.clean_discontinuities(e)
# e = ebsd.orient.fill_isolated_with_median(e)
# u = ebsd.tvflow.denoise(e, weighted=True, beta=0.0005, on_quats=False)
# errors.append(ebsd.orientation.mean_misorientation_ang(clean, u, symmetry_op = None))
# plt.figure(); plt.imshow(u/np.pi); plt.show()

# hybrid = np.load('/home/emmanuel/Desktop/images_for_IMMI_paper/pconv_paper_996_epochs/im_threeblock/0_hybrid_00.npy')*np.pi; e = hybrid[:,:,:]
# errors.append(ebsd.orientation.mean_misorientation_ang(clean, hybrid, symmetry_op = None))
# e = ebsd.orient.clean_discontinuities(e)
# e = ebsd.orient.fill_isolated_with_median(e)
# u = ebsd.tvflow.denoise(e, weighted=True, beta=0.0005, on_quats=False)
# errors.append(ebsd.orientation.mean_misorientation_ang(clean, u, symmetry_op = None))
# plt.figure(); plt.imshow(u/np.pi); plt.show()

# hybrid = np.load('/home/emmanuel/Desktop/images_for_IMMI_paper/pconv_paper_996_epochs/im_boundary/0_hybrid_00.npy')*np.pi; e = hybrid[:,:,:]
# errors.append(ebsd.orientation.mean_misorientation_ang(clean, hybrid, symmetry_op = None))
# e = ebsd.orient.clean_discontinuities(e)
# e = ebsd.orient.fill_isolated_with_median(e)
# u = ebsd.tvflow.denoise(e, weighted=True, beta=0.0005, on_quats=False)
# errors.append(ebsd.orientation.mean_misorientation_ang(clean, u, symmetry_op = None))
# plt.figure(); plt.imshow(u/np.pi); plt.show()

# hybrid = np.load('/home/emmanuel/Desktop/images_for_IMMI_paper/pconv_paper_996_epochs/im_random/0_hybrid_00.npy')*np.pi; e = hybrid[:,:,:]
# errors.append(ebsd.orientation.mean_misorientation_ang(clean, hybrid, symmetry_op = None))
# e = ebsd.orient.clean_discontinuities(e)
# e = ebsd.orient.fill_isolated_with_median(e)
# u = ebsd.tvflow.denoise(e, weighted=True, beta=0.0005, on_quats=False)
# errors.append(ebsd.orientation.mean_misorientation_ang(clean, u, symmetry_op = None))
# plt.figure(); plt.imshow(u/np.pi); plt.show()

# hybrid = np.load('/home/emmanuel/Desktop/images_for_IMMI_paper/pconv_paper_996_epochs/im_zigzag/255_hybrid_00.npy')*np.pi; e = hybrid[:,:,:]
# errors.append(ebsd.orientation.mean_misorientation_ang(clean, hybrid, symmetry_op = None))
# e = ebsd.orient.clean_discontinuities(e)
# e = ebsd.orient.fill_isolated_with_median(e)
# u = ebsd.tvflow.denoise(e, weighted=True, beta=0.0005, on_quats=False)
# errors.append(ebsd.orientation.mean_misorientation_ang(clean, u, symmetry_op = None))
# plt.figure(); plt.imshow(u/np.pi); plt.show()




# crim_stand = np.load('/home/emmanuel/Desktop/images_for_IMMI_paper/pconv_paper_996_epochs/im_oneblock/0_Criminisi (Standard).npy')*np.pi; e = crim_stand[:,:,:]
# e = ebsd.orient.clean_discontinuities(e)
# e = ebsd.orient.fill_isolated_with_median(e)
# u = ebsd.tvflow.denoise(e, weighted=True, beta=0.0005, on_quats=False)
# errors.append(ebsd.orientation.mean_misorientation_ang(clean, u, symmetry_op = None))
# plt.figure(); plt.imshow(u/np.pi); plt.show()

# crim_ssd = np.load('/home/emmanuel/Desktop/images_for_IMMI_paper/pconv_paper_996_epochs/im_oneblock/0_Criminisi (SSDelta).npy')*np.pi; e = crim_ssd[:,:,:]
# e = ebsd.orient.clean_discontinuities(e)
# e = ebsd.orient.fill_isolated_with_median(e)
# u = ebsd.tvflow.denoise(e, weighted=True, beta=0.0005, on_quats=False)
# errors.append(ebsd.orientation.mean_misorientation_ang(clean, u, symmetry_op = None))
# plt.figure(); plt.imshow(u/np.pi); plt.show()

# crim_ssd_eu = np.load('/home/emmanuel/Desktop/images_for_IMMI_paper/pconv_paper_996_epochs/im_oneblock/0_Criminisi (SSDelta, Euclidean (1)).npy')*np.pi; e = crim_ssd_eu[:,:,:]
# e = ebsd.orient.clean_discontinuities(e)
# e = ebsd.orient.fill_isolated_with_median(e)
# u = ebsd.tvflow.denoise(e, weighted=True, beta=0.0005, on_quats=False)
# errors.append(ebsd.orientation.mean_misorientation_ang(clean, u, symmetry_op = None))
# plt.figure(); plt.imshow(u/np.pi); plt.show()

# og_pconv = np.load('/home/emmanuel/Desktop/images_for_IMMI_paper/pconv_paper_996_epochs/im_oneblock/0_og_pconv.npy')*np.pi; e = og_pconv[:,:,:]
# e = ebsd.orient.clean_discontinuities(e)
# e = ebsd.orient.fill_isolated_with_median(e)
# u = ebsd.tvflow.denoise(e, weighted=True, beta=0.0005, on_quats=False)
# errors.append(ebsd.orientation.mean_misorientation_ang(clean, u, symmetry_op = None))
# plt.figure(); plt.imshow(u/np.pi); plt.show()

# ml = np.load('/home/emmanuel/Desktop/images_for_IMMI_paper/pconv_paper_996_epochs/im_oneblock/0_ml.npy')*np.pi; e = ml[:,:,:]
# e = ebsd.orient.clean_discontinuities(e)
# e = ebsd.orient.fill_isolated_with_median(e)
# u = ebsd.tvflow.denoise(e, weighted=True, beta=0.0005, on_quats=False)
# errors.append(ebsd.orientation.mean_misorientation_ang(clean, u, symmetry_op = None))
# plt.figure(); plt.imshow(u/np.pi); plt.show()

# hybrid = np.load('/home/emmanuel/Desktop/images_for_IMMI_paper/pconv_paper_996_epochs/im_oneblock/0_hybrid_00.npy')*np.pi; e = hybrid[:,:,:]
# e = ebsd.orient.clean_discontinuities(e)
# e = ebsd.orient.fill_isolated_with_median(e)
# u = ebsd.tvflow.denoise(e, weighted=True, beta=0.0005, on_quats=False)
# errors.append(ebsd.orientation.mean_misorientation_ang(clean, u, symmetry_op = None))
# plt.figure(); plt.imshow(u/np.pi); plt.show()









































# original = np.load('/home/emmanuel/Desktop/images_for_IMMI_paper/pconv_paper_776_epochs/im_zigzag/255_Original.npy')*np.pi; e = original[:,:,:]
# original = ebsd.ipf.saveIPF(original); plt.imsave('/home/emmanuel/Desktop/images_for_IMMI_paper/pconv_paper_776_epochs/im_zigzag/ipf_original.png', original)

# e = ebsd.orient.clean_discontinuities(e)
# e = ebsd.orient.fill_isolated_with_median(e)
# u = ebsd.tvflow.denoise(e, weighted=True, beta=0.0005, on_quats=False)
# plt.figure();
# plt.imshow(u/np.pi);
# plt.show()
# plt.figure()
# plt.imshow(e/np.pi)
# plt.show()
# denoised = ebsd.ipf.saveIPF(u); plt.imsave('/home/emmanuel/Desktop/images_for_IMMI_paper/pconv_paper_776_epochs/im_zigzag/ipf_denoised.png', denoised)

# clean = np.load('/home/emmanuel/Desktop/images_for_IMMI_paper/pconv_paper_776_epochs/im_zigzag/clean.npy')*np.pi;
# clean = ebsd.ipf.saveIPF(clean); plt.imsave('/home/emmanuel/Desktop/images_for_IMMI_paper/pconv_paper_776_epochs/im_zigzag/ipf_clean.png', clean)



# og_pconv = np.load('/home/emmanuel/Desktop/images_for_IMMI_paper/pconv_paper_776_epochs/im_zigzag/255_og_pconv.npy')*np.pi
# og_pconv = ebsd.ipf.saveIPF(og_pconv); plt.imsave('/home/emmanuel/Desktop/images_for_IMMI_paper/pconv_paper_776_epochs/im_zigzag/ipf_og_pconv.png', og_pconv)

# ml = np.load('/home/emmanuel/Desktop/images_for_IMMI_paper/pconv_paper_776_epochs/im_zigzag/255_ml.npy')*np.pi
# ml = ebsd.ipf.saveIPF(ml); plt.imsave('/home/emmanuel/Desktop/images_for_IMMI_paper/pconv_paper_776_epochs/im_zigzag/ipf_ml.png', ml)

# hybrid = np.load('/home/emmanuel/Desktop/images_for_IMMI_paper/pconv_paper_776_epochs/im_zigzag/255_hybrid_00.npy')*np.pi
# hybrid = ebsd.ipf.saveIPF(hybrid); plt.imsave('/home/emmanuel/Desktop/images_for_IMMI_paper/pconv_paper_776_epochs/im_zigzag/ipf_hybrid.png', hybrid)

# damaged = np.load('/home/emmanuel/Desktop/images_for_IMMI_paper/pconv_paper_776_epochs/im_zigzag/255_Damaged.npy')*np.pi
# damaged1 = ebsd.ipf.saveIPF(damaged); x = plt.imread('/home/emmanuel/Desktop/images_for_IMMI_paper/pconv_paper_776_epochs/im_zigzag/255_Damaged.png')
# for i in range(damaged.shape[0]):
#     for j in range(damaged.shape[1]):
#         if x[i,j,0]>0.98 and x[i,j,1]>0.98 and x[i,j,2]>0.98:
#             damaged1[i,j,:]=1
# plt.imsave('/home/emmanuel/Desktop/images_for_IMMI_paper/pconv_paper_776_epochs/im_zigzag/ipf_damaged.png', damaged1)

# criminisi_ssdelta = np.load('/home/emmanuel/Desktop/images_for_IMMI_paper/pconv_paper_776_epochs/im_zigzag/255_Criminisi (SSDelta).npy')*np.pi
# criminisi_ssdelta = ebsd.ipf.saveIPF(criminisi_ssdelta); plt.imsave('/home/emmanuel/Desktop/images_for_IMMI_paper/pconv_paper_776_epochs/im_zigzag/ipf_criminisi_ssdelta.png', criminisi_ssdelta)

# criminisi_standard = np.load('/home/emmanuel/Desktop/images_for_IMMI_paper/pconv_paper_776_epochs/im_zigzag/255_Criminisi (Standard).npy')*np.pi
# criminisi_standard = ebsd.ipf.saveIPF(criminisi_standard); plt.imsave('/home/emmanuel/Desktop/images_for_IMMI_paper/pconv_paper_776_epochs/im_zigzag/ipf_criminisi_standard.png', criminisi_standard)

# criminisi_ssdelta_eu = np.load('/home/emmanuel/Desktop/images_for_IMMI_paper/pconv_paper_776_epochs/im_zigzag/255_Criminisi (SSDelta, Euclidean (1)).npy')*np.pi
# criminisi_ssdelta_eu = ebsd.ipf.saveIPF(criminisi_ssdelta_eu); plt.imsave('/home/emmanuel/Desktop/images_for_IMMI_paper/pconv_paper_776_epochs/im_zigzag/ipf_criminisi_ssdelta_eu.png', criminisi_ssdelta_eu)
