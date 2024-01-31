# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 17:39:24 2024

@author: David
"""

from astropy.io import fits
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.optimize import curve_fit
from scipy.interpolate import CubicSpline
from scipy.interpolate import lagrange
from scipy import ndimage
from scipy.optimize import minimize
from scipy.ndimage import median_filter
import random

hdulist = fits.open("Fits_Data\mosaic.fits")
data = hdulist[0].data
data_trans = np.transpose(data)
data_minus_background = data

binary_check = np.zeros((4611,2570))
for i in range(4611):
    for j in range(2570):
        if data[i][j] > 3450:
            binary_check[i][j] = 1            

s = [[0,1,0],
      [1,1,1],
      [0,1,0]]

labeled_data, num_features = ndimage.label(binary_check,s)
print(num_features)
labeled_areas = np.array(ndimage.sum(binary_check, labeled_data, np.arange(labeled_data.max()+1)))
mask1 = labeled_areas > 50
remove_small_area = mask1[labeled_data.ravel()].reshape(labeled_data.shape)

big_labeled_data, num_features = ndimage.label(remove_small_area,s)
print(num_features)
labeled_areas = np.array(ndimage.sum(binary_check, big_labeled_data, np.arange(big_labeled_data.max()+1)))

'''Replace this with a lower value at some point.'''
mask2 = labeled_areas < np.inf
remove_large_area = mask2[big_labeled_data.ravel()].reshape(big_labeled_data.shape)

final_labeled_data, num_features = ndimage.label(remove_large_area,s)
print(num_features)

background = np.zeros((4611,2570))
for i in range(500):
    xp =  np.array([])
    fp = np.array([])
    for j in range(2570):
        if final_labeled_data[i][j] == 0:
            background[i][j] = data[i][j]
            xp = np.append(xp,j)
            fp= np.append(fp,background[i][j])
    for j in range(2570):
        if final_labeled_data[i][j] != 0:
            background[i][j] = np.interp(j,xp,fp)

radius = 20
filtered_data = 0
blurred = ndimage.median_filter(background, size=20)


# counts = np.array([])
# for feature in range(1,2):
#     count = 0
#     for i in range(500):
#         for j in range(2570):
#             if final_labeled_data[i][j] == feature:
#                 count += data_minus_background[i][j]
#     counts = np.append(counts,count)

plt.imshow(blurred)

