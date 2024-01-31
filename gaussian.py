# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 09:46:06 2024

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

def gaussian(x, μ, σ, A):
   return (A / (σ * np.sqrt(2 * np.pi))) * np.exp(-((x - μ) ** 2) / (2 * σ** 2))

def cost_function(params, X, Y, data, model_function):
    model = model_function((X, Y), *params)
    difference = model - data
    return np.sum(difference**2)

def calculate_bic(params, X, Y, data, model_function):
    k = len(params)  # Number of parameters
    n = np.prod(data.shape)  # Number of data points
    L = -0.5 * cost_function(params, X, Y, data, model_function)  # Maximized likelihood
    bic = k * np.log(n) - 2 * L
    return bic

hdulist = fits.open("Fits_Data\mosaic.fits")
data = hdulist[0].data
data_trans = np.transpose(data)

hist, bins = np.histogram(data, 1000)
width = 0.7 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
plt.bar(center, hist, align='center', width=width)
plt.show()

fig, axes = plt.subplots(figsize=(10,6))
hist_zoom, bins_zoom = np.histogram(data, 150, range=(3350,3500))
width_zoom = 0.7 * (bins_zoom[1] - bins_zoom[0])
center_zoom = (bins_zoom[:-1] + bins_zoom[1:]) / 2
plt.bar(center_zoom, hist_zoom, align='center', width=width_zoom, label = 'Histogram')

x_data = center_zoom
y_data = hist_zoom
popt, pcov = curve_fit(gaussian, x_data, y_data, p0=(3400,50, 400000))
mu_fit, sd_fit, A_fit = popt
y_fit = gaussian(x_data, mu_fit, sd_fit, A_fit)
plt.plot(x_data, y_fit, 'r', label='Gaussian Fit')
plt.xlabel('Count (Fits Value)',size=20)
plt.ylabel('Frequency',size=20)
plt.title('Histogram of pixel brightness count', size=24)
plt.legend(loc='upper right',fancybox=True, shadow=True, prop={'size': 18})
plt.xticks(size=18,color='#4f4e4e')
plt.yticks(size=18,color='#4f4e4e')
sns.set(style='whitegrid')
plt.show()

# plt.savefig("gaussian_histogram.jpg")

# #Need to determine the sigma level we want to look at - something to investigate.
# print(mu_fit, sd_fit, A_fit)
# print(np.sqrt(np.diag(pcov)))
# print(mu_fit + 3 * sd_fit)
# print(mu_fit + 5 * sd_fit)

# fig, axes = plt.subplots(figsize=(10,6))
# x=np.linspace(0,4611,len(data_trans[0]))
# plt.plot(x,data_trans[500])
# plt.ylim(3300,3800)
# plt.xlim(1380,1660)
# plt.ylabel('Count (Fits Value)',size=20)
# plt.xlabel('Vertical Pixel',size=20)
# plt.xticks(size=18,color='#4f4e4e')
# plt.yticks(size=18,color='#4f4e4e')
# plt.title('Vertical Splice at 500 Pixels', size=24)
# plt.text(1485,3615,"D",size=24,color='red')
# plt.text(1537,3690,"E",size=24,color='red')
# plt.text(1550,3720,"B",size=24,color='red')
# sns.set(style='whitegrid')
# plt.show()

# fig, axes = plt.subplots(figsize=(10,6))
# x=np.linspace(0,4611,len(data_trans[0]))
# plt.plot(x,data_trans[510])
# plt.ylim(3300,3800)
# plt.xlim(1380,1660)
# plt.ylabel('Count (Fits Value)',size=20)
# plt.xlabel('Vertical Pixel',size=20)
# plt.xticks(size=18,color='#4f4e4e')
# plt.yticks(size=18,color='#4f4e4e')
# plt.title('Vertical Splice at 510 Pixels', size=24)
# plt.text(1485,3625,"D",size=24,color='red')
# plt.text(1518,3460,"F",size=24,color='red')
# sns.set(style='whitegrid')
# plt.show()

# hdubackground = fits.open("Fits_Data\\background.fits", mode='update')

# binary_check = np.zeros((4611,2570))
# data_background = data

# for i in range(4611):
#     for j in range(2570):
#         if data_background[i][j] > 3450:
#             binary_check[i][j] = 1
#             data_background[i][j] = 3421
#     print(i)
            
    

# hdubackground[0].data = data_background

# hdubackground.close()

# plt.imshow(data_background)



## CREATES A BACKGROUND FITS FILE FOR FUTURE ANALYSIS
# hdubackground = fits.open("Fits_Data\\background.fits", mode='update')

# binary_check = np.zeros((4611,2570))
# data_background = data

# for i in range(4611):
#     x1 = np.array([])
#     y1 = np.array([])
#     x2 = np.array([])
#     y2 = np.array([])
#     x3 = np.array([])
#     y3 = np.array([])
#     count = 0
#     for j in range(2570):
#         if data_background[i][j] < 3430:
#             count+=1
#             binary_check[i][j] = 1
#             if count % 9 == 0:
#                 x1 = np.append(x1, j)
#                 y1 = np.append(y1, data_background[i][j])
#             if count % 9 == 3:
#                 x2 = np.append(x2, j)
#                 y2 = np.append(y2, data_background[i][j])
#             if count % 9 == 6:
#                 x3 = np.append(x3, j)
#                 y3 = np.append(y3, data_background[i][j])
#     cs1 = CubicSpline(x1,y1)
#     cs2 = CubicSpline(x2,y2)
#     cs3 = CubicSpline(x3,y3)
#     for j in range(2570):
#         if binary_check[i][j] == 0:
#             data_background[i][j] = (cs1(j) +cs2(j)+cs3(j))/3
#     print(i)
            
    

# hdubackground[0].data = data_background

# hdubackground.close()

# plt.imshow(data_background)

binary_check = np.zeros((4611,2570))
for i in range(4611):
    for j in range(2570):
        if data[i][j] > 3450:
            binary_check[i][j] = 1            

s = [[0,1,0],
     [1,1,1],
     [0,1,0]]

labeled_data, num_features = ndimage.label(binary_check,s)
labeled_areas = np.array(ndimage.sum(binary_check, labeled_data, np.arange(labeled_data.max()+1)))
mask1 = labeled_areas > 112
remove_small_area = mask1[labeled_data.ravel()].reshape(labeled_data.shape)

big_labeled_data, num_features = ndimage.label(remove_small_area,s)
labeled_areas = np.array(ndimage.sum(binary_check, big_labeled_data, np.arange(big_labeled_data.max()+1)))
mask2 = labeled_areas < 1000
remove_large_area = mask2[big_labeled_data.ravel()].reshape(big_labeled_data.shape)

final_labeled_data, num_features = ndimage.label(remove_large_area,s)

#     component = np.where(labeled_data == label, 1, 0)
#     print(f"Group {label}: {component}")





def multivariate_gaussian_fit(xy, amp, x0, y0, sigma_x, sigma_y, theta):
    x, y = xy
    a = (np.cos(theta)**2) / (2 * sigma_x**2) + (np.sin(theta)**2) / (2 * sigma_y**2)
    b = -(np.sin(2 * theta)) / (4 * sigma_x**2) + (np.sin(2 * theta)) / (4 * sigma_y**2)
    c = (np.sin(theta)**2) / (2 * sigma_x**2) + (np.cos(theta)**2) / (2 * sigma_y**2)
    return amp * np.exp(-(a * (x - x0)**2 + 2 * b * (x - x0) * (y - y0) + c * (y - y0)**2))

Y = np.linspace(0,len(data)-1,len(data))
X = np.linspace(0,len(data[0]-1),len(data[0]))
X, Y = np.meshgrid(X, Y)
N_X = len(data)
N_Y = len(data[0])

params = np.empty((0, 6))
fitted_model = np.zeros((N_X,N_Y))
init_params = np.empty((0, 8))
bics = np.array([])

for feature in range(1, 2):
    reduced_space = np.zeros((len(data), len(data[0])))
    old_value = 0
    max_i = 0
    max_j = 0

    for i in range(len(final_labeled_data)):
        for j in range(len(final_labeled_data[0])):
            if final_labeled_data[i][j] == feature + 1:
                reduced_space[i][j] = data[i][j] - 3450
                if data[i][j] > old_value:
                    maximum = np.array([i, j])
                    old_value = data[i][j]
                if i > max_i:
                    max_i = i
                if j > max_j:
                    max_j = j

    new_space = reduced_space

    amp = old_value
    y0, x0 = maximum
    sigma_x, sigma_y = 0.33 * max_i, 0.33 * max_j
    width = (sigma_x + sigma_y) / 2
    init_params = np.append(init_params, [[amp, x0, y0, sigma_x, sigma_y, np.pi / 4, width, 1]], axis=0)
    initial_params = init_params[0]
    gaussian_params = initial_params[0:6]

    print(f"Feature {feature}")
    print("Initial Parameters:", gaussian_params)

    result_gaussian = minimize(
        cost_function, gaussian_params, args=(X, Y, new_space, multivariate_gaussian_fit),
        bounds=[(0.001, None), (0, None), (0, None), (0.0001, None), (0.0001, None), (0, 2 * np.pi)]
    )

    print('Fitted parameters:')
    print(result_gaussian.x)
    bic = calculate_bic(result_gaussian.x, X, Y, new_space, multivariate_gaussian_fit)
    print("BIC:", bic)

    params = np.append(params, [result_gaussian.x], axis=0)
    bics = np.append(bics, bic)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(new_space, origin='lower', cmap='Blues', interpolation='none')
    plt.title('Original Data')

    fitted_model = multivariate_gaussian_fit((X, Y), *result_gaussian.x)
    plt.subplot(1, 2, 2)
    plt.imshow(fitted_model, origin='lower', cmap='Blues', interpolation='none')
    plt.title('Fitted Model')
    plt.show()



    # plt.figure(figsize=(12, 6))
    
    # plt.subplot(1, 2, 1)
    # plt.imshow(new_space, origin='lower', cmap='Blues', interpolation='none')
    # plt.title('Original Data')
    
    # plt.subplot(1, 2, 2)
    # plt.imshow(fitted_model, origin='lower', cmap='Blues', interpolation='none')
    # plt.title('Fitted Model')
    
    # plt.show()

plt.hist(bics)
plt.show