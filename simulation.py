# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 16:02:00 2024

@author: David
"""

"generates astronomical image data for analysis script testing"

from astropy.io import fits
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.colors import ListedColormap,LinearSegmentedColormap
from scipy.optimize import curve_fit

import random

from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import maximum_filter
from scipy.optimize import minimize
from scipy import ndimage




def find_peaks(space, radius=5):
    num_peaks = 0
    locations = np.empty((0, 2), dtype=int)
    local_maxima = maximum_filter(space, size=2*radius+1, mode='constant', cval=-np.inf)

    for i in range(radius, space.shape[0] - radius):
        for j in range(radius, space.shape[1] - radius):
            if space[i, j] == local_maxima[i, j]:
                num_peaks += 1
                locations = np.append(locations,[[i,j]],axis=0)
                
    return num_peaks, locations


# -----------------------------------------------------------------------------------------

# generate empty data

# rows, cols = (10, 10)
# empty_data=[[0 for i in range(cols)] for j in range(rows)]
# plt.imshow(empty_data, interpolation='none')
# plt.xlabel("x")
# plt.ylabel("y")
# plt.show()

# -----------------------------------------------------------------------------------------

# plot single galaxy

# Our 2-dimensional distribution will be over variables X and Y
N = 100
X = np.linspace(-2, 2, N)
Y = np.linspace(-2, 2, N)
X, Y = np.meshgrid(X, Y)

# Mean vector and covariance matrix
mu = np.array([0., 0.])
Sigma = np.array([[ 1. , -0.5], [-0.5,  0.3]])

# Pack X and Y into a single 3-dimensional array
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X
pos[:, :, 1] = Y

def multivariate_gaussian(pos, mu, Sigma):
    """Return the multivariate Gaussian distribution on array pos."""

    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2*np.pi)**n * Sigma_det)
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)

    return np.exp(-fac / 2) / N

# The distribution on the variables X, Y packed into pos.
Z = multivariate_gaussian(pos, mu, Sigma)

# plot using subplots
fig = plt.figure()
# ax1 = fig.add_subplot(1,2,1,projection='3d')

# ax1.plot_surface(X, Y, Z, rstride=3, cstride=3, linewidth=1, antialiased=True,
#                 cmap=cm.viridis)
# ax1.view_init(55,-70)
# ax1.set_xticks([])
# ax1.set_yticks([])
# ax1.set_zticks([])
# ax1.set_xlabel(r'x')
# ax1.set_ylabel(r'y')

ax = fig.add_subplot(1,1,1)
ax.contourf(X, Y, Z, zdir='z', offset=0)
ax.grid(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel(r'x')
ax.set_ylabel(r'y')

plt.show()

# -----------------------------------------------------------------------------------------

# generate gaussian based on PDF of values

# Our 2-dimensional distribution will be over variables X and Y
N = 100
X = np.linspace(-2, 2, N)
Y = np.linspace(-2, 2, N)
X, Y = np.meshgrid(X, Y)

# Mean vector and covariance matrix
mu = np.array([0., 0.])
Sigma = np.array([[ 1.5 , 0.3], [-2.,  5.6]])

# Pack X and Y into a single 3-dimensional array
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X
pos[:, :, 1] = Y

def multivariate_gaussian(pos, mu, Sigma):
    """Return the multivariate Gaussian distribution on array pos."""

    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2*np.pi)**n * Sigma_det)
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)

    return np.exp(-fac / 2) / N

Z = multivariate_gaussian(pos, mu, Sigma)
Z = Z/np.max(Z)

fig=plt.figure()
ax = fig.add_subplot(1,1,1)
ax.contourf(X, Y, Z, zdir='z', offset=0)
ax.grid(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel(r'x')
ax.set_ylabel(r'y')
plt.show()

length = len(Z)
for y in range(length):
    for x in range(length):
        rand_num = random.random()
        if Z[y][x] > rand_num:
            Z[y][x] = 1
        else:
            Z[y][x] = 0
print(Z)

# binary_gray = cm.get_cmap('gray', 2)# visualize with the new_inferno colormaps

# fig=plt.figure()
# ax = fig.add_subplot(1,1,1)
# ax.contourf(X, Y, Z, zdir='z', offset=0, cmap=binary_gray)
# ax.grid(False)
# ax.set_xticks([])
# ax.set_yticks([])
# ax.set_xlabel(r'x')
# ax.set_ylabel(r'y')
# plt.show()
# -----------------------------------------------------------------------------------------

# generate multiple randomly distributed galaxies

def galaxy(N):
    sigma_1 = random.uniform(50,100)
    sigma_4 = random.uniform(50, 100)
    max_val = np.sqrt(sigma_1*sigma_4)
    sigma_2 = -max_val + 2*random.random()*max_val
    sigma_3 = sigma_2
    sigma = np.array([[sigma_1,sigma_2], [sigma_3, sigma_4]])
    print(sigma)

    mu_x = random.uniform(0,N)
    mu_y = random.uniform(0,N)
    mu = np.array([mu_x,mu_y])
    return mu, sigma

# Our 2-dimensional distribution will be over variables X and Y
N = 1000
X = np.linspace(0, N, N)
Y = np.linspace(0, N, N)
X, Y = np.meshgrid(X, Y)

# Pack X and Y into a single 3-dimensional array
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X
pos[:, :, 1] = Y

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

def multivariate_gaussian(pos, mu, Sigma):
    """Return the multivariate Gaussian distribution on array pos."""

    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2*np.pi)**n * Sigma_det)
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)

    return np.exp(-fac / 2) / N

def circular_top_hat(pos, center, radius, amp):
    x, y = pos.T  # Assuming pos is an array of shape (N, 2)
    x0, y0 = center
    distances = np.sqrt((x - x0)**2 + (y - y0)**2)
    return amp * (distances <= radius)


def multivariate_gaussian_fit(xy, amp, x0, y0, sigma_x, sigma_y, theta):
    x, y = xy
    a = (np.cos(theta)**2) / (2 * sigma_x**2) + (np.sin(theta)**2) / (2 * sigma_y**2)
    b = -(np.sin(2 * theta)) / (4 * sigma_x**2) + (np.sin(2 * theta)) / (4 * sigma_y**2)
    c = (np.sin(theta)**2) / (2 * sigma_x**2) + (np.cos(theta)**2) / (2 * sigma_y**2)
    return amp * np.exp(-(a * (x - x0)**2 + 2 * b * (x - x0) * (y - y0) + c * (y - y0)**2))

def top_hat_gaussian_fit(xy, amp, x0, y0, sigma_x, sigma_y, theta, width, likelihood):
    x, y = xy
    a = (np.cos(theta)**2) / (2 * sigma_x**2) + (np.sin(theta)**2) / (2 * sigma_y**2)
    b = -(np.sin(2 * theta)) / (4 * sigma_x**2) + (np.sin(2 * theta)) / (4 * sigma_y**2)
    c = (np.sin(theta)**2) / (2 * sigma_x**2) + (np.cos(theta)**2) / (2 * sigma_y**2)
    gaussian = likelihood*(amp * np.exp(-(a * (x - x0)**2 + 2 * b * (x - x0) * (y - y0) + c * (y - y0)**2)))
    distances = np.sqrt((x - x0)**2 + (y - y0)**2)
    top_hat = (1-likelihood)*(distances<=width)
    return gaussian+top_hat

def objective_function(params, xy, target):
    amp, x0, y0, sigma_x, sigma_y, theta, width, likelihood = params
    x, y = xy
    a = (np.cos(theta)**2) / (2 * sigma_x**2) + (np.sin(theta)**2) / (2 * sigma_y**2)
    b = -(np.sin(2 * theta)) / (4 * sigma_x**2) + (np.sin(2 * theta)) / (4 * sigma_y**2)
    c = (np.sin(theta)**2) / (2 * sigma_x**2) + (np.cos(theta)**2) / (2 * sigma_y**2)
    gaussian = likelihood * (amp * np.exp(-(a * (x - x0)**2 + 2 * b * (x - x0) * (y - y0) + c * (y - y0)**2)))
    distances = np.sqrt((x - x0)**2 + (y - y0)**2)
    top_hat = (1 - likelihood) * (distances <= width)
    model = gaussian + top_hat
    # Define a suitable objective function to minimize
    error = np.sum((model - target)**2)
    return error

no_of_galaxies = int(input("How many galaxies would you like to generate today?"))
space = np.zeros((N,N))
for i in range(no_of_galaxies):
    print("Generating galaxy no. "+str(i+1)+"...")
    mu,sigma = galaxy(N)
    Z = multivariate_gaussian(pos, mu, sigma)
    space+=Z

if no_of_galaxies != 0:
    space = space/np.max(space)

no_of_stars = int(input("How many stars would you like to generate today?"))
for i in range(no_of_stars):
    print("Generating star no. " + str(i+1) + "...")
    center = np.array([random.uniform(0, N), random.uniform(0, N)])
    radius = 10
    Z = circular_top_hat(pos, center, radius, 1)
    space += Z

for i in range(len(space)):
    for j in range(len(space[i])):
        if space[i][j] > 1:
            space[i][j] = 1

binary_check = np.zeros((N,N))
for i in range(N):
    for j in range(N):
        if space[i][j] > 0.1:
            binary_check[i][j] = 1  

s = [[0,1,0],
     [1,1,1],
     [0,1,0]]
labeled_data, num_features = ndimage.label(binary_check,s)
labeled_areas = np.array(ndimage.sum(binary_check, labeled_data, np.arange(labeled_data.max()+1)))
mask = labeled_areas > 10
remove_small_area = mask[labeled_data.ravel()].reshape(labeled_data.shape)
final_labeled_data, num_features = ndimage.label(remove_small_area,s)

init_params = np.empty((0, 8))

for feature in range(num_features):
    print(feature)
    old_value = 0
    min_i = N
    max_i = 0
    min_j = N
    max_j = 0
    for i in range(len(final_labeled_data)):
        for j in range(len(final_labeled_data[0])):
            if final_labeled_data[i][j] == feature+1:
                if space[i][j]>old_value:
                    maximum = np.array([i,j])
                    old_value = space[i][j]
                if i < min_i:
                    min_i = i
                if i > max_i:
                    max_i = i
                if j < min_j:
                    min_j = j
                if j > max_j:
                    max_j = j
                else:
                    pass
            else:
                pass
    amp = old_value
    y0,x0 = maximum
    sigma_x, sigma_y = 0.33*(max_i-min_i), 0.33*(max_j-min_j)
    width = (sigma_x+sigma_y)/2
    init_params = np.append(init_params,[[amp,x0,y0,sigma_x,sigma_y,np.pi/4,width,1]],axis=0)


# fig=plt.figure()
# ax = fig.add_subplot(1,1,1)
# mappable=ax.contourf(X, Y, space, zdir='z', offset=0)
# fig.colorbar(mappable)
# ax.grid(False)
# ax.set_xticks([])
# ax.set_yticks([])
# ax.set_xlabel(r'x')
# ax.set_ylabel(r'y')
# plt.show()


fig=plt.figure()
ax = fig.add_subplot(1,1,1)
ax.contourf(X, Y, space)
ax.grid(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel(r'x')
ax.set_ylabel(r'y')
plt.show()
        
fit = np.shape((len(X),len(Y)))
params = np.empty((0, 6))
fitted_model = np.zeros((N,N))

for feature in range(num_features):
    print(feature)
    initial_params = init_params[feature]
    gaussian_params = initial_params[0:6]
    print(gaussian_params)
    
    new_space = np.zeros((N,N))
    for i in range(len(space)):
        for j in range(len(space[0])):
            if final_labeled_data[i][j] == feature+1:
                new_space[i][j] = space[i][j]
    result_gaussian = minimize(cost_function, gaussian_params, args=(X, Y, new_space, multivariate_gaussian_fit),
                                bounds=[(0.001, None), (0.0001, None), (0.001, None), (0.0001, None), (0.0001, None), (0, 2*np.pi)])

    print('Fitted parameters:')
    print(result_gaussian.x)
    print(calculate_bic(result_gaussian.x, X, Y, new_space, multivariate_gaussian_fit))
    params = np.append(params, [result_gaussian.x], axis=0)
    fitted_model += multivariate_gaussian_fit((X, Y), *result_gaussian.x)


plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(space, origin='lower', cmap='Blues', interpolation='none')
plt.title('Original Data')

plt.subplot(1, 2, 2)
plt.imshow(fitted_model, origin='lower', cmap='Blues', interpolation='none')
plt.title('Fitted Model')

plt.show()