# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 09:35:31 2024

@author: David
"""
from astropy.io import fits
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.optimize import curve_fit
from scipy.optimize import minimize
import random
from scipy import ndimage
from matplotlib import cm

def multivariate_gaussian_fit(xy, amp, x0, y0, sigma_x, sigma_y, theta):
    x, y = xy
    a = (np.cos(theta)**2) / (2 * sigma_x**2) + (np.sin(theta)**2) / (2 * sigma_y**2)
    b = -(np.sin(2 * theta)) / (4 * sigma_x**2) + (np.sin(2 * theta)) / (4 * sigma_y**2)
    c = (np.sin(theta)**2) / (2 * sigma_x**2) + (np.cos(theta)**2) / (2 * sigma_y**2)
    return amp * np.exp(-(a * (x - x0)**2 + 2 * b * (x - x0) * (y - y0) + c * (y - y0)**2))

# generate gaussian based on PDF of values

# Our 2-dimensional distribution will be over variables X and Y
N = 1000
max_val = 10
X = np.linspace(-max_val, max_val, N)
Y = np.linspace(-max_val, max_val, N)
X, Y = np.meshgrid(X, Y)

# Mean vector and covariance matrix
mu = np.array([3., 0.])
Sigma = np.array([[1. , 1.], [-0.,  1.]])

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

length = len(Z)
for y in range(length):
    for x in range(length):
        rand_num = random.random()
        if Z[y][x] > rand_num:
            Z[y][x] = 1
        else:
            Z[y][x] = 0

new_inferno = cm.get_cmap('gray', 2)# visualize with the new_inferno colormaps

fig=plt.figure()
ax = fig.add_subplot(1,1,1)
ax.contourf(X, Y, Z, zdir='z', offset=0, cmap=new_inferno)
ax.grid(False)
ax.set_xlabel(r'x')
ax.set_ylabel(r'y')
plt.show()

#Get the average x and y values (find the centre of the galaxy)
xdata = np.vstack((X.ravel(), Y.ravel()))
y_mean=(np.argmax(np.mean(Z, axis=1))*2*max_val/N)-max_val
Z_trans = np.transpose(Z)
x_mean=(np.argmax(np.mean(Z_trans, axis=1))*2*max_val/N)-max_val


# Initial guess for the parameters
initial_guess = [1, 0, 0, 0.5, 0.5, 0]

# Perform the curve fit
popt, pcov = curve_fit(multivariate_gaussian_fit, (X.flatten(), Y.flatten()), Z.flatten(), p0=initial_guess)
fit = multivariate_gaussian_fit((X, Y), *popt)
print('Fitted parameters:')
print(popt)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot_surface(X, Y, fit, cmap='plasma')
cset = ax.contourf(X, Y, fit, zdir='z', offset=-4, cmap='plasma')
ax.set_zlim(-4,np.max(fit))
plt.show()

# Plot the test data as a 2D image and the fit as overlaid contours.
fig = plt.figure()
ax = fig.add_subplot(111)
ax.contourf(X, Y, Z)
cnt = ax.contour(X, Y, fit, levels=[0.01,0.1],colors='r')
ax.clabel(cnt, cnt.levels, inline = True, fontsize = 10)
ax.set_xlabel(r'x')
ax.set_ylabel(r'y')
plt.show()

weighted_count_inside_contour = np.sum((fit > 0.01) * Z)
print('Number of pixels inside the contour level:', weighted_count_inside_contour)
            
binary_check = np.zeros((N,N))
for i in range(N):
    for j in range(N):
        if Z[i][j] > 0.25:
            binary_check[i][j] = 1            

s = [[0,1,0],
     [1,1,1],
     [0,1,0]]
labeled_data, num_features = ndimage.label(binary_check,s)
labeled_areas = np.array(ndimage.sum(binary_check, labeled_data, np.arange(labeled_data.max()+1)))
mask = labeled_areas > 112
remove_small_area = mask[labeled_data.ravel()].reshape(labeled_data.shape)
final_labeled_data, num_features = ndimage.label(remove_small_area,s)

init_params = np.empty((0, 6))

for feature in range(1,2):
    old_value = 0
    for i in range(len(final_labeled_data)):
        for j in range(len(final_labeled_data[0])):
            if final_labeled_data[i][j] == feature:
                if Z[i][j]>old_value:
                    maximum = np.array([i,j])
                    old_value = Z[i][j]
                    print(old_value)
                else:
                    pass
            else:
                pass
    amp = old_value
    x0, y0 = maximum
    sigma_x, sigma_y = 1,1
    init_params = np.append(init_params, [[amp,x0,y0,sigma_x,sigma_y,0]])