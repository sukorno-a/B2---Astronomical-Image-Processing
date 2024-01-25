"generates astronomical image data for analysis script testing"

from astropy.io import fits
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.colors import ListedColormap,LinearSegmentedColormap
from scipy.optimize import curve_fit

import random

from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


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
N = 40
X = np.linspace(-2, 2, N)
Y = np.linspace(-2, 2, N)
X, Y = np.meshgrid(X, Y)

# Mean vector and covariance matrix
mu = np.array([0., 0.])
Sigma = np.array([[ 1. , -0.5], [-0.5,  1.]])

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
ax.contourf(X, Y, Z, zdir='z', offset=0, cmap=cm.viridis)
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
Sigma = np.array([[ 1. , -0.5], [-0.5,  1.]])

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
print(Z)

new_inferno = cm.get_cmap('gray', 2)# visualize with the new_inferno colormaps

fig=plt.figure()
ax = fig.add_subplot(1,1,1)
ax.contourf(X, Y, Z, zdir='z', offset=0, cmap=new_inferno)
ax.grid(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel(r'x')
ax.set_ylabel(r'y')
plt.show()
# -----------------------------------------------------------------------------------------

# generate multiple randomly distributed galaxies

def galaxy():
    sigma_val = random.uniform(-1, 1)
    sigma = np.array([[1.0,sigma_val], [sigma_val, 1.0]])

    N = 100
    X = np.linspace(-2, 2, N)
    Y = np.linspace(-2, 2, N)
    X, Y = np.meshgrid(X, Y)

    # Mean vector and covariance matrix
    mu = np.array([0., 0.])
    Sigma = np.array([[ 1. , -0.5], [-0.5,  1.]])

    # Pack X and Y into a single 3-dimensional array
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y
        
