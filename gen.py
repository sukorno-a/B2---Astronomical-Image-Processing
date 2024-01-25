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
ax.contourf(X, Y, Z, zdir='z', offset=0, cmap=cm.gray)
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
ax.contourf(X, Y, Z, zdir='z', offset=0, cmap=cm.inferno)
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

binary_gray = cm.get_cmap('gray', 2)# visualize with the new_inferno colormaps

fig=plt.figure()
ax = fig.add_subplot(1,1,1)
ax.contourf(X, Y, Z, zdir='z', offset=0, cmap=binary_gray)
ax.grid(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel(r'x')
ax.set_ylabel(r'y')
plt.show()
# -----------------------------------------------------------------------------------------

# generate multiple randomly distributed galaxies

def galaxy(N):
    sigma_1 = random.uniform(-1,1)
    sigma_2 = random.uniform(-1,1)
    sigma_3 = random.uniform(-1, 1)
    sigma_4 = random.uniform(-1, 1)
    sigma = np.array([[sigma_1,sigma_2], [sigma_3, sigma_4]])

    mu_x = random.uniform(-N,N)
    mu_y = random.uniform(-N,N)
    mu = np.array([mu_x,mu_y])
    return mu, sigma

# Our 2-dimensional distribution will be over variables X and Y
N = 1000
X = np.linspace(-20, 20, N)
Y = np.linspace(-20, 20, N)
X, Y = np.meshgrid(X, Y)

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

no_of_galaxies = int(input("How many galaxies would you like to generate today?"))
gaussians_list = []
for i in range(no_of_galaxies):
    print("Generating galaxy no. "+str(i+1)+"...")
    mu,sigma = galaxy(20)
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
    gaussians_list.append(Z)

space = np.zeros((N,N))
for i in gaussians_list:
    space += i

length = len(space)
for y in range(length):
    for x in range(length):
        if space[y][x] > 1:
            space[y][x] = 1
        else:
            pass

fig=plt.figure()
ax = fig.add_subplot(1,1,1)
mappable=ax.contourf(X, Y, space, zdir='z', offset=0, cmap=cm.inferno)
fig.colorbar(mappable)
ax.grid(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel(r'x')
ax.set_ylabel(r'y')
plt.show()

binary_gray = cm.get_cmap('gray', 2)# visualize with the new_inferno colormaps

fig=plt.figure()
ax = fig.add_subplot(1,1,1)
ax.contourf(X, Y, space, zdir='z', offset=0, cmap=binary_gray)
ax.grid(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel(r'x')
ax.set_ylabel(r'y')
plt.show()
        
