# 

from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt

hdulist = fits.open("Fits_Data/mosaic.fits")

hist, bins = np.histogram(hdulist[0].data, 1000)
width = 0.7 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
plt.bar(center, hist, align='center', width=width)
plt.show()