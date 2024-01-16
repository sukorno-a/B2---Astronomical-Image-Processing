
from astropy.io import fits
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.optimize import curve_fit
import seaborn as sns

def gaussian(x, μ, σ, A):
   return (A / (σ * np.sqrt(2 * np.pi))) * np.exp(-((x - μ) ** 2) / (2 * σ** 2))

hdulist = fits.open("Fits_Data\mosaic.fits")
data = hdulist[0].data

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
plt.xticks(size=14,color='#4f4e4e')
plt.yticks(size=14,color='#4f4e4e')
sns.set(style='whitegrid')
plt.show()

plt.savefig("gaussian_histogram.jpg")

#Need to determine the sigma level we want to look at - something to investigate.
print(mu_fit, sd_fit, A_fit)
print(np.sqrt(np.diag(pcov)))
print(mu_fit + 3 * sd_fit)
print(mu_fit + 5 * sd_fit)