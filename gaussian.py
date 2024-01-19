# -- FITS image analysis script -- #
# takes histogram of FITS data, isolates background peak
# fits Gaussian distribuition to background
# determines background - object threshold using standard dev.
# masks out astronomical objects using threshold

from astropy.io import fits
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.optimize import curve_fit
from scipy.interpolate import CubicSpline
from scipy.interpolate import lagrange
from scipy import ndimage

def gaussian(x, μ, σ, A):
   return (A / (σ * np.sqrt(2 * np.pi))) * np.exp(-((x - μ) ** 2) / (2 * σ** 2))

hdulist = fits.open("Fits_Data\mosaic.fits")
data = hdulist[0].data
data_trans = np.transpose(data)

# hist, bins = np.histogram(data, 1000)
# width = 0.7 * (bins[1] - bins[0])
# center = (bins[:-1] + bins[1:]) / 2
# plt.bar(center, hist, align='center', width=width)
# plt.show()

# fig, axes = plt.subplots(figsize=(10,6))
# hist_zoom, bins_zoom = np.histogram(data, 150, range=(3350,3500))
# width_zoom = 0.7 * (bins_zoom[1] - bins_zoom[0])
# center_zoom = (bins_zoom[:-1] + bins_zoom[1:]) / 2
# plt.bar(center_zoom, hist_zoom, align='center', width=width_zoom, label = 'Histogram')

# x_data = center_zoom
# y_data = hist_zoom
# popt, pcov = curve_fit(gaussian, x_data, y_data, p0=(3400,50, 400000))
# mu_fit, sd_fit, A_fit = popt
# y_fit = gaussian(x_data, mu_fit, sd_fit, A_fit)
# plt.plot(x_data, y_fit, 'r', label='Gaussian Fit')
# plt.xlabel('Count (Fits Value)',size=20)
# plt.ylabel('Frequency',size=20)
# plt.title('Histogram of pixel brightness count', size=24)
# plt.legend(loc='upper right',fancybox=True, shadow=True, prop={'size': 18})
# plt.xticks(size=18,color='#4f4e4e')
# plt.yticks(size=18,color='#4f4e4e')
# sns.set(style='whitegrid')
# plt.show()

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
        if data[i][j] > 3430:
            binary_check[i][j] = 1            

s = [[0,1,0],
     [1,1,1],
     [0,1,0]]
labeled_data, num_features = ndimage.label(binary_check,s)
labeled_areas = np.array(ndimage.sum(binary_check, labeled_data, np.arange(labeled_data.max()+1)))
mask = labeled_areas > 112
remove_small_area = mask[labeled_data.ravel()].reshape(labeled_data.shape)
final_labeled_data, num_features = ndimage.label(remove_small_area,s)

#     component = np.where(labeled_data == label, 1, 0)
#     print(f"Group {label}: {component}")