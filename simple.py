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
import pandas as pd

def gaussian(x, μ, σ, A):
   return (A / (σ * np.sqrt(2 * np.pi))) * np.exp(-((x - μ) ** 2) / (2 * σ** 2))

def image_binary(data, threshold, min_pixel, s = [[0,1,0],[1,1,1],[0,1,0]]):
    '''Determines which values should be used to calculate the background using a binary check
    array and then labels each object that should be removed from background'''
    #Change background threshold to desired value (Recommended = 3430).
    height = len(data)
    if height == 0:
        width = 0
    else:
        width = len(data[0])
    
    binary_check = np.zeros((height,width))
    if height==0 or width==0:
        pass
    else:
        for i in range(height):
            for j in range(width):
                if data[i][j] > threshold:
                    binary_check[i][j] = 1            

    
    #Labels the array based on the previous binary array (connecting adjacent points)
    labeled_data, num_features = ndimage.label(binary_check,s)
    labeled_areas = np.array(ndimage.sum(binary_check, labeled_data, np.arange(labeled_data.max()+1)))
    
    #Removes all groups that are less than a desired number of pixels.
    mask = labeled_areas > min_pixel
    remove_small_area = mask[labeled_data.ravel()].reshape(labeled_data.shape)
    
    #Relabels the data after removing small pixels.
    final_labeled_data, num_features = ndimage.label(remove_small_area,s)
    labeled_areas = np.array(ndimage.sum(binary_check, final_labeled_data, np.arange(final_labeled_data.max()+1)))    
    
    return(binary_check, final_labeled_data, num_features)
    
'''Opens the data and the header.'''
hdulist = fits.open("Fits_Data\mosaic.fits")
data = hdulist[0].data
data_header = hdulist[0].header


'''ZP instument was obtained from the header'''
ZPinst = 2.530e+01
gain = 900
exposure_time = 75


'''Crops the outside pixels of the image as they do not contain anything physical and
are just background.'''
cropped_value = 120
cropped_image = data[cropped_value:4611-cropped_value,cropped_value:2570-cropped_value]


'''Defines the height and width of the image for loop statements.'''
height = len(cropped_image)
width = len(cropped_image[0])


'''Determines which values should be used to calculate the background using a binary check
array and then labels each object that should be removed from background'''
#Change background threshold to desired value (Recommended = 3430).
binary_check, final_labeled_data, num_features = image_binary(data, 3430, 24)


'''Linearly interpolates across the removed values and then replaces every value
with the median value of all pixels within 20 pixels.'''
background = np.zeros((height,width))
for i in range(height):
    xp =  np.array([])
    fp = np.array([])
    for j in range(width):
        if binary_check[i][j] == 0:
            background[i][j] = data[i][j]
            xp = np.append(xp,j)
            fp= np.append(fp,background[i][j])
    for j in range(width):
        if binary_check[i][j] != 0:
            background[i][j] = np.interp(j,xp,fp)

blurred = ndimage.median_filter(background, size=20)


'''Saves and shows the "background" image.'''
hdubackground = fits.open("Fits_Data\\background.fits", mode='update')
plt.imshow(blurred)
plt.show()
hdubackground[0].data = blurred
hdubackground.close()


'''Plots a histogram of the background values'''
plt.hist(blurred.flatten())
plt.title("Histogram of fits background values")
plt.xlabel("Fits value")
plt.ylabel("Count")
plt.show()


'''Shows the "true" image (original - background)'''
image = cropped_image - blurred
plt.imshow(image)
plt.show()


'''Plots a histogram of the "true" image (original - background)'''
plt.hist(image.flatten(),bins=50)
plt.title("Histogram of fits background values")
plt.xlabel("Fits value")
plt.ylabel("Count")
plt.show()


'''Plots a histogram of the "true" image (original - background) zoomed between +-200'''
fig, axes = plt.subplots(figsize=(10,6))
hist_zoom, bins_zoom = np.histogram(image.flatten(),150, range=(-200,200))
width_zoom = 0.7 * (bins_zoom[1] - bins_zoom[0])
center_zoom = (bins_zoom[:-1] + bins_zoom[1:]) / 2
plt.bar(center_zoom, hist_zoom, align='center', width=width_zoom, label = 'Histogram')

'''Fits a Gaussian to the previous histogram.'''
x_data = center_zoom
y_data = hist_zoom
popt, pcov = curve_fit(gaussian, x_data, y_data, p0=(0,20,1e6))
mu_fit, sd_fit, A_fit = popt
y_fit = gaussian(x_data, mu_fit, sd_fit, A_fit)
plt.plot(x_data, y_fit, 'r', label='Gaussian Fit')
plt.xlabel('Count (Fits Value)',size=20)
plt.ylabel('Frequency',size=20)
plt.title('Histogram of pixel brightness count', size=24)
plt.legend(loc='upper right',fancybox=True, shadow=True, prop={'size': 18})
plt.xticks(size=18,color='#4f4e4e')
plt.yticks(size=18,color='#4f4e4e')
# sns.set(style='whitegrid')
plt.show()


'''Does another 'binary check' for detecting an object. Sigma level uses the SD
of the previous Gaussian to determine what count is required to be considered
an object.'''
sigma_level = 5

binary_check_2, final_labeled_data, num_features = image_binary(image, sigma_level*popt[1],24)

'''NEED TO FIND A WAY TO REMOVE STARS HERE'''
headers = ['Class', 'Num_Galaxy', 'Avg_Mag', 'Avg_Count', 'Total_Mag', 'Total_Count',
            'size', 'centre_height', 'centre_width', 'max_count', 'min_height', 'max_height', 'min_width', 'max_width']
ll = len(headers)
galaxies_df = [[0,0,0,0,0,0,0,0,0,0,height,0,width,0] for j in range(num_features+2)]
galaxies_df = np.array(galaxies_df)
galaxies_df =  pd.DataFrame(galaxies_df[1:], columns = headers)

galaxies = {}

for feature in range(num_features+1):
    galaxies.update({feature:[0,0,0,0,0,height,0,width,0]})

galaxies[0] = [1,0,0,0,0,height,0,width,0]
for i in range(height):
    for j in range(width):
        feature = final_labeled_data[i][j]
        if feature == 0:
            pass
        else:
            galaxies_df.loc[feature,'Total_Count'] = float(galaxies_df.loc[feature,'Total_Count']) + image[i][j]
            galaxies_df.loc[feature,'size'] = int(galaxies_df.loc[feature,'size']) + 1
            galaxies[feature][1] += image[i][j]
            galaxies[feature][2] += 1
            if image[i][j]>galaxies_df.loc[feature,'max_count']:
                galaxies_df.loc[feature,'max_count'] = image[i][j]
                galaxies_df.loc[feature,'centre_height'] = i
                galaxies_df.loc[feature,'centre_width'] = j
                galaxies[feature][3] = np.array([i,j])
                galaxies[feature][4] = image[i][j]
            if i < galaxies_df.loc[feature,'min_height']:
                galaxies_df.loc[feature,'min_height'] = i
                galaxies[feature][5] = i
            if i > galaxies_df.loc[feature,'max_height']:
                galaxies_df.loc[feature,'max_height'] = i
                galaxies[feature][6] = i
            if j < galaxies_df.loc[feature,'min_width']:
                galaxies_df.loc[feature,'min_width'] = j
                galaxies[feature][7] = j
            if j > galaxies_df.loc[feature,'max_width']:
                galaxies_df.loc[feature,'max_width'] = j
                galaxies[feature][8] = j
            else:
                pass
    # galaxies[feature] = [0,count,size,centre,old_value,min_i,max_i,min_j,max_j]

for feature in range(1,num_features+1):
    galaxies_df.loc[feature,'Total_Mag'] = ZPinst - (2.5*np.log10(galaxies_df.loc[feature,'Total_Count']/(gain*exposure_time)))
    galaxies[feature][0] = ZPinst -2.5*np.log10(galaxies[feature][1]/720)

    
# print(galaxies)
sorted_df = galaxies_df.sort_values(by = ['Total_Count'],ascending=False)
sorted_df = sorted_df.reset_index(drop=True)
sorted_galaxies = sorted(galaxies, key=lambda k: galaxies[k][0])
# print(sorted_galaxies)

count_values = [galaxies_df.loc[k,'Total_Count'] for k in range(len(galaxies_df))]
count_array = np.array(count_values)
plt.hist(count_array,50)

magnitude_values = [galaxies_df.loc[k,'Total_Mag'] for k in range(len(galaxies_df))]
magnitude_array = np.array(magnitude_values)

sorted_values = [galaxies[k] for k in sorted_galaxies]

# print(sorted_galaxies)
# print(sorted_values)

# print(galaxies[4])


# for i in range(1,27):
#     feature = sorted_df.loc[feature]
#     print(feature)
#     galaxy = galaxies.get(feature)
#     sample = image[galaxy[5]:galaxy[6],galaxy[7]:galaxy[8]]
#     plt.imshow(image[galaxy[5]:galaxy[6],galaxy[7]:galaxy[8]])
#     plt.show()

plt.hist(magnitude_array,50)

magnitude_array = np.array([])
num_galaxies_array = np.array([])
for feature in range(num_features):
    sample = image[sorted_df.loc[feature,'min_height']:sorted_df.loc[feature,'max_height'],sorted_df.loc[feature,'min_width']:sorted_df.loc[feature,'max_width']]         
                
    object_binary, final_labeled_data, num_galaxies = image_binary(sample,100,1)
    if num_galaxies == 0:
        num_galaxies = 1
    sorted_df.loc[feature,'Num_Galaxy'] = num_galaxies
    sorted_df.loc[feature,'Avg_Count'] =  sorted_df.loc[feature,'Total_Count']/sorted_df.loc[feature,'Num_Galaxy']
    magnitude = ZPinst -2.5*np.log10(sorted_df.loc[feature,'Avg_Count']/(gain*exposure_time))
    sorted_df.loc[feature,'Avg_Mag'] = magnitude
    num_galaxies_array = np.append(num_galaxies_array,num_galaxies)
    
    if feature < 5:
        plt.imshow(sample)
        plt.show()
        print("This image has", sorted_df.loc[feature,'Num_Galaxy'], "galaxies identified.")
        while True:
            classification = int(input("Is this background (0), a star (1), cluster (2) or galaxy (3)? "))
            if classification == 0:
                sorted_df.loc[feature,'Class'] = "Background"
                break
            if classification == 1:
                sorted_df.loc[feature,'Class'] = "Star"
                break
            elif classification == 2:
                sorted_df.loc[feature,'Class'] = "Cluster"
                break
            elif classification == 3:
                sorted_df.loc[feature,'Class'] = "Galaxy"
                break
            else:
                print("Please write 0, 1, 2 or 3.")   
    else:
        sorted_df.loc[feature,'Class'] = "Galaxy"
    
    if sorted_df.loc[feature,'Class'] == "Cluster" or sorted_df.loc[feature,'Class'] == "Galaxy":
        for galaxy in range(num_galaxies):
            magnitude_array = np.append(magnitude_array, magnitude)

print(num_features)
print(np.max(num_galaxies_array))
print(np.sum(num_galaxies_array))

'''Finds the number of counts in each object and converts it to magnitude.'''
# counts = np.zeros(num_features+1)
# counts[0] = 1
# for i in range(height):
#     for j in range(width):
#         feature = final_labeled_data[i][j]
#         if feature==0:
#             pass
#         else:
#             counts[feature] = counts[feature] + image[i][j]
# mag_i = -2.5*np.log10(counts/720)
# m = mag_i + ZPinst

plt.hist(magnitude_array,50)
plt.xlabel("Object Magnitude")
plt.ylabel("Count")
plt.show()

m_hist, bins= np.histogram(magnitude_array,100000,range=(20,29))
width = 0.7 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2

N_m = np.cumsum(m_hist)

plt.plot(center,np.log10(N_m),label="log10(N(m))")
plt.plot(center,0.6*center-12,label="0.6m - 12")
plt.plot(center,0.5*center-10,label="0.5m - 10")
plt.plot(center,0.4*center-8,label="0.4m - 8")
plt.plot(center,0.3*center-5.5,label="0.3m - 5.5")
plt.xlabel("Object Magnitude")
plt.ylabel("Log10(N(m)), N(m) is cum. freq.")
plt.legend()
plt.show()
