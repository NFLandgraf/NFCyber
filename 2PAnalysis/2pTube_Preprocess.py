#%%
# IMPORT and read tiff file
%matplotlib qt
import os
import tifffile
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.ndimage import gaussian_filter
from scipy.ndimage import uniform_filter1d

img = tifffile.imread("tret\\CA1Dopa_2pTube_2025-07-28_mKate_1001f_day2_airpuff_green.tiff")  # works for 3D, 4D, or 5D
print(img.shape)


#%%

def preprocess(img):

    def z_projection(img, mean=True):
        # Mean/Max projection over Z
        mean_proj = img.mean(axis=1) if mean==True else img.max(axis=1) # resulting Shape: (T, Y, X)
        print(f'z_proj: {img.shape}, {mean_proj.shape}')
        return mean_proj
        
    def temp_smooth(img, window_size=3):
        # moving window average filter
        temp_smooth = uniform_filter1d(img, size=window_size, axis=0, mode='nearest')
        print(f'temp_smooth: {img.shape}, {temp_smooth.shape}')
        return temp_smooth

    def spat_smooth(img, sig=1.5):
        # spatial gaussian blur
        spat_smooth = gaussian_filter(img, sigma=(0, sig, sig))  # (frames, y, x)
        print(f'spat_smooth: {img.shape}, {spat_smooth.shape}')
        return spat_smooth

    img = z_projection(img)
    img = temp_smooth(img)
    img = spat_smooth(img)

    return img

prepr = preprocess(img)

tifffile.imwrite('my_processed_movie.tif', prepr.astype('float32'))

