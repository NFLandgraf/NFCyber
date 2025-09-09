#%%
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

# Split channels: green = channel 0, red = channel 1
green = img[0]
red   = img[1]
print(green.shape)

