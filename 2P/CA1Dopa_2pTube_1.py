#%%
import os
import tifffile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.ndimage import gaussian_filter
from scipy.ndimage import uniform_filter1d
import suite2p
from suite2p.run_s2p import run_s2p
from scipy import signal, optimize, stats
from pathlib import Path

'''
This code takes the original TIFs of the axonal recording (one channel) and prepares them for the FIJI step
1. Z-Projection
2. Temporal smooting
3. Spatial Smooting
4. Motion Correction
'''

fold                            = Path(r'E:\2pTube')
fold_Suite2p                    = 'suite2p\\plane0'
fold_RawImg                     = fold / 'Raw'
fold_RawImg_Preprocess          = fold / 'Raw_Preprocess'
fold_RawImg_Preprocess_MotCorr  = fold / 'Raw_Preprocess_MotCorr'

#%%
'''
PRE-PROCESS
Z-projection was created via the mean signal intensity along the Z-axis.
Temporal and spatial smoothing was performed using a moving average filter along time axis and a Gaussian blur, respectively.
'''
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

files_raw = sorted(fold_RawImg.glob('*.tif'))
os.makedirs(fold_RawImg_Preprocess, exist_ok=True)

for file_raw in files_raw:
    img_raw = tifffile.imread(file_raw)
    img_raw_preprocess = preprocess(img_raw)
    tifffile.imwrite(fold_RawImg_Preprocess / f'{file_raw.stem}_Preprocess.tif', img_raw_preprocess.astype('float32'))


#%%
'''
MOTION CORRECTION
Motion correction was applied via Suite2p.
'''
def motcorr(tif_path, output_folder):
    # Motion correction of a tiff file
    ops = {
        'data_path': [os.path.dirname(tif_path)],
        'save_path0': output_folder,
        'fast_disk': output_folder,
        'tiff_list': [tif_path],

        'do_registration': True,                # make motion correction
        'nonrigid': False,                      # False to do rigid
        'save_movie': True,                     # safe movie as tiff, not working though

        'nplanes': 1,                           # we have 1 z layer
        'nchannels': 1,                         # we have 1 channel
        'functional_chan': 1,
        'input_format': 'tif',                  # input video is tif

        # Skip everything else
        'roidetect': False,                     
        'do_extract': False,
        'neuropil_extract': False,
        'deconvolution': False,
        'save_mat': False,

        # Motion correction
        'maxregshift': 0.1,                     # Maximum allowed frame shift (in fraction of frame size, e.g. 10% of frame width/height)
        'subpixel': 10,                         # Determines registration precision. Higher values mean finer subpixel accuracy (1/10 pixel here)
        'batch_size': 2000,                     # Number of frames loaded and processed in each batch. Adjust if you run into memory issues.
    }

    run_s2p(ops=ops)
def video_from_MotCorr(suite2p_folder):
    # loads info and binary file
    ops = np.load(os.path.join(suite2p_folder, 'ops.npy'), allow_pickle=True).item()
    bin_file = os.path.join(suite2p_folder, 'data.bin')
    frames = np.memmap(bin_file, dtype='int16', mode='r', shape=(ops['nframes'], ops['Ly'], ops['Lx']))
    return frames

files_raw_preprocess = sorted(fold_RawImg_Preprocess.glob('*.tif'))
os.makedirs(fold_RawImg_Preprocess_MotCorr, exist_ok=True)

for file_raw_preprocess in files_raw_preprocess:
    motcorr(file_raw_preprocess, fold_RawImg_Preprocess_MotCorr)
    frames = video_from_MotCorr(fold_Suite2p, fold_RawImg_Preprocess_MotCorr)
    tifffile.imwrite(fold_RawImg_Preprocess_MotCorr / f'{file_raw.stem}_MotCorr.tif', frames, dtype='int16', photometric='minisblack')
