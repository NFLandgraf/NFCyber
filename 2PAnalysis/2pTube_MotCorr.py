#%%
%matplotlib qt
import os
import tifffile
import numpy as np
import suite2p
from suite2p.run_s2p import run_s2p

# Path to your tif file
tif_path =      "D:\\Nico\\Coding\\tret\\CA1Dopa_2pTube_2025-07-28_mKate_1001f_day2_airpuff_green_prepr.tif"
output_folder = "D:\\Nico\\Coding\\tret"
suite2p_folder = output_folder + '\\suite2p\plane0'
output_video = os.path.join(suite2p_folder, 'MotCorr.tif')

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

    # Run Suite2p motion correction
    run_s2p(ops=ops)
    print('MotCorr finished')

def video_from_MotCorr(suite2p_folder, output_video):
    # creates a video from the suite2p pipeline

    
    # loads info
    ops = np.load(os.path.join(suite2p_folder, 'ops.npy'), allow_pickle=True).item()

    # loads binary file
    bin_file = os.path.join(suite2p_folder, 'data.bin')
    frames = np.memmap(bin_file, dtype='int16', mode='r', shape=(ops['nframes'], ops['Ly'], ops['Lx']))

    # Save as TIFF
    tifffile.imwrite(output_video, frames, dtype='int16', photometric='minisblack')

    print(f"Saved registered movie to: {output_video}")


motcorr(tif_path, output_folder)
video_from_MotCorr(suite2p_folder, output_video)