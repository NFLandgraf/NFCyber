#%% 1. Preprocess
from scipy.ndimage import uniform_filter1d
from scipy.ndimage import gaussian_filter
from pathlib import Path
import numpy as np
import tifffile
import suite2p
import os


'''
This script takes the original TIFs of the axonal recording (one channel) and prepares them for the FIJI step
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

os.makedirs(fold_RawImg_Preprocess, exist_ok=True)
os.makedirs(fold_RawImg_Preprocess_MotCorr, exist_ok=True)


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


for file_raw in sorted(fold_RawImg.glob('*.tif')):
    img_raw = tifffile.imread(file_raw)
    img_raw_preprocess = preprocess(img_raw)
    tifffile.imwrite(fold_RawImg_Preprocess / f'{file_raw.stem}_Preprocess.tif', img_raw_preprocess.astype('float32'))


#%% 2. Motion correction of tiff
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

    suite2p.run_s2p(ops=ops)

def video_from_MotCorr(suite2p_folder):
    # loads info and binary file
    ops = np.load(os.path.join(suite2p_folder, 'ops.npy'), allow_pickle=True).item()
    bin_file = os.path.join(suite2p_folder, 'data.bin')
    frames = np.memmap(bin_file, dtype='int16', mode='r', shape=(ops['nframes'], ops['Ly'], ops['Lx']))
    return frames


for file_raw_preprocess in sorted(fold_RawImg_Preprocess.glob('*.tif')):
    motcorr(file_raw_preprocess, fold_RawImg_Preprocess_MotCorr)
    frames = video_from_MotCorr(fold_Suite2p)
    tifffile.imwrite(fold_RawImg_Preprocess_MotCorr / f'{file_raw.stem}_MotCorr.tif', frames, dtype='int16', photometric='minisblack')


#%% 3. FIJI MACRO: Max projection of tiff
'''
FIJI MACRO
Create a max projection of the motion corrected tiff
'''

// Makro1 — Create Max Projection from a Frames×Y×X TIFF
// Use: open your TIFF (time series) so it’s the active window, then run.

macro "Makro1 - Create Max Projection (Frames x Y x X)" {
    
    // get base name + directory of the active image
    dir  = getDirectory("image");
    name = getTitle();
    
    // strip extension (handles .tif and .tiff)
    lname = toLowerCase(name);
    extLen = 0;
    if (endsWith(lname, ".tif"))  extLen = 4;
    if (endsWith(lname, ".tiff")) extLen = 5;
    if (extLen>0)
        base = substring(name, 0, lengthOf(name)-extLen);
    else
        base = name;

    // create max-intensity projection
    run("Z Project...", "projection=[Max Intensity]");
    rename("MaxProjection");

    // save next to the original with an addendum
    saveAs("Tiff", dir + base + "_MaxProj.tif");
}


#%% 4. FIJI MACRO: mean intensity of ROI
'''
FIJI MACRO
Gets the mean intensity of a user-defined ROI across all frames in a time series TIFF
Saves the results as a CSV file
'''

// ===== Macro2: Apply ROI from Max Projection to the time series, measure mean over time,
//                and save outputs with filenames based on the original TIFF =====
// What it does:
// - Prompts you to select the ORIGINAL time series window.
// - Uses the FIRST ROI in ROI Manager (draw it on the MaxProjection and press 't' before running).
// - Measures mean intensity in that ROI across all frames and saves CSV.
// - Saves Max Projection as <basename>_MAX.tif
// - Saves Max Projection with ROI painted in YELLOW as <basename>_MAX_withROI_yellow.png

macro "Makro2 - ROI trace + save max + save max+ROI (yellow)" {
    
    // --- Build base name and directory from the original stack ---
    dir = getDirectory("image");           // directory of the active image
    title = getTitle();                    // window title (usually the filename)
    lname = toLowerCase(title);
    extLen = 0;
    if (endsWith(lname, ".tif")) extLen = 4;
    else if (endsWith(lname, ".tiff")) extLen = 5;
    if (extLen > 0)
        base = substring(title, 0, lengthOf(title)-extLen);
    else
        base = title;

    // --- Compute and save Max Projection from the ORIGINAL stack ---
    run("Z Project...", "projection=[Max Intensity]");
    saveAs("Tiff", dir + base + "_MaxProj.tif");

    // --- Create a copy to paint the ROI (yellow) and save ---
    roiManager("Select", 0);
    setForegroundColor(255, 255, 0);
    roiManager("Set Line Width", 1);
    roiManager("Draw");
    saveAs("Tiff", dir + base + "_Roi.tif");
    
    // --- Measure mean intensity over all frames in original stack ---
    selectWindow(title);
    run("Set Measurements...", "mean median decimal=6");
    roiManager("Select", 0);
    roiManager("Multi Measure");  // does the loop for you!

    // --- Save Results as CSV ---
    selectWindow("Results");
    saveAs("Results", dir + base + "_Roi_Trace.csv");
    
    // Optional: tidy up the extra max window we created for saving
    if (isOpen("MaxProjection_forSave")) close("MaxProjection_forSave");
