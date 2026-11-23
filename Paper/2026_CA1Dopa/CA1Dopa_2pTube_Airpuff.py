#%% 1. Import and define
from scipy.ndimage import uniform_filter1d
from scipy.ndimage import gaussian_filter
from scipy import signal, optimize, stats
from matplotlib.widgets import Slider
from suite2p.run_s2p import run_s2p
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
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

#%% 2. Pre-process of tiff
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


#%% 3. Motion correction of tiff
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


#%% 4. FIJI MACRO: Max projection of tiff
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


#%% 5. FIJI MACRO: mean intensity of ROI
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


#%% 6. Peri-event
'''
This code looks at the Peri-event response and takes the raw brightness files from the axon pixels from FIJI
1. Raw fluorescent trace
2. Detrended/ bleaching-corrected trace
3. dF/F normalization
4. Cut out windows around each event
5. Calculates ΔF/F using the pre-airpuff baseline
6. Average the recording response across airpuffs
7. Saves the mean trace and the max df/f
'''

fold                                    = Path(r'E:\CA1Dopa_2pTube')
file_RawTraces                          = fold / 'CA1Dopa_2pTube_Results_Airpuff_RawTraces.csv'
file_RawTraces_Detrend                  = fold / 'CA1Dopa_2pTube_Results_Airpuff_RawTraces_Detrend.csv'
file_RawTraces_Detrend_Dff              = fold / 'CA1Dopa_2pTube_Results_Airpuff_RawTraces_Detrend_Dff.csv'
fold_RawTraces_Detrend_PeriEvent        = fold / 'CA1Dopa_2pTube_Results_Airpuff_RawTraces_Detrend_PeriEvent'
file_RawTraces_Detrend_PeriEvent_MaxDff = fold / 'CA1Dopa_2pTube_Results_Airpuff_RawTraces_Detrend_PeriEvent_MaxDff.txt'
file_RawTraces_Detrend_PeriEvent_Means  = fold / 'CA1Dopa_2pTube_Results_Airpuff_RawTraces_Detrend_PeriEvent_Mean.csv'
file_RawTraces_Detrend_PeriEvent_Max_AUC= fold / 'CA1Dopa_2pTube_Results_Airpuff_RawTraces_Detrend_PeriEvent_Max_AUC.csv'

def plot(df, ylabel):
    a53t = ['1002', '976', '972']
    mKate = ['1001', '975', '971']

    plt.figure(figsize=(10, 5))
    x = df['Time [s]']
    for col in df.columns:
        if col == 'Time [s]':
            continue
        elif any(s in col for s in mKate):
            color = 'gray'
        elif any(s in col for s in a53t):
            color = 'red'
        plt.plot(x, df[col], color=color, alpha=0.8, linewidth=1)

    plt.xlabel('Time [s]')
    plt.ylabel(ylabel)
    plt.show()

'''
POST-PROCESSING
Raw fluorescent traces were divided by a fitted double-exponential function for bleaching correction.
ΔF/F was calculated for each trace as ΔF/F = (F - F0) / F0, where F0 was defined as the mean fluorescence of the detrended trace
'''
def bleaching(arr_time, arr_trace):

    # fits a double-exponential function to trace to correct for all possible bleaching reasons
    def double_exponential(t, const, amp_fast, amp_slow, tau_slow, tau_multiplier):
        '''
        Compute a double exponential function with constant offset.
        t               : Time vector in seconds.
        const           : Amplitude of the constant offset. 
        amp_fast        : Amplitude of the fast component.  
        amp_slow        : Amplitude of the slow component.  
        tau_slow        : Time constant of slow component in seconds.
        tau_multiplier  : Time constant of fast component relative to slow. 
        '''
        
        tau_fast = tau_slow * tau_multiplier
        return const+amp_slow*np.exp(-t/tau_slow)+amp_fast*np.exp(-t/tau_fast)
    
    def get_parameters(time_sec, trace):
        max_sig = np.max(trace)
        initial_params = [max_sig/2, max_sig/4, max_sig/4, 3600, 0.1] # initial parmeters
        bounds = ([0, 0, 0, 600, 0], [max_sig, max_sig, max_sig, 36000, 1])
        trace_params, parm_conv = optimize.curve_fit(double_exponential, time_sec, trace, p0=initial_params, 
                                                     bounds=bounds, maxfev=1000)
        trace_expfit = double_exponential(time_sec, *trace_params)

        return trace_expfit
    
    # If bleaching results from a reduction of autofluorescence, signal amplitude should not be affected
    # --> subtract function from trace (signal is V)
    # If bleaching comes from bleaching of the fluorophore, the amplitude suffers as well 
    # --> division is needed (signal is df/f)

    trace_expofit = get_parameters(arr_time, arr_trace)
    trace_detrend = arr_trace / trace_expofit
    
    return trace_detrend
def dff(trace, baseline_range=None, eps=1e-6):
    """
    Compute ΔF/F from a 1D trace using the trace MEAN as baseline.
    trace: 1D array-like (length T)
    baseline_range: optional (start, stop) frame indices to compute the mean baseline on [start, stop)
                    use None to use the entire trace
    eps: small constant to avoid divide-by-zero
    Returns: dff (np.ndarray)
    """
    trace = np.asarray(trace, dtype=np.float64)

    if baseline_range is None:
        f0 = np.nanmean(trace)
    else:
        start, end = baseline_range
        f0 = np.nanmean(trace[start:end])

    denom = max(f0, eps)
    dff = (trace - denom) / denom
    return dff

df = pd.read_csv(file_RawTraces)
df.index = df['Frames']
time_arr = df['Time [s]']
df = df.drop('Frames', axis=1)
df = df.drop('Time [s]', axis=1)

# creates new df for storing the new traces
df_detrend = pd.DataFrame(index=df.index)
df_detrend['Time [s]'] = time_arr
df_dff = pd.DataFrame(index=df.index)
df_dff['Time [s]'] = time_arr

# creates new columns for each recording
for col in df.columns:
    df_detrend[col] = bleaching(time_arr, df[col].values)
    df_dff[col] = dff(df_detrend[col].values)

df_detrend.to_csv(file_RawTraces_Detrend)
df_dff.to_csv(file_RawTraces_Detrend_Dff)

#plot(df_detrend, 'detrend')
#plot(df_dff, 'df/f')


'''
PERI-EVENT
Detrended fluorescence traces were aligned to each airpuff so that stimulus onsets corresponded to frame 0.
For each event, a peri-event window from 2s before to 15s after airpuff onset was extracted.
For each peri-event window, the mean fluorescence during the pre-stimulus period (-2s to -0.1s) was subtracted from the window trace for normalization.
The maximum ΔF/F was extracted from each event window and averaged for the recording. 
Similarly, individual airpuff responses were averaged to receive one recording mean.
'''
def peri_event_dff_frames(series, col, fps, event_frames, window=(-2, 15), baseline=(-2,-0.1), analysis_window=(0,10)):

    # to calculate everything in frames
    window_pre = int(round(window[0] * fps))
    window_post = int(round(window[1] * fps))
    baseline_pre = int(round(baseline[0] * fps))
    baseline_post = int(round(baseline[1] * fps))
    analysis_pre = int(round(analysis_window[0] * fps))
    analysis_post = int(round(analysis_window[1] * fps))

    rel_index = pd.Index(range(window_pre, window_post), dtype=int)
    df_all_events = pd.DataFrame(index=rel_index)

    event_max_values = []
    event_auc_values = []

    for i, event in enumerate(event_frames):

        # move event frame to zero
        shifted = series.copy()
        shifted.index = shifted.index - int(event)

        # cuts out the window
        event_window = shifted.loc[window_pre:window_post]
        baseline_values = shifted.loc[baseline_pre:baseline_post]

        # calc dff
        f0 = baseline_values.mean()
        f0 = max(float(f0), 1e-6)  # numeric safety
        dff = (event_window - f0) / f0   # we already divided during the bleaching correction, so this should just be the subtraction
        df_all_events[f'Air{i}'] = dff.reindex(rel_index)

        # get the max and AUC of the trace
        analysis_trace = dff.loc[analysis_pre:analysis_post].dropna()
        event_max_values.append(analysis_trace.max())
        event_auc_values.append(np.trapezoid(analysis_trace.to_numpy(), x=analysis_trace.index.to_numpy()/fps))

    # adds row means
    mean_trace = df_all_events.mean(axis=1)

    # Average max and AUC across all events
    mean_max = np.nanmean(event_max_values)
    mean_auc = np.nanmean(event_auc_values)

    df_event_max_auc.loc[col, "max"] = mean_max
    df_event_max_auc.loc[col, "auc"] = mean_auc

    return mean_trace

event_frames = [323, 647, 971, 1296, 1620]
fps = 10.8056
df_event_means = pd.DataFrame()
df_event_max_auc = pd.DataFrame()

df = pd.read_csv(file_RawTraces_Detrend)
df.index = df['Frames']
df = df.drop('Frames', axis=1)
df = df.drop('Time [s]', axis=1)

for col in df:
    
    # for the first run, add the frame and Time[s] index
    mean_trace = peri_event_dff_frames(df[col], col, fps, event_frames)
    if df_event_means.empty:
        df_event_means = pd.DataFrame(index=mean_trace.index)
        df_event_means.index.name = 'Frames'
        df_event_means['Time [s]'] = df_event_means.index / fps

    df_event_means[col] = mean_trace
    

df_event_means.to_csv(file_RawTraces_Detrend_PeriEvent_Means)
df_event_max_auc.to_csv(file_RawTraces_Detrend_PeriEvent_Max_AUC)

plot(df_event_means, 'df/f')
