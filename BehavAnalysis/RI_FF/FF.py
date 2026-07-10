#%%
'''
Takes YM recordings and checks correlations between behavioral events and GRAB signal
'''
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal, optimize, stats
from pathlib import Path
import os
from scipy.ndimage import gaussian_filter1d
from matplotlib.animation import FuncAnimation, FFMpegWriter
import cv2


path = r"E:\AGG-FF\test"
file_useless_strings = ['']


def manage_filename(file):
    name = Path(file).stem
    return "".join(name.replace(word, "") for word in file_useless_strings)
def get_files(path, common_name):

    # get files
    files = [file for file in Path(path).iterdir() if file.is_file() and common_name in file.name]
    
    # print(f'\n{len(files)} files found')
    # for file in files:
    #     print(file)
    # print('\n')

    return files


def get_signals_1IO(file_doric):
    # do this when you have 1 IO-TTLs, e.g. IO2-Cam_TTL

    def h5print(item, leading=''):
        # prints out the h5 data structure
        for key in item:
            if isinstance(item[key], h5py.Dataset):
                print(leading + key + ':' + str(item[key].shape))
            else:
                print(leading + key)
                h5print(item[key], leading + '  ')
    
    def extend_high_signal(df, column, extra_rows):
        # for one IO channel, extend the ocurrence of 1 to also see the high state when downsampling to 100Hz
        # Extend the 1s for x additional rows after a transition
        
        transitions = (df[column].shift(1) == 1) & (df[column] == 0)  # where does it change from 1 to 0
        for i in range(len(df)):
            if transitions.iloc[i]:  # If there was a transition from 1 to 0
                for j in range(0, extra_rows + 1):  # Add `1` for x extra rows
                    if i + j < len(df):  # Ensure we stay within bounds
                        df.iloc[i + j, df.columns.get_loc(column)] = 1
        return df

    def get_hightimes(df_io, value_threshold=0.9):
        # returns a list with the onset (and offset) timepoints, where certain channel is 'high' 
        s = df_io['digi_Cam'].to_numpy()
        t = df_io.index.to_numpy()

        high = s >= value_threshold
        edges = np.diff(high.astype(np.int8))

        onset_idx  = np.where(edges == 1)[0] + 1   # low→high
        offset_idx = np.where(edges == -1)[0] + 1  # high→low

        hightimes = np.c_[t[onset_idx], t[offset_idx]]
        onsets = np.array(hightimes)[:, 0]
        diffs = np.diff(onsets)

        # unique_vals, counts = np.unique(diffs, return_counts=True)
        # for val, count in zip(unique_vals, counts):
        #     print(f"{val} → {count}")

        return hightimes


    with h5py.File(file_doric, 'r') as f:
        #h5print(f, '')     # prints the whole structure
        path = 'DataAcquisition/NC500/Signals/Series0001/'

        # collect raw data from h5 file
        sig_isos  = np.array(f[path + 'LockInAOUT01/AIN01'])
        sig_fluo  = np.array(f[path + 'LockInAOUT02/AIN01'])
        sig_time  = np.array(f[path + 'LockInAOUT01/Time'])

        digi_Cam    = np.array(f[path + 'DigitalIO/DIO01']).astype(int)
        digi_time   = np.array(f[path + 'DigitalIO/Time'])

    # get signal parameters
    sig_time = np.round(sig_time, 2)
    duration = max(sig_time)
    sig_sampling_rate = int(len(sig_isos) / duration)

    # create frames
    df_signal  = pd.DataFrame({'Isos': sig_isos, 'Fluo': sig_fluo}, index=pd.Index(sig_time, name='Time'))
    df_io      = pd.DataFrame({'digi_Cam': digi_Cam}, index=pd.Index(digi_time, name='Time'))

    # get IO parameters
    df_io = df_io.iloc[0::10]      # downsample to take every 10th frame -> one datapoint every ms
    #io = extend_high_signal(io, 'digi_Cam', 6)   # increase the duration to see every TTL at 100Hz
    io_hightimes = get_hightimes(df_io)

    return df_signal, df_io, io_hightimes

def get_DLC(file_DLC, dist_bp='Center', DLC_mm_per_px = 0.12, fps_Cam=30, thresh_likelihood=0.5):
    # calculates the distance and speed between frames
    
    def cleaning_raw_df(file_DLC):

        df = pd.read_csv(file_DLC, header=None, low_memory=False)

        # create new column names from first two rows
        new_columns = [f"{col[0]}_{col[1]}" for col in zip(df.iloc[1], df.iloc[2])]
        new_columns = [("Resi_" + s) if "_1_" in s else s for s in new_columns]
        new_columns = [("Intr_" + s) if "_2_" in s else s for s in new_columns]
        new_columns = [s.replace('_1_', '_') for s in new_columns]
        new_columns = [s.replace('_2_', '_') for s in new_columns]
        df.columns = new_columns
        df = df.drop(labels=[0, 1, 2], axis="index")

        # set index and convert to numeric
        df.set_index('frame_nan', inplace=True)
        df.index.names = ['frames']
        df = df.astype(float)
        df.index = df.index.astype(float).astype(int)

        bps_all         = ['Ear_left', 'Ear_right', 'Nose', 'Center', 'Lat_left', 'Lat_right', 'Tail_base', 'Tail_end']
        bps_all_Resi = [("Resi_" + s) for s in bps_all]
        bps_all_Intr = [("Intr_" + s) for s in bps_all]
        bps_all = bps_all_Resi + bps_all_Intr
        bps_head        = ['Resi_Ear_left', 'Resi_Ear_right', 'Resi_Nose', 'Intr_Ear_left', 'Intr_Ear_right', 'Intr_Nose']
        bps_postbody    = ['Resi_Center', 'Resi_Lat_left', 'Resi_Lat_right', 'Resi_Tail_base', 'Intr_Center', 'Intr_Lat_left', 'Intr_Lat_right', 'Intr_Tail_base']

        # remove low-confidence points
        for bodypart in bps_all:
            filter = df[f'{bodypart}_likelihood'] <= thresh_likelihood
            df.loc[filter, f'{bodypart}_x'] = np.nan
            df.loc[filter, f'{bodypart}_y'] = np.nan
            df = df.drop(columns=f'{bodypart}_likelihood')

        # interpolate to skip nans
        df = df.interpolate(method="linear")

        # mean 'trustworthy' bodyparts (ears, ear tips, eyes, midpoint, neck) to 'head' column
        for animal in ('Resi', 'Intr'):
            for c in ('_x', '_y'):
                df[f'{animal}_Head{c}'] = df[[bp+c for bp in bps_head]].mean(axis=1, skipna=True)
                df[f'{animal}_Postbody{c}'] = df[[bp+c for bp in bps_postbody]].mean(axis=1, skipna=True)

        # smoothing along time via gaussian filter
        for col in df.columns:
            df[col] = gaussian_filter1d(df[col].values, sigma=2, mode="nearest")

        return df
    
    df = cleaning_raw_df(file_DLC)
    
    # takes boypart and calculates everything from that
    for anim in ('Resi', 'Intr'):
        dx = np.diff(df[f'{anim}_{dist_bp}_x'].to_numpy())
        dy = np.diff(df[f'{anim}_{dist_bp}_y'].to_numpy())
        dist = np.sqrt(dx*dx + dy*dy)              # length = n_frames - 1
        dist = np.r_[0.0, dist]                    # prepend 0 for first frame (align length)
        dist = dist * DLC_mm_per_px
        df[f'{anim}_Distance'] = dist
        df[f'{anim}_Speed'] = dist * fps_Cam
        df[f'{anim}_Distance_cum'] = df[f'{anim}_Distance'].cumsum()

    return df

def merge_signal_DLC(df_trace, df_DLC, io_hightimes):

    # set the correct index for df_trace
    hightimes = np.array([i[0] for i in io_hightimes])
    df_trace_crop = df_trace.copy()
    df_trace_crop.index = pd.Index(pd.to_numeric(df_trace_crop.index), name="time")
    df_trace_crop = df_trace_crop.sort_index()

    # trim hightimes or DLC
    n = min(len(hightimes), len(df_DLC))
    hightimes = hightimes[:n]
    df_DLC = df_DLC.iloc[:n].copy()
    
    # make TTL hightimes the index of DLC and merge
    df_DLC.index = hightimes
    df_DLC.index.name = 'time'
    window_start, window_end = float(df_DLC.index.min()), float(df_DLC.index.max())

    # crop trace to DLC window
    df_trace_crop = df_trace_crop.loc[(df_trace_crop.index >= window_start) & (df_trace_crop.index <= window_end)]
    dlc_num = df_DLC.select_dtypes(include=[np.number]).copy()
    dlc_on_trace = (dlc_num.reindex(dlc_num.index.union(df_trace_crop.index)).sort_index().interpolate(method="index").reindex(df_trace_crop.index))
    df_merged = df_trace_crop.join(dlc_on_trace, how="left")

    return df_merged

def dff(df, bleaching_correct=True):
    # for the FootShock stuff, dont use bleaching_correction

    time_sec = df.index.to_numpy(dtype=float)
    fluo_raw = df['Fluo'].to_numpy(dtype=float)
    isos_raw = df['Isos'].to_numpy(dtype=float)
    sampling_rate = (len(time_sec) - 1) / (time_sec[-1] - time_sec[0])

    # Denoising
    # low-pass cutoff freq depends on indicator (2-10Hz for GCaMP6f)
    # use zero-phase filter that changes amplitude but not phase of freq components/ no signal distortion
    b, a = signal.butter(2, 10, btype='low', fs=sampling_rate)
    fluo_process = signal.filtfilt(b, a, fluo_raw)
    isos_process = signal.filtfilt(b, a, isos_raw)

    # Bleaching correction
    if bleaching_correct:
        def double_exponential(t, const, amp_fast, amp_slow, tau_slow, tau_multiplier):
            # fits a double-exponential function to trace to correct for all possible bleaching reasons
            '''Compute a double exponential function with constant offset.
            t               : Time vector in seconds.
            const           : Amplitude of the constant offset. 
            amp_fast        : Amplitude of the fast component.  
            amp_slow        : Amplitude of the slow component.  
            tau_slow        : Time constant of slow component in seconds.
            tau_multiplier  : Time constant of fast component relative to slow. '''
            
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
        fluo_expofit = get_parameters(time_sec, fluo_process)
        fluo_process = fluo_process / fluo_expofit
        isos_expofit = get_parameters(time_sec, isos_process)
        isos_process = isos_process / isos_expofit

    # Motion
    slope, intercept, r_value, p_value, std_err = stats.linregress(x=isos_process, y=fluo_process)
    fluo_est_motion = intercept + slope * isos_process

    # Normalization
    fluo_dff = (fluo_process - fluo_est_motion) / fluo_est_motion * 100
    df['dff'] = fluo_dff

    # Z-Score: baseline is mean of whole trace
    baseline = np.mean(fluo_dff)
    st_dev = np.std(fluo_dff, ddof=1)
    fluo_zscore = (fluo_dff - baseline) / (st_dev if st_dev > 0 else np.nan)
    df['zscore'] = fluo_zscore

    return df

def create_video_dff(df, z_window=5, flip_x=False, flip_y=True):
    '''
    Creates a video that shows the DLC animals and the FF zscore signal 
    '''
    bps_all =  ['Ear_left', 'Ear_right', 'Nose', 'Center', 'Lat_left', 'Lat_right', 'Tail_base', 'Head', 'Postbody']
    bps_all_Resi = [("Resi_" + s) for s in bps_all]
    bps_all_Intr = [("Intr_" + s) for s in bps_all]
    bps_all = bps_all_Resi + bps_all_Intr

    connections = [
                    ('Nose', 'Ear_left'),
                    ('Nose', 'Ear_right'),
                    ('Nose', 'Center'),
                    ('Center', 'Tail_base'),
                    ('Ear_left', 'Lat_left'),
                    ('Ear_right', 'Lat_right'),
                    ('Lat_right', 'Tail_base'),
                    ('Lat_left', 'Tail_base'),
                ]
    
    colors = ['tab:red'] * len(bps_all_Resi) + ['tab:blue'] * len(bps_all_Intr)
    sizes = []
    for bp in bps_all:
        if bp in ['Resi_Head', 'Intr_Head']:
            sizes.append(80)
        elif bp in ['Resi_Postbody', 'Intr_Postbody']:
            sizes.append(40)
        else:
            sizes.append(1)
  
    df = df.iloc[::10]
    df = df[:50]
    time = df.index.to_numpy()
    n_frames = len(time)
    
    # get bodypart coordinates
    x_cols, y_cols = [], []
    for bp in bps_all:
        xcol, ycol = f'{bp}_x', f'{bp}_y'
        x_cols.append(xcol)
        y_cols.append(ycol)

    # get values
    X = df[x_cols].to_numpy(dtype=float)
    Y = df[y_cols].to_numpy(dtype=float)
    Z = df['dff'].to_numpy(dtype=float)

    # flip video if necessary
    if flip_x:
        xmin = np.nanmin(X)
        xmax = np.nanmax(X)
        X = xmin + xmax - X
    if flip_y:
        ymin = np.nanmin(Y)
        ymax = np.nanmax(Y)
        Y = ymin + ymax - Y

    # intitiate plot
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_xlim(np.nanmin(X), np.nanmax(X))
    ax.set_ylim(np.nanmin(Y), np.nanmax(Y))
    scat = ax.scatter([], [], s=10, c='black')

    # add lines between bodyparts
    lines = []
    for animal, color in [('Resi', 'tab:red'), ('Intr', 'tab:green')]:
        for bp1, bp2 in connections:
            line, = ax.plot([], [], color=color, linewidth=1)
            lines.append((line, animal, bp1, bp2))

    # inset for signal
    inset = ax.inset_axes([0.9, 0.7, 0.35, 0.25])
    inset.plot(time, Z, linewidth=0.7, color='black')
    vline = inset.axvline(time[0], linewidth=1)
    inset.set_ylim(np.nanmin(Z), np.nanmax(Z))
    inset.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

    def update(i):
        pts = np.column_stack([X[i], Y[i]])
        scat.set_offsets(pts)
        scat.set_sizes(sizes)
        scat.set_color(colors)
        for line, animal, bp1, bp2 in lines:
            col1 = bps_all.index(f'{animal}_{bp1}')
            col2 = bps_all.index(f'{animal}_{bp2}')
            line.set_data([X[i, col1], X[i, col2]], [Y[i, col1], Y[i, col2]])
        current_time = time[i]
        inset.set_xlim(current_time - z_window, current_time + z_window)
        vline.set_xdata([current_time, current_time])
        return [scat, vline] + [line[0] for line in lines]

    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000/100, blit=True)
    writer = FFMpegWriter(fps=10)
    anim.save('sicko.mp4', writer=writer)
    plt.close(fig)


files = get_files(path, '.doric')

for i, file_doric in enumerate(files):

    file_DLC = str(file_doric).replace('.doric', '_fps_sLEAP_DLC.csv')
    file_short = manage_filename(file_doric)
    print(f'----- {file_short} -----')

    # extract data
    df_signal, df_io, io_hightimes = get_signals_1IO(file_doric)
    df_DLC = get_DLC(file_DLC)
    df_DLC.to_csv('ee.csv')
    main_df = merge_signal_DLC(df_signal, df_DLC, io_hightimes)
    main_df = dff(main_df, file_short)
    main_df.to_csv('ee.csv')

    create_video_dff(main_df)