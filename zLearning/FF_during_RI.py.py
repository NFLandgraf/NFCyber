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


path = r"C:\Users\landgrafn\Desktop\FF_RI\87"
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
    io = extend_high_signal(df_io, 'digi_Cam', 6)   # increase the duration to see every TTL at 100Hz
    io_hightimes = get_hightimes(df_io)

    return df_signal, df_io, io_hightimes

def dff(df, filename, bleaching_correct=True, del_values=True):
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

    if del_values:
        # for times where it is clear that the values are wrong
        if '691' in filename:
            df.loc[111.00 : 112.50, ["dff","zscore"]] = np.nan
        if '701' in filename:
            df.loc[130.3 : 130.5, ["dff","zscore"]] = np.nan
        if '407' in filename:
            df.loc[41.4 : 41.6, ["dff","zscore"]] = np.nan

    return df

def create_video(df, z_window=5, flip_x=True):
    '''
    Creates a video that shows the DLC animals and the FF zscore signal 
    '''
    bps_all =  ['nose', 'left_ear', 'right_ear', 'left_ear_tip', 'right_ear_tip', 'left_eye', 'right_eye', 'head_midpoint', 
                'neck', 'mid_back', 'mouse_center', 'mid_backend', 'mid_backend2', 'mid_backend3', 
                'tail_base', 'tail1', 'tail2', 'tail3', 'tail4', 'tail5', 'tail_end',
                'left_shoulder', 'left_midside', 'left_hip', 'right_shoulder', 'right_midside', 'right_hip']
    
    df = df.iloc[::10]
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
    Z = df['zscore'].to_numpy(dtype=float)

    if flip_x:
        xmin = np.nanmin(X)
        xmax = np.nanmax(X)
        X = xmin + xmax - X

    # intitiate plot
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(np.nanmin(X), np.nanmax(X))
    ax.set_ylim(np.nanmin(Y), np.nanmax(Y))
    scat = ax.scatter([], [], s=10)

    # inset for signal
    inset = ax.inset_axes([0.6, 0.7, 0.35, 0.25])
    inset.plot(time, Z, linewidth=0.7, color='black')
    vline = inset.axvline(time[0], linewidth=1)
    inset.set_ylim(np.nanmin(Z), np.nanmax(Z))
    inset.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

    def update(i):
        pts = np.column_stack([X[i], Y[i]])
        scat.set_offsets(pts)
        current_time = time[i]
        inset.set_xlim(current_time - z_window, current_time + z_window)
        vline.set_xdata([current_time, current_time])
        return scat, vline

    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000/30, blit=True)
    writer = FFMpegWriter(fps=30)
    anim.save('sicko.mp4', writer=writer)
    plt.close(fig)




files = get_files(path, '.doric')
groups = ["mKate", "A53T", "GFP"]
results = []
df_distance_cum = pd.DataFrame()
df_perievents = pd.DataFrame()


for i, file_doric in enumerate(files):

    file_DLC = str(file_doric).replace('.doric', '_DLC.csv')
    file_short = manage_filename(file_doric)
    print(f'----- {file_short} -----')

    # extract data
    df_signal, df_io, io_hightimes = get_signals_1IO(file_doric)

    onsets = np.array(io_hightimes)[:, 0]
    onsets = np.array([round(a, 2) for a in onsets])
    print(onsets)

    main_df = dff(df_signal, file_short)
    print(main_df)


    idx = main_df.index.get_indexer(onsets, method='nearest')
    result = main_df.iloc[idx].reset_index(drop=True)

    result.to_csv('86.csv')
    break


#%%
import cv2
import numpy as np
import pandas as pd

def create_video_on_mp4(df, video_path, out_path, z_window=100, downsample=1):
    df = df.iloc[::downsample].copy()
    time = df.index.to_numpy()
    Z = df['dff'].to_numpy(dtype=float)

    cap = cv2.VideoCapture(video_path)
    fps_in = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fps_out = fps_in / downsample
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps_out, (width, height))

    zmin, zmax = np.nanmin(Z), np.nanmax(Z)

    # inset position
    x0, y0 = int(width * 0.58), int(height * 0.08)
    w, h = int(width * 0.36), int(height * 0.22)

    frame_idx = 0
    df_idx = 0

    while cap.isOpened() and df_idx < len(df):
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % downsample == 0:
            current_time = time[df_idx]

            # normalize once
            zmin, zmax = np.nanmin(Z), np.nanmax(Z)

            # avoid division by zero
            zr = zmax - zmin if zmax != zmin else 1.0

            # semi-transparent white background
            overlay = frame.copy()
            cv2.rectangle(overlay, (x0, y0), (x0 + w, y0 + h), (255, 255, 255), -1)
            frame = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)

            # select window around current time
            mask = (time >= current_time - z_window) & (time <= current_time + z_window)
            t_win = time[mask]
            z_win = Z[mask]

            if len(t_win) > 1:
                xs = x0 + ((t_win - t_win.min()) / (t_win.max() - t_win.min()) * w).astype(int)
                ys = y0 + h - ((z_win - zmin) / (zmax - zmin) * h).astype(int)

                pts = np.column_stack([xs, ys]).astype(np.int32)
                cv2.polylines(frame, [pts], False, (0, 0, 0), 2)

                # vertical current-time line
                x_now = x0 + int((current_time - t_win.min()) / (t_win.max() - t_win.min()) * w)
                cv2.line(frame, (x_now, y0), (x_now, y0 + h), (150, 150, 150), 1)
                
                if zmin <= 0 <= zmax:
                    y_zero = y0 + h - int((0 - zmin) / zr * h)
                    cv2.line(frame, (x0, y_zero), (x0 + w, y_zero), (150, 150, 150), 1)

            writer.write(frame)
            df_idx += 1

        frame_idx += 1

    cap.release()
    writer.release()

df = pd.read_csv(r"C:\Users\landgrafn\Desktop\FF_RI\87\2026-05-05_FF-PFC_87_RI_dff.csv")

path_in = r"C:\Users\landgrafn\Desktop\FF_RI\87\2026-05-05_FF-PFC_87_RI_edit_frames.mp4"
path_out = r"C:\Users\landgrafn\Desktop\FF_RI\87\2026-05-05_FF-PFC_87_RI_edit_frames_NE.mp4"

create_video_on_mp4(df, path_in, path_out)
