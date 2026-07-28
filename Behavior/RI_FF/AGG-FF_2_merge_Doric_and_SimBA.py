#%%
import h5py
import numpy as np
import pandas as pd
from scipy import signal, optimize, stats
from pathlib import Path

'''
In the same folder, place the raw .doric and sLEAP_DLC_SimBA file of one or many recordings, check that the length of the SimBA file makes sense
    1. extract the fluo, isos and IO data from the .doric file
    2. cleans the behavioral data
    3. merges the .doric and the SimBA file into one df, the doric is cropped to the behav
    4. calculates the dff and zscore from the cropped .doric recording
    5. saves the resulting df
'''

behavs = ["Investigate", "Following", "AnogenitalSniff", "Nose2nose", "Approach",
          "Agitated", "Chase", "Circle", "Mounting", "TailRattle",
          "Attack"]

path = r"E:\AGG-FF\Doric_Raw"

def get_files(path, common_name, print=False):

    # get files
    files = [file for file in Path(path).iterdir() if file.is_file() and common_name in file.name]
    if print:
        print(f'\n{len(files)} files found')
        for file in files:
            print(file)
        print('\n')

    return files


def get_Doric_1IO(file_doric):
    # do this when you have 1 IO-TTLs, e.g. IO2-Cam_TTL
    print('get Doric')
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

    def get_dff(df, denoising=True, bleaching_correct=True):

        time_sec = df.index.to_numpy(dtype=float)
        fluo_raw = df['Fluo'].to_numpy(dtype=float)
        isos_raw = df['Isos'].to_numpy(dtype=float)
        sampling_rate = (len(time_sec) - 1) / (time_sec[-1] - time_sec[0])

        # Denoising
        if denoising:
            # low-pass cutoff freq depends on indicator (2-10Hz for GCaMP6f)
            # use zero-phase filter that changes amplitude but not phase of freq components/ no signal distortion
            b, a = signal.butter(2, 10, btype='low', fs=sampling_rate)
            fluo_process = signal.filtfilt(b, a, fluo_raw)
            isos_process = signal.filtfilt(b, a, isos_raw)

        # Bleaching correction
        if bleaching_correct:
            def bleaching_double_exponential(t, const, amp_fast, amp_slow, tau_slow, tau_multiplier):
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
            def bleaching_get_parameters(time_sec, trace):
                max_sig = np.max(trace)
                initial_params = [max_sig/2, max_sig/4, max_sig/4, 3600, 0.1] # initial parmeters
                bounds = ([0, 0, 0, 600, 0], [max_sig, max_sig, max_sig, 36000, 1])
                trace_params, parm_conv = optimize.curve_fit(bleaching_double_exponential, time_sec, trace, p0=initial_params, 
                                                            bounds=bounds, maxfev=1000)
                trace_expfit = bleaching_double_exponential(time_sec, *trace_params)

                return trace_expfit
            fluo_expofit = bleaching_get_parameters(time_sec, fluo_process)
            fluo_process = fluo_process / fluo_expofit
            isos_expofit = bleaching_get_parameters(time_sec, isos_process)
            isos_process = isos_process / isos_expofit

        # Motion normalization
        slope, intercept, r_value, p_value, std_err = stats.linregress(x=isos_process, y=fluo_process)
        fluo_est_motion = intercept + slope * isos_process
        fluo_dff = (fluo_process - fluo_est_motion) / fluo_est_motion * 100
        df['dff'] = fluo_dff

        # Z-Score: baseline is mean of whole trace
        baseline = np.mean(fluo_dff)
        st_dev = np.std(fluo_dff, ddof=1)
        fluo_zscore = (fluo_dff - baseline) / (st_dev if st_dev > 0 else np.nan)
        df['zscore'] = fluo_zscore

        return df



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

    df_signal = get_dff(df_signal)

    return df_signal, io_hightimes

def get_SimBA(file_DLC, dist_bp='Center', fps_Cam=30, thresh_likelihood=0.2):
    print('get SimBA')
    def cleaning_raw_df(df):

        # create new column names from first two rows
        new_columns = df.columns
        new_columns = [("Resi_" + s) if "_1_" in s else s for s in new_columns]
        new_columns = [("Intr_" + s) if "_2_" in s else s for s in new_columns]
        new_columns = [s.replace('_1_', '_') for s in new_columns]
        new_columns = [s.replace('_2_', '_') for s in new_columns]
        df.columns = new_columns

        # set index and convert to numeric
        df.set_index('Unnamed: 0', inplace=True)
        df.index.names = ['frames']
        df = df.astype(float)
        df.index = df.index.astype(float).astype(int)

        bps_all         = ['Ear_left', 'Ear_right', 'Nose', 'Center', 'Lat_left', 'Lat_right', 'Tail_base', 'Tail_end']
        bps_head        = ['Resi_Ear_left', 'Resi_Ear_right', 'Resi_Nose', 'Intr_Ear_left', 'Intr_Ear_right', 'Intr_Nose']
        bps_postbody    = ['Resi_Center', 'Resi_Lat_left', 'Resi_Lat_right', 'Resi_Tail_base', 'Intr_Center', 'Intr_Lat_left', 'Intr_Lat_right', 'Intr_Tail_base']
        bps_all         = [("Resi_" + s) for s in bps_all] + [("Intr_" + s) for s in bps_all]
        
        # remove low-confidence points
        for bodypart in bps_all:
            filter = df[f'{bodypart}_p'] <= thresh_likelihood
            df.loc[filter, f'{bodypart}_x'] = np.nan
            df.loc[filter, f'{bodypart}_y'] = np.nan
            df = df.drop(columns=f'{bodypart}_p')
        
        # interpolate to skip nans
        #df = df.interpolate(method="linear")

        # mean 'trustworthy' bodyparts (ears, ear tips, eyes, midpoint, neck) to 'head' column
        new_cols = {}
        for animal in ('Resi', 'Intr'):
            for c in ('_x', '_y'):
                new_cols[f'{animal}_Head{c}'] = (df[[bp + c for bp in bps_head]].mean(axis=1, skipna=True))
                new_cols[f'{animal}_Postbody{c}'] = (df[[bp + c for bp in bps_postbody]].mean(axis=1, skipna=True))
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

        # smoothing along time via gaussian filter
        # for col in df.columns:
        #     df[col] = gaussian_filter1d(df[col].values, sigma=2, mode="nearest")

        return df
    
    def movement_parameters(df):

        new_cols = {}
        for anim in ('Resi', 'Intr'):

            dx = np.diff(df[f'{anim}_{dist_bp}_x'].to_numpy())
            dy = np.diff(df[f'{anim}_{dist_bp}_y'].to_numpy())
            dist = np.r_[0.0, np.sqrt(dx * dx + dy * dy)]

            new_cols[f'{anim}_Distance'] = dist
            new_cols[f'{anim}_Speed'] = dist * fps_Cam
            new_cols[f'{anim}_Distance_cum'] = np.cumsum(dist)

        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

        return df

    def correct_Intr_occurence(df, min_block_len_s=2, valid_Intr_range_s=(270, 930)):

        # if Intruder positions occur before or after valid_intr_range_s, make 
        # 1. Intruder positions NaN
        # 2. behaviors and probabilities to 0

        min_block_len_frames = int(min_block_len_s * fps_Cam)
        first_exp_Intr_frame, last_exp_Intr_frame = (int(point * fps_Cam) for point in valid_Intr_range_s)
        df_out = df.copy()

        # if more than 50% of all Intruder position columns are not NaN, count it as Intruder available
        intr_x_cols = [col for col in df_out.columns if col.startswith("Intr_") and col.endswith("_x")]
        intr_present = (df_out[intr_x_cols].notna().sum(axis=1) > (len(intr_x_cols) / 2))
        intr_frames = pd.Series(df_out.index[intr_present].astype(int)).sort_values(ignore_index=True)

        block_id = intr_frames.diff().ne(1).cumsum()
        valid_blocks = [block for _, block in intr_frames.groupby(block_id) if len(block) >= min_block_len_frames and block.iloc[0] >= first_exp_Intr_frame and block.iloc[-1] <= last_exp_Intr_frame]

        if len(valid_blocks) == 0:
            print("\nNo valid Intr block found.")
            print(f"Found Intr frames from {intr_frames.min()} to {intr_frames.max()}")
            return df_out

        first_valid_Intr_frame = valid_blocks[0].iloc[0]
        last_valid_Intr_frame = valid_blocks[-1].iloc[-1]

        # change frames outside the blocks
        outside_intr = ((df_out.index < first_valid_Intr_frame) | (df_out.index > last_valid_Intr_frame))
        cols_intr_posi = [c for c in df_out.columns if c.startswith("Intr_") and (c.endswith("_x") or c.endswith("_y"))]
        cols_behav = [c for c in df_out.columns if c.startswith("Probability_") or c in behavs]
        df_out.loc[outside_intr, cols_intr_posi] = np.nan
        df_out.loc[outside_intr, cols_behav] = 0

        return df_out

    df = pd.read_csv(file_DLC,low_memory=False)
    
    df = cleaning_raw_df(df)
    df = movement_parameters(df)
    df = correct_Intr_occurence(df)

    return df

def merge_Doric_SimBA(df_trace, df_SimBA, io_hightimes, closest_trace_values=3):
    print('merge Doric + SimBA')
    # set the correct index for df_trace
    hightimes = np.array([i[0] for i in io_hightimes])

    df_trace_crop = df_trace.copy()
    df_trace_crop.index = pd.Index(pd.to_numeric(df_trace_crop.index), name="time")
    df_trace_crop = df_trace_crop.sort_index()

    # match DLC length to available hightimes and use hightimes as DLC time index
    assert len(hightimes) == len(df_SimBA)
    n = min(len(hightimes), len(df_SimBA))
    hightimes = hightimes[:n]
    df_SimBA = df_SimBA.iloc[:n].copy()

    # add and set indices
    df_SimBA['behav_idx'] = np.arange(len(df_SimBA))
    df_SimBA['hightimes'] = hightimes.round(2)
    df_SimBA.index = pd.Index(hightimes, name="mastertime")

    # crop trace to DLC window
    window_start = float(df_SimBA.index.min())
    window_end   = float(df_SimBA.index.max())
    df_trace_crop = df_trace_crop.loc[(df_trace_crop.index >= window_start) & (df_trace_crop.index <= window_end)]
    trace_times = df_trace_crop.index.to_numpy(dtype=float)

    # averages the 3 closest df_trace valuee
    averaged_trace_rows = []
    closest_trace_indices = []
    for t in df_SimBA.index.to_numpy(dtype=float):
        closest_idx = np.argsort(np.abs(trace_times - t))[:closest_trace_values]
        avg_row = df_trace_crop.iloc[closest_idx].mean(axis=0)
        averaged_trace_rows.append(avg_row)
        closest_trace_indices.append(trace_times[closest_idx[0]])
    df_trace_avg = pd.DataFrame(averaged_trace_rows, index=df_SimBA.index)

    # merged df keeps DLC index
    df_merged = df_SimBA.join(df_trace_avg, how="left")

    # subtract first value to start at zero
    df_merged["trace_idx"] = closest_trace_indices
    df_merged.index = (df_merged.index - df_merged.index[0]).round(2)

    # reorder columns
    front_cols = ['hightimes', "behav_idx", "trace_idx", 'dff', 'zscore']
    other_cols = [c for c in df_merged.columns if c not in front_cols]
    df_merged = df_merged[front_cols + other_cols]
    
    return df_merged


files = get_files(path, '.doric')

for _, file_doric in enumerate(files):

    # get file names
    file_DLC = str(file_doric).replace('.doric', '_fps_sLEAP_check_DLC_SimBA.csv')
    file_short = "".join(Path(file_doric).stem.replace(word, "") for word in [''])
    print(f'\n----- {file_short} -----')

    # extract data
    df_signal, io_hightimes = get_Doric_1IO(file_doric)
    df_SimBA                = get_SimBA(file_DLC)
    main_df                 = merge_Doric_SimBA(df_signal, df_SimBA, io_hightimes)

    main_df.to_csv(f"E:\\AGG-FF\\Videos_Raw_fps_sLEAP_check_DLC_SimBA_Doric_new\\{file_short}_fps_sLEAP_check_DLC_SimBA_Doric.csv")
