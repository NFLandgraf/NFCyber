#%%
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal, optimize, stats
from pathlib import Path
import os
from scipy.ndimage import gaussian_filter1d

'''
This code looks at the FF signal and DLC for Peri-event responses
1. From the .doric files, get Fluo, Isos, Shock-Times and Cam-Times
2. From the DLC files, get the distances travelled
3. Merge the dfs according to the Cam-Times
4. Trim the resulting df before the first and after the last Shock-Time
5. Calculate the df/f from the trimmed time series
6. Do Peri-event analysis around each Shock, then average them for each animal and save it
'''

path = r"Y:\_proj_CA1Dopa\CA1Dopa_FF-NE\CA1Dopa_FF-NE(3m)_2025-08-07_FS\CA1Dopa_FF-NE(3m)_2025-08-07_FS_Data_Doric"
file_useless_string = ['CA1Dopa_FF-NE(3m)_2025-08-07_FS_', '']

def manage_filename(file):

    # managing file names
    file_name = os.path.basename(file)
    file_name_short = os.path.splitext(file_name)[0]
    for word in file_useless_string:
        file_name_short = file_name_short.replace(word, '')
    
    return file_name_short
def get_files(path, common_name):

    # get files
    files = [file for file in Path(path).iterdir() if file.is_file() and common_name in file.name]
    
    print(f'\n{len(files)} files found')
    for file in files:
        print(file)
    print('\n')

    return files


def get_Doric_2IO(file_doric):
    # do this when you have 2 IO-TTLs, e.g. IO2-Cam_TTL and IO3-FCbox

    def h5print(item, leading=''):
        # prints out the h5 data structure
        for key in item:
            if isinstance(item[key], h5py.Dataset):
                print(leading + key + ':' + str(item[key].shape))
            else:
                print(leading + key)
                h5print(item[key], leading + '  ')
    def event_list(df, channel_name, value_threshold=0.9):
        # returns a list with the onset (and offset) timepoints, where certain channel is 'high' 
        
        # iterate through rows and save on- and offsets
        high_times, curr_high_time = [], []
        value_low = True

        for idx, value in df[channel_name].items():
            if value_low and value >= value_threshold:
                curr_high_time.append(idx)
                value_low = False
            elif not value_low and value <= value_threshold:
                curr_high_time.append(idx)
                high_times.append(curr_high_time)
                curr_high_time = []
                value_low = True

        return high_times
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

    with h5py.File(file_doric, 'r') as f:
        #h5print(f, '')     # prints the whole structure
        path = 'DataAcquisition/NC500/Signals/Series0001/'

        # collect raw data from h5 file
        sig_isos  = np.array(f[path + 'LockInAOUT01/AIN01'])
        sig_fluo  = np.array(f[path + 'LockInAOUT02/AIN01'])
        sig_time  = np.array(f[path + 'LockInAOUT01/Time'])

        digi_Cam    = np.array(f[path + 'DigitalIO/DIO05']).astype(int)
        digi_Shock  = np.array(f[path + 'DigitalIO/DIO06']).astype(int)
        digi_time   = np.array(f[path + 'DigitalIO/Time'])

    # get signal parameters
    sig_time = np.round(sig_time, 3)
    duration = max(sig_time)
    sig_sampling_rate = int(len(sig_isos) / duration)

    # create frames
    signal = pd.DataFrame({'Isos': sig_isos, 'Fluo': sig_fluo}, index=pd.Index(sig_time, name='Time'))
    io = pd.DataFrame({'digi_Cam': digi_Cam, 'digi_Shock': digi_Shock}, index=pd.Index(digi_time, name='Time'))

    # get IO parameters
    io = io.iloc[0::10]      # downsample to take every 10th frame -> one datapoint every ms
    io.index = io.index.round(3)
    io = extend_high_signal(io, 'digi_Cam', 6)   # increase the duration to see every TTL at 100Hz

    # fuse Fluo, Isos and IO datapoints (somehow signal starts at 0.1 and not 0) and delete rows with nans
    df = pd.concat([signal, io], axis=1)
    df.index.names = ['Time']
    df = df[['Fluo', 'Isos', 'digi_Cam', 'digi_Shock']]
    df = df.dropna(subset=['Fluo', 'Isos'])
    df[['digi_Cam','digi_Shock']] = df[['digi_Cam','digi_Shock']].fillna(0).astype(int)

    # get TTL events from a digital channel
    events_digi_Cam =   event_list(df, 'digi_Cam')
    events_digi_Shock = event_list(df, 'digi_Shock')
    events_digi_Shock = [i[0] for i in events_digi_Shock]

    # print(  f'Datapoints: Time {len(sig_time)}, Isos {len(sig_isos)}, Fluo {len(sig_fluo)}\n'
    #         f'Datapoints: digi_Cam {len(digi_Cam)}, digi_Shock {len(digi_Shock)}\n'
    #         f'Duration: {duration}s, SamplingRate: {sig_sampling_rate}Hz\n'
    #         f'digi_Cam: {len(events_digi_Cam)}x {[ev[0] for ev in events_digi_Cam]}\n'
    #         f'digi_Shock: {len(events_digi_Shock)}x {[ev[0] for ev in events_digi_Shock]}')
    
    return df, events_digi_Shock, events_digi_Cam

def get_DLC(file_DLC, mm_per_px=1, bps_position='mid_backend', min_distance=0.001, likelihood=0.2):

    # calculates the distance travelled between frames
    def cleaning_raw_df(csv_file):

        df = pd.read_csv(csv_file, header=None, low_memory=False)

        # create new column names from first two rows
        new_columns = [f"{col[0]}_{col[1]}" for col in zip(df.iloc[1], df.iloc[2])]
        df.columns = new_columns
        df = df.drop(labels=[0, 1, 2], axis="index")

        # set index and convert to numeric
        df.set_index('bodyparts_coords', inplace=True)
        df.index.names = ['frames']
        df = df.astype(float)
        df.index = df.index.astype(int)

        bps_all = ['nose', 'left_ear', 'right_ear', 'left_ear_tip', 'right_ear_tip', 'left_eye', 'right_eye', 'head_midpoint', 
                 'neck', 'mid_back', 'mouse_center', 'mid_backend', 'mid_backend2', 'mid_backend3', 
                 'tail_base', 'tail1', 'tail2', 'tail3', 'tail4', 'tail5', 'tail_end',
                 'left_shoulder', 'left_midside', 'left_hip', 'right_shoulder', 'right_midside', 'right_hip']

        # remove low-confidence points
        for bodypart in bps_all:
            filter = df[f'{bodypart}_likelihood'] <= likelihood
            df.loc[filter, f'{bodypart}_x'] = np.nan
            df.loc[filter, f'{bodypart}_y'] = np.nan
            df = df.drop(columns=f'{bodypart}_likelihood')

        return df
    
    def euclidian_dist(point1x, point1y, point2x, point2y):
        # calculates distance between 2 points in carthesic coordinate system
        return np.sqrt((point1x - point2x) ** 2 + (point1y - point2y) ** 2)

    df = cleaning_raw_df(file_DLC)

    x_col, y_col = f'{bps_position}_x', f'{bps_position}_y'
    
    # creates a full index range (to include missing frames), fill Series with NaNs
    full_index = np.arange(df.index.min(), df.index.max() + 1)
    dist_series = pd.Series(index=full_index, dtype=float)

    # goes though all frames/rows and calcs distance between the frames with values (not NaN)
    last_valid_idx = None
    for i in df.index:
        if not np.isnan(df.at[i, x_col]) and not np.isnan(df.at[i, y_col]):
            if last_valid_idx is not None:
                dist = euclidian_dist(df.at[last_valid_idx, x_col], df.at[last_valid_idx, y_col], df.at[i, x_col], df.at[i, y_col])
                dist_series[i] = dist if dist >= min_distance else np.nan  
            last_valid_idx = i
    dist_travelled = round(dist_series.sum(), 1)

    # goes through the frames and distributes the distance evenly across NaNs
    nan_indices = np.where(dist_series.isna())[0]
    if len(nan_indices) > 0:
        last_valid_idx = None
        for i in range(len(dist_series)):
            if not np.isnan(dist_series[i]):  
                if last_valid_idx is not None:
                    num_missing = i - last_valid_idx - 1  # Frames between valid distances
                    if num_missing > 0:
                        total_dist = dist_series[i]  # Distance to be spread
                        per_frame_dist = total_dist / (num_missing + 1)
                        for j in range(1, num_missing + 2):
                            dist_series[last_valid_idx + j] = per_frame_dist
                last_valid_idx = i
    dist_series[0] = 0.0
    dist_travelled_distr = round(dist_series.sum(), 1)

    # sanity check
    if dist_travelled != dist_travelled_distr:
        print(f'Error: different distances calculated: {dist_travelled} vs {dist_travelled_distr}')
    
    # convert to mm and to DataFrame
    dist_series = dist_series * mm_per_px
    dist_df = dist_series.to_frame(name='Distances')

    return dist_df

def merge_Doric_DLC(df_trace, df_DLC):

    df_merged = df_trace.copy()

    # Ensure the trace index is numeric and sorted
    df_merged.index = pd.Index(pd.to_numeric(df_merged.index), name=df_merged.index.name)
    df_merged = df_merged.sort_index()

    # Create empty Distances column
    df_merged["Distances"] = np.nan

    # Rows corresponding to DLC frames
    cam_mask = df_merged["digi_Cam"].eq(1)
    cam_indices = df_merged.index[cam_mask]

    # Distance values from DLC
    distances = df_DLC["Distances"].to_numpy()

    # Only use the number of values available in both DataFrames
    n = min(len(cam_indices), len(distances))

    # Assign DLC distances to the corresponding camera rows
    df_merged.loc[cam_indices[:n], "Distances"] = distances[:n]

    return df_merged


def trim_trace(df, events, time_passed_pre=30, time_passed_post=30):

    # checks that the events are correct
    if len(events) != 5:
        print(f"Warning: expected 5 shocks, detected {len(events)}")
    if not np.allclose(np.diff(events), np.diff(events)[0], atol=0.1):
        print(f"Warning: Shock intervals are not constant: {np.diff(events)}")

    # trims the recording to time_passed_pre before event[0] and time_passed_post after event[-1]
    skip_time_pre   = round(events[0] - time_passed_pre, 3)
    skip_time_post  = round(events[-1] + time_passed_post, 3)

    # delete last
    df = df.loc[:skip_time_post]

    # delete first
    df.index = df.index - skip_time_pre
    df = df.loc[0:]
    df.index = df.index.round(3)

    # subtract the skip_time_pre from the events to have the events correctly
    events = [round(ev - skip_time_pre, 3) for ev in events]

    return df, events

def dff(df, bleaching_correct=False, del_values=False, move=False):

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

    # trace cleaning
    if del_values:
        fluo_dff[fluo_dff < -1.0] = np.nan
        fluo_dff[fluo_dff > 1.0] = np.nan

    # Z-Score: baseline is mean of whole trace
    baseline = np.mean(fluo_dff)
    st_dev = np.std(fluo_dff, ddof=1)
    fluo_zscore = (fluo_dff - baseline) / (st_dev if st_dev > 0 else np.nan)

    # merge dff and zscore and distances into one df
    if move:
        trace = pd.DataFrame({"dff":fluo_dff, "zscore":fluo_zscore, 'Distances':df['Distances'].to_numpy(dtype=float)}, index=time_sec)
    else:
        trace = pd.DataFrame({"dff":fluo_dff, "zscore":fluo_zscore}, index=time_sec)

    # trim animal trace if longer than the summary df
    # if len(all_animals_dff) >= 0:
    #     target_len = len(all_animals_dff.index)
    #     print(target_len)
    #     if len(fluo_dff) > target_len:
    #         print('yo')
    #         fluo_dff = fluo_dff[:target_len]
    #         fluo_zscore = fluo_zscore[:target_len]
    # all_animals_dff[file_name_short] = fluo_dff

    return trace

def perievent(df, events, file_name_short, trace_window=(-5, 10), baseline_window=(-1, -0.1), move=False):

    # takes one event at a time, gets the window around it and subtracts the bsas
    df_dff = df['dff']
    animal_trials = []
    event_maxima = []
    event_aucs = []

    if move:
        df_move = df['Distances']

    for event_number, event_time in enumerate(events):

        # everything for the FF signal
        relative_time = df_dff.index.to_numpy() - event_time
        trial = pd.Series(df_dff.to_numpy(), index=relative_time, name=f"trial_{event_number}")

        trial = trial.loc[trace_window[0]:trace_window[1]]
        baseline = trial.loc[baseline_window[0]:baseline_window[1]]

        trial_normalized = trial - baseline.mean()
        trial_normalized.index = trial_normalized.index.round(3)

        # add trace to the summary
        all_animals_ind_events[f"{file_name_short}_FS_{event_number}"] = trial_normalized
        animal_trials.append(trial_normalized)

        # event maximum
        response = trial_normalized.loc[0:10]
        event_max = response.max(skipna=True)
        event_maxima.append(event_max)

        # positive AUC
        response_positive = response.clip(lower=0)
        event_auc = np.trapezoid(response_positive.values, response_positive.index.values)
        event_aucs.append(event_auc)


        # everything for the DLC signal
        if move:
            relative_time = df_move.index.to_numpy() - event_time
            trial = pd.Series(df_move.to_numpy(), index=relative_time, name=f"trial_{event_number}")
            trial = trial.loc[trace_window[0]:trace_window[1]]
            trial.index = trial.index.round(3)
            all_animals_ind_event_moves[f"{file_name_short}_FS_{event_number}"] = trial


    animal_trials_df = pd.concat(animal_trials, axis=1)
    all_animals_event_mean[file_name_short] = (animal_trials_df.mean(axis=1))

    # mean event maximum and positive AUC
    all_animals_event_max_auc.loc[file_name_short] = {"file": file_name_short, "mean_event_max": np.mean(event_maxima), "mean_event_auc": np.mean(event_aucs)}


def draw_perievent_groups(all_animals_event_mean, title):

    groups = ["mKate", "GFP", "A53T"]
    mean_df = pd.DataFrame({f"{g}_Mean": all_animals_event_mean.filter(like=g).mean(axis=1) for g in groups})
    sem_df  = pd.DataFrame({f"{g}_SEM":  all_animals_event_mean.filter(like=g).sem(axis=1)  for g in groups})

    plt.figure(figsize=(8,5))
    x = mean_df.index
    col = ['gray', 'green', 'red']
    for i, g in enumerate(groups):
        mean_trace = mean_df[f"{g}_Mean"]
        sem_trace  = sem_df[f"{g}_SEM"]
        plt.plot(x, mean_trace, label=f"{g} Mean", color=col[i])
        plt.fill_between(x, mean_trace-sem_trace, mean_trace+sem_trace, alpha=0.3, color=col[i])
    plt.xlabel("Time [s]")
    plt.ylabel("df/f")
    plt.title(title)
    plt.ylim(-0.5, 2.0)
    plt.legend()
    plt.tight_layout()
    plt.show()


files = get_files(path, '.doric')

all_animals_dff = pd.DataFrame()
all_animals_ind_events = pd.DataFrame()
all_animals_event_mean = pd.DataFrame()
all_animals_ind_event_moves = pd.DataFrame()
all_animals_event_max_auc = pd.DataFrame(columns=["file", "mean_event_max", "mean_event_auc"])

for file_doric in files:

    # get the filenames
    file_DLC = str(file_doric).replace('.doric', '_DLC.csv')
    file_name_short = manage_filename(file_doric)
    print(f'----- {file_name_short} -----')

    # Main
    main_df, events_shock, events_cam   = get_Doric_2IO(file_doric)
    #df_DLC                              = get_DLC(file_DLC)
    #main_df                             = merge_Doric_DLC(main_df, df_DLC)
    
    main_df, events_shock               = trim_trace(main_df, events_shock) 
    main_df                             = dff(main_df)
    perievent(main_df, events_shock, file_name_short)


draw_perievent_groups(all_animals_event_mean, 'FF-DA_FS1')

if True:
    all_animals_dff.to_csv(path + '\\AllAnimalsDff.csv')
    all_animals_ind_events.to_csv(path + '\\AllAnimalsIndEvents.csv')
    all_animals_event_mean.to_csv(path + '\\AllAnimalsEventMean.csv')
    all_animals_ind_event_moves.to_csv(path + '\\AllAnimalsIndEventMoves.csv')
    all_animals_event_max_auc.to_csv(path + '\\AllAnimalsEventMaxAUC.csv', index=False)

