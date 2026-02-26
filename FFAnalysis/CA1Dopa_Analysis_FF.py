#%%
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal, optimize, stats
from pathlib import Path
import os
from scipy.ndimage import gaussian_filter1d


path = r"E:\FF-DA_YM\a"
file_useless_string = ['2024-11-19_FF-Weilin_YMaze_']#, '_FS']

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
files = get_files(path, '.doric')


#%%
def extract_signals_1IO(file_doric):
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

        digi_Cam    = np.array(f[path + 'DigitalIO/DIO02']).astype(int)
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

def extract_signals_2IO(file_doric):
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
        h5print(f, '')     # prints the whole structure
        path = 'DataAcquisition/NC500/Signals/Series0001/'

        # collect raw data from h5 file
        sig_isos  = np.array(f[path + 'LockInAOUT01/AIN01'])
        sig_fluo  = np.array(f[path + 'LockInAOUT02/AIN01'])
        sig_time  = np.array(f[path + 'LockInAOUT01/Time'])

        digi_Cam    = np.array(f[path + 'DigitalIO/DIO02']).astype(int)
        digi_Shock    = np.array(f[path + 'DigitalIO/DIO03']).astype(int)
        digi_time   = np.array(f[path + 'DigitalIO/Time'])

    # get signal parameters
    sig_time = np.round(sig_time, 2)
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

    # print(  f'Datapoints: Time {len(sig_time)}, Isos {len(sig_isos)}, Fluo {len(sig_fluo)}\n'
    #         f'Datapoints: digi_Cam {len(digi_Cam)}, digi_Shock {len(digi_Shock)}\n'
    #         f'Duration: {duration}s, SamplingRate: {sig_sampling_rate}Hz\n'
    #         f'digi_Cam: {len(events_digi_Cam)}x {[ev[0] for ev in events_digi_Cam]}\n'
    #         f'digi_Shock: {len(events_digi_Shock)}x {[ev[0] for ev in events_digi_Shock]}')
    
    return df, events_digi_Cam, events_digi_Shock

def get_movement(file_DLC, dist_bp='postbody', DLC_mm_per_px = 0.12, fps=30):
    # calculates the distance and speed between frames
    
    def cleaning_raw_df(file_DLC):

        df = pd.read_csv(file_DLC, header=None, low_memory=False)

        # create new column names from first two rows
        new_columns = [f"{col[0]}_{col[1]}" for col in zip(df.iloc[1], df.iloc[2])]
        df.columns = new_columns
        df = df.drop(labels=[0, 1, 2], axis="index")
        print(len(df))

        # set index and convert to numeric
        df.set_index('bodyparts_coords', inplace=True)
        df.index.names = ['frames']
        df = df.astype(float)
        df.index = df.index.astype(int)

        bps_all         = ['nose', 'left_ear', 'right_ear', 'left_ear_tip', 'right_ear_tip', 'left_eye', 'right_eye', 'head_midpoint', 
                            'neck', 'mid_back', 'mouse_center', 'mid_backend', 'mid_backend2', 'mid_backend3', 
                            'tail_base', 'tail1', 'tail2', 'tail3', 'tail4', 'tail5', 'tail_end',
                            'left_shoulder', 'left_midside', 'left_hip', 'right_shoulder', 'right_midside', 'right_hip']
        bps_head        = ['nose', 'left_eye', 'right_eye', 'head_midpoint', 'left_ear', 'right_ear', 'neck']
        bps_postbody    = ['mid_backend2', 'mid_backend3', 'tail_base']

        # remove low-confidence points
        for bodypart in bps_all:
            filter = df[f'{bodypart}_likelihood'] <= 0.9
            df.loc[filter, f'{bodypart}_x'] = np.nan
            df.loc[filter, f'{bodypart}_y'] = np.nan
            df = df.drop(columns=f'{bodypart}_likelihood')

        # interpolate to skip nans
        df = df.interpolate(method="linear")

        # mean 'trustworthy' bodyparts (ears, ear tips, eyes, midpoint, neck) to 'head' column
        for c in ('_x', '_y'):
            df[f'head{c}'] = df[[bp+c for bp in bps_head]].mean(axis=1, skipna=True)
            df[f'postbody{c}'] = df[[bp+c for bp in bps_postbody]].mean(axis=1, skipna=True)

        # smoothing along time via gaussian filter
        for col in df.columns:
            df[col] = gaussian_filter1d(df[col].values, sigma=2, mode="nearest")

        return df
    
    df = cleaning_raw_df(file_DLC)
    
    x = df[f'{dist_bp}_x'].to_numpy()
    y = df[f'{dist_bp}_y'].to_numpy()

    dx = np.diff(x)
    dy = np.diff(y)

    dist = np.sqrt(dx*dx + dy*dy)              # length = n_frames - 1
    dist = np.r_[0.0, dist]                    # prepend 0 for first frame (align length)
    dist = dist * DLC_mm_per_px
    df['Distance'] = dist
    df['Speed'] = dist * fps
        
    return df

def align_to_events(df, events, time_passed_pre=30, time_passed_post=30):

    # we have different amounts of time passed before the first event onset and after the last event onset. 
    # Therefore, delete anything time_passed_post s after last event and subtract time_passed_pre s before first event
    # time_passed_pre/_post is the number of seconds before/after the first/last event (do not take more than available)
    skip_time_pre   = round(events[0][0] - time_passed_pre, 2)
    skip_time_post  = round(events[-1][0] + time_passed_post, 2)

    # delete last
    df = df.loc[:skip_time_post]

    # delete first
    df.index = df.index - skip_time_pre
    df = df.loc[0:]
    df.index = df.index.round(2)

    return df

def dff(main_df, file_name_short):

    time_sec = main_df.index
    fluo_raw = main_df['Fluo']
    isos_raw = main_df['Isos']
    sampling_rate = (len(time_sec) - 1) / (time_sec[-1] - time_sec[0])

    # Denoising
    # low-pass cutoff freq depends on indicator (2-10Hz for GCaMP6f)
    # use zero-phase filter that changes amplitude but not phase of freq components/ no signal distortion
    b, a = signal.butter(2, 10, btype='low', fs=sampling_rate)
    fluo_denoised = signal.filtfilt(b, a, fluo_raw)
    isos_denoised = signal.filtfilt(b, a, isos_raw)

    # Motion
    slope, intercept, r_value, p_value, std_err = stats.linregress(x=isos_denoised, y=fluo_denoised)
    fluo_est_motion = intercept + slope * isos_denoised

    # Normalization
    fluo_dff = (fluo_denoised - fluo_est_motion) / fluo_est_motion * 100
    fluo_dff = pd.Series(fluo_dff, index=time_sec)

    all_animals_dff[file_name_short] = fluo_dff

    return fluo_dff

def perievent(df_dff, file_name_short):

    # takes one event at a time and gets the window around it
    events = [30, 60, 90, 120, 150]
    in_recording_all_events = pd.DataFrame()
    for ev in events:
        df_event = df_dff.copy()

        # reindex so the event is 0
        df_event.index = df_event.index - ev
        df_event.index = df_event.index.round(2)

        # crop the recording to the area around the event
        window = df_event.loc[-5 : 20]
        baseline = df_event.loc[-1 : -0.1]
        baseline_mean = baseline.mean()

        # normalize everything to the baseline_mean
        window_norm = window - baseline_mean
        all_animals_ind_events[f'{file_name_short}_{ev}'] = window_norm
        in_recording_all_events[f'{ev}'] = window_norm
    
    all_animals_event_mean[file_name_short] = in_recording_all_events.mean(axis=1)

def perievent_move(main_df, file_name_short, window=(-5, 20)):

    move = main_df['Distances']
    move = move.dropna()
    move.to_csv(f'{path}\{file_name_short}.csv')




    # print(move)
    # arr = np.arange(0.0, 180.0, 0.1)
    # move.index = arr
    # print(move)

    # # takes one event at a time and gets the window around it
    # events = [30, 60, 90, 120, 150]
    # win_start, win_end = window
    # in_recording_all_event_moves = pd.DataFrame()
    

    # # for each event, reindex so the event is 0 and crop around event
    # for ev in events:
    #     event_move = move.copy()
    #     event_move.index = (event_move.index - ev).round(1)
    #     window = event_move.loc[win_start : win_end]   

    #     in_recording_all_event_moves[f'{ev}'] = window
    all_animals_ind_event_moves[f'{file_name_short}'] = move
        



all_animals_dff = pd.DataFrame()
all_animals_ind_events = pd.DataFrame()
all_animals_event_mean = pd.DataFrame()
all_animals_ind_event_moves = pd.DataFrame()


for i, file_doric in enumerate(files):

    file_name_short = manage_filename(file_doric)
    print(f'----- {file_name_short} -----\n')

    file_DLC = str(file_doric).replace('.doric', '_DLC.csv')


    df_signal, df_io, io_hightimes = extract_signals_1IO(file_doric)
    #main_df, events_IO2, events_IO3         = extract_signals_2IO(file)

    df_DLC                                 = get_movement(file_DLC)
    #main_df                                 = align_to_events(main_df, events_IO3)  
    #df_dff                                  = dff(main_df, file_name_short)
    #perievent(df_dff, file_name_short)
    #perievent_move(main_df, file_name_short)
    

# all_animals_dff.to_csv(path + '\\all_animals_dff.csv')
# all_animals_ind_events.to_csv(path + '\\all_animals_ind_events.csv')
# all_animals_event_mean.to_csv(path + '\\all_animals_eventmean.csv')
# all_animals_ind_event_moves.to_csv(path + '\\all_animals_ind_event_moves.csv')

#%%

groups = ["mKate", "GFP", "A53T"]

# build mean and SEM DataFrames
mean_df = pd.DataFrame({f"{g}_Mean": all_animals_event_mean.filter(like=g).mean(axis=1) for g in groups})
sem_df  = pd.DataFrame({f"{g}_SEM":  all_animals_event_mean.filter(like=g).sem(axis=1)  for g in groups})

# plot
plt.figure(figsize=(8,5))
x = mean_df.index

for g in groups:
    mean_trace = mean_df[f"{g}_Mean"]
    sem_trace  = sem_df[f"{g}_SEM"]

    plt.plot(x, mean_trace, label=f"{g} Mean")
    plt.fill_between(x,
                     mean_trace - sem_trace,
                     mean_trace + sem_trace,
                     alpha=0.3)

plt.xlabel("Time [s]")
plt.ylabel("df/f")
plt.title("DA_Nomi_Pre")
plt.legend()
plt.tight_layout()
plt.show()

