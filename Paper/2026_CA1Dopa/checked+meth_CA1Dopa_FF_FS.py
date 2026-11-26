#%%
import scipy
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import numpy as np
import h5py
import os

'''
This code looks at the FF signal (.doric) (and DLC (.csv)) for Peri-event responses
1. From the .doric files, get Fluo, Isos, FS-Times and Cam-Times
2. From the DLC files, get the distances travelled
3. Merge the dfs according to the Cam-Times
4. Trim the resulting df before the first and after the last Shock-Time
5. Calculate the df/f from the trimmed time series
6. Do Peri-event analysis around each FS: subtract the baseline from trace
7. Calculate Peak and AUC per event, then average them and the traces for each animal
'''

path = r"E:\FF-DA_Nomi"
file_useless_string = ['CA1Dopa_FF-DA(3m)_2024-12-09_Nomifensine_', '']
include_DLC = False


def manage_filename(file):
    # managing file names
    file_name = os.path.basename(file)
    file_name_short = os.path.splitext(file_name)[0]
    for word in file_useless_string:
        file_name_short = file_name_short.replace(word, '')
    
    return file_name_short

def get_files(path, common_name, printing=False):

    files = [file for file in Path(path).iterdir() if file.is_file() and common_name in file.name]
    
    if printing:
        print(f'\n{len(files)} files found')
        for file in files:
            print(file)
        print('\n')

    return files


def get_Doric_2IO(file_doric, io_cam_fs=(2,3)):
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
                high_times.append([round(t, 2) for t in curr_high_time])
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

    io_channel_cam, io_channel_fs = io_cam_fs

    with h5py.File(file_doric, 'r') as f:
        #h5print(f, '')     # prints the whole structure
        path = 'DataAcquisition/NC500/Signals/Series0001/'

        # collect raw data from h5 file
        sig_isos  = np.array(f[path + 'LockInAOUT01/AIN01'])
        sig_fluo  = np.array(f[path + 'LockInAOUT02/AIN01'])
        sig_time  = np.array(f[path + 'LockInAOUT01/Time'])

        digi_Cam    = np.array(f[path + f'DigitalIO/DIO0{io_channel_cam}']).astype(int)
        digi_Shock  = np.array(f[path + f'DigitalIO/DIO0{io_channel_fs}']).astype(int)
        digi_time   = np.array(f[path + 'DigitalIO/Time'])

    # get signal parameters
    sig  = pd.DataFrame({'Isos': sig_isos, 'Fluo': sig_fluo},             index=pd.Index(np.round(sig_time, 3), name='Time'))
    digi = pd.DataFrame({'digi_Cam': digi_Cam, 'digi_Shock': digi_Shock}, index=pd.Index(digi_time, name='Time'))

    # get the event timepoints
    events_digi_Cam   = event_list(digi, 'digi_Cam')
    events_digi_Shock = [event[0] for event in event_list(digi, 'digi_Shock')]

    # Prepare digital channels for the merged table
    digi = digi.iloc[0::10]                                               # downsample to take every 10th frame -> one datapoint every ms
    digi.index = pd.Index(np.round(digi.index.to_numpy(dtype=float), 3))
    digi = extend_high_signal(digi, 'digi_Cam', 6)                        # increase the duration to see every TTL at 100Hz

    # Merge and replace missing digital values with zero
    df = (pd.concat([sig, digi], axis=1).rename_axis('Time')[['Fluo', 'Isos', 'digi_Cam', 'digi_Shock']].dropna(subset=['Fluo', 'Isos']))
    cols = ['digi_Cam', 'digi_Shock']
    df[cols] = df[cols].fillna(0).astype(int)


    if False:
        print(  f'Datapoints: Time {len(sig_time)}, Isos {len(sig_isos)}, Fluo {len(sig_fluo)}\n'
                f'Datapoints: digi_Cam {len(digi_Cam)}, digi_Shock {len(digi_Shock)}\n'
                f'Duration: {max(sig_time)}s, SamplingRate: {int(len(sig_isos) / max(sig_time))}Hz\n'
                f'digi_Cam: {len(events_digi_Cam)}x {[ev[0] for ev in events_digi_Cam]}\n'
                f'digi_Shock: {len(events_digi_Shock)}x {[ev[0] for ev in events_digi_Shock]}')
    
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
        if pd.notna(df.at[i, x_col]) and pd.notna(df.at[i, y_col]):
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

def dff(df):

    # for the FootShock stuff, dont use bleaching_correction
    time_sec = df.index.to_numpy(dtype=float)
    fluo_raw = df['Fluo'].to_numpy(dtype=float)
    isos_raw = df['Isos'].to_numpy(dtype=float)
    sampling_rate = (len(time_sec) - 1) / (time_sec[-1] - time_sec[0])

    # Denoising: low-pass cutoff freq depends on indicator (2-10Hz for GCaMP6f); use zero-phase filter that changes amplitude but not phase of freq components/ no signal distortion
    b, a = scipy.signal.butter(2, 10, btype='low', fs=sampling_rate)
    fluo_process = scipy.signal.filtfilt(b, a, fluo_raw)
    isos_process = scipy.signal.filtfilt(b, a, isos_raw)

    # Motion-correction
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x=isos_process, y=fluo_process)
    fluo_est_motion = intercept + slope * isos_process

    # Normalization
    fluo_dff = (fluo_process - fluo_est_motion) / fluo_est_motion * 100
    baseline = np.mean(fluo_dff)
    st_dev = np.std(fluo_dff, ddof=1)
    fluo_zscore = (fluo_dff - baseline) / (st_dev if st_dev > 0 else np.nan)

    # merge dff and zscore and distances into one df
    if include_DLC:
        trace = pd.DataFrame({"dff":fluo_dff, "zscore":fluo_zscore, 'Distances':df['Distances'].to_numpy(dtype=float)}, index=time_sec)
    else:
        trace = pd.DataFrame({"dff":fluo_dff, "zscore":fluo_zscore}, index=time_sec)

    return trace

def perievent(df, events, file_name_short, window_trace=(-5,10), window_baseline=(-1,-0.1)):

    animal_trials = []
    event_maxima  = []
    event_aucs    = []

    for event_number, event_time in enumerate(events):

        # everything for the FF signal
        relative_time = df['dff'].index.to_numpy() - event_time
        trial = pd.Series(df['dff'].to_numpy(), index=relative_time, name=f"trial_{event_number}")

        trial    = trial.loc[window_trace[0]:window_trace[1]]
        baseline = trial.loc[window_baseline[0]:window_baseline[1]]

        trial_norm       = trial - baseline.mean()
        trial_norm.index = pd.Index(np.round(trial_norm.index.to_numpy(dtype=float), 3), name=trial_norm.index.name)

        # add trace to the summary
        all_animals_events_ind[f"{file_name_short}_FS_{event_number}"] = trial_norm
        animal_trials.append(trial_norm)

        # event maximum
        response = trial_norm.loc[0:10]
        event_max = response.max(skipna=True)
        event_maxima.append(event_max)

        # positive AUC
        response_positive = response.clip(lower=0)
        event_auc = np.trapezoid(response_positive.to_numpy(dtype=float),x=response_positive.index.to_numpy(dtype=float))
        event_aucs.append(event_auc)

        # everything for the DLC signal
        if include_DLC:
            relative_time = df['Distances'].index.to_numpy() - event_time
            trial = pd.Series(df['Distances'].to_numpy(), index=relative_time, name=f"trial_{event_number}")
            trial = trial.loc[window_trace[0]:window_trace[1]]
            trial.index = pd.Index(np.round(trial.index.to_numpy(dtype=float), 3), name=trial.index.name)
            all_animals_events_moves_ind[f"{file_name_short}_FS_{event_number}"] = trial


    animal_trials_df = pd.concat(animal_trials, axis=1)
    all_animals_events_mean[file_name_short] = (animal_trials_df.mean(axis=1))
    all_animals_events_max_auc.loc[file_name_short] = {"file": file_name_short, "mean_event_max": np.mean(event_maxima), "mean_event_auc": np.mean(event_aucs)}


def plot_perievent(df):

    groups = ["mKate", "GFP", "A53T"]
    mean_df = pd.DataFrame({f"{g}_Mean": df.filter(like=g).mean(axis=1) for g in groups})
    sem_df  = pd.DataFrame({f"{g}_SEM":  df.filter(like=g).sem(axis=1)  for g in groups})

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
    plt.title('Perievent response')
    plt.legend()
    plt.tight_layout()
    plt.show()

def ttest(df, col):

    data = df[['file', col]].copy()
    data['Group'] = data['file'].astype(str).apply(lambda name: next((group for group in ['mKate', 'A53T', 'GFP'] if group in name),None))
    data[col] = pd.to_numeric(data[col], errors='coerce')

    fig, ax = plt.subplots(figsize=(5, 5))
    values_by_group = {}

    for x, group in enumerate(groups):
        values = data.loc[data['Group'] == group, col].dropna().to_numpy()
        values_by_group[group] = values

        if not len(values):
            continue

        jitter = np.linspace(-0.06, 0.06, len(values)) if len(values) > 1 else 0
        ax.scatter(np.full(len(values), x) + jitter, values,color=colors[group], s=60, zorder=3)
        ax.errorbar(x, values.mean(),yerr=scipy.stats.sem(values) if len(values) > 1 else None,fmt='_', color='black', markersize=25, capsize=5)

    # t-test
    a, b = values_by_group['mKate'], values_by_group['A53T']
    _, p = scipy.stats.ttest_ind(a, b, equal_var=False)
    ax.text(0.5, 0.95, f'mKate vs A53T: p={p:.4g}',transform=ax.transAxes, ha='left', va='top')

    ax.set_xticks(range(len(groups)),[g for g in groups])
    ax.set_ylabel(col)
    ax.set_xlim(-0.5, 2.5)
    ax.margins(y=0.25)
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    plt.show()

    return p

groups = ['mKate', 'A53T', 'GFP']
colors = {'mKate': 'gray', 'A53T': 'red', 'GFP': 'green'}

all_animals_dff              = pd.DataFrame()
all_animals_events_ind       = pd.DataFrame()
all_animals_events_mean      = pd.DataFrame()
all_animals_events_moves_ind = pd.DataFrame()
all_animals_events_max_auc   = pd.DataFrame(columns=["file", "mean_event_max", "mean_event_auc"])


files = get_files(path, '.doric')

for file_doric in files:

    # get the filenames
    file_DLC = str(file_doric).replace('.doric', '_DLC.csv')
    file_name_short = manage_filename(file_doric)
    print(f'----- {file_name_short} -----')

    # Main
    main_df, events_shock, events_cam = get_Doric_2IO(file_doric)

    if include_DLC:
        df_DLC                        = get_DLC(file_DLC)
        main_df                       = merge_Doric_DLC(main_df, df_DLC)
    
    main_df, events_shock             = trim_trace(main_df, events_shock) 
    main_df                           = dff(main_df)


    perievent(main_df, events_shock, file_name_short)


if True:
    plot_perievent(all_animals_events_mean)
    ttest(all_animals_events_max_auc, 'mean_event_max')
    ttest(all_animals_events_max_auc, 'mean_event_auc')

if False:
    all_animals_dff.to_csv(             path + '\\AllAnimals_Dff.csv')
    all_animals_events_ind.to_csv(      path + '\\AllAnimals_Events_Ind.csv')
    all_animals_events_mean.to_csv(     path + '\\AllAnimals_Events_Mean.csv')
    all_animals_events_moves_ind.to_csv(path + '\\AllAnimals_Events_MovesInd.csv')
    all_animals_events_max_auc.to_csv(  path + '\\AllAnimals_Events_MaxAUC.csv', index=False)

