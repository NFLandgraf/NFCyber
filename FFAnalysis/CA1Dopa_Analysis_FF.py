#%%
# takes csv files and creates df/f etc 

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal, optimize, stats
from pathlib import Path
import os


path = r"D:\FF_DA\Analysis\Raw"
file_useless_string = ['2024-11-20_FF-Weilin_FS_', '_LockIn']


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
files = get_files(path, 'csv')


#%%


def dff(time_sec, fluo_raw, isos_raw):

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

    return fluo_dff

def perievent(df_dff, file_name_short):

    # Peri-Event
    events = [30, 60, 90, 120, 150]
    df_all_events = pd.DataFrame()
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
        allanimals_events[f'{file_name_short}_{ev}'] = window_norm
        df_all_events[f'{ev}'] = window_norm

    return df_all_events
    


allanimals_dff = pd.DataFrame()
allanimals_events = pd.DataFrame()
allanimals_eventmean = pd.DataFrame()

for i, file in enumerate(files):

    file_name_short = manage_filename(file)
    print(file_name_short)

    df = pd.read_csv(file)
    time_sec = np.array(df['Time'])
    fluo_raw = df['Fluo']
    isos_raw = df['Isos']


    df_dff = dff(time_sec, fluo_raw, isos_raw)
    allanimals_dff[file_name_short] = df_dff

    df_all_events = perievent(df_dff, file_name_short)
    df_all_events.to_csv(f'{file_name_short}_events.csv')
    event_mean = df_all_events.mean(axis=1)
    allanimals_eventmean[file_name_short] = event_mean

allanimals_dff.to_csv('allanimals_dff.csv')
allanimals_events.to_csv('allanimals_events.csv')
allanimals_eventmean.to_csv('allanimals_eventmean.csv')


#%%

groups = ["mKate", "A53T", "GFP"]

# build mean and SEM DataFrames
mean_df = pd.DataFrame({f"{g}_Mean": allanimals_eventmean.filter(like=g).mean(axis=1) for g in groups})
sem_df  = pd.DataFrame({f"{g}_SEM":  allanimals_eventmean.filter(like=g).sem(axis=1)  for g in groups})

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

plt.xlabel("Time (frames or seconds)")
plt.ylabel("Signal")
plt.title("DA: no detrend, minus motion, divide by motion, mal 100")
plt.legend()
plt.tight_layout()
plt.show()

