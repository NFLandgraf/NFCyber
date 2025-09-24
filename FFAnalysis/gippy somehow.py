
#%%
# takes csv files and creates df/f etc 

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal as ss
from pathlib import Path
import os


path = r"C:\Users\landgrafn\Desktop\FFneg"
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

def FF_analyze(time_sec, fluo_raw, isos_raw):


    # function to compute deltaF/F using fitted control channel and filtered signal channel
    def deltaFF(signal, control):
        
        res = np.subtract(signal, control)
        normData = np.divide(res, control)
        #deltaFF = normData
        normData = normData*100

        return normData

    # function to fit control channel to signal channel
    def controlFit(control, signal):
        
        p = np.polyfit(control, signal, 1)
        arr = (p[0]*control)+p[1]
        return arr

    def filterSignal(filter_window, signal):
        if filter_window==0:
            return signal
        elif filter_window>1:
            b = np.divide(np.ones((filter_window,)), filter_window)
            a = 1
            filtered_signal = ss.filtfilt(b, a, signal)
            return filtered_signal
        else:
            raise Exception("Moving average filter window value is not correct.")

    filter_window = 4

    control_smooth = filterSignal(filter_window, isos_raw) #ss.filtfilt(b, a, control)
    signal_smooth = filterSignal(filter_window, fluo_raw)    #ss.filtfilt(b, a, signal)
    control_fit = controlFit(control_smooth, signal_smooth)
    norm_data = deltaFF(signal_smooth, control_fit)

    return norm_data





    # Peri-Event
    events = [30, 60, 90, 120, 150]
    df_all_events = pd.DataFrame()
    for ev in events:
        df_event = fluo_dff.copy()

        # reindex so the event is 0
        df_event.index = df_event.index - ev
        df_event.index = df_event.index.round(2)

        # crop the recording to the area around the event
        window = df_event.loc[-5 : 20]
        baseline = df_event.loc[-2 : -0.1]
        baseline_mean = baseline.mean()

        # normalize everything to the baseline_mean
        window_norm = window - baseline_mean
        df_all_events[f'{ev}'] = window_norm
    
    event_mean = df_all_events.mean(axis=1)

    return event_mean

all_animals = pd.DataFrame()

for file in files:
    file_name_short = manage_filename(file)
    print(file_name_short)

    df = pd.read_csv(file)
    df.index = df['Time']
    del df['Time']

    time_sec = df.index
    fluo_raw = df['Fluo']
    isos_raw = df['Isos']

    event_mean = FF_analyze(time_sec, fluo_raw, isos_raw)
    all_animals[file_name_short] = event_mean

all_animals.to_csv('negGFP_Fluodff.csv')


#%%

groups = ["mKate", "A53T", "GFP"]

# build mean and SEM DataFrames
mean_df = pd.DataFrame({f"{g}_Mean": all_animals.filter(like=g).mean(axis=1) for g in groups})
sem_df  = pd.DataFrame({f"{g}_SEM":  all_animals.filter(like=g).sem(axis=1)  for g in groups})

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
plt.title("divide, minus motion, mal 100")
plt.legend()
plt.tight_layout()
plt.show()



#%%

all_animals.plot(figsize=(10,6))
plt.xlabel("Index")
plt.ylabel("Value")
plt.title("All columns of DataFrame")
plt.legend()
plt.show()

