
#%%
# takes csv files and creates df/f etc 

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal, optimize, stats
from pathlib import Path
import os



path = "C:\\Users\\landgrafn\\Desktop\\FF-DA\\try\\"
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

    sampling_rate = (len(time_sec) - 1) / (time_sec[-1] - time_sec[0])

    # Denoising
    # low-pass cutoff freq depends on indicator (2-10Hz for GCaMP6f)
    # use zero-phase filter that changes amplitude but not phase of freq components/ no signal distortion
    b, a = signal.butter(2, 10, btype='low', fs=sampling_rate)
    fluo_denoised = signal.filtfilt(b, a, fluo_raw)
    isos_denoised = signal.filtfilt(b, a, isos_raw)


    # Photobleaching Correction
    max_sig = np.max(fluo_denoised)
    initial_params = [max_sig/2, max_sig/4, max_sig/4, 3600, 0.1]
    bounds = ([0, 0, 0, 600, 0], [max_sig, max_sig, max_sig, 36000, 1])
    fluo_params, parm_conv = optimize.curve_fit(double_exponential, time_sec, fluo_denoised, p0=initial_params, bounds=bounds, maxfev=1000)
    fluo_expfit = double_exponential(time_sec, *fluo_params)

    max_sig = np.max(isos_denoised)
    initial_params = [max_sig/2, max_sig/4, max_sig/4, 3600, 0.1]
    bounds = ([0, 0, 0, 600, 0], [max_sig, max_sig, max_sig, 36000, 1])
    isos_params, parm_conv = optimize.curve_fit(double_exponential, time_sec, isos_denoised, p0=initial_params, bounds=bounds, maxfev=1000)
    isos_expfit = double_exponential(time_sec, *isos_params)

    # If bleaching results from a reduction of autofluorescence, signal amplitude should not be affected
    # --> subtract function from trace (signal is V)
    # If bleaching comes from bleaching of the fluorophore, the amplitude suffers as well 
    # --> division is needed (signal is df/f)
    fluo_detrend = fluo_denoised / fluo_expfit
    isos_detrend = isos_denoised / isos_expfit


    # Motion Correction
    slope, intercept, r_value, p_value, std_err = stats.linregress(x=isos_detrend, y=fluo_detrend)
    fluo_est_motion = intercept + slope * isos_detrend   # fit Isos to Fluo
    fluo_corrected = fluo_detrend - fluo_est_motion


    # Normalization
    fluo_dff = 100 * fluo_corrected
    fluo_dff = pd.Series(fluo_dff, index=time_sec)

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
    
all_animals.to_csv('all_animals.csv')




#%%

def get_data_csv(file):

    df = pd.read_csv(file)
    df.index = df['Time']
    del df['Time']

    return df

def main(files):

    df_base = pd.DataFrame()
    for file in files:

        file_name_short = manage_filename(file)
        print(file_name_short)


        # preprocess data
        df = get_data_csv(file)
        df = old_preprocess(df)


        df = df['Fluo_dff']
        df = df.rename(f'{file_name_short}_Fluo_dff', inplace=True)

        df_base[file_name_short] = df
        print(f'{file_name_short} added as new column to df_base')


    #df_base.to_csv(path + 'Fluo_corrected_allfiles.csv')

    return df_base



df_base = main(files)


print(df_base)



#%%

# Mean and align everything to events
# You give events as input (e.g. FS) and for each animal, it means the signal around each event, therefore means the animal response to the event

def perievent(df, events, window_pre=5, window_post=20, baseline_pre=2):
    df_perievent_means = pd.DataFrame()

    # go through each column in df_base
    for column in df_base:

        df = df_base[column]
        df_all_events = pd.DataFrame()

        for ev in events:
            df_event = df.copy()

            # reindex so the event is 0
            df_event.index = df_event.index - ev
            df_event.index = df_event.index.round(2)

            # crop the recording to the time around the event
            df_event = df_event.loc[-window_pre : window_post]

            # get the mean of the Xs baseline before the event
            baseline_mean = df_event.loc[-baseline_pre : 0].mean()

            # normalize everything to the baseline_mean
            df_event = df_event - baseline_mean
            df_all_events[f'{ev}'] = df_event

        # add the row means to the main df
        df_all_events['mean'] = df_all_events.mean(axis=1)
        df_perievent_means[f'{column}'] = df_all_events['mean']

    df_perievent_means.to_csv(path + "Perievent_means.csv")
    print('Done')

    return df_perievent_means


events = [30, 60, 90, 120, 150]
df_perievent_means = perievent(df_base, events)