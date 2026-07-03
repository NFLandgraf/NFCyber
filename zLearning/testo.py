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


path = r"C:\Users\landgrafn\Desktop\FF-PFC"
file_useless_strings = []


def manage_filename(file):
    name = Path(file).stem
    return name
def get_files(path, common_name):

    # get files
    files = [file for file in Path(path).iterdir() if file.is_file() and common_name in file.name]
    
    print(f'\n{len(files)} files found')
    for file in files:
        print(file)
    print('\n')

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
    
    with h5py.File(file_doric, 'r') as f:
        #h5print(f, '')     # prints the whole structure
        path = 'DataAcquisition/NC500/Signals/Series0001/'

        # collect raw data from h5 file
        sig_isos  = np.array(f[path + 'LockInAOUT01/AIN01'])
        sig_fluo  = np.array(f[path + 'LockInAOUT02/AIN01'])
        sig_time  = np.array(f[path + 'LockInAOUT01/Time'])


    # get signal parameters
    sig_time = np.round(sig_time, 2)
    duration = max(sig_time)
    sig_sampling_rate = int(len(sig_isos) / duration)

    # create frames
    df_signal  = pd.DataFrame({'Isos': sig_isos, 'Fluo': sig_fluo}, index=pd.Index(sig_time, name='Time'))

    return df_signal

def dff(df, filename, bleaching_correct=True, del_values=False):
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


files = get_files(path, '')

for i, file_doric in enumerate(files):

    file_short = manage_filename(file_doric)
    print(f'----- {file_short} -----')

    # extract data
    df_signal = get_signals_1IO(file_doric)
    
    df_dff = dff(df_signal, file_doric)
    df_dff.to_csv(f'{file_doric}_{file_short}.csv')
