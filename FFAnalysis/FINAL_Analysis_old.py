
#%%
# takes csv files and creates df/f etc 

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal, optimize, stats
from pathlib import Path
import os

path = r"C:\Users\landgrafn\Desktop\FF-DA"
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

def a_filter_butterworth(arr_time, arr_signal):

    # low-pass cutoff freq depends on indicator (2-10Hz for GCaMP6f)
    # use zero-phase filter that changes amplitude but not phase of freq components/ no signal distortion
    sampling_rate = len(arr_signal) / np.max(arr_time)
    b, a = signal.butter(2, 10, btype='low', fs=sampling_rate)
    arr_signal_filtered = signal.filtfilt(b, a, arr_signal)   

    return arr_signal_filtered

def b_bleaching(arr_time, arr_signal):

    # fits a double-exponential function to trace to correct for all possible bleaching reasons
    def double_exponential(t, const, amp_fast, amp_slow, tau_slow, tau_multiplier):
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
    
    # If bleaching results from a reduction of autofluorescence, signal amplitude should not be affected
    # --> subtract function from trace (signal is V)
    # If bleaching comes from bleaching of the fluorophore, the amplitude suffers as well 
    # --> division is needed (signal is df/f)
    signal_expofit = get_parameters(arr_time, arr_signal)
    signal_detrend = arr_signal / signal_expofit
    
    return signal_detrend, signal_expofit

def c_fitted_Isos(arr_signal, arr_control):
    '''Although quite similar, movement artefacts will be different between signal and control channel.
    -> necessary to scale movement signals from control channels to match artefacts in signal channel
    linear regression predicts the signal channel from control channel,
    any difference of the signal from the predicted value is considered real signal
    Important not to have bleed through and that channels are independent except for artefacts'''
    
    def plot_sig_correlation(fluo, isos, intercept, slope, r_value):
        plt.scatter(fluo, isos, c='gray')
        x = np.array(plt.xlim())
        plt.plot(x, intercept + slope*x, c='r', )

        plt.text(plt.xlim()[1]/2, plt.ylim()[1]*0.8, f'slope={round(slope, 2)}\nr_value={round(r_value, 2)}', c='r', fontsize=20)
        plt.xlabel('Fluo [V]')
        plt.ylabel('Isos [V]')
        plt.title('Signal correlation')
        plt.show()

    slope, intercept, r_value, p_value, std_err = stats.linregress(x=arr_signal, y=arr_control)
    # plot_sig_correlation(fluo, isos, intercept, slope, r_value)

    Isos_fitted = intercept + slope * arr_control   # fit Isos to Fluo

    return Isos_fitted

def d_normalization(arr_signal, arr_control, arr_expofit):

    Fluo_dff = 100 * ((arr_signal - arr_control) / arr_expofit)

    return Fluo_dff

def old_preprocess(df):

    # filters
    df['Isos_smooth'] = a_filter_butterworth(df.index, df['Isos'])
    df['Fluo_smooth'] = a_filter_butterworth(df.index, df['Fluo'])

    # bleaching correction
    df['Isos_detrend'], df['Isos_expofit'] = b_bleaching(df.index, df['Isos_smooth'])
    df['Fluo_detrend'], df['Fluo_expofit'] = b_bleaching(df.index, df['Fluo_smooth'])

    # fitting Isos to Fluo
    df['Isos_fitted'] = c_fitted_Isos(df['Fluo_detrend'], df['Isos_detrend'])

    # normalization
    df['Fluo_dff'] = d_normalization(df['Fluo_detrend'], df['Isos_fitted'], df['Fluo_expofit'])

    return df

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


    df_base.to_csv(path + 'dff_allfiles.csv')

    return df_base



df_base = main(files)


print(df_base)


#%%

df = pd.read_csv(r"D:\FF\proc\dff_allfiles.CSV")
df.index = df['Time']
del df['Time']
print(df)
#%%
df_base = df

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



#%%
# Calculate baseline means and stds

def baseline_mean(df, pre):
    baseline_means, baseline_stds = {}, {}

    for col in df.columns:
        datapoints_baseline = df_perievent_means.loc[pre:0.00, col]

        # calculate baseline means
        baseline_mean = datapoints_baseline.mean()
        baseline_means[col] = baseline_mean

        # calculate baseline stds
        baseline_std = datapoints_baseline.std()
        baseline_stds[col] = baseline_std

        print(f'{col}: {baseline_mean}, {baseline_std}')

    print(baseline_means)
    print(baseline_stds)
    return baseline_means, baseline_stds

baseline_means, baseline_stds = baseline_mean(df_perievent_means, -5.00)


#%%
# calculate, when traces reach a certain threshold



def threshold_reach(df, thresholds):
    # calculate, when a signal reaches a threshold

    first_responses = {}
    for col, threshold in zip(df.columns, thresholds):

        first_index = df[col][df[col] > threshold].index.min()
        first_responses[col] = first_index
    
    print(first_responses)
    return first_responses

thresholds = [0.2029, -0.3028, 0.7057, 0.2695, 0.3749, -0.0499, 0.1954, -0.0044, 0.0429, -0.0701]
first_responses = threshold_reach(df_perievent_means, thresholds)













#%%
# old code:
slope, intercept, r_value, p_value, std_err = stats.linregress(x=df['Fluo_detrend'], y=df['Isos_detrend'])
estimated_motion = intercept + slope * df['Isos_detrend']   # calculate estimated motion acc to isos

# new code:
p = np.polyfit(control, signal, 1)
Isos_fitted = (p[0]*control)+p[1]






#%%
# just for the movement vector

def get_data_csv(file):

    df = pd.read_csv(file)
    df.index = df['Time']
    del df['Time']

    return df


def perievent(file, df, events, window_pre=5, window_post=20):

    df_all_events = pd.DataFrame()

    for ev in events:
        df_event = df.copy()

        # reindex so the event is 0
        df_event.index = df_event.index - ev
        df_event.index = df_event.index.round(2)

        # crop the recording to the time around the event
        df_event = df_event.loc[-window_pre : window_post]
        df_all_events[ev] = df_event



    # add the row means to the main df
    df_all_events['mean'] = df_all_events.mean(axis=1)
        

    df_all_events['mean'].to_csv(path + file + "Perievent_means.csv")

    return df_all_events


events = [30, 60, 90, 120, 150]


for file in files:
    df = get_data_csv(file)
    file_short = manage_filename(file)
    

    df = df['Distances']

    df_perievent = perievent(file_short, df, events)


    print(df)