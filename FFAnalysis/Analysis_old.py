
#%%

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal, optimize, stats
from pathlib import Path
import os


path = 'C:\\Users\\landgrafn\\Desktop\\FF\\'
file_useless_string = ['2024-11-20_FF-Weilin_FS_', '_LockIn']


def get_files(path, common_name):

    # get files
    files = [file for file in Path(path).iterdir() if file.is_file() and common_name in file.name]
    
    print(f'\n{len(files)} files found')
    #for file in files:
        #print(file)

    return files
files = get_files(path, '2024')



def plot_sig(time_sec, fluo, isos, title):

    #set default plot properties
    plt.rcParams['figure.figsize'] = [14, 12] # Make default figure size larger.
    plt.rcParams['axes.xmargin'] = 0          # Make default margin on x axis zero.
    plt.rcParams['axes.labelsize'] = 12     #Set default axes label size 
    plt.rcParams['axes.titlesize']=15
    plt.rcParams['axes.titleweight']='heavy'
    plt.rcParams['ytick.labelsize']= 10
    plt.rcParams['xtick.labelsize']= 10
    plt.rcParams['legend.fontsize']=12
    plt.rcParams['legend.markerscale']=2

    # plots raw Fluo and Isos on different y-axis
    fig, ax1 = plt.subplots()
    ax2=plt.twinx()
    plot1 = ax1.plot(time_sec, fluo, 'g', label='Fluo') # fluo on new right y-axis
    plot2 = ax2.plot(time_sec, isos, 'm', label='Isos') # isos on regular left axis

    #ax1.set_ylim(1.25, 1.65)
    #ax2.set_ylim(1.35, 1.75)
    #ax1.set_xlim(100, 200)
    ax1.set_xlabel('Time [sec]')
    ax1.set_ylabel('Fluo [V]', color='g')
    ax2.set_ylabel('Isos [V]', color='m')
    ax1.set_title(title)

    lines = plot1 + plot2 # line handle for legend
    labels = [l.get_label() for l in lines]  #get legend labels
    legend = ax1.legend(lines, labels, loc='upper right')
    plt.show()

def plot_sig_fluo(time, fluo, title):

    #set default plot properties
    plt.rcParams['figure.figsize'] = [14, 12] # Make default figure size larger.
    plt.rcParams['axes.xmargin'] = 0          # Make default margin on x axis zero.
    plt.rcParams['axes.labelsize'] = 12     #Set default axes label size 
    plt.rcParams['axes.titlesize']=15
    plt.rcParams['axes.titleweight']='heavy'
    plt.rcParams['ytick.labelsize']= 10
    plt.rcParams['xtick.labelsize']= 10
    plt.rcParams['legend.fontsize']=12
    plt.rcParams['legend.markerscale']=2

    plt.plot(time, fluo, 'g', label='Fluo') # fluo on new right y-axis

    #ax1.set_ylim(1.25, 1.65)
    #ax2.set_ylim(1.35, 1.75)
    #ax1.set_xlim(100, 200)
    plt.xlabel('Time [sec]')
    plt.ylabel('Fluo [V]', color='g')
    plt.title(title)

    plt.legend(loc='upper right')
    plt.show()

def get_data(file):

    with h5py.File(file, 'r') as f:
        path = 'DataAcquisition/NC500/Signals/Series0001/'

        if rec_type == 'NormFilter':
            time_sec    = np.array(f[path + 'AnalogIN/Time'])
            sig         = np.array(f[path + 'AnalogIN/AIN01'])
            output      = np.array(f[path + 'AnalogOUT/AOUT01'])

            duration = max(time_sec)
            sampling_rate = round(len(sig) / duration, 2)

            print(f'NormFilter\n'
                  f'Output: {[round(min(output), 1), round(np.mean(output), 1), round(max(output), 1)]}V\n'
                  f'Datapoints: {len(time_sec)} (Time), {len(sig)} (Signal)\n'
                  f'Duration: {duration}s, Sampling_rate: {sampling_rate}Hz\n')

            return time_sec, sig, output, sampling_rate
        
        
        
        elif rec_type == 'LockIn':

            time_sec    = np.array(f[path + 'LockInAOUT01/Time'])
            isos        = np.array(f[path + 'LockInAOUT01/AIN01'])
            fluo        = np.array(f[path + 'LockInAOUT02/AIN01'])
            output_isos = np.array(f[path + 'AnalogOut/AOUT01'])
            output_fluo = np.array(f[path + 'AnalogOut/AOUT02'])
            digital_io  = np.array(f[path + 'DigitalIO/DIO01'])
            digital_time= np.array(f[path + 'DigitalIO/Time'])


            duration = max(time_sec)
            sampling_rate = round(len(isos) / duration, 2)

            print(f'LockIn\n'
                  f'Output_Isos: {[round(min(output_isos), 1), round(np.mean(output_isos), 1), round(max(output_isos), 1)]}V\n'
                  f'Output_Fluo: {[round(min(output_fluo), 1), round(np.mean(output_fluo), 1), round(max(output_fluo), 1)]}V\n'
                  f'Datapoints: {len(time_sec)} (Time), {len(isos)} (Isos), {len(fluo)} (Fluo), {len(digital_io)} (IO)\n'
                  f'Duration: {duration}s, Sampling_rate: {sampling_rate}Hz\n')

            return time_sec, fluo, isos, output_fluo, output_isos, sampling_rate, digital_io, digital_time
        
def get_data_csv(file):

    df = pd.read_csv(file)
    df.index = df['Time']
    del df['Time']

    return df

def a_denoising(trace):

    # low-pass cutoff freq depends on indicator (2-10 Hz for GCaMP6f)
    # use zero-phase filter that changes amplitude but not phase of freq components/ no signal distortion
    time = np.array(trace.index.values)
    sampling_rate = len(time) / np.max(time)
    b, a = signal.butter(2, 10, btype='low', fs=sampling_rate)

    trace_denoised = signal.filtfilt(b, a, trace)   

    return trace_denoised

def b_bleaching(trace):
    # fits a double-exponential function to trace to correct for all possible bleaching reasons

    def double_exponential(t, const, amp_fast, amp_slow, tau_slow, tau_multiplier):
        '''Compute a double exponential function with constant offset.
        t               : Time vector in seconds
        const           : Amplitude of the constant offset
        amp_fast        : Amplitude of the fast component
        amp_slow        : Amplitude of the slow component
        tau_slow        : Time constant of slow component in seconds
        tau_multiplier  : Time constant of fast component relative to slow '''
        
        tau_fast = tau_slow * tau_multiplier
        return const+amp_slow*np.exp(-t/tau_slow)+amp_fast*np.exp(-t/tau_fast)
    
    def get_parameters(time_sec, trace):
        max_sig = np.max(trace)
        initial_params = [max_sig/2, max_sig/4, max_sig/4, 3600, 0.1] # initial parmeters
        bounds = ([0, 0, 0, 600, 0], [max_sig, max_sig, max_sig, 36000, 1])
        trace_params, parm_conv = optimize.curve_fit(double_exponential, time_sec, trace, p0=initial_params, bounds=bounds, maxfev=1000)
        
        return double_exponential(time_sec, *trace_params)
    
    # If bleaching results from a reduction of autofluorescence, signal amplitude should not be affected
    # --> subtract function from trace (signal is V)
    # If bleaching comes from bleaching of the fluorophore, the amplitude suffers as well 
    # --> division is needed (signal is df/f)
    time = np.array(trace.index.values)
    trace_expofit = get_parameters(time, trace)
    trace_detrend = trace / trace_expofit
        
    return trace_detrend, trace_expofit

def c_motion_correction(fluo, isos):
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

    slope, intercept, r_value, p_value, std_err = stats.linregress(x=fluo, y=isos)
    estimated_motion = intercept + slope * isos     # calculate estimated motion acc to isos
    fluo_corrected = fluo - estimated_motion        # subtract to get corrected
    # plot_sig_correlation(fluo, isos, intercept, slope, r_value)

    return fluo_corrected

def d_normalization(trace, trace_expofit):

    Fluo_dff = 100 * trace #/ trace_expofit
    Fluo_zscore = trace / np.std(trace)         # makes sense if std is similar between the two groups

    return Fluo_dff, Fluo_zscore

def main(files):

    df_base = pd.DataFrame()
    for file in files:

        # managing file names
        print(f'\nProcessing file: {file}')
        file_name = os.path.basename(file)
        file_name_short = os.path.splitext(file_name)[0]
        for word in file_useless_string:
            file_name_short = file_name_short.replace(word, '')
            

        # process data
        df = get_data_csv(file)
        df['Fluo_denoised']                     = a_denoising(df['Fluo'])
        df['Isos_denoised']                     = a_denoising(df['Isos'])
        df['Fluo_detrend'], df['Fluo_expofit']  = b_bleaching(df['Fluo_denoised'])
        df['Isos_detrend'], df['Isos_expofit']  = b_bleaching(df['Isos_denoised'])
        df['Fluo_motcorrected']                 = c_motion_correction(df['Fluo_detrend'], df['Isos_detrend'])
        df['Fluo_dff'], df['Fluo_zscore']       = d_normalization(df['Fluo_motcorrected'], df['Fluo_expofit'])


        # plot results
        # time = np.array(df.index.values)
        # plot_sig(time, df['Fluo'], df['Isos'], 'Raw Signal')
        # plot_sig(time, df['Fluo_denoised'], df['Isos_denoised'], 'Denoised Signal')
        # plot_sig(time, df['Fluo_detrend'], df['Isos_detrend'], 'Detrended Signal')
        # plot_sig_fluo(time, df['Fluo_motcorrected'], 'Motion Corrected Signal')
        # plot_sig_fluo(time, df['Fluo_dff'], 'dF/F')
        # plot_sig_fluo(time, df['Fluo_zscore'], 'Z-Score')

        
        # add all files to df_base, each file is a new column
        df_base[file_name_short] = df['Fluo_dff']
        print(f'{file_name_short} added to df_base')
    
    df_base.to_csv(path + 'dff_allfiles.csv')
    return df_base

df_base = main(files)



# Mean and align everything to events

def perievent(df, events, pre_interval=5, post_interval=10, pre_interval_baseline=2):
    df_perievent_means = pd.DataFrame()

    # go through each column in df_base
    for column in df_base:

        df = df_base[column]
        df_perievent = pd.DataFrame()

        for e in events:
            df_trial = df.copy()
            
            # reindex so the event is 0
            df_trial.index = df_trial.index - e
            df_trial.index = df_trial.index.round(2)

            # collect everything from -x to +y
            df_trial = df_trial.loc[- pre_interval : post_interval]

            # for each event, subtract the pre-event baseline from the trial to normalize to baseline
            pre_baseline_mean = df_trial.loc[- pre_interval_baseline : 0].mean()
            df_trial = df_trial - pre_baseline_mean

            df_perievent[f'{e}'] = df_trial

        # add the row means to the main df
        df_perievent['mean'] = df_perievent.mean(axis=1)
        df_perievent_means[f'{column}'] = df_perievent['mean']

    df_perievent_means.to_csv(path + "Perievent_means.csv")

    return df_perievent_means


events = [30, 60, 90, 120, 150]
df_perievent_means = perievent(df_base, events, pre_interval=5, post_interval=10)





for column in df_perievent_means:
    print(column)

    if 'A53T' in column:
        plt.plot(df_perievent_means[column], c='r')
    elif 'mKate' in column:
        plt.plot(df_perievent_means[column], c='g')
    elif 'GFP' in column:
        plt.plot(df_perievent_means[column], c='gray')

plt.show()


#%%
# Std
baselines = [[10.00, 30.00], [40.00, 60.00], [70.00, 90.00], [100.00, 120.00], [130.00, 150.00]]

df = pd.DataFrame()
for column in df_base:
    print(column)

    arr = []
    for bl in baselines:
        arr.extend(df_base.loc[bl[0]:bl[1], column].tolist())
    arr = np.array(arr)
    
    df[column] = arr

df.to_csv('std.csv')







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
