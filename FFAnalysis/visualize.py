#%%
# takes csv files and creates df/f etc 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal, optimize, stats
from pathlib import Path
import os

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


path = "C:\\Users\\landgrafn\\Desktop\\FF-NE\\"
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


    # # Plot signals
    reward_cue_times = [30,60,90,120,150]
    fig,ax1=plt.subplots()  # create a plot to allow for dual y-axes plotting
    plot1=ax1.plot(time_sec, fluo_raw, 'g', label='Fluo') #plot dLight on left y-axis
    ax2=plt.twinx()# create a right y-axis, sharing x-axis on the same plot
    plot2=ax2.plot(time_sec, isos_raw, 'r', label='Isos') # plot TdTomato on right y-axis

    # Plot rewards times as ticks.
    #reward_ticks = ax1.plot(reward_cue_times, np.full(np.size(reward_cue_times), 0.151), label='Foot Shock', color='w', marker="|", mec='k', ms=10)


    #ax1.set_ylim(0.134, 0.152)
    #ax2.set_ylim(0.119, 0.137)
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Fluo Signal (V)', color='g')
    ax2.set_ylabel('Isos Signal (V)', color='r')
    ax1.set_title('Raw signals')
    ax1.set_xlim(140,170)

    lines = plot1 + plot2# +reward_ticks #line handle for legend
    labels = [l.get_label() for l in lines]  #get legend labels
    legend = ax1.legend(lines, labels, loc='upper right', bbox_to_anchor=(0.98, 0.93)) #add legend















    # Denoising
    # low-pass cutoff freq depends on indicator (2-10Hz for GCaMP6f)
    # use zero-phase filter that changes amplitude but not phase of freq components/ no signal distortion
    b, a = signal.butter(2, 10, btype='low', fs=sampling_rate)
    fluo_denoised = signal.filtfilt(b, a, fluo_raw)
    isos_denoised = signal.filtfilt(b, a, isos_raw)


    fig,ax1=plt.subplots()
    plot1=ax1.plot(time_sec, fluo_denoised, 'g', label='Fluo denoised')
    ax2=plt.twinx()
    plot2=ax2.plot(time_sec, isos_denoised, 'r', label='Isos denoised')
    reward_ticks = ax1.plot(reward_cue_times, np.full(np.size(reward_cue_times), 0.151), label='Foot Shock', color='w', marker="|", mec='k', ms=10)

    #ax1.set_ylim(0.134, 0.152)
    #ax2.set_ylim(0.119, 0.137)
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Fluo Signal (V)', color='g')
    ax2.set_ylabel('Isos Signal (V)', color='r')
    ax1.set_title('Denoised signals')
    ax1.set_xlim(140,170)

    lines = plot1+plot2 +reward_ticks #line handle for legend
    labels = [l.get_label() for l in lines]  #get legend labels
    legend = ax1.legend(lines, labels, loc='upper right', bbox_to_anchor=(0.98, 0.92)) #add legend










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


    #plot fits over denoised data
    fig,ax1=plt.subplots()  
    plot1=ax1.plot(time_sec, fluo_denoised, 'g', label='Fluo')
    plot3=ax1.plot(time_sec, fluo_expfit, color='k', linewidth=1.5, label='Exponential fit') 
    ax2=plt.twinx()
    plot2=ax2.plot(time_sec, isos_denoised, color='r', label='Isos') 
    plot4=ax2.plot(time_sec, isos_expfit,color='k', linewidth=1.5) 

    #reward_ticks = ax1.plot(reward_cue_times, np.full(np.size(reward_cue_times), 0.151), label='Foot Shock', color='w', marker="|", mec='k', ms=10)



    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Fluo Signal (V)', color='g')
    ax2.set_ylabel('Isos Signal (V)', color='r')
    ax1.set_title('Denoised signals with double exponential fits')
    ax1.set_xlim(140,170)

    lines = plot1 + plot2 + plot3
    labels = [l.get_label() for l in lines]  
    #legend = ax1.legend(lines, labels, loc='upper right')
    #ax1.set_ylim(0.134, 0.152)
    #ax2.set_ylim(0.119, 0.137)



    fig,ax1=plt.subplots()  
    plot1=ax1.plot(time_sec, fluo_detrend, 'g', label='Fluo')
    ax2=plt.twinx()
    plot2=ax2.plot(time_sec, isos_detrend, color='r', label='Isos') 
    #reward_ticks = ax1.plot(reward_cue_times, np.full(np.size(reward_cue_times), 0.0035), label='Foot Shock', color='w', marker="|", mec='k', ms=10)


    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Fluo Signal (V)', color='g')
    ax2.set_ylabel('Isos Signal (V)', color='r')
    ax1.set_title('Bleaching Correction by Double Exponential Fit')

    lines = plot1+plot2 
    labels = [l.get_label() for l in lines]  
    #legend = ax1.legend(lines, labels, loc='upper right')
    #ax1.set_ylim(-0.01, 0.01)
    #ax2.set_ylim(-0.01, 0.01)
    ax1.set_xlim(140,170)
    plt.show()




    # Motion Correction
    slope, intercept, r_value, p_value, std_err = stats.linregress(x=isos_detrend, y=fluo_detrend)
    fluo_est_motion = intercept + slope * isos_detrend   # fit Isos to Fluo
    fluo_corrected = fluo_detrend - fluo_est_motion


    # plt.scatter(isos_detrend[::5], fluo_detrend[::5],alpha=0.1, marker='.')
    # x = np.array(plt.xlim())
    # plt.plot(x, intercept+slope*x)
    # plt.xlabel('Isos')
    # plt.ylabel('Fluo')
    # plt.title('Isos - Fluo correlation.')

    # print('Slope    : {:.3f}'.format(slope))
    # print('R-squared: {:.3f}'.format(r_value**2))



    fig,ax1=plt.subplots()  
    plot1=ax1.plot(time_sec, fluo_detrend, 'b' , label='Fluo - not motion corrected', alpha=0.5)
    plot3=ax1.plot(time_sec, fluo_corrected, 'g', label='Fluo - motion corrected', alpha=1)
    ax2=plt.twinx()
    plot4=ax2.plot(time_sec, fluo_est_motion - 0.05, 'y', label='estimated motion')
    #reward_ticks = ax1.plot(reward_cue_times, np.full(np.size(reward_cue_times), 0.004), label='Foot Shock', color='w', marker="|", mec='k', ms=10)

    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Fluo Signal (V)', color='g')
    ax1.set_title('Motion Correction')

    lines = plot1+plot3+plot4#+ reward_ticks
    labels = [l.get_label() for l in lines]  
    legend = ax1.legend(lines, labels, loc='upper right', bbox_to_anchor=(0.95, 0.98))

    ax1.set_xlim(140,170)
    #ax1.set_ylim(-0.01, 0.01)
    #ax2.set_ylim(-0.055, -0.035)
    plt.plot()









    # Normalization
    fluo_dff = 100 * fluo_corrected / fluo_expfit
    fluo_dff = pd.Series(fluo_dff, index=time_sec)



    fig,ax1=plt.subplots()  
    plot1=ax1.plot(time_sec, fluo_dff, 'g', label='Fluo dF/F')
    #reward_ticks = ax1.plot(reward_cue_times, np.full(np.size(reward_cue_times), 1.5), label='Foot Shock', color='w', marker="|", mec='k', ms=10)

    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Fluo dF/F (%)')
    ax1.set_title('Fluo dF/F')

    lines = plot1#+ reward_ticks
    labels = [l.get_label() for l in lines]  
    #legend = ax1.legend(lines, labels, loc='upper right', bbox_to_anchor=(0.95, 0.98))

    ax1.set_xlim(140,170)
    #ax1.set_ylim(-3, 3)














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
    
#all_animals.to_csv('all_animals.csv')


