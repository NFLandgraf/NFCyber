
#%%
import doric as dr
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal, optimize, stats


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

path = 'C:\\Users\\nicol\\NFCyber\\FFAnalysis\\data\\'
file = '2024-08-27_FF-Test_m626_LockIn.doric'
#dr.h5print(path)

    
#%%

def plot_sig(time_sec, fluo, isos, title):

    # plots raw Fluo and Isos on different y-axis
    fig, ax1 = plt.subplots()
    ax2=plt.twinx()
    plot1 = ax1.plot(time_sec, isos, 'm', label='Isos') # isos on regular left axis
    plot2 = ax2.plot(time_sec, fluo, 'g', label='Fluo') # fluo on new right y-axis

    #ax1.set_ylim(1.25, 1.65)
    #ax2.set_ylim(1.35, 1.75)
    #ax1.set_xlim(100, 200)
    ax1.set_xlabel('Time [sec]')
    ax1.set_ylabel('Isos [V]', color='m')
    ax2.set_ylabel('Fluo [V]', color='g')
    ax1.set_title(title)

    lines = plot1 + plot2 # line handle for legend
    labels = [l.get_label() for l in lines]  #get legend labels
    legend = ax1.legend(lines, labels, loc='upper right')

def get_data(path, file):
    with h5py.File(path+file, 'r') as f:
        path = '/DataAcquisition/NC500/Signals/Series0001/'
        time_sec = np.array(f[path + 'LockInAOUT01/Time'])
        isos = np.array(f[path + 'LockInAOUT01/AIN01'])
        fluo = np.array(f[path + 'LockInAOUT02/AIN01'])

        duration = max(time_sec)
        sampling_rate = round(len(isos) / duration, 2)

        print(f'time {len(time_sec)}, isos {len(isos)}, fluo {len(fluo)} datapoints')
        print(f'duration: {duration}s, sampling_rate: {sampling_rate}Hz')

    return time_sec, fluo, isos, sampling_rate


def a_filtering(fluo, isos, sampling_rate):

    # low-pass cutoff freq depends on indicator (2-10Hz for GCaMP6f)
    # use zero-phase filter that changes amplitude but not phase of freq components/ no signal distortion'''
    b, a = signal.butter(2, 10, btype='low', fs=sampling_rate)

    fluo_denoised = signal.filtfilt(b, a, fluo)   
    isos_denoised = signal.filtfilt(b, a, isos)

    return fluo_denoised, isos_denoised

def b_bleaching(time_sec, fluo, isos):

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
    
    fluo_expfit = get_parameters(time_sec, fluo)
    isos_expfit = get_parameters(time_sec, isos)

    # If bleaching results from a reduction of autofluorescence, signal amplitude should not be affected
    # --> subtract function from trace (signal is V)
    # If bleaching comes from bleaching of the fluorophore, the amplitude suffers as well 
    # --> division is needed (signal is df/f)
    fluo_detrend = fluo - fluo_expfit
    isos_detrend = isos - isos_expfit
    
    return fluo_detrend, isos_detrend, fluo_expfit

def c_movement_correction(fluo, isos):
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
    plot_sig_correlation(fluo, isos, intercept, slope, r_value)

    estimated_motion = intercept + slope * isos   # calculate estimated motion acc to isos
    fluo_motcorrected = fluo - estimated_motion   # subtract to get corrected

    return fluo_motcorrected, estimated_motion

def d_normalization(signal, expfit):
    signal_dff = 100 * signal / expfit
    signal_zscore = (signal - np.mean(signal)) / np.std(signal)

    return signal_dff, signal_zscore





time_sec, fluo_raw, isos_raw, sampling_rate = get_data(path, file)

fluo_denoised, isos_denoised = a_filtering(fluo_raw, isos_raw, sampling_rate)
fluo_detrend, isos_detrend, fluo_expfit = b_bleaching(time_sec, fluo_denoised, isos_denoised)
fluo_motcorrected, estimated_motion = c_movement_correction(fluo_detrend, isos_detrend)
fluo_dff, fluo_zscore = d_normalization(fluo_motcorrected, fluo_expfit)

plot_sig(time_sec, fluo_raw, isos_raw, 'Raw Signal')
plot_sig(time_sec, fluo_denoised, isos_denoised, 'Denoised Signal')
plot_sig(time_sec, fluo_detrend, 'Detrended Signal')








#%%

plt.plot(time, isos, label='isos', c='purple')
plt.plot(time, fluo, label='fluo', c='green')

plt.xlabel('Time [s]')
plt.ylabel('Signal [V]')
plt.title(file)
plt.legend()
plt.show()