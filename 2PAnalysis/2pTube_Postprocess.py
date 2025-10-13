#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal, optimize, stats
from pathlib import Path
import os

#%%
# does processing of the traces

df = pd.read_csv(r"D:\CA1Dopa_2pTube\Traces\Airpuff_RawTraces.CSV")
df.index = df['Frames']
time_arr = df['Time [s]']
df = df.drop('Frames', axis=1)
df = df.drop('Time [s]', axis=1)
print(df)


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
    
    return signal_detrend#, signal_expofit

def dff_from_trace(trace, baseline_range=None, eps=1e-6):
    """
    Compute ΔF/F from a 1D trace using the trace MEAN as baseline.
    trace: 1D array-like (length T)
    baseline_range: optional (start, stop) frame indices to compute the mean baseline on [start, stop)
                    use None to use the entire trace
    eps: small constant to avoid divide-by-zero
    Returns: dff (np.ndarray), F0 (float)
    """
    trace = np.asarray(trace, dtype=np.float64)

    if baseline_range is None:
        f0 = np.nanmean(trace)
    else:
        start, end = baseline_range
        f0 = np.nanmean(trace[start:end])

    denom = max(f0, eps)
    dff = (trace - denom) / denom
    return dff

df_detrend = pd.DataFrame(index=df.index)
df_dff = pd.DataFrame(index=df.index)


for col in df.columns:
    df_detrend[col] = b_bleaching(df.index, df[col].values)

#for col in df_detrend:
#    df_dff[col] = dff_from_trace(df_detrend[col].values)

#print(df_detrend)
df_detrend['Time [s]'] = time_arr

df_detrend.to_csv('detrend.csv')

# print(df_dff)
# df_dff.to_csv('dff.csv')


#%%
# does peri event 

df = pd.read_csv(r"D:\CA1Dopa_2pTube\Traces\Airpuff_RawTraces_detrend.csv")
df.index = df['Frames']
df = df.drop('Frames', axis=1)
df = df.drop('Time [s]', axis=1)

event_frames = [323, 647, 971, 1296, 1620]
fps = 10.8056
all_means = pd.DataFrame()
allevents_allrois = pd.DataFrame()


def peri_event_dff_frames(series, event_frames, fps, col, window_pre=-2, window_post=15, baseline_pre=-2, baseline_post=-0.1):

    # to calculate everything in frames
    window_pre = int(round(window_pre * fps))
    window_post = int(round(window_post * fps))
    baseline_pre = int(round(baseline_pre * fps))
    baseline_post = int(round(baseline_post * fps))

    rel_index = pd.Index(range(window_pre, window_post), dtype=int)

    df_all_events = pd.DataFrame(index=rel_index)

    for i, event in enumerate(event_frames):
        # subtract the event from df.index so that event is at zero
        shifted = series.copy()
        shifted.index = shifted.index - int(event)

        # cuts out the window
        window = shifted.loc[window_pre:window_post]
        baseline = shifted.loc[baseline_pre:baseline_post]

        # calc dff
        f0 = baseline.mean()
        f0 = max(float(f0), 1e-6)  # numeric safety
        dff = (window - f0) / f0

        with open(r"D:\CA1Dopa_2pTube\Traces\lulu.txt", "a") as f:
            f.write(f'\n{max(dff)}')

        df_all_events[f'Air{i}'] = dff.reindex(rel_index)
        allevents_allrois[f'{col}_event{i}'] = dff.reindex(rel_index)

    # adds row means
    df_all_events['mean'] = df_all_events.mean(axis=1)

    return df_all_events

for col in df:
    with open(r"D:\CA1Dopa_2pTube\Traces\lulu.txt", "a") as f:
        f.write(f'\n{col}')

    df_all_events = peri_event_dff_frames(df[col], event_frames, fps, col)
    #df_all_events.to_csv(f'{col}.csv')

    all_means[col] = df_all_events['mean']
    all_means.to_csv('all_means.csv')

#%%

print(all_means)
all_means.to_csv('all_means.csv')


#%%
print(allevents_allrois)
allevents_allrois.to_csv('allevents_allrois.csv')



#%%

# calcualte the area under the curve

df = pd.read_csv(r"D:\2P\Airpuff_detrend_frames.csv")
df.index = df['Frames']
df = df.drop('Frames', axis=1)



#%%

with open(r"D:\2P_Analysis\lulu.txt", "a") as f:
    f.write('\nlel')
