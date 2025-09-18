import numpy as np
import pandas as pd
from scipy import signal, optimize, stats

time_sec = np.array()
fluo_raw = np.array()
isos_raw = np.array()


# Denoising
sampling_rate = (len(time_sec) - 1) / (time_sec[-1] - time_sec[0])
b, a = signal.butter(2, 10, btype='low', fs=sampling_rate)
fluo_denoised = signal.filtfilt(b, a, fluo_raw)
isos_denoised = signal.filtfilt(b, a, isos_raw)



# Photobleaching Correction and Normalization
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

fluo_detrend = fluo_denoised / fluo_expfit
isos_detrend = isos_denoised / isos_expfit



# Motion Correction
slope, intercept, r_value, p_value, std_err = stats.linregress(x=isos_detrend, y=fluo_detrend)
fluo_est_motion = intercept + slope * isos_detrend
fluo_corrected = fluo_detrend - fluo_est_motion



# Normalization
fluo_dff = 100 * fluo_corrected



# Peri-Event
events = [30, 60, 90, 120, 150]
df_all_events = pd.DataFrame()
for ev in events:
    df_event = fluo_dff.copy()

    df_event.index = df_event.index - ev
    df_event.index = df_event.index.round(2)

    window = df_event.loc[-5 : 20]
    baseline = df_event.loc[-2 : -0.1]
    baseline_mean = baseline.mean()

    window_norm = window - baseline_mean
    df_all_events[f'{ev}'] = window_norm

event_mean = df_all_events.mean(axis=1)


