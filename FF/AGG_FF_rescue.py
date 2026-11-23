#%%
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal, optimize, stats
from pathlib import Path
import os
from scipy.ndimage import gaussian_filter1d
from scipy.signal import welch, find_peaks
from scipy.signal import butter, sosfiltfilt, resample_poly
from pathlib import Path
from fractions import Fraction

path = r"E:\AGG_FF-NE_2"

freq_fluo = 215.054
freq_isos = 333.333

file_useless_string = 'd'

def get_files(path, common_name):

    # get files
    files = [file for file in Path(path).iterdir() if file.is_file() and common_name in file.name]
    
    print(f'\n{len(files)} files found')
    for file in files:
        print(file)
    print('\n')

    return files
def manage_filename(file):

    # managing file names
    file_name = os.path.basename(file)
    file_name_short = os.path.splitext(file_name)[0]
    for word in file_useless_string:
        file_name_short = file_name_short.replace(word, '')
    
    return file_name_short


def demodulate_from_raw(raw_time, raw_sig, printing=False):
    def check_freqs(raw_sig, raw_time, fs, printing=False):

        freq, power = welch(raw_sig,fs=fs,nperseg=65536)
        mask = (freq >= 100) & (freq <= 500)

        peaks, properties = find_peaks(10 * np.log10(power[mask] + 1e-30),prominence=5)

        peak_freq = freq[mask][peaks]
        peak_power = power[mask][peaks]
        order = np.argsort(peak_power)[::-1]

        if printing:
            print("Strongest frequency peaks:")
            for i in order[:2]:
                print(f"{peak_freq[i]:.3f} Hz")
    def demodulate(raw, time, frequency, fs, filter_order=4, lowpass_cutoff=10):

        # Generate sine and cosine references
        phase = 2 * np.pi * frequency * (time - time[0])

        reference_cos = np.cos(phase)
        reference_sin = np.sin(phase)

        # Multiply raw signal by both references
        X = raw * reference_cos
        Y = raw * reference_sin

        # Butterworth low-pass filter
        sos = butter(filter_order,lowpass_cutoff,btype="low",fs=fs,output="sos")

        # Zero-phase filtering
        X_filtered = sosfiltfilt(sos, X)
        Y_filtered = sosfiltfilt(sos, Y)

        # Reconstruct modulation amplitude
        amplitude = 2 * np.sqrt(X_filtered**2 + Y_filtered**2)

        return amplitude
    def downsample(time, fs, sig_fluo, sig_isos, output_fps=100, edge_trim=1, printing=True):

        # Rational resampling factor
        ratio = Fraction(output_fps / fs).limit_denominator(10000)
        up = ratio.numerator
        down = ratio.denominator
        actual_output_fs = fs * up / down

        sig_isos_ds = resample_poly(sig_isos, up, down)
        sig_fluo_ds = resample_poly(sig_fluo, up, down)

        # Reconstruct time axis
        time_ds = time[0] + np.arange(len(sig_isos_ds)) / actual_output_fs

        # Remove filter edge artifacts
        valid = ((time_ds >= time[0] + edge_trim) &(time_ds <= time[-1] - edge_trim))

        time_ds = time_ds[valid]
        sig_isos_ds = sig_isos_ds[valid]
        sig_fluo_ds = sig_fluo_ds[valid]

        if printing:
            print(f"Output sampling rate: {actual_output_fs:.3f} Hz")
            print(f"Output samples per channel: {len(time_ds)}")

        return time_ds, sig_isos_ds, sig_fluo_ds

    
    # Determine actual sampling rate and checks
    dt = np.median(np.diff(raw_time))
    fs = 1 / dt

    check_freqs(raw_sig, raw_time, fs)

    if printing:
        print(f"Raw sampling rate: {fs:.3f} Hz")
        print(f"Recording duration: {raw_time[-1] - raw_time[0]:.2f} s")
        print(f"Raw samples: {len(raw_sig)}")

    if fs <= 2 * max(freq_isos, freq_fluo):
        raise ValueError("Raw sampling rate is too low for demodulation.")
    
    if np.any(np.diff(raw_time) <= 0):
        raise ValueError("Timestamps must be strictly increasing.")
    
    if not np.allclose(np.diff(raw_time), dt, rtol=0.01, atol=1e-6):
        raise ValueError("Irregular timestamps detected. Inspect timing before demodulation.")


    sig_isos = demodulate(raw_sig, raw_time, freq_isos, fs)
    sig_fluo = demodulate(raw_sig, raw_time, freq_fluo, fs)

    time_ds, sig_isos_ds, sig_fluo_ds = downsample(raw_time, fs, sig_fluo, sig_isos)

    df = pd.DataFrame({"Fluo": sig_fluo_ds, "Isos": sig_isos_ds}, index=pd.Index(time_ds, name="Time"))

    return df

def get_Doric_2IO(file_doric):
    # do this when you have 2 IO-TTLs, e.g. IO2-Cam_TTL and IO3-FCbox

    def h5print(item, leading=''):
        # prints out the h5 data structure
        for key in item:
            if isinstance(item[key], h5py.Dataset):
                print(leading + key + ':' + str(item[key].shape))
            else:
                print(leading + key)
                h5print(item[key], leading + '  ')
    def get_ttl_events(signal, time):
    
        high = np.asarray(signal) > 0.9
        onsets = np.flatnonzero(high & ~np.r_[False, high[:-1]])
        offsets = np.flatnonzero(high & ~np.r_[high[1:], False]) + 1
        onset_times = time[onsets]

        # End-of-recording pulses have no measured falling edge
        offset_times = np.full(len(offsets), np.nan)
        valid = offsets < len(time)
        offset_times[valid] = time[offsets[valid]]

        return onset_times, offset_times
    def add_ttl_onsets(df, onset_times, column):

        df[column] = 0

        time_values = df.index.to_numpy()

        # Find nearest fluorescence sample for each TTL onset
        for onset in onset_times:

            if onset < time_values[0] or onset > time_values[-1]:
                continue

            idx = np.argmin(np.abs(time_values - onset))

            df.iloc[idx, df.columns.get_loc(column)] = 1

        return df
    
    with h5py.File(file_doric, 'r') as f:
        #h5print(f, '')     # prints the whole structure
        path = 'DataAcquisition/NC500/Signals/Series0001/'

        # collectdata from h5 file
        raw_sig = np.array(f[path + 'AnalogIn/AIN01'])
        raw_time = np.array(f[path + 'AnalogIn/Time'])

        sig_isos  = np.array(f[path + 'LockInAOUT01/AIN01'])
        sig_fluo  = np.array(f[path + 'LockInAOUT02/AIN01'])
        sig_time  = np.array(f[path + 'LockInAOUT01/Time'])

        digi_Cam    = np.array(f[path + 'DigitalIO/DIO01']).astype(int)
        digi_Shock  = np.array(f[path + 'DigitalIO/DIO02']).astype(int)
        digi_time   = np.array(f[path + 'DigitalIO/Time'])



    df_sig = demodulate_from_raw(raw_time, raw_sig)
    df = df_sig.copy()

    cam_onsets, cam_offsets     = get_ttl_events(digi_Cam, digi_time)
    shock_onsets, shock_offsets = get_ttl_events(digi_Shock, digi_time)

    df = add_ttl_onsets(df, cam_onsets, "digi_Cam")
    df = add_ttl_onsets(df, shock_onsets, "digi_Shock")

    events_digi_Cam = cam_onsets.tolist()
    events_digi_Shock = shock_onsets.tolist()

    print(
        f"Datapoints: {len(df)}\n"
        f"Isos: {len(df['Isos'])}\n"
        f"Fluo: {len(df['Fluo'])}\n"
        f"Camera TTL events: {len(events_digi_Cam)}\n"
        f"Shock TTL events: {len(events_digi_Shock)}")


    return df, events_digi_Cam, events_digi_Shock

def dff(df, bleaching_correct=False, del_values=False, move=False):

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

    # trace cleaning
    if del_values:
        fluo_dff[fluo_dff < -1.0] = np.nan
        fluo_dff[fluo_dff > 1.0] = np.nan

    # Z-Score: baseline is mean of whole trace
    baseline = np.mean(fluo_dff)
    st_dev = np.std(fluo_dff, ddof=1)
    fluo_zscore = (fluo_dff - baseline) / (st_dev if st_dev > 0 else np.nan)

    df['Fluo_dff'] = fluo_dff
    df['Fluo_zscore'] = fluo_zscore

    return df

def perievent(df, events, file_name_short, trace_window=(-5, 15), baseline_window=(-1, -0.1), move=False):

    # takes one event at a time, gets the window around it and subtracts the bsas
    df_dff = df['Fluo_dff']
    animal_trials = []
    event_maxima = []
    event_aucs = []

    if move:
        df_move = df['Distances']

    for event_number, event_time in enumerate(events):

        # everything for the FF signal
        relative_time = df_dff.index.to_numpy() - event_time
        trial = pd.Series(df_dff.to_numpy(), index=relative_time, name=f"trial_{event_number}")

        trial = trial.loc[trace_window[0]:trace_window[1]]
        baseline = trial.loc[baseline_window[0]:baseline_window[1]]

        trial_normalized = trial - baseline.mean()
        trial_normalized.index = trial_normalized.index.round(3)

        # add trace to the summary
        animal_trials.append(trial_normalized)

        # event maximum
        response = trial_normalized.loc[0:10]
        event_max = response.max(skipna=True)
        event_maxima.append(event_max)


        # everything for the DLC signal
        if move:
            relative_time = df_move.index.to_numpy() - event_time
            trial = pd.Series(df_move.to_numpy(), index=relative_time, name=f"trial_{event_number}")
            trial = trial.loc[trace_window[0]:trace_window[1]]
            trial.index = trial.index.round(3)


    animal_trials_df = pd.concat(animal_trials, axis=1)
    all_animals_event_mean[file_name_short] = (animal_trials_df.mean(axis=1))

    # mean event maximum and positive AUC
    all_animals_event_max_auc.loc[file_name_short] = {"file": file_name_short, "mean_event_max": np.mean(event_maxima), "mean_event_auc": np.mean(event_aucs)}

def draw_perievent_groups(all_animals_event_mean, title):

    groups = ["mKate", "GFP", "A53T"]
    groups = ['RI']
    mean_df = pd.DataFrame({f"{g}_Mean": all_animals_event_mean.filter(like=g).mean(axis=1) for g in groups})
    sem_df  = pd.DataFrame({f"{g}_SEM":  all_animals_event_mean.filter(like=g).sem(axis=1)  for g in groups})

    plt.figure(figsize=(8,5))
    x = mean_df.index
    col = ['gray', 'green', 'red']
    for i, g in enumerate(groups):
        mean_trace = mean_df[f"{g}_Mean"]
        sem_trace  = sem_df[f"{g}_SEM"]
        plt.plot(x, mean_trace, label=f"{g} Mean", color=col[i])
        plt.fill_between(x, mean_trace-sem_trace, mean_trace+sem_trace, alpha=0.3, color=col[i])
    plt.xlabel("Time [s]")
    plt.ylabel("df/f")
    plt.title(title)
    plt.ylim(-0.5, 2.0)
    plt.legend()
    plt.tight_layout()
    plt.show()



files = get_files(path, '.doric')
events_shock = [38.8, 69.8, 100.8, 131.8, 162.8]
all_animals_dff = pd.DataFrame()
all_animals_ind_events = pd.DataFrame()
all_animals_event_mean = pd.DataFrame()
all_animals_ind_event_moves = pd.DataFrame()
all_animals_event_max_auc = pd.DataFrame(columns=["file", "mean_event_max", "mean_event_auc"])


for file_doric in files:

    # get the filenames
    file_name_short = manage_filename(file_doric)
    print(f'----- {file_name_short} -----')

    # Main
    df, events_digi_Cam, events_digi_Shock = get_Doric_2IO(file_doric)
    df = dff(df)

    perievent(df, events_shock, file_name_short)

draw_perievent_groups(all_animals_event_mean, 'dd')