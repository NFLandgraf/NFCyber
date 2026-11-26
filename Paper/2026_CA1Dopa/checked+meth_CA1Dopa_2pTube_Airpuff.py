#%%
# CA1Dopa_2pTube_Airpuff
from scipy import optimize, stats
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import numpy as np

'''
Calculates the peri-event response from axons upon an airpuff
INPUT: Raw brightness from the axon pixels from FIJI
1. Raw fluorescent trace
2. Detrended trace: Raw traces divided by fitted double-exponential function
3. Whole ΔF/F:      ΔF/F = (F-F0)/F0 (F0 is mean fluorescence of detrended trace)

4. Detrended fluorescence traces were aligned to each airpuff so that stimulus onsets corresponded to frame 0
5. For each event, a peri-event window [-2s, 15s] from airpuff onset (0s) is extracted
6. For each peri-event window, mean fluorescence during pre-stimulus period [-2s, -0.1s] is subtracted from the window trace and divided: ΔF/F = (F-F0)/F0
7. The maximum ΔF/F is taken from each event window and averaged through all event for that recording
8. Similarly, individual airpuff responses were averaged to receive one recording average
'''

fold                                    = Path(r"E:\2pTube")
file_RawTraces                          = fold / 'CA1Dopa_2pTube_Results_Airpuff_RawTraces.csv'
file_RawTraces_Detrend                  = fold / 'CA1Dopa_2pTube_Results_Airpuff_RawTraces_Detrend.csv'
file_RawTraces_Detrend_Dff              = fold / 'CA1Dopa_2pTube_Results_Airpuff_RawTraces_Detrend_Dff.csv'
fold_RawTraces_Detrend_PeriEvent        = fold / 'CA1Dopa_2pTube_Results_Airpuff_RawTraces_Detrend_PeriEvent'
file_RawTraces_Detrend_PeriEvent_Means  = fold / 'CA1Dopa_2pTube_Results_Airpuff_RawTraces_Detrend_PeriEvent_Mean.csv'
file_RawTraces_Detrend_PeriEvent_Max_AUC= fold / 'CA1Dopa_2pTube_Results_Airpuff_RawTraces_Detrend_PeriEvent_Max_AUC.csv'

event_frames = [323, 647, 971, 1296, 1620]
fps = 10.8056


def clean(file_RawTraces):

    df = pd.read_csv(file_RawTraces)
    df.index = df['Frames']
    time_arr = df['Time [s]']
    df = df.drop('Frames', axis=1)
    df = df.drop('Time [s]', axis=1)

    # creates new df for storing the new traces
    df_detrend = pd.DataFrame(index=df.index)
    df_detrend['Time [s]'] = time_arr
    df_dff = pd.DataFrame(index=df.index)
    df_dff['Time [s]'] = time_arr

    return df, df_detrend, df_dff, time_arr

def bleaching(arr_time, arr_trace):
    # fits a double-exponential function to trace to correct for all possible bleaching reasons
    def double_exponential(t, const, amp_fast, amp_slow, tau_slow, tau_multiplier):
        '''
        Compute a double exponential function with constant offset.
        t               : Time vector in seconds.
        const           : Amplitude of the constant offset. 
        amp_fast        : Amplitude of the fast component.  
        amp_slow        : Amplitude of the slow component.  
        tau_slow        : Time constant of slow component in seconds.
        tau_multiplier  : Time constant of fast component relative to slow. 
        '''
        
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

    trace_expofit = get_parameters(arr_time, arr_trace)
    trace_detrend = arr_trace / trace_expofit
    
    return trace_detrend

def dff_normal(trace, baseline_range=None):
    """
    Compute ΔF/F from a 1D trace using the trace MEAN as baseline.
    trace: 1D array-like (length T)
    baseline_range: optional (start, stop) frame indices to compute the mean baseline on [start, stop)
                    use None to use the entire trace
    eps: small constant to avoid divide-by-zero
    Returns: dff (np.ndarray)
    """
    trace = np.asarray(trace, dtype=np.float64)

    if baseline_range is None:
        f0 = np.nanmean(trace)
    else:
        start, end = baseline_range
        f0 = np.nanmean(trace[start:end])

    denom = max(f0, 1e-6)
    dff = (trace - denom) / denom
    return dff

def peri_event_dff_frames(series, col, fps, event_frames, window=(-2, 15), baseline=(-2,-0.1), analysis_window=(0,10)):

    # to calculate everything in frames
    window_pre    = int(round(window[0] * fps))
    window_post   = int(round(window[1] * fps))
    baseline_pre  = int(round(baseline[0] * fps))
    baseline_post = int(round(baseline[1] * fps))
    analysis_pre  = int(round(analysis_window[0] * fps))
    analysis_post = int(round(analysis_window[1] * fps))

    rel_index     = pd.Index(range(window_pre, window_post), dtype=int)
    df_all_events = pd.DataFrame(index=rel_index)

    event_max_values, event_auc_values = [], []

    for i, event in enumerate(event_frames):

        # move event frame to zero
        shifted = series.copy()
        shifted.index = shifted.index - int(event)

        # cuts out the window
        event_window = shifted.loc[window_pre:window_post]
        baseline_values = shifted.loc[baseline_pre:baseline_post]

        # calc dff
        f0 = baseline_values.mean()
        f0 = max(float(f0), 1e-6)  # numeric safety
        dff = (event_window - f0) / f0   # we already divided during the bleaching correction, so this should just be the subtraction
        df_all_events[f'Air{i}'] = dff.reindex(rel_index)

        # get the max and AUC of the trace
        analysis_trace = dff.loc[analysis_pre:analysis_post].dropna()
        event_max_values.append(analysis_trace.max())
        event_auc_values.append(np.trapezoid(analysis_trace.to_numpy(), x=analysis_trace.index.to_numpy()/fps))

    # adds row means
    mean_trace = df_all_events.mean(axis=1)

    # Average max and AUC across all events
    mean_max = np.nanmean(event_max_values)
    mean_auc = np.nanmean(event_auc_values)

    df_event_max_auc.loc[col, "max"] = mean_max
    df_event_max_auc.loc[col, "auc"] = mean_auc

    return mean_trace

def plot_response(df, individ):

    def animal_id(name):
        name = str(name)
        return name[:next((i for i, char in enumerate(name) if not char.isdigit()), len(name))]

    traces = df.drop(columns='Time [s]')
    time   = df['Time [s]'].to_numpy()

    title  = 'Axons averages' if individ else 'Animal averages'
    unit   = 'Axons'          if individ else 'Animals'
    traces = traces           if individ else traces.T.groupby([animal_id(c) for c in traces.columns]).mean().T


    fig, ax = plt.subplots(figsize=(8, 5))

    for group, ids in groups.items():
        cols = [c for c in traces.columns if animal_id(c) in ids]
        data = traces[cols]

        mean = data.mean(axis=1).to_numpy()
        sem = data.sem(axis=1).to_numpy()

        ax.plot(time, mean, color=colors[group],label=f'{group} (n={len(cols)} {unit})')
        ax.fill_between(time, mean - sem, mean + sem,color=colors[group], alpha=0.2)

    ax.axvline(0, color='black', linestyle='--', linewidth=1)
    ax.set(xlabel='Time from airpuff [s]', ylabel='ΔF/F', title=title)
    ax.set_xlim(-2,10)
    ax.legend()
    plt.tight_layout()
    plt.show()

def plot_max_auc(df, individ):

    def animal_id(name):
        name = str(name)
        return name[:next((i for i, char in enumerate(name) if not char.isdigit()), len(name))]
    
    title = 'Axons' if individ else 'Animals'
    df   = df      if individ else df.groupby([animal_id(i) for i in df.index]).mean()

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    for ax, metric in zip(axes, ['max', 'auc']):
        values = {}

        for x, (group, ids) in enumerate(groups.items()):
            mask = [animal_id(i) in ids for i in df.index]
            y = df.loc[mask, metric].dropna().to_numpy()
            values[group] = y

            ax.scatter(np.full(len(y), x), y, color=colors[group], alpha=0.7)
            if len(y):
                ax.errorbar(x, y.mean(), yerr=stats.sem(y) if len(y) > 1 else None, fmt='_', color='black', markersize=25, capsize=5)

        a, b = values['mKate'], values['A53T']

        t_stat, p_value = stats.ttest_ind(a, b, equal_var=False)
        
        ax.set_xticks([0, 1],[f'mKate\nn={len(a)}', f'A53T\nn={len(b)}'])
        ax.set_xlim(-0.5, 1.5)
        ax.text(0.5, 0.95, f'p={p_value:.4g}', transform=ax.transAxes, ha='center', va='top')
        ax.set_ylabel('Maximum ΔF/F' if metric == 'max' else 'AUC [ΔF/F × s]')
        ax.set_title(f'{title}_{metric}')

    plt.tight_layout()
    plt.show()


groups = {'mKate': ['971', '975', '1001'], 'A53T':  ['972', '976', '1002']}
colors = {'mKate': 'gray', 'A53T': 'red'}
df_event_means, df_event_max_auc = pd.DataFrame(), pd.DataFrame()


df, df_detrend, df_dff, time_arr = clean(file_RawTraces)

for col in df.columns:
    df_detrend[col] = bleaching(time_arr, df[col].values)
    df_dff[col]     = dff_normal(df_detrend[col].values)
    mean_trace      = peri_event_dff_frames(df_detrend[col], col, fps, event_frames)

    if df_event_means.empty:
        df_event_means = pd.DataFrame(index=mean_trace.index)
        df_event_means.index.name = 'Frames'
        df_event_means['Time [s]'] = df_event_means.index / fps

    df_event_means[col] = mean_trace


if True:
    df_detrend.to_csv(      file_RawTraces_Detrend)
    df_dff.to_csv(          file_RawTraces_Detrend_Dff)
    df_event_means.to_csv(  file_RawTraces_Detrend_PeriEvent_Means)
    df_event_max_auc.to_csv(file_RawTraces_Detrend_PeriEvent_Max_AUC)

plot_response(df_event_means, individ=True)
plot_response(df_event_means, individ=False)
plot_max_auc(df_event_max_auc, individ=True)
plot_max_auc(df_event_max_auc, individ=False)
