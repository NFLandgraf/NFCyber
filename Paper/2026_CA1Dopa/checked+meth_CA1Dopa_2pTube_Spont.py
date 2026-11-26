#%%
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import optimize, stats, signal
from suite2p.extraction import dcnv
import torch


'''
GET TRACE PROPERTIES
1. Import raw traces
2. Correct each axons fluorescence for bleaching by dividing a double-exponential function with constant offset from the raw trace
3. Calculate ΔF/F by subtracting and diving the mean of the entire detrended trace from the detrended trace
4. Calculate fluorescence variability properties for each axon
5. Calculate the power spectrum from the df/f trace
'''

fold = Path(r"C:\Users\landgrafn\Desktop\CA1Dopa_FINAL\CA1Dopa_FINAL_Data\2pTube")
file = 'CA1Dopa_2pTube_TwoAltLasers_Raw_Preprocess_MotCorr_axGCaMP_AxonROIs_Trace_Spont_Raw'

file_RawTraces                           = fold / f'{file}.csv'
file_RawTraces_detrend                   = fold / f'{file}_Detrend.csv'
file_RawTraces_detrend_dff               = fold / f'{file}_Detrend_Dff.csv'
file_RawTraces_detrend_dff_stats         = fold / f'{file}_Detrend_Dff_Stats.csv'
file_RawTraces_detrend_dff_spectrum      = fold / f'{file}_Detrend_Dff_Spectrum.csv'
file_RawTraces_detrend_dff_spectrum_norm = fold / f'{file}_Detrend_Dff_Spectrum_Norm.csv'


fps = 10.8056

groups = {'mKate': ['971', '975', '1001'], 'A53T':  ['972', '976', '1002']}
colors = {'mKate': 'gray', 'A53T': 'red'}

def animal_id(name):
    id = pd.Series([str(name)]).str.extract(r'(?<!\d)(971|972|975|976|1001|1002)(?!\d)',expand=False).iloc[0]
    return id

def check_group(name):
    animal = animal_id(name)
    return next((group for group, ids in groups.items() if animal in ids),None)


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


def calc_trace_properties(trace, column, group):

    # get info
    trace = trace[np.isfinite(trace)]
    frame_changes = np.diff(trace)

    data = {"Group":            group,
            "Recording":        column,
            "n_frames":         len(trace),
            "Mean_dff":         np.mean(trace),
            "SD_dff":           np.std(trace, ddof=1),
            "Variance_dff":     np.var(trace, ddof=1),
            "Mean_abs_change":  np.mean(np.abs(frame_changes)),
            "SD_frame_change":  np.std(frame_changes, ddof=1)}
        
    return data

def calc_power_spectrum(trace):

    trace = trace[np.isfinite(trace)]
    trace = signal.detrend(trace, type="linear")

    frequencies, power = signal.welch(trace, fs=fps, window="hann", nperseg=min(1024, len(trace)), noverlap=min(512, len(trace) // 2), scaling="density")

    total_power = np.trapezoid(power, frequencies)
    power_norm = power / total_power

    return frequencies, power, power_norm


def assemble_export(traces_detrend, traces_dff, trace_properties, spectra, spectra_norm):

    # Assemble tables
    df_detrend      = pd.DataFrame(traces_detrend)
    df_dff          = pd.DataFrame(traces_dff)
    df_props        = pd.DataFrame(trace_properties)
    df_spectra      = pd.DataFrame(spectra)
    df_spectra_norm = pd.DataFrame(spectra_norm)

    for table in [df_detrend, df_dff]:
        table.index.name = 'Frames'
        table.insert(0, 'Time [s]', time_arr)

    for table in [df_spectra, df_spectra_norm]:
        table.index.name = 'Frequency [Hz]'

    # Save
    df_detrend.to_csv(     file_RawTraces_detrend)
    df_dff.to_csv(         file_RawTraces_detrend_dff)
    df_props.to_csv(       file_RawTraces_detrend_dff_stats, index=False)
    df_spectra.to_csv(     file_RawTraces_detrend_dff_spectrum)
    df_spectra_norm.to_csv(file_RawTraces_detrend_dff_spectrum_norm)

    return df_props, df_spectra, df_spectra_norm

def plot_ttest(df, value_column, individ):

    data = df[['Group', 'Recording', value_column]].dropna().copy()
    unit = 'Axons' if individ else 'Animals'
    group_order=("mKate","A53T")

    if not individ:
        data['Animal'] = data['Recording'].map(animal_id)
        data = data.groupby(['Group', 'Animal'], as_index=False)[value_column].mean()


    fig, ax = plt.subplots(figsize=(5, 5))
    group_values = {}

    for x, group in enumerate(group_order):
        values = data.loc[data['Group'] == group, value_column].to_numpy()
        group_values[group] = values

        jitter = np.linspace(-0.05, 0.05, len(values))
        ax.scatter(np.full(len(values), x) + jitter, values, color=colors[group], s=65, zorder=3)

        if len(values):
            ax.errorbar(x, values.mean(), yerr=stats.sem(values) if len(values) > 1 else None, fmt='_', color='black', markersize=20, capsize=5)

    a, b = [group_values[g] for g in group_order]
    if len(a) >= 2 and len(b) >= 2:
        _, pvalue = stats.ttest_ind(a, b, equal_var=False)
        ax.text(0.5, 0.95, f'p = {pvalue:.4g}',transform=ax.transAxes, ha='center', va='top')

    ax.margins(y=0.2)
    ax.set_xticks(range(len(group_order)), [f'{g}\nn={len(group_values[g])}' for g in group_order])
    ax.set_ylabel(value_column)
    ax.set_title(f'{value_column} — {unit}')
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    plt.show()

def plot_spectra(df, individ):

    data = df.copy()
    unit = 'Axons' if individ else 'Animals'

    if not individ:
        data = data.T.groupby([animal_id(c) for c in data.columns]).mean().T

    fig, ax = plt.subplots(figsize=(6, 4))

    for group in ['mKate', 'A53T']:
        cols = [c for c in data.columns if check_group(c) == group]
        values = data[cols]

        mean = values.mean(axis=1).to_numpy()
        sem = values.sem(axis=1).to_numpy()

        ax.plot(data.index, mean, color=colors[group],label=f'{group} (n={len(cols)})')
        ax.fill_between(data.index, mean - sem, mean + sem,color=colors[group], alpha=0.25)

    ax.set_yscale('log')
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Power spectral density')
    ax.set_title(f'ΔF/F spectra — {unit}')
    ax.legend(frameon=False)

    plt.tight_layout()
    plt.show()


# Load raw fluorescence
df = pd.read_csv(file_RawTraces).iloc[:-2].reset_index(drop=True)
time_arr = np.arange(len(df)) / fps

traces_detrend, traces_dff = {}, {}
spectra, spectra_norm = {}, {}
trace_properties = []


for col in df.columns:

    group = check_group(col)
    if group is None or col in ['Frames', 'Time [s]']:
        continue

    # do trace processing
    trace_raw     = pd.to_numeric(df[col], errors='coerce').to_numpy()
    trace_detrend = bleaching(time_arr, trace_raw)
    trace_dff     = dff_normal(trace_detrend)

    # calc trace properties
    trace_props             = calc_trace_properties(trace_dff, col, group)
    freq, power, power_norm = calc_power_spectrum(trace_dff)
    
    traces_detrend[col] = trace_detrend
    traces_dff[col]     = trace_dff
    spectra[col]        = pd.Series(power, index=freq)
    spectra_norm[col]   = pd.Series(power_norm, index=freq)
    trace_properties.append(trace_props)

if True:
    df_props, df_spectra, df_spectra_norm = assemble_export(traces_detrend, traces_dff, trace_properties, spectra, spectra_norm)

if True:
    print(df_props.columns)
    plot_ttest(df_props, 'SD_frame_change', individ=True)
    plot_ttest(df_props, 'Mean_abs_change', individ=True)
    plot_ttest(df_props, 'Variance_dff', individ=True)
    #plot_ttest(df_props, 'SD_frame_change', individ=False)

    #plot_spectra(df_spectra, individ=True)
    #plot_spectra(df_spectra, individ=False)


#%%
'''
GET AMPLITUDE HISTOGRAMS
1. Import raw traces
2. Correct baseline via Suite2P with a 60s window and Gaussian smooting SD of 10 frames
3. Calculates non-negative estimate of underlying neuronal activity via OASIS
4. Create amplitude histogram of amplitude bins of 0.2 width for each axon
5. Fit exponential curve to mean of all axons and subtract from all axonal histograms
6. Summarize amplitudes within amplitude bands
'''

file_RawTraces_Fcorr_Deconv                       = fold / 'CA1Dopa_2pTube_Results_Spont_RawTraces_Fcorr_Deconv.csv'
file_RawTraces_Fcorr_Deconv_Histo                 = fold / 'CA1Dopa_2pTube_Results_Spont_RawTraces_Fcorr_Deconv_Histo.csv'
file_RawTraces_Fcorr_Deconv_Histo_Residuals       = fold / 'CA1Dopa_2pTube_Results_Spont_RawTraces_Fcorr_Deconv_Histo_Residuals.csv'
file_RawTraces_Fcorr_Deconv_Histo_CurveFit        = fold / 'CA1Dopa_2pTube_Results_Spont_RawTraces_Fcorr_Deconv_Histo_CurveFit.csv'
file_RawTraces_Fcorr_Deconv_Histo_Residuals_Stats = fold / 'CA1Dopa_2pTube_Results_Spont_RawTraces_Fcorr_Deconv_Histo_Residuals_Stats.csv'

bin_size = 0.02
bands = ((0.0, 0.5), (0.5, 3.1))


def deconvolve_trace(trace_raw):

    baseline = 'maximin'   # Suite2p's rolling max-of-min baseline
    sig_baseline = 10.0    # in "bins" (internal units used by Suite2p)
    win_baseline = 60.0    # seconds for baseline window
    batch_size   = 3000    # safe default for long traces
    tau          = 0.8

    signal = np.asarray(trace_raw, dtype=np.float32)[None, :]

    F_corr   = dcnv.preprocess(F=signal.copy(), baseline=baseline, win_baseline=win_baseline, sig_baseline=sig_baseline, fs=fps, device=torch.device("cpu"))
    F_deconv = dcnv.oasis(F=F_corr.copy(), batch_size=batch_size, tau=tau, fs=fps)
    
    return F_deconv.ravel()

def expofit(df_counts, bin_centers):

    def exponential(x, amplitude, decay):
        return amplitude*np.exp(-decay*x)

    pooled_mean = df_counts.mean(axis=1).to_numpy()

    param, _ = optimize.curve_fit(exponential, bin_centers, pooled_mean, p0=(pooled_mean.max(), 1.0), bounds=(0, np.inf), maxfev=10000)
    fitted_curve = exponential(bin_centers, *param)
    df_fitted_curve = pd.DataFrame({"Amplitude_bin_center": bin_centers,"curve": fitted_curve})

    df_residuals = df_counts.subtract(fitted_curve, axis=0)

    return df_residuals, df_fitted_curve

def band_mean_residuals(df_residuals, bin_centers, bands):

    band_means = {}

    for low, high in bands:

        label = f"{low}-{high}"

        # Select amplitude bins within this range
        in_band = (bin_centers >= low) & (bin_centers < high)
        selected_bins = df_residuals.loc[in_band]

        # Mean residual for each axon
        band_means[label] = selected_bins.mean(axis=0)

    # Rows = amplitude ranges; columns = axons
    df_band_means = pd.DataFrame(band_means).T
    df_band_means.index.name = "Amplitude_range"

    return df_band_means


def plot_histogram(tabel_histo, reference=None, individ=True):

    data = tabel_histo              if individ else tabel_histo.T.groupby([animal_id(c) for c in tabel_histo.columns]).mean().T
    unit = "Axons"                  if individ else "Animals"
    ylabel = "Sample count per bin" if reference is not None else "Residual sample count per bin"


    fig, ax = plt.subplots(figsize=(8, 4))

    for group in groups:
        cols = [c for c in data.columns if check_group(c) == group]
        values = data[cols]
        mean = values.mean(axis=1)
        sem = values.sem(axis=1)

        ax.plot(data.index, mean, color=colors[group],label=f"{group} (n={len(cols)})")
        ax.fill_between(data.index.to_numpy(), (mean - sem).to_numpy(), (mean + sem).to_numpy(),color=colors[group], alpha=0.2)

    if reference is not None:
        ax.plot(tabel_histo.index, reference, color="black", ls="--", label="Pooled exponential fit")

    ax.set(xlabel="Deconvolved amplitude [a.u.]", ylabel=ylabel, title=unit)
    ax.legend(frameon=False)
    fig.tight_layout()
    plt.show()

def plot_ttests(table, individ):

    data = table    if individ else table.T.groupby([animal_id(c) for c in table.columns]).mean().T
    unit = "Axons"  if individ else "Animals"


    fig, axes = plt.subplots(1, len(data.index), figsize=(4.5*len(data.index), 4), squeeze=False)
    results = []

    for ax, metric in zip(axes.ravel(), data.index):
        values_by_group = {}
        summary = {"Metric": metric, "Unit": unit}
        for x, group in enumerate(groups):
            cols = [c for c in data.columns if check_group(c) == group]
            values = data.loc[metric, cols].dropna().to_numpy(dtype=float)
            values_by_group[group] = values
            sem = stats.sem(values) if len(values) > 1 else np.nan
            summary.update({f"{group}_mean": values.mean(),
                            f"{group}_SEM": sem, f"{group}_n": len(values)})
            jitter = np.linspace(-0.05, 0.05, len(values)) if len(values) > 1 else 0
            ax.scatter(np.full(len(values), x)+jitter, values,
                       color=colors[group], alpha=0.8)
            ax.errorbar(x, values.mean(), yerr=sem if len(values) > 1 else None,
                        fmt="_", color="black", markersize=22, capsize=4)

        a, b = values_by_group["mKate"], values_by_group["A53T"]
        t, p = stats.ttest_ind(a, b, equal_var=False) if min(len(a), len(b)) >= 2 else (np.nan, np.nan)
        summary.update(t_stat=t, p_value=p)  # Two-sided, uncorrected Welch tests
        results.append(summary)
        ax.text(0.5, 0.95, f"p = {p:.4g}", transform=ax.transAxes, ha="center", va="top")
        ax.set_xticks([0, 1], [f"{g}\nn={len(values_by_group[g])}" for g in groups])
        ax.set(xlim=(-0.5, 1.5), ylabel="Sample count per bin", title=f"{metric} — {unit}")
        ax.margins(y=0.25)
        ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    plt.show()


df = pd.read_csv(file_RawTraces).iloc[:-2].reset_index(drop=True)
time_arr = np.arange(len(df)) / fps

histo_edges = np.arange(0, 5, bin_size)
bin_centers = (histo_edges[:-1] + histo_edges[1:]) / 2
traces_deconvolved, counts = {}, {}

for col in df.columns:

    trace_raw    = pd.to_numeric(df[col], errors='coerce').to_numpy()
    trace_deconv = deconvolve_trace(trace_raw)
    
    traces_deconvolved[col] = trace_deconv
    positive = trace_deconv[trace_deconv >= 1e-5]
    counts[col] = np.histogram(positive, bins=histo_edges)[0]


df_deconv = pd.DataFrame(traces_deconvolved)
df_histo  = pd.DataFrame(counts, index=pd.Index(bin_centers, name="Amplitude_bin_center"))


df_residuals, df_curve_fit = expofit(df_histo, bin_centers)
df_band_means              = band_mean_residuals(df_residuals, bin_centers, bands)

if False:
    df_deconv.to_csv(    file_RawTraces_Fcorr_Deconv)
    df_histo.to_csv(     file_RawTraces_Fcorr_Deconv_Histo)
    df_curve_fit.to_csv( file_RawTraces_Fcorr_Deconv_Histo_CurveFit)
    df_residuals.to_csv( file_RawTraces_Fcorr_Deconv_Histo_Residuals, index=False)
    df_band_means.to_csv(file_RawTraces_Fcorr_Deconv_Histo_Residuals_Stats)

if True:
    plot_histogram(df_histo, df_curve_fit['curve'], individ=True)
    plot_histogram(df_residuals, individ=True)
    plot_ttests(df_band_means, individ=True)
    plot_ttests(df_band_means, individ=False)
