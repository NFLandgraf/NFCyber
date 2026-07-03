#%%
import os
from pathlib import Path
import numpy as np
import pandas as pd
import glob
from suite2p.extraction import dcnv
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.stats import sem, ttest_ind


'''
This code looks at the spontanteous recordings only and takes the raw brightness files from the axon pixels from FIJI and checks for underlying differences in the firing activity
1. Raw fluorescence trace
2. baseline-corrected fluorescence trace, F_corr
3. deconvolved activity trace, F_deconv
4. histogram of non-zero F_deconv values
5. exponential fit to mean histogram
6. residual histograms after subtracting the common exponential shape
'''

fold                                              = Path(r'E:\2pTube')
file_RawTraces                                    = fold / 'Spont_RawTraces.csv'
file_RawTraces_Fcorr                              = fold / 'Spont_RawTraces_Fcorr.csv'
file_RawTraces_Fcorr_Deconv                       = fold / 'Spont_RawTraces_Fcorr_Deconv.csv'
file_RawTraces_Fcorr_Deconv_Histo                 = fold / 'Spont_RawTraces_Fcorr_Deconv_Histo.csv'
file_RawTraces_Fcorr_Deconv_Histo_Residuals       = fold / 'Spont_RawTraces_Fcorr_Deconv_Histo_Residuals.csv'
file_RawTraces_Fcorr_Deconv_Histo_CurveFit        = fold / 'Spont_RawTraces_Fcorr_Deconv_Histo_CurveFit.csv'
file_RawTraces_Fcorr_Deconv_Histo_Residuals_Stats = fold / 'Spont_RawTraces_Fcorr_Deconv_Histo_Residuals_Stats.csv'


#%%
'''
DECONVOLUTION
F_corr: Slow baseline fluctuations were corrected using the Suite2p maximin baseline method with a 60s baseline window and a smooting parameter of 10.
F_deconv: The baseline-corrected fluorescence trace was then deconvoluted using the OASIS algorithm implemented in Suite2p with a calcium decay time constant of 0.8s.
The resulting deconcoluted signal was used as an estimate of inferred calcium-event activity over time.
'''
def deconvolve_raw_trace(F_raw, fps):

    tau = 0.8              # s, axonal GCaMP often 0.15–0.35 s (faster than somata)
    baseline = 'maximin'   # Suite2p's rolling max-of-min baseline
    sig_baseline = 10.0    # in "bins" (internal units used by Suite2p)
    win_baseline = 60.0    # seconds for baseline window
    batch_size = 3000      # safe default for long traces

    # ensure float32 numpy and finite, Suite2p expects shape: n_traces x time
    F_raw = np.asarray(F_raw, dtype=np.float32)
    F_raw = np.nan_to_num(F_raw, nan=np.nanmedian(F_raw), posinf=np.max(F_raw), neginf=np.min(F_raw))
    F_raw_reshape = F_raw[None, :]

    # Baseline correction
    F_corr = dcnv.preprocess(F=F_raw_reshape, baseline=baseline, win_baseline=win_baseline, sig_baseline=sig_baseline, fs=fps, device='cpu')
    
    # deconvolution (OASIS)
    F_deconv = dcnv.oasis(F=F_corr, batch_size=batch_size, tau=tau, fs=fps)

    return F_corr.reshape(-1), F_deconv.reshape(-1)

fps = 10.8056
df = pd.read_csv(file_RawTraces)
F_corr_all = pd.DataFrame()
F_deconv_all = pd.DataFrame()

for F_raw in df:
    F_corr, F_deconv = deconvolve_raw_trace(df[F_raw], fps)
    F_corr_all[F_raw] = F_corr
    F_deconv_all[F_raw] = F_deconv

# add index
F_corr_all.index = range(len(F_corr_all))
F_deconv_all.index = range(len(F_deconv_all))
F_corr_all.index.name = 'Frames'
F_deconv_all.index.name = 'Frames'
F_corr_all.insert(0, 'Time [s]', F_corr_all.index / fps)
F_deconv_all.insert(0, 'Time [s]', F_deconv_all.index / fps)

F_corr_all.to_csv(file_RawTraces_Fcorr)
F_deconv_all.to_csv(file_RawTraces_Fcorr_Deconv)


#%%
'''
HISTOGRAM
To compare the distribution of deconvolved activity amplitudes across recordings, histograms were generated from the F_deconv traces.
Non-zero values were binned using a band width of 0.02 and the number of samples falling into each amplitude bin was counted for each recording.
'''
def get_bin_counts(trace, bins):

    trace = np.asarray(trace)
    trace = trace[trace >= 0.00001]
    counts, _ = np.histogram(trace, bins=bins)
    return counts
def plot(df):
    a53t = ['1002', '976', '972']
    mKate = ['1001', '975', '971']

    plt.figure(figsize=(10, 5))
    x = df.index
    for col in df.columns:
        if any(s in col for s in mKate):
            color = 'gray'
        elif any(s in col for s in a53t):
            color = 'red'
        plt.plot(x, df[col], color=color, alpha=0.8, linewidth=1)

    plt.xlabel('amplitude bin')
    plt.ylabel('bin count')
    plt.show()

df = pd.read_csv(file_RawTraces_Fcorr_Deconv)
bins = np.arange(0, 3.1, 0.02)
bin_centers = np.round(0.5 * (bins[:-1] + bins[1:]), 2)

counts_all = []
recordings = [col for col in df.columns if '_' in col]
for col in recordings:
    counts_all.append(get_bin_counts(df[col], bins))

counts_all = np.vstack(counts_all)
df_counts = pd.DataFrame(data=np.asarray(counts_all).T, columns=recordings, index=bin_centers)
df_counts.index.name = 'bin_centers'
df_counts.to_csv(file_RawTraces_Fcorr_Deconv_Histo)
plot(df_counts)


#%%
'''
CURVE FIT
To remove the common exponential decay component of the amplitude distribution, bin counts were averaged across all recordings.
An exponential function was fitted to this mean histogram and subtracted from the histogram of each recording.
'''
def exponential_fit(df, bin_centers):

    # Fit exponential function to the mean distribution and subtract it from each individual histogram
    files_mean = df.mean(axis=1).values
    popt, _ = curve_fit(lambda x, a, b: a * np.exp(-b * x), bin_centers, files_mean, p0=(np.max(files_mean), 1))
    a_fit, b_fit = popt
    fit_y = a_fit * np.exp(-b_fit * bin_centers)

    # save the fitted exponential curve
    fit_y = np.array([float(a) for a in fit_y])
    fit_df = pd.DataFrame({"bin_centers": np.round(bin_centers, 2), "Fitted_mean": fit_y})
    fit_df.to_csv(file_RawTraces_Fcorr_Deconv_Histo_CurveFit, index=False)

    # subtract exponential function from traces
    residuals = df.T - fit_y[None, :]

    return residuals, fit_y
def plot(df, expofit=None):

    a53t = ['1002', '976', '972']
    mKate = ['1001', '975', '971']
    cols_a53t, cols_mKate = [], []

    plt.figure(figsize=(10, 5))
    x = df.index

    for col in df.columns:
        if any(s in col for s in mKate):
            color = 'gray'
            cols_mKate.append(col)
        elif any(s in col for s in a53t):
            color = 'red'
            cols_a53t.append(col)
        plt.plot(x, df[col], color=color, alpha=0.25, linewidth=1)

    # get the mean as well
    mean_mKate = df[cols_mKate].mean(axis=1)
    plt.plot(x, mean_mKate, color='gray', linewidth=3, label='mKate mean')
    mean_a53t = df[cols_a53t].mean(axis=1)
    plt.plot(x, mean_a53t, color='red', linewidth=3, label='A53T mean')

    # add expofit if wanted
    if expofit is not None:
        plt.plot(x, expofit, color='black', linewidth=2, label='exponential fit')

    plt.xlabel('amplitude bin')
    plt.ylabel('bin count')
    plt.legend()
    plt.show()

# average the counts for each bin of all files and fit exponential, then subtract exponential from histogram
df = pd.read_csv(file_RawTraces_Fcorr_Deconv_Histo, index_col=0)
bin_centers = df.index.values.astype(float)
residuals, fit_y = exponential_fit(df, bin_centers)

# save residuals (each row = one trace, each column = amplitude bin)
df_residuals = residuals.T
df_residuals.index = np.round(bin_centers, 2)
df_residuals.index.name = 'bin_centers'
df_residuals.to_csv(file_RawTraces_Fcorr_Deconv_Histo_Residuals)
plot(df, fit_y)
plot(df_residuals, None)


#%%
'''
STATISTICS
Splits the Histogram into several bins and calculates the mean per axon
'''
def bin_means(df):

    bins = [[0.01, 0.31], [0.31, 0.61], [0.61, 0.91], [0.91, 1.5]]
    all_bins = []
    for start, stop in bins:
        all_bins.append(float(df.loc[start:stop].mean()))
    return all_bins
def get_stast(df, cols_a53t, cols_mKate):


    df_stats = pd.DataFrame(index=df.index)
    df_stats.index.name = 'bin'

    for bin_label in df.index:
        values_a53t = df.loc[bin_label, cols_a53t].dropna().astype(float)
        values_mKate = df.loc[bin_label, cols_mKate].dropna().astype(float)

        t_stat, p_value = ttest_ind(values_a53t, values_mKate, equal_var=False)

        df_stats.loc[bin_label, 'A53T_mean'] = values_a53t.mean()
        df_stats.loc[bin_label, 'A53T_SEM'] = sem(values_a53t)
        df_stats.loc[bin_label, 'A53T_n'] = len(values_a53t)

        df_stats.loc[bin_label, 'mKate_mean'] = values_mKate.mean()
        df_stats.loc[bin_label, 'mKate_SEM'] = sem(values_mKate)
        df_stats.loc[bin_label, 'mKate_n'] = len(values_mKate)

        df_stats.loc[bin_label, 't_stat'] = t_stat
        df_stats.loc[bin_label, 'p_value'] = p_value
    
    return df_stats
def plot_stat(df, cols_a53t, cols_mKate, df_stats):

    plt.figure(figsize=(8, 5))
    bar_width = 0.25
    x_positions = np.arange(len(df.index))

    for i, bin_label in enumerate(df.index):

        # bar and scatter plot
        values_a53t = df.loc[bin_label, cols_a53t].dropna().astype(float)
        values_mKate = df.loc[bin_label, cols_mKate].dropna().astype(float)
        mean_a53t = values_a53t.mean()
        sem_a53t = sem(values_a53t)
        mean_mKate = values_mKate.mean()
        sem_mKate = sem(values_mKate)
        x_mKate = x_positions[i] - bar_width / 2
        x_a53t = x_positions[i] + bar_width / 2
        plt.bar(x_mKate, mean_mKate, width=bar_width, color='gray', alpha=0.5, yerr=sem_mKate, capsize=4, label='mKate')
        plt.bar(x_a53t, mean_a53t, width=bar_width, color='red', alpha=0.5, yerr=sem_a53t, capsize=4, label='A53T')
        plt.scatter(np.full(len(values_mKate),x_mKate)+np.random.uniform(-0.04,0.04,size=len(values_mKate)), values_mKate, color='gray', alpha=1, s=35)
        plt.scatter(np.full(len(values_a53t),x_a53t)+np.random.uniform(-0.04,0.04,size=len(values_a53t)), values_a53t, color='red', alpha=1, s=35)

        # stats
        p = round(df_stats.loc[bin_label, 'p_value'], 4)
        y_max = max(values_mKate.max(), values_a53t.max())
        y_min = min(values_mKate.min(), values_a53t.min())
        y_range = y_max - y_min
        y = y_max + 0.15 * y_range
        h = 0.05 * y_range
        plt.plot([x_mKate, x_mKate, x_a53t, x_a53t], [y, y + h, y + h, y], color='black', linewidth=1)
        plt.text((x_mKate + x_a53t) / 2, y + h, p, ha='center', va='bottom', color='black')

    plt.xticks(x_positions, df.index)
    plt.axhline(y=0, color='black', linewidth=1)
    plt.xlabel('bin')
    plt.ylabel('mean residual bin count')
    plt.tight_layout()
    plt.show()

df = pd.read_csv(file_RawTraces_Fcorr_Deconv_Histo_Residuals, index_col=0)
df_all = pd.DataFrame(index=[1,2,3,4])
df_all.index.name = 'bin'
a53t, mKate = ['1002', '976', '972'], ['1001', '975', '971']

for col in df:
    df_all[col] = bin_means(df[col])
df_all.to_csv(file_RawTraces_Fcorr_Deconv_Histo_Residuals_Stats)

cols_a53t = [col for col in df_all.columns if any(s in col for s in a53t)]
cols_mKate = [col for col in df_all.columns if any(s in col for s in mKate)]
df_stats = get_stast(df_all, cols_a53t, cols_mKate)
plot_stat(df_all, cols_a53t, cols_mKate, df_stats)
