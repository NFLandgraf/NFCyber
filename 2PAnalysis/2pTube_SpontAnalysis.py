#%%
import numpy as np
import glob
from suite2p.extraction import dcnv
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from pathlib import Path




#%%
# DECONVOLUTION

fs = 10.8067         # Hz, your frame rate
tau = 0.8        # s, axonal GCaMP often 0.15–0.35 s (faster than somata)
baseline = 'maximin'   # Suite2p's rolling max-of-min baseline
sig_baseline = 10.0    # in "bins" (internal units used by Suite2p)
win_baseline = 60.0    # seconds for baseline window
batch_size = 3000      # safe default for long traces

def load_trace_csv(path):
    # tries to load a single-column CSV (no header) or a column named 'F'
    try:
        arr = np.loadtxt(path, delimiter=',')
        if arr.ndim == 2:  # take first column if multiple
            arr = arr[:, 0]
    except Exception:
        # fallback if it has a header & named column
        import pandas as pd
        df = pd.read_csv(path)
        col = 'F' if 'F' in df.columns else df.columns[0]
        arr = df[col].to_numpy()
    return arr.astype(np.float32)

def deconvolve_raw_trace(F_raw, fs, tau, baseline='maximin', win_baseline=60.0, sig_baseline=10.0, batch_size=3000):
    # ensure float32 numpy and finite
    F = np.asarray(F_raw, dtype=np.float32)
    F = np.nan_to_num(F, nan=np.nanmedian(F), posinf=np.max(F), neginf=np.min(F))

    # preprocess expects shape (n_rois, T)
    F2 = F[None, :] if F.ndim == 1 else F
    # baseline / detrend (Suite2p-style)
    Fc2 = dcnv.preprocess(
        F=F2,
        baseline=baseline,
        win_baseline=win_baseline,
        sig_baseline=sig_baseline,
        fs=fs,
    )
    # deconvolution (OASIS)
    spks2 = dcnv.oasis(F=Fc2, batch_size=batch_size, tau=tau, fs=fs)

    # squeeze back to 1-D if input was 1-D
    Fc   = Fc2[0]   if F.ndim == 1 else Fc2
    spks = spks2[0] if F.ndim == 1 else spks2
    return Fc, spks

csv_paths = sorted(glob.glob(r"D:\2P_Analysis\Traces\raw\air\*.csv"))
results = {}
for p in csv_paths:
    F_raw = load_trace_csv(p)
    Fc, spks = deconvolve_raw_trace(F_raw, fs=fs, tau=tau)
    results[p] = {"F": F_raw, "Fc": Fc, "spks": spks}

    # make a DataFrame for saving
    df_out = pd.DataFrame({"F_raw": F_raw, "F_corr": Fc, "spks": spks})

    # construct output filename: same as input + "_deconv.csv"
    base = os.path.splitext(os.path.basename(p))[0]
    out_path = os.path.join(os.path.dirname(p), base + "_deconv.csv")
    df_out.to_csv(out_path, index=False)

    print(f"Saved {out_path}")


#%%
# EVENT DETECTION



path = r"D:\2P_Analysis\Traces\events"
fps = 10.8067

def get_files(path, common_name):

    # get files
    files = [file for file in Path(path).iterdir() if file.is_file() and common_name in file.name]
    
    print(f'\n{len(files)} files found')
    for file in files:
        print(file)
    print('\n')

    return files

files = get_files(path, 'Spont')



def mad_threshold_positive_tail(x, k, frac_low_pos, cap_q, min_pos=50):
    
    # Robust threshold when many samples are exactly zero.
    # Uses the lower 'frac_low_pos' of the *positive* values to estimate noise.
   
    x = np.asarray(x, dtype=np.float32).ravel()
    pos = x[x > 0]

    if pos.size >= min_pos:
        qpos = np.quantile(pos, frac_low_pos)          # e.g., 20th percentile of positives
        quiet = pos[pos <= qpos]                        # small positives ~ noise floor
    else:
        # Fallback: use lower bulk of all values (may still be okay if a few positives exist)
        qall = np.quantile(x, 0.6)
        quiet = x[x <= qall]

    med = np.median(quiet)
    mad = np.median(np.abs(quiet - med))
    sigma = 1.4826 * mad if mad > 0 else np.std(quiet)
    thr = med + k * sigma

    if cap_q is not None:
        thr = min(thr, float(np.quantile(x, cap_q)))   # cap threshold so it can't explode

    # If everything collapsed (all zeros), set a tiny epsilon threshold
    if not np.isfinite(thr) or sigma == 0:
        thr = float(np.quantile(x[x>0], 0.5)) if (x>0).any() else 0.0

    return float(thr), float(sigma), float(med)

def detect_events_spks(x, fs, k=3.5, frac_low_pos=0.2, refr_ms=500, cap_q=0.98, prom_k=1.0, smooth_win=3):
    # frac_low_pos: what fraction of all positive values should be considered as noise

    x = np.asarray(x, dtype=np.float32).ravel()
    if smooth_win and smooth_win > 1:
        x = np.convolve(x, np.ones(smooth_win)/smooth_win, mode='same')

    thr, sigma, med = mad_threshold_positive_tail(x, k=k, frac_low_pos=frac_low_pos, cap_q=cap_q)

    distance = int(round(refr_ms/1000.0 * fs))
    kwargs = {"height": thr, "distance": distance}
    if prom_k is not None and sigma > 0:
        kwargs["prominence"] = max(1e-9, prom_k * sigma)   # helps catch smaller but sharp peaks

    peaks, props = find_peaks(x, **kwargs)
    amps = props.get("peak_heights", x[peaks])
    return peaks, amps, thr

for file in files:
    print(file)

    spks = pd.read_csv(file)
    spks = spks.squeeze("columns")   

    idx, amps, thr = detect_events_spks(spks, fps)
    event_rate_hz = len(idx) / (len(spks)/fps)

    base = os.path.splitext(os.path.basename(file))[0]
    out_path = os.path.join(os.path.dirname(file), base + "_thresh.csv")
    pd.DataFrame({"idx": idx, "amps": amps}).to_csv(out_path, index=False)

    print(f'Thresh {thr}')
    print(f'Ratess {event_rate_hz}')
    print('\n\n')





#%%
# HISTOGRAM


# Load your data
df = pd.read_csv(r"D:\Traces\Spont_RawTraces_Fcorr_Spikes.csv")

# Define bins
bins = np.arange(0, 3.1, 0.02)
bin_centers = 0.5 * (bins[:-1] + bins[1:])

def get_bin_counts(trace, bins):
    trace = np.asarray(trace)
    trace = trace[trace >= 0.00001]
    counts, _ = np.histogram(trace, bins=bins)
    return counts

# Example: define groups (adapt to your naming convention)
groupA_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ["1001", "975", "971"])]
groupB_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ["1002", "976", "972"])]
groupAll_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ["_"])]
counts_A = []
counts_B = []
counts_all = []

for col in groupA_cols:
    counts_A.append(get_bin_counts(df[col], bins))
for col in groupB_cols:
    counts_B.append(get_bin_counts(df[col], bins))
for col in groupAll_cols:
    counts_all.append(get_bin_counts(df[col], bins))

# Convert to arrays for easy averaging
counts_A = np.vstack(counts_A)
counts_B = np.vstack(counts_B)
counts_all = np.vstack(counts_all)
mean_A = counts_A.mean(axis=0)
mean_B = counts_B.mean(axis=0)
mean_all = counts_all.mean(axis=0)
sem_A = counts_A.std(axis=0, ddof=1) / np.sqrt(counts_A.shape[0])
sem_B = counts_B.std(axis=0, ddof=1) / np.sqrt(counts_B.shape[0])
sem_all = counts_all.std(axis=0, ddof=1) / np.sqrt(counts_all.shape[0])

def save(counts_all):
    all_counts = pd.DataFrame(data = counts_all, columns = [f"bin_{b:.2f}" for b in bin_centers])
    all_counts.insert(0, "Trace", groupAll_cols)
    all_counts.to_csv(r"D:\2pTube\Spont_RawTraces_Fcorr_Spikes_Histo.csv", index=False)

    amp_bins_df = pd.DataFrame({"bin_index": np.arange(len(bin_centers)),"amplitude_center": bin_centers})
    amp_bins_df.to_csv(r"D:\2pTube\Spont_RawTraces_Fcorr_Spikes_HistoAmps.csv", index=False)

save(counts_all)

plt.figure(figsize=(7,4))
plt.plot(bin_centers, mean_A, label='Control', color='steelblue')
plt.fill_between(bin_centers, mean_A - sem_A, mean_A + sem_A, color='steelblue', alpha=0.3)
plt.plot(bin_centers, mean_B, label='aSyn', color='firebrick')
plt.fill_between(bin_centers, mean_B - sem_B, mean_B + sem_B, color='firebrick', alpha=0.3)
plt.xlabel("Spike amplitude")
plt.ylabel("Mean frequency (count)")
plt.title("Amplitude distribution across recordings")
plt.legend()
plt.tight_layout()
plt.show()


#%%
# CURVE FIT

'''
We see differences but its hidden under the exponential curve both groups are following, so we fit a curve, subtract it,
then everything is around zero and we compute differences via t test
'''

def exp_func(x, a, b):
    return a * np.exp(-b * x)

mean_all = counts_all.mean(axis=0)

# Fit exponential to the combined mean distribution
popt, _ = curve_fit(exp_func, bin_centers, mean_all, p0=(np.max(mean_all), 1))
a_fit, b_fit = popt
fit_y = exp_func(bin_centers, *popt)

residuals = counts_all - fit_y[None, :]
resA =      counts_A - fit_y[None, :]
resB =      counts_B - fit_y[None, :]

# (1) Save residuals (each row = one trace, each column = amplitude bin)
residuals_df = pd.DataFrame(residuals, columns=[f"bin_{b:.2f}" for b in bin_centers])
residuals_df.insert(0, "Trace", groupAll_cols)
residuals_df.to_csv(r"D:\2pTube\Spont_RawTraces_Fcorr_Spikes_Histo_Residuals.csv", index=False)
# (2) Save the fitted exponential curve
fit_y = np.array([float(a) for a in fit_y])
fit_df = pd.DataFrame({"Amplitude_bin_center": bin_centers, "Fitted_mean": fit_y})
fit_df.to_csv(r"D:\2pTube\Spont_RawTraces_Fcorr_Spikes_Histo_CurveFit.csv", index=False)

# Compute mean and SEM of residuals
mean_resA = resA.mean(axis=0)
mean_resB = resB.mean(axis=0)
sem_resA = resA.std(axis=0, ddof=1) / np.sqrt(resA.shape[0])
sem_resB = resB.std(axis=0, ddof=1) / np.sqrt(resB.shape[0])


plt.figure(figsize=(8,5))
plt.axhline(0, color='k', linestyle='--', linewidth=1, label='Global fit baseline')
plt.plot(bin_centers, mean_resA, color='steelblue', label='Group A (Control)')
plt.fill_between(bin_centers, mean_resA - sem_resA, mean_resA + sem_resA,
                 color='steelblue', alpha=0.3)
plt.plot(bin_centers, mean_resB, color='firebrick', label='Group B (aSyn)')
plt.fill_between(bin_centers, mean_resB - sem_resB, mean_resB + sem_resB,
                 color='firebrick', alpha=0.3)
plt.xlabel("Spike amplitude")
plt.ylabel("Residual count (obs - fit)")
plt.title("Residual amplitude distribution after global exponential subtraction")
plt.legend()
plt.tight_layout()
plt.show()


#%%
# check group differences


# --- Define your group identifiers ---
groupA_keys = ["1001", "975", "971"]
groupB_keys = ["1002", "976", "972"]

# --- Assign each row to a group based on Trace string ---
def assign_group(trace):
    trace_lower = str(trace).lower()
    if any(k in trace_lower for k in groupA_keys):
        return "GroupA"
    elif any(k in trace_lower for k in groupB_keys):
        return "GroupB"
    else:
        return "Unknown"

residuals_df["Group"] = residuals_df["Trace"].apply(assign_group)

# --- Separate groups ---
resA = residuals_df.loc[residuals_df["Group"] == "GroupA"].iloc[:, 1:-1].to_numpy()
resB = residuals_df.loc[residuals_df["Group"] == "GroupB"].iloc[:, 1:-1].to_numpy()

# --- Extract bin centers from column names ---
bin_centers = np.array([float(c.replace("bin_", "")) for c in residuals_df.columns[1:-1]])

# --- Compute mean and SEM ---
mean_resA = resA.mean(axis=0)
mean_resB = resB.mean(axis=0)
sem_resA = resA.std(axis=0, ddof=1) / np.sqrt(resA.shape[0])
sem_resB = resB.std(axis=0, ddof=1) / np.sqrt(resB.shape[0])

plt.figure(figsize=(8,5))
plt.axhline(0, color='k', linestyle='--', linewidth=1, label='Global fit baseline')
plt.plot(bin_centers, mean_resA, color='steelblue', label='Group A (Control)')
plt.fill_between(bin_centers, mean_resA - sem_resA, mean_resA + sem_resA,
                 color='steelblue', alpha=0.3)
plt.plot(bin_centers, mean_resB, color='firebrick', label='Group B (aSyn)')
plt.fill_between(bin_centers, mean_resB - sem_resB, mean_resB + sem_resB,
                 color='firebrick', alpha=0.3)
plt.xlabel("Spike amplitude")
plt.ylabel("Residual count (obs - fit)")
plt.title("Residual amplitude distribution per group (auto-detected)")
plt.legend()
plt.tight_layout()
plt.show()
