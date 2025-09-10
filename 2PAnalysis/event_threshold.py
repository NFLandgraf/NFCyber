#%%
import numpy as np
from scipy.signal import find_peaks
from pathlib import Path
import pandas as pd
import os

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

#%%

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
    
