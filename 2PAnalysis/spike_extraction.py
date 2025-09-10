#%%
import numpy as np
import glob
from suite2p.extraction import dcnv
import pandas as pd
import os

#%%
# ---- set your dataset-specific params ----
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

# ---- run over your ROI CSVs ----
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
print(results)

