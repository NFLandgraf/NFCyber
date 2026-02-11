#%%
"""
extract_heatmaps.py

Usage:
 - Edit DATA_DIR and SUFFIX below, or call from another script.
 - Produces one PNG per CSV with matching columns, and optionally a combined PNG.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

def find_columns_with_suffix(df, suffix):
    """Return sorted list of column names that end with suffix."""
    return [c for c in df.columns if str(c).endswith(suffix)]

def shuffle_rows(mat, labels=None, seed=None):
    """
    Shuffle rows of mat (and labels, if provided) consistently.
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(mat.shape[0])

    mat_shuffled = mat[idx]
    labels_shuffled = None
    if labels is not None:
        labels_shuffled = [labels[i] for i in idx]

    return mat_shuffled, labels_shuffled

def build_matrix_from_columns(df, cols):
    """
    Build 2D numpy array shaped (n_cols, n_rows)
    where row i is column cols[i] across all dataframe rows (timepoints).
    """
    if len(cols) == 0:
        return None, []
    arr = df[cols].to_numpy()  # shape (n_rows, n_cols)
    # transpose -> (n_cols, n_rows)
    return arr.T, cols

def zscore_rows(mat, eps=1e-8):
    """Z-score each row (row mean 0, std 1)."""
    mean = np.nanmean(mat, axis=1, keepdims=True)
    std = np.nanstd(mat, axis=1, keepdims=True)
    std[std < eps] = 1.0
    return (mat - mean) / std

def plot_and_save_heatmap(mat, row_labels, out_path,
                          cmap="viridis",
                          vmin=None, vmax=None,
                          base_width=10,
                          height_per_row=0.2,
                          min_height=4,
                          max_height=25,
                          xlabel="Time (frames / samples)",
                          ylabel="Columns / cells",
                          show_colorbar=True):
    """
    mat: shape (n_rows, n_cols)
    """

    n_rows = mat.shape[0]

    # dynamic figure height
    height = height_per_row * n_rows
    height = max(min_height, min(height, max_height))
    figsize = (base_width, height)

    plt.figure(figsize=figsize)

    im = plt.imshow(
        mat,
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        origin="lower"
    )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if row_labels:
        n = len(row_labels)
        max_ticks = 10
        if n <= max_ticks:
            yticks = np.arange(n)
            yticklabels = row_labels
        else:
            idx = np.linspace(0, n - 1, max_ticks, dtype=int)
            yticks = idx
            yticklabels = [row_labels[i] for i in idx]
        plt.yticks(yticks, yticklabels)
    else:
        plt.yticks([])

    if show_colorbar:
        cbar = plt.colorbar(im)
        cbar.set_label("Value")

    plt.tight_layout()
    plt.savefig(r"C:\Users\landgrafn\Desktop\CA1Dopa_Paper\Mini\Heatmap\A53T.pdf", format='pdf')
    plt.close()
    print("Saved heatmap:", out_path)

def find_min_length(csv_files, suffix):
    min_len = None
    for csv in csv_files:
        df = pd.read_csv(csv)
        cols = [c for c in df.columns if str(c).endswith(suffix)]
        if len(cols) == 0:
            continue
        n = len(df)
        if min_len is None or n < min_len:
            min_len = n
    return min_len

def process_folder(data_dir, suffix,
                   out_dir=None,
                   do_zscore=False,
                   vmin=None, vmax=None,
                   use_percentile_vmin_vmax=None,
                   combined_output_name="combined_heatmap.png"):
    
    
    data_dir = Path(data_dir)
    if out_dir is None:
        out_dir = data_dir / "heatmaps"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(data_dir.glob("*.csv"))
    combined_mats = []
    combined_labels = []

    min_len = 2078

    print(f"Trimming all CSVs to shortest length: {min_len}")

    for csv in csv_files:
        try:
            df = pd.read_csv(csv, index_col=None)
        except Exception as e:
            print(f"Failed to read {csv}: {e}")
            continue

        cols = find_columns_with_suffix(df, suffix)
        if len(cols) == 0:
            print(f"No columns ending with {suffix} in {csv.name}")
            continue

        mat, labels = build_matrix_from_columns(df.iloc[:min_len], cols)
        # optional z-score each row
        if do_zscore:
            mat = zscore_rows(mat)

        # compute vmin/vmax if percentile requested
        vmn, vmx = vmin, vmax
        if use_percentile_vmin_vmax is not None:
            lo_pct, hi_pct = use_percentile_vmin_vmax
            vmn = np.nanpercentile(mat, lo_pct)
            vmx = np.nanpercentile(mat, hi_pct)

        # save per-file heatmap
        out_png = out_dir / f"{csv.stem}{suffix}_heatmap.png"
        plot_and_save_heatmap(mat, labels, out_png, vmin=vmn, vmax=vmx)

        # store for combined plotting
        combined_mats.append(mat)
        # prefix labels with filename to keep them unique
        combined_labels.extend([f"{csv.stem}:{lab}" for lab in labels])

    # optional combined heatmap (stack rows from all files)
    if len(combined_mats) > 0:
        combined = np.vstack(combined_mats)  # shape (sum(n_cols), n_timepoints)
        # If datasets have different time lengths (columns), pad or trim to min length:
        # Here we trim to minimum columns length for simplicity
        min_len = min(m.shape[1] for m in combined_mats)
        if any(m.shape[1] != min_len for m in combined_mats):
            print("Warning: varying time length across files — trimming to min length", min_len)
            combined = np.vstack([m[:, :min_len] for m in combined_mats])

        # compute combined vmin/vmax if requested
        vmn, vmx = vmin, vmax
        if use_percentile_vmin_vmax is not None:
            vmn = np.nanpercentile(combined, use_percentile_vmin_vmax[0])
            vmx = np.nanpercentile(combined, use_percentile_vmin_vmax[1])

        out_combined = out_dir / combined_output_name

        combined, combined_labels = shuffle_rows(combined, combined_labels, seed=None)
        print(len(combined_labels))
        plot_and_save_heatmap(combined, combined_labels, out_combined, vmin=vmn, vmax=vmx)
    else:
        print("No matching columns found in any CSVs.")


DATA_DIR = r"C:\Users\landgrafn\Desktop\CA1Dopa_Paper\Mini\Heatmap\a"
SUFFIX = "spikeprob"
OUT_DIR = None
DO_ZSCORE = False
VMN, VMX = 0, 1.5

process_folder(DATA_DIR, SUFFIX,
                out_dir=OUT_DIR,
                do_zscore=DO_ZSCORE,
                vmin=VMN, vmax=VMX,
                combined_output_name="all_files_heatmap.png")
