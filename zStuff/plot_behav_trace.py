#%%
from pathlib import Path
from collections.abc import Sequence
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.colors import PowerNorm
from scipy.ndimage import gaussian_filter


csv_file = r"C:\Users\landgrafn\Desktop\YM.csv"


def plot_mean_trace(df, canvas_size=(500, 500), linewidth=0.6):

    bodyparts = ["nose", "left_ear", "right_ear", "left_ear_tip", "right_ear_tip", "left_eye", "right_eye", "neck", "mid_back", "mouse_center", "mid_backend", "mid_backend2", "mid_backend3",
                 "tail_base", "left_shoulder", "left_midside", "left_hip", "right_shoulder", "right_midside", "right_hip", "head_midpoint"]
    width, height = canvas_size

    # Each resulting table has frames as rows and selected body parts as columns.
    x = df.xs("x", axis=1, level=1)[bodyparts].apply(pd.to_numeric, errors="coerce")
    y = df.xs("y", axis=1, level=1)[bodyparts].apply(pd.to_numeric, errors="coerce")

    valid = (np.isfinite(x) & np.isfinite(y)).all(axis=1)

    mean_x = x.mean(axis=1).where(valid)
    mean_y = y.mean(axis=1).where(valid)

    plt.plot(mean_x, mean_y, linewidth=linewidth)
    #plt.xlim(0, width)
    plt.ylim(height+80, 0)
    
    #plt.savefig('trace.pdf', format="pdf", bbox_inches=None, pad_inches=0)
    plt.show()
    plt.close()


def plot_mean_heatmap(df, bin_size=5, blur_sigma=1.5, gamma=1.0, vmin=0, vmax=None):

    bodyparts = ["nose", "left_ear", "right_ear", "left_ear_tip", "right_ear_tip", "left_eye", "right_eye", "neck", "mid_back", "mouse_center", "mid_backend", "mid_backend2", "mid_backend3",
                     "tail_base", "left_shoulder", "left_midside", "left_hip", "right_shoulder", "right_midside", "right_hip", "head_midpoint"]
    
    
    x = df.xs("x", axis=1, level=1)[bodyparts].apply(pd.to_numeric, errors="coerce")
    y = df.xs("y", axis=1, level=1)[bodyparts].apply(pd.to_numeric, errors="coerce")
    valid = (np.isfinite(x) & np.isfinite(y)).all(axis=1)

    mean_x = x.mean(axis=1).where(valid)
    mean_y = y.mean(axis=1).where(valid)

    # Passing y first gives an array with rows=y and columns=x.
    nx = int(np.ceil(max(mean_x) / bin_size))
    ny = int(np.ceil(max(mean_y) / bin_size))
    heat, _, _ = np.histogram2d(mean_y, mean_x, bins=(ny, nx))

    # Smooth spatial counts, then convert to percentage occupancy.
    if blur_sigma > 0:
        heat = gaussian_filter(heat, sigma=blur_sigma, mode="reflect")

    heat = 100 * heat / heat.sum()

    plt.imshow(heat, cmap="viridis", origin="upper", interpolation="nearest", norm=PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax))

    #plt.savefig('heatmap.pdf',format="pdf",dpi=300,bbox_inches="tight",pad_inches=0.05)
   






df = pd.read_csv(csv_file, header=[0, 1, 2], index_col=0)
df.columns = df.columns.droplevel(0)

plot_mean_trace(df)
plot_mean_heatmap(df)