#%%
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt


csv = r"C:\Users\landgrafn\Desktop\CA1Dopa_Paper\Mini\traces\clean.csv"


def gaussian_smooth_columns(df, sigma_seconds=0.1, time_column='Time [s]', axis_time=0, mode="reflect", truncate=4.0):
    """
    Apply a 1D Gaussian filter to each column (trace) in a DataFrame where rows are time samples.

    Parameters
    ----------
    df : pd.DataFrame
        Input data. Rows are time points; columns are traces.
    sigma_seconds : float
        Gaussian std dev in *seconds*.
    time_column : str | None
        If provided, use this column as the time vector (in seconds) and do not smooth it.
        If None, assumes df.index is the time vector (in seconds) if numeric/float-like.
    axis_time : int
        Time axis (0 for rows, which matches your description).
    mode : str
        Boundary handling for scipy.ndimage.gaussian_filter1d (e.g., "reflect", "nearest", "mirror", "wrap").
    truncate : float
        Truncate filter at this many standard deviations.

    Returns
    -------
    pd.DataFrame
        Smoothed DataFrame with the same shape/labels.
    """
   
    
    t = pd.to_numeric(df[time_column]).to_numpy()
    data_cols = [c for c in df.columns if c != time_column]
    out = df.copy()
    
    dt = np.median(np.diff(t))
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Time vector must be increasing (positive dt).")

    sigma_samples = sigma_seconds / dt

    smoothed = {}
    for c in data_cols:
        x = pd.to_numeric(df[c], errors="coerce").to_numpy()
        smoothed[c] = gaussian_filter1d(x, sigma=sigma_samples, axis=axis_time, mode=mode, truncate=truncate)

    smoothed_df = pd.DataFrame(smoothed, index=df.index)

    
    out.loc[:, data_cols] = smoothed_df[data_cols].to_numpy()
    return out
   

df = pd.read_csv(csv)
df_smooth = gaussian_smooth_columns(df)

plt.figure()
plt.plot(df.index, df['66_Zost2_YM_C160_dfovernoise'], alpha=1.0)
plt.tight_layout()
plt.xlim(500,1000)
plt.show()

plt.figure()
plt.plot(df_smooth.index, df_smooth['66_Zost2_YM_C160_dfovernoise'], alpha=1.0)
plt.tight_layout()
plt.xlim(500,1000)
plt.show()

