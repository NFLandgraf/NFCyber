#%%
'''
We have the raw traces that came from CNMFe and want to use CASCADE in the next step to identify events/spikes
Therefore, we need to clean the df_traces and transform the traces into df/f
'''
import pandas as pd
import numpy as np
from pathlib import Path

input = Path(r"D:\zest")
common_name = ''
baseline_percentile = 8

import pandas as pd
import matplotlib.pyplot as plt

def plot_neuron_traces(time, traces, title):
    
    plt.figure(figsize=(12, 3))
    plt.plot(time, traces)
    plt.xlabel("Time (s)")
    plt.title(title)
    plt.tight_layout()
    plt.show()






def get_files(path, common_name, suffix=".csv"):
    path = Path(path)
    files = sorted([p for p in path.iterdir() if p.is_file() and (common_name in p.name) and p.name.endswith(suffix)])
    print(f"\n{len(files)} files found")
    for f in files:
        print(f)
    print('\n\n')
    return files

def compute_dff(F_neurons_x_time, percentile):
    
    # F_neurons_x_time: array with shape (n_neurons, n_timepoints), baseline F0 per neuron (row-wise percentile)
    F0 = np.percentile(F_neurons_x_time, percentile, axis=1, keepdims=True)
    print(F0)
    F0[F0 == 0] = np.nan
    dff = (F_neurons_x_time - F0) / F0
    return dff

def zscore_per_neuron(data_neurons_x_time):
    
    # Z-score per neuron over time (row-wise).
    mean = np.nanmean(data_neurons_x_time, axis=1, keepdims=True)
    std = np.nanstd(data_neurons_x_time, axis=1, keepdims=True)
    std[std == 0] = np.nan
    z = (data_neurons_x_time - mean) / std
    return z

def process_csv_file(csv_path, baseline_percentile):

    filebase = csv_path.stem
    print(f"Processing {filebase}")
    
    df = pd.read_csv(csv_path)
    df_data = df.iloc[1:].copy()
    
    # get the time vector from the first column
    time_col_name = df.columns[0]
    time = pd.to_numeric(df_data[time_col_name], errors="coerce")
    
    # keep the neuron columns (C00, C01, ...)
    neuron_cols = df.columns[1:]
    neuron_cols_new = [f"{filebase}_{col}" for col in neuron_cols]
    neuron_cols_new = [col.replace("motcor_trim_traces_ ", "") for col in neuron_cols_new]
    F_time_x_neurons = (df_data[neuron_cols].apply(pd.to_numeric, errors="coerce").copy())
    F_time_x_neurons.columns = neuron_cols_new
    
    # Transpose to get neurons x time (row=neurons, columns=timepoints)
    F_neurons_x_time = F_time_x_neurons.T
    
    # Optional: set column names to time (so each column is the time stamp)
    F_neurons_x_time.columns = time.values
    F_neurons_x_time.index.name = "neuron"
    raw_array = F_neurons_x_time.values
    trace = F_neurons_x_time.loc["48_Week8_OF_C002"].values
    time  = F_neurons_x_time.columns.values
    plot_neuron_traces(time, trace, 'raw')

    
    # get df/f
    F_array = F_neurons_x_time.values  # (n_neurons, n_timepoints)
    dff_array = compute_dff(F_array, percentile=baseline_percentile)
    dff_df = pd.DataFrame(dff_array, index=F_neurons_x_time.index, columns=F_neurons_x_time.columns)
    trace = dff_df.loc["48_Week8_OF_C002"].values
    time  = dff_df.columns.values
    plot_neuron_traces(time, trace, 'df/f')

    # get zscore
    z_array = zscore_per_neuron(dff_array)
    z_df = pd.DataFrame(z_array, index=F_neurons_x_time.index, columns=F_neurons_x_time.columns)
    trace = z_df.loc["48_Week8_OF_C002"].values
    time  = z_df.columns.values
    plot_neuron_traces(time, trace, 'zscore')














    # --- Build output filenames ---
    base = csv_path.stem
    raw_out = csv_path.with_name(base + "_raw.csv")
    dff_out = csv_path.with_name(base + "_dff.csv")
    z_out   = csv_path.with_name(base + "_zscore.csv")
    raw_out_npy = csv_path.with_name(base + "_raw_npy.npy")
    dff_out_npy = csv_path.with_name(base + "_dff_npy.npy")
    
    # save
    F_neurons_x_time.to_csv(raw_out)
    dff_df.to_csv(dff_out)
    z_df.to_csv(z_out)
    np.save(raw_out_npy, raw_array)
    np.save(dff_out_npy, dff_array)
    
    print(f"✅ {raw_out.name}")
    print(f"✅ {dff_out.name}")
    print(f"✅ {z_out.name}\n")



csv_files = get_files(input, common_name)
for file in csv_files:
    process_csv_file(file, baseline_percentile=baseline_percentile)
