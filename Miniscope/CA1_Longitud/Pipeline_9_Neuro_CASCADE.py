#%%
import os
os.chdir(r"C:\Users\landgrafn\Cascade-master")
import pandas as pd
import numpy as np
import ruamel.yaml as yaml
yaml = yaml.YAML(typ='rt')
from pathlib import Path
from cascade2p import cascade
from cascade2p.utils import plot_dFF_traces, plot_noise_level_distribution, plot_noise_matched_ground_truth
from scipy.stats import zscore
import matplotlib.pyplot as plt

'''
Takes traces (best case df/f) and models the underlying spike probabilities via CASCADE
CASCADE needs to be installed on PC
'''

folder_traces   = Path(r"E:\CA1Dopa_Miniscope\2026-11-23_NewRound_InvestigatingSmallSpikeprobValues\C_Neuro_7_CNMFe_Pyth_traces_raw_dff")
folder_out      = Path(r"E:\CA1Dopa_Miniscope\2026-11-23_NewRound_InvestigatingSmallSpikeprobValues\C_Neuro_9_Cascade_dff_scale3")
model_path      = Path(r"C:\Users\landgrafn\Cascade-master\Pretrained_models\Global_EXC_10Hz_smoothing100ms")

frame_rate = 10.0
dff_scaling = 3

replace_what = '_motcor_trim_traces_raw_dff'
replace_with = ''

def load_neurons_x_time(csv_path):
   
    # Skip the "Time(s)/Cell Status" row
    df = pd.read_csv(str(csv_path), skiprows=[1])

    time_col = df.columns[0]
    time = df[time_col].to_numpy(dtype=np.float32)

    cell_names = list(df.columns[1:])

    neurons = df.iloc[:, 1:].to_numpy(dtype=np.float32)
    flipped = neurons.T# / 10.0
    #np.save(r"D:\new\Neuro_7_CNMFe\flip.npy", flipped)

    return flipped, time, cell_names

def plot(cell_idx=7, start=0, end=None):

    if end is None:
        end = time[-1]

    # Select time window
    mask = (time >= start) & (time <= end)

    fig, ax1 = plt.subplots(figsize=(14, 5))

    # ΔF/F
    ax1.plot(time[mask],traces_dff[cell_idx, mask],color="black",linewidth=1,label="ΔF/F")
    ax1.set_ylabel("ΔF/F", color="black")
    ax1.tick_params(axis="y", labelcolor="black")

    # CASCADE
    ax2 = ax1.twinx()
    ax2.plot(time[mask],spikeprob[cell_idx, mask],color="red",linewidth=1,alpha=0.8,label="CASCADE")
    ax2.set_ylabel("Expected spikes/frame", color="red")
    ax2.tick_params(axis="y", labelcolor="red")

    ax1.set_xlabel("Time (s)")
    ax1.set_title(f"Neuron: {cell_names[cell_idx].strip()}")

    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)

    plt.tight_layout()
    plt.show()

files = sorted([p for p in folder_traces.iterdir() if p.is_file() and ('.csv' in p.name)])
print(f'{len(files)} files found')

df_QC_noise_all = pd.DataFrame(columns=["QC_Cascade_noise"])
df_QC_noise_all.index.name = "cell_ID"

for trace_file in files:

    print(f'------- {trace_file.stem} -------')
    print(f'dff_scaling: {dff_scaling}')

    # calculate
    traces_dff, time, cell_names = load_neurons_x_time(trace_file)

    # Scale ΔF/F for CASCADE
    traces_dff_scaled = traces_dff * dff_scaling
    traces_z = zscore(traces_dff, axis=1)

    # calculate
    spikeprob = cascade.predict(model_path, traces_dff_scaled, threshold=0)

    #plot(cell_idx=10, start=0, end=100)
    total_spikes = np.nansum(spikeprob, axis=1)
    max_spikes = np.nanmax(spikeprob, axis=1)
    print("\nCASCADE diagnostics")
    print("Median total inferred spikes:", np.nanmedian(total_spikes))
    print("Maximum total inferred spikes:", np.nanmax(total_spikes))
    print("Median peak prediction:", np.nanmedian(max_spikes))
    print("Maximum peak prediction:", np.nanmax(max_spikes))
    print("Cells with > 1 inferred spike:", np.sum(total_spikes > 1))
    print("Cells with > 5 inferred spikes:", np.sum(total_spikes > 5))
    print("Cells with > 10 inferred spikes:", np.sum(total_spikes > 10))

    # check noise
    noise_levels = plot_noise_level_distribution(traces_dff_scaled, frame_rate)
    print("Noise levels:")
    print("Min:", np.nanmin(noise_levels))
    print("Median:", np.nanmedian(noise_levels))
    print("Max:", np.nanmax(noise_levels))

    # renaming all cells
    prefix = trace_file.stem.replace(replace_what, replace_with)
    cell_names_dff = [f"{prefix}_{c.replace(' C', 'C')}_dff" for c in cell_names]
    cell_names_zscore = [f"{prefix}_{c.replace(' C', 'C')}_dff_zscore" for c in cell_names]
    cell_names_spikeprob = [f"{prefix}_{c.replace(' C', 'C')}_dff_spikeprob" for c in cell_names]
    cell_names_QCnoise = [f"{prefix}_{c.replace(' C', 'C')}" for c in cell_names]

    # create dfs
    df_dff          = pd.DataFrame(traces_dff.T, columns=cell_names_dff)
    df_zscore       = pd.DataFrame(traces_z.T, columns=cell_names_zscore)
    df_spikeprob    = pd.DataFrame(spikeprob.T, columns=cell_names_spikeprob)
    df_QC_noise     = pd.DataFrame({"QC_Cascade_noise": noise_levels}, index=cell_names_QCnoise)
    df_QC_noise.index.name = "cell_ID"

    # merge and save
    df_out = pd.concat([df_spikeprob, df_dff, df_zscore], axis=1)
    df_out.to_csv(f'{folder_out}\\{prefix}_cascade_dff_scale{dff_scaling}.csv')
    print(f'orig: {np.shape(traces_dff)}')
    print(f'outt: {df_out.shape}')

    # append to global df (row-wise)
    df_QC_noise_all = pd.concat([df_QC_noise_all, df_QC_noise], axis=0)

df_QC_noise_all.to_csv(f'{folder_out}\\QC_Cascade_final.csv')


