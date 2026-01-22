#%%

import os, sys
os.chdir(r"C:\Users\landgrafn\Cascade-master")
import pandas as pd
import numpy as np
import scipy.io as sio
import ruamel.yaml as yaml
yaml = yaml.YAML(typ='rt')
from pathlib import Path
from cascade2p import cascade
from cascade2p.utils import plot_dFF_traces, plot_noise_level_distribution, plot_noise_matched_ground_truth
from scipy.stats import zscore


folder_traces   = Path(r"D:\new\Neuro_7_CNMFe\traces")
folder_out      = Path(r"D:\new\Neuro_7_CNMFe")
model_path      = Path(r"C:\Users\landgrafn\Cascade-master\Pretrained_models\Global_EXC_10Hz_smoothing100ms")
frame_rate = 10.0

replace_what = '_motcor_trim_traces'
replace_with = ''

def load_neurons_x_time(csv_path):
   
    # Skip the "Time(s)/Cell Status" row
    df = pd.read_csv(str(csv_path), skiprows=[1])

    time_col = df.columns[0]
    time = df[time_col].to_numpy(dtype=np.float32)

    cell_names = list(df.columns[1:])

    neurons = df.iloc[:, 1:].to_numpy(dtype=np.float32)
    flipped = neurons.T / 10.0
    #np.save(r"D:\new\Neuro_7_CNMFe\flip.npy", flipped)

    return flipped, time, cell_names


files = sorted([p for p in folder_traces.iterdir() if p.is_file() and ('.csv' in p.name)])
print(f'{len(files)} files found')

df_QC_noise_all = pd.DataFrame(columns=["QC_Cascade_noise"])
df_QC_noise_all.index.name = "cell_ID"

for trace_file in files:

    print(f'------- {trace_file.stem} -------')

    # calculate
    traces_df, time, cell_names = load_neurons_x_time(trace_file)
    #traces_z = zscore(traces_df, axis=1)
    #spikeprob = cascade.predict(model_path, traces_df)
    noise_levels = plot_noise_level_distribution(traces_df, frame_rate)

    # renaming all cells
    prefix = trace_file.stem.replace(replace_what, replace_with)
    #cell_names_dfovernoise = [f"{prefix}_{c.replace(' C', 'C')}_dfovernoise" for c in cell_names]
    #cell_names_zscore = [f"{prefix}_{c.replace(' C', 'C')}_zscore" for c in cell_names]
    #cell_names_spikeprob = [f"{prefix}_{c.replace(' C', 'C')}_spikeprob" for c in cell_names]
    cell_names_QCnoise = [f"{prefix}_{c.replace(' C', 'C')}" for c in cell_names]

    # create dfs
    #df_dfovernoise  = pd.DataFrame(traces_df.T, columns=cell_names_dfovernoise)
    #df_zscore       = pd.DataFrame(traces_z.T, columns=cell_names_zscore)
    #df_spikeprob    = pd.DataFrame(spikeprob.T, columns=cell_names_spikeprob)
    df_QC_noise     = pd.DataFrame({"QC_Cascade_noise": noise_levels}, index=cell_names_QCnoise)
    df_QC_noise.index.name = "cell_ID"

    # merge and save
    #df_out = pd.concat([df_spikeprob, df_dfovernoise, df_zscore], axis=1)
    #df_out.to_csv(f'{folder_out}\\{prefix}_cascade.csv')
    #print(f'orig: {np.shape(traces_df)}')
    #print(f'outt: {df_out.shape}')

    # append to global df (row-wise)
    df_QC_noise_all = pd.concat([df_QC_noise_all, df_QC_noise], axis=0)

df_QC_noise_all.to_csv(f'{folder_out}\\QC_Cascade_final.csv')
        

