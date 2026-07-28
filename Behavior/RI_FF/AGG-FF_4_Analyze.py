#%%
import pandas as pd
import numpy as np
from pathlib import Path
import os
import glob
import matplotlib.pyplot as plt
from scipy.stats import sem
from tqdm import tqdm

path = r"E:\AGG-FF\_test2\Attack"

Habit_end_s = 270
Recov_len_s = 100
signal_col = 'dff'
fps = 30


groups = {"WT":[], "TG":[], "GFP":[]}
groups_AllRI = {"RI1":[], "RI2":[], "RI3":[]}
groups_Behavs = {'Attacks':[], 'Threat':[], 'Social':[]}

files = glob.glob(os.path.join(path, "*.csv"))

def get_Intr_timepoints(df):

    def correct_Intr_occurence(df, min_block_s=2, valid_Intr_range_s=(270, 930), frame_col='behav_idx', fps=30):

        min_block_frames = int(min_block_s * fps)
        first_exp_Intr_frame, last_exp_Intr_frame = (int(point * fps) for point in valid_Intr_range_s)
        df_out = df.copy()

        # get all columns where all Intr_bodyparts are available
        intr_x_cols = [col for col in df_out.columns if col.startswith("Intr_") and col.endswith("_x")]
        intr_present = df_out[intr_x_cols].notna().all(axis=1)

        intr_frames = (df_out.loc[intr_present, frame_col].dropna().astype(int).sort_values())

        block_id = intr_frames.diff().ne(1).cumsum()
        valid_blocks = [block for _, block in intr_frames.groupby(block_id) if len(block) >= min_block_frames and block.iloc[0] >= first_exp_Intr_frame and block.iloc[-1] <= last_exp_Intr_frame]

        if len(valid_blocks) == 0:
            print("\nNo valid Intr block found.")
            print(f"Found Intr frames from {intr_frames.min()} to {intr_frames.max()}")
            return df_out

        first_valid_Intr_frame = valid_blocks[0].iloc[0]
        last_valid_Intr_frame = valid_blocks[-1].iloc[-1]

        outside_intr = ((df_out[frame_col] < first_valid_Intr_frame) | (df_out[frame_col] > last_valid_Intr_frame))
        intr_cols = [c for c in df_out.columns if c.startswith("Intr_") and (c.endswith("_x") or c.endswith("_y"))]
        df_out.loc[outside_intr, intr_cols] = np.nan

        return df_out

    df = correct_Intr_occurence(df)

    df.index = df['mastertime']
    first_Intr_occ = df["Intr_Center_x"].first_valid_index()
    last_Intr_occ  = df["Intr_Center_x"].last_valid_index()

    # change behavs to 0 before first_Intr_occ and after last_Intr_occ
    behavior_cols = ["Attack", "Threat", "Investigate", "Approach", "Following","Nose2nose", "Anogenital_Sniff", "Circle", "Chase","Mounting", "Agitated", "TailRattle"]

    outside_intr = (df.index < first_Intr_occ) | (df.index > last_Intr_occ)
    df.loc[outside_intr, behavior_cols] = 0

    mask = (df.index > first_Intr_occ) & (df["Investigate"] == 1)
    first_Investigate = df.loc[mask].index[0]

    return df, first_Intr_occ, last_Intr_occ, first_Investigate

def get_behav_timepoints(df, behaviors, min_len):

    active = df[behaviors].eq(1)

    # Label consecutive 0/1 blocks
    block_id = active.ne(active.shift(fill_value=False)).cumsum()

    starts = []

    # Iterate over only the active blocks
    for _, bout in df[active].groupby(block_id[active]):
        if len(bout) >= min_len:
            starts.append(float(bout.index[0]))

    return starts

for file in tqdm(files):

    # get files
    filename = os.path.basename(file)
    if '' in filename:

        df = pd.read_csv(file)
        df["Threat"] = (df[['Chase', 'Circle', 'Mounting']].eq(1).any(axis=1).astype(int))

        # get Intr data
        df, first_Intr_occ, last_Intr_occ, first_Investigate = get_Intr_timepoints(df)

        attack_starts = get_behav_timepoints(df, 'Attack', min_len=5)
        threat_starts = get_behav_timepoints(df, 'Threat', min_len=10)
        social_starts = get_behav_timepoints(df, 'Investigate', min_len=10)
        behaviors = [social_starts]

        for i, behav in enumerate(behaviors):
            # get range
            pre, x0, post = 10, behav, 30
            baseline = [-10, -5]

            for start_point in x0:
                range_start = int(start_point - pre)
                range_end = int(start_point + post)
                signal = df.loc[(df.index > range_start) & (df.index <= range_end), signal_col].copy()

                # get baseline and normalize to baseline
                signal.index = signal.index - start_point
                idx = signal.index.to_numpy()
                baseline_start = idx[np.argmin(np.abs(idx - baseline[0]))]
                baseline_end   = idx[np.argmin(np.abs(idx - baseline[1]))]
                baseline_values = signal.loc[baseline_start:baseline_end].mean()
                trace_baseline = signal - baseline_values
                time = idx


                if "WT" in filename:
                    groups['WT'].append(trace_baseline)
                elif "TG" in filename:
                    groups['TG'].append(trace_baseline)


                # if i == 0:
                #     groups_Behavs['Attacks'].append(trace_baseline)
                # elif i == 1:
                #     groups_Behavs['Threat'].append(trace_baseline)
                # elif i == 2:
                #     groups_Behavs['Social'].append(trace_baseline)

            # if "RI1" in filename:
            #     groups_AllRI['RI1'].append(trace_baseline)
            # elif "RI2" in filename:
            #     groups_AllRI['RI2'].append(trace_baseline)
            # elif "RI3" in filename:
            #     groups_AllRI['RI3'].append(trace_baseline)


            # # add traces to respective dictionaries
            # if "WT" in filename:
            #     groups_FirstContact["WT"].append(trace_baseline)
            # elif "TG" in filename:
            #     groups_FirstContact["TG"].append(trace_baseline)
            # elif "GFP" in filename:
            #     groups_FirstContact["GFP"].append(trace_baseline)

#%%

colors = {"WT":"tab:gray", "TG":"tab:red", "GFP":"tab:green"}
#colors = {"RI1":"tab:blue", "RI2":"tab:orange", "RI3":"tab:purple"}
#colors = {"Attacks":"tab:red", "Threat":"tab:orange", "Social":"tab:green"}



groups.pop("GFP", None)
plt.figure(figsize=(12,5))

for group, signals in groups.items():

    signals = np.vstack(signals)

    # Plot individual animals
    #for signal in signals:
        #plt.plot(time, signal, color=colors[group])

    # Group mean ± SEM
    mean = signals.mean(axis=0)
    error = sem(signals, axis=0)

    plt.vlines(x=0, ymin=-10, ymax=10, color='black')
    plt.ylim(-0.1, 0.45)

    if group == 'WT':
        plt.plot(time, mean, color=colors[group], lw=3, label=f"{group} (n={len(signals)}, N=3)")
    elif group == 'TG':
        plt.plot(time, mean, color=colors[group], lw=3, label=f"{group} (n={len(signals)}, N=2)")
    plt.fill_between(time, mean-error, mean+error, color=colors[group], alpha=0.25)

plt.xlabel("Time (s)")
plt.ylabel("ΔF/F")
plt.title(f"AGG-FF_Social_PeriEvents (Only AGG videos)")
plt.legend(frameon=False)
plt.tight_layout()
plt.show()

