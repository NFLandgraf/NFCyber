#%%
import pandas as pd
import numpy as np
from pathlib import Path
import os
import glob
import matplotlib.pyplot as plt
from scipy.stats import sem
from tqdm import tqdm

path = r"E:\AGG-FF\_test2\new_all"


all_behaviors = ["Attack", "Threat", "Social"]
all_RI = ["RI1", "RI2", "RI3"]
all_groups = ["WT", "TG", "GF"]

groups_RI = {f"{ri}_{group}": [] for group in all_groups for ri in all_RI}
groups_Behavs = {f"{behav}_{ri}_{group}": [] for behav in all_behaviors for ri in all_RI for group in all_groups}

files = glob.glob(os.path.join(path, f"**.csv"))

def get_Intr_timepoints(df):

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

def get_behav_timepoints(df, behavior, min_len):

    active = df[behavior].eq(1)

    # Label consecutive 0/1 blocks
    block_id = active.ne(active.shift(fill_value=False)).cumsum()

    starts = []

    # Iterate over only the active blocks
    for _, bout in df[active].groupby(block_id[active]):
        if len(bout) >= min_len:
            starts.append(float(bout.index[0]))

    return starts


def peri_event(df, event, pre, post, baseline, signal_col='dff'):

    # calculates the peri-event trace from the df
    range_start = int(event - pre)
    range_end = int(event + post)
    signal = df.loc[(df.index > range_start) & (df.index <= range_end), signal_col].copy()

    # get baseline and normalize to baseline
    signal.index = signal.index - event
    idx = signal.index.to_numpy()
    baseline_start = idx[np.argmin(np.abs(idx - baseline[0]))]
    baseline_end   = idx[np.argmin(np.abs(idx - baseline[1]))]
    baseline_values = signal.loc[baseline_start:baseline_end].mean()
    trace_baseline = signal - baseline_values

    return trace_baseline, idx


def compare_groups_in_RI(filename):

    group = next((g for g in ["WT", "TG", "GF"] if g in filename), None)
    ri = next((r for r in ["RI1", "RI2", "RI3"] if r in filename), None)

    if group is not None and ri is not None:
        return group, ri
           
    else:
        print('wrong')


for file in files:

    filename = os.path.basename(file)
    group, ri = compare_groups_in_RI(filename)
    print(filename)

    df = pd.read_csv(file).copy()
    df["Threat"] = (df[["Chase", "Circle", "Mounting"]].eq(1).any(axis=1).astype(int))

    # Intruder-related trace
    df, first_Intr_occ, last_Intr_occ, first_Investigate = get_Intr_timepoints(df)
    trace_baseline, time_Intr = peri_event(df, last_Intr_occ, pre=10, post=60, baseline=[-10, -5])
    groups_RI[f"{ri}_{group}"].append(trace_baseline)


    # Get behavior onset times
    behavior_events = {"Attack": get_behav_timepoints(df, "Attack", min_len=5),
                       "Threat": get_behav_timepoints(df, "Threat", min_len=10),
                       "Social": get_behav_timepoints(df, "Investigate", min_len=10)}
                            
    # Store all event traces for this animal
    rec_behav = {"Attack":[], "Threat":[], "Social":[]}
    for behav, behav_events in behavior_events.items():
        for event in behav_events:
            trace_baseline, time_Behav = peri_event(df, event, pre=10, post=60, baseline=[-10, -5])
            rec_behav[behav].append(trace_baseline)

    # get the mean trace for each behavior of that recording
    rec_behav_mean = {}
    for behav, signals in rec_behav.items():
        if len(signals) > 0:
            rec_behav_mean[behav] = np.mean(np.vstack(signals), axis=0)

    # add one animal mean trace to the large dictionary
    for behav, mean_trace in rec_behav_mean.items():
        groups_Behavs[f"{behav}_{ri}_{group}"].append(mean_trace)


groups_Behav = {behav: {ri: {group: groups_Behavs[f"{behav}_{ri}_{group}"] for group in all_groups} for ri in all_RI} for behav in all_behaviors}
groups_RI = {ri: {group: groups_RI[f"{ri}_{group}"]for group in all_groups} for ri in all_RI}



        

#%%

colors = {"WT":"tab:gray", "TG":"tab:red", "GF":"tab:green"}
#colors = {"RI1":"tab:blue", "RI2":"tab:orange", "RI3":"tab:purple"}
#colors = {"Attacks":"tab:red", "Threat":"tab:orange", "Social":"tab:green"}

attack_WT = [trace for ri in ["RI1", "RI2", "RI3"] for trace in groups_Behav["Attack"][ri]["WT"]]
attack_TG = [trace for ri in ["RI1", "RI2", "RI3"] for trace in groups_Behav["Attack"][ri]["TG"]]
attack_WT = np.vstack(attack_WT)
attack_TG = np.vstack(attack_TG)


def plot_common_struc(comparisons, plot_ind=False):

    plt.figure(figsize=(12,5))

    for group, signal in comparisons.items():
        if 'GF' in group:
            continue

        # get x and y coordinates
        time = time_Intr
        signals = np.vstack(signal)
        mean = signals.mean(axis=0)
        error = sem(signals, axis=0)

        if not plot_ind:
            plt.plot(time, mean, color=colors[group], lw=3, label=f"{group} (n={len(signals)})")
            plt.fill_between(time, mean-error, mean+error, color=colors[group], alpha=0.25)
            continue

        for signal in signals:
            plt.plot(time, signal, color=colors[group])
        
    ymin, ymax = plt.ylim()
    plt.vlines(0, ymin, ymax, color='black')
    plt.xlabel("Time (s)")
    plt.ylabel("ΔF/F")
    plt.title(f"AGG-FF_Social_RI3")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()

def plot_spec_struc(comparisons, plot_ind=False):

    plt.figure(figsize=(12, 5))

    for group in ["WT", "TG"]:

        # Combine this group's animals across RI1, RI2 and RI3
        time = time_Behav
        signal = [trace for ri in comparisons.values() for trace in ri[group]]
        signals = np.vstack(signal)
        mean = signals.mean(axis=0)
        error = sem(signals, axis=0)

        if not plot_ind:
            plt.plot(time,mean,color=colors[group],lw=3,label=f"{group} (n={len(signals)})")
            plt.fill_between(time,mean - error,mean + error,color=colors[group],alpha=0.25)
            continue
        for signal in signals:
                plt.plot(time, signal, color=colors[group])

    plt.axvline(0, color="black")
    plt.xlabel("Time (s)")
    plt.ylabel("ΔF/F")
    plt.title("AGG-FF Attack — all RIs")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()


#plot_common_struc(groups_Behav["Attack"]['RI1'])
plot_spec_struc(groups_Behav["Attack"])
