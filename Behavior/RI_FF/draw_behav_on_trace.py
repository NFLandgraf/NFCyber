#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_trace_with_behaviors(trace_csv, behavior_csv):
    trace = pd.read_csv(trace_csv)
    beh = pd.read_csv(behavior_csv)

    behaviors = [
        "Approach","Investigate","Following","AnogenitalSniff",
        "Nose2nose","Agitated","Mounting","Circle","Chase","Attack"
    ]

    behavior_colors = {
        "social": (0, 200, 0),
        "active": (255, 165, 0),
        "attack": (255, 0, 0)
    }
    behavior_colors = {k: tuple(np.array(v) / 255) for k, v in behavior_colors.items()}

    df = pd.merge(trace, beh, on="Frames", how="left").fillna(0)

    social_behaviors = ["Approach", "Investigate", "Following", "AnogenitalSniff", "Nose2nose"]
    active_behaviors = ["Circle", "Chase", "Agitated", "Mounting"]

    df["behavior_level"] = 0
    df.loc[df[social_behaviors].eq(1).any(axis=1), "behavior_level"] = 1
    df.loc[df[active_behaviors].eq(1).any(axis=1), "behavior_level"] = 2
    df.loc[df["Attack"].eq(1), "behavior_level"] = 3

    # 🚫 Remove behavior coloring for first 4730 frames
    df.loc[df["Frames"] < 4730, "behavior_level"] = 0

    level_colors = {
        1: behavior_colors["social"],
        2: behavior_colors["active"],
        3: behavior_colors["attack"]
    }

    x = df["Frames"].values
    x = x /30/60
    y = df["dff"].values

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x, y, color='k')

    vals = df["behavior_level"].values

    for level, color in level_colors.items():
        mask = (vals == level).astype(int)
        diff = np.diff(np.concatenate(([0], mask, [0])))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]

        for s, e in zip(starts, ends):
            ax.axvspan(
                x[s],
                x[e - 1],
                color=color,
                alpha=0.5
            )

    ax.set_xlabel("Time [min]")
    ax.set_ylabel("Noradrenaline release mPFC [df/f]")
    plt.tight_layout()
    plt.rcParams["savefig.dpi"] = 300
    plt.savefig(r"C:\Users\landgrafn\Desktop\2026-05-05_FF-PFC_87_RI_edit_pic.pdf", bbox_inches="tight")
    plt.show()

trace = r"E:\87\2026-05-05_FF-PFC_87_RI_dff.csv"
behav = r"E:\87\2026-05-05_FF-PFC_87_RI_edit_sLEAP_recover_DLC_SimBA.csv"

plot_trace_with_behaviors(trace, behav)