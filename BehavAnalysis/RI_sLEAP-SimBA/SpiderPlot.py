#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def spider_plot_from_csv(csv_file, group_col="Group", groups=None):
    df = pd.read_csv(csv_file)

    sem_alpha=0.18
    fill_alpha=0.08



    colors = ['gray', 'red']

    behaviors = list(df.columns[2:])
    n = len(behaviors)
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    angles = np.r_[angles, angles[0]]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})

    for i, group in enumerate(groups):
        sub = df[df[group_col] == group]
        mean = sub[behaviors].mean().clip(lower=1e-3)
        sem = sub[behaviors].sem()

        mean_vals = np.r_[mean.values, mean.values[0]]
        upper_vals = np.r_[(mean + sem).values, (mean + sem).values[0]]
        lower_vals = np.r_[(mean - sem).values, (mean - sem).values[0]]

        color = colors[i % len(colors)]

        ax.plot(angles, mean_vals, linewidth=2.5, label=group, color=color)
        ax.fill_between(angles, lower_vals, upper_vals, alpha=sem_alpha, color=color)

    ax.set_yscale("log")
    ax.set_ylim(1, 10000)

    # log tick marks
    ax.set_yticks([1, 10, 100, 1000, 10000])
    ax.set_yticklabels([r"$10^0$", r"$10^1$", r"$10^2$", r"$10^3$", r"$10^4$"], fontsize=10)

    # place radial labels at the top
    ax.set_rlabel_position(90)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(behaviors, fontsize=10)

    ax.set_title("Spider plot", fontsize=14, fontweight="bold", pad=25)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.15), frameon=False)
    ax.spines["polar"].set_linewidth(1.2)
    ax.grid(True, alpha=0.4)

    plt.tight_layout()
    plt.show()


file = r"C:\Users\landgrafn\Desktop\Pyth.csv"

spider_plot_from_csv(csv_file=file, group_col="Group", groups=["WT", "TG"])