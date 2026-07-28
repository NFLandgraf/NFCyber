#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

path_DA     = r"W:\_proj_CA1Dopa\Paper\FF\Heatmap\FF-DA_csv.csv"
path_DA_out = r"W:\_proj_CA1Dopa\Paper\FF\Heatmap\FF-DA_Heatmap_xsmooth_5.pdf"

path_NE     = r"W:\_proj_CA1Dopa\Paper\FF\Heatmap\FF-NE_csv.csv"
path_NE_out = r"W:\_proj_CA1Dopa\Paper\FF\Heatmap\FF-NE_Heatmap_xsmooth_5.pdf"

path_2P     = r"W:\_proj_CA1Dopa\Paper\2pTube\Heatmap\Heatmap_csv.csv"
path_2P_out = r"W:\_proj_CA1Dopa\Paper\2pTube\Heatmap\Heatmap_raw.pdf"


def heatmap_animals_time(df, path_out, vmin=-1.5, vmax=3, cmap="viridis"):

    t = df.index
    data = df.apply(pd.to_numeric, errors="coerce")

    matrix = data.to_numpy().T
    matrix = gaussian_filter1d(matrix, sigma=5, axis=1)
    animal_names = data.columns.to_list()

    print(matrix.shape)


    plt.figure(figsize=(10, 6))
    #im = plt.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    im = plt.pcolormesh(np.arange(matrix.shape[1]+1), np.arange(matrix.shape[0]+1), matrix, cmap=cmap, vmin=vmin, vmax=vmax, shading="flat")
    plt.gca().invert_yaxis()

    plt.colorbar(im)
    plt.yticks(np.arange(len(animal_names)), animal_names)
    plt.ylabel("Animal")

    n_time = len(t)
    xticks = np.arange(0, n_time, 1000)
    plt.xticks(xticks, [f"{t[i]:g}" for i in xticks])
    plt.xlabel("Time [s]")
    plt.tight_layout()

    plt.savefig(path_out, dpi=300)
    plt.show()
    plt.close()


df = pd.read_csv(path_DA)
df = df.set_index('Time[s]')
heatmap_animals_time(df, path_DA_out)

df = pd.read_csv(path_NE)
df = df.set_index('Time[s]')
heatmap_animals_time(df, path_NE_out)

# df = pd.read_csv(path_2P)
# df = df.set_index('Time[s]')
# df = df.drop(columns=['Frames'])
# heatmap_animals_time(df, path_2P_out)
