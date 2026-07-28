#%%
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm


def shuffle_rows(mat, seed):
    
    rng = np.random.default_rng(seed)
    idx = rng.permutation(mat.shape[0])

    mat_shuffled = mat[idx]
    
    return mat_shuffled

def sort_rows(combined):
    # sort rows by their total activity (row sum)
    row_sums = combined.sum(axis=1)
    sort_idx = np.argsort(row_sums)  # descending
    combined = combined[sort_idx]

    return combined

def heatmap_vector(mat, out_path, vmin, vmax, cmap="viridis", figsize=(5,10)):

    fig, ax = plt.subplots(figsize=figsize)

    # pcolormesh expects grid edges; this simple call works for regular grids
    mesh = ax.pcolormesh(mat, cmap=cmap, vmin=vmin, vmax=vmax, shading="nearest")

    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("Value")

    ax.set_xlabel("Time (frames / samples)")
    ax.set_ylabel("Cells")

    # match imshow-style orientation (origin="lower")
    ax.set_ylim(0, mat.shape[0])

    fig.tight_layout()
    fig.savefig(out_path+'_heatmap_vector.pdf', format="pdf", bbox_inches="tight")
    plt.close(fig)

def heatmap(mat, out_path, vmin, vmax, cmap="viridis", figsize=(5,10)):
   
    plt.figure(figsize=figsize)
    im = plt.imshow(mat, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")

    cbar = plt.colorbar(im)
    cbar.set_label("Value")

    plt.xlabel("Time (frames / samples)")
    plt.ylabel("Cells")
    plt.tight_layout()
    plt.savefig(out_path+'_heatmap.pdf', format='pdf')
    #plt.show()
    plt.close()

def process_folder(folder_in, file_out, vmin=0, vmax=0.5, min_length=2078):

    folder_in = Path(folder_in)

    csv_files = sorted(folder_in.glob("*.csv"))
    combined_mats = []

    # goes through csvs in folder and creates a combined heatmap from all
    for csv in csv_files:

        df = pd.read_csv(csv, index_col=None)
        cols = [c for c in df.columns if str(c).endswith('spikeprob')]

        # trim recording and create matrix
        df = df.iloc[32:min_length]
        arr = df[cols].to_numpy()
        mat = arr.T

        combined_mats.append(mat)

    # create shuffled cells of same number
    combined = np.vstack(combined_mats)
    combined = shuffle_rows(combined, seed=None)
    combined = combined[:251]

    combined = sort_rows(combined)

    print((combined.shape))
    heatmap(combined, file_out, vmin, vmax)
    heatmap_vector(combined, file_out, vmin, vmax)



#group = 'mKate'
group = 'A53T'

folder_in = f"C:\\Users\\landgrafn\\Desktop\\a\\{group}"
file_out = f"C:\\Users\\landgrafn\\Desktop\\a\\{group}"

process_folder(folder_in, file_out)