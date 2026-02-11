#%%
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

csv_path = Path(r"C:\Users\landgrafn\Desktop\CA1Dopa_Paper\FF\Heatmap\FFDA_Heatmap_csv.csv")
out_pdf  = Path(r"C:\Users\landgrafn\Desktop\CA1Dopa_Paper\FF\Heatmap\FFDA_Heatsrgrmap.pdf")

cmap = "viridis"
vmin = -3
vmax = 4
row_start = None
row_stop  = None

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["font.family"] = "DejaVu Sans"

df = pd.read_csv(csv_path)
if row_start is not None or row_stop is not None:
    df = df.iloc[row_start:row_stop]
time = df['Time[s]'].to_numpy()
df = df.drop(columns=['Time[s]'])
data = df.to_numpy(dtype=float).T
n_cells, n_time = data.shape

fig, ax = plt.subplots(figsize=(10, max(4, n_cells * 0.15)))
im = ax.imshow(data, aspect="auto", interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax, extent=[time[0], time[-1], n_cells, 0])

ax.set_xlabel("Time")
ax.set_ylabel("Cells")
ax.set_yticks(np.arange(0, n_cells))
ax.set_yticklabels(df.columns)

cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Activity")

fig.tight_layout()
fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
plt.show()
plt.close(fig)
