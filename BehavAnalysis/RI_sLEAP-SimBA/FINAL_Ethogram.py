#%%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

folder = r"C:\Users\landgrafn\Desktop\SimBA_MaschineResults2\RI1\TG"
out_pdf = r"C:\Users\landgrafn\Desktop\SimBA_MaschineResults2\Ethogram_TG.png"

green_behaviors = ["Investigate","AnogenitalSniff","Nose2nose","Following","Approach"]
orange_behaviors = ["Mounting","Chase","Circle"]
red_behaviors = ["Attack"]

csv_files = sorted([f for f in os.listdir(folder) if f.lower().endswith(".csv")])

dfs = []
lengths = []

for file in csv_files:
    df = pd.read_csv(os.path.join(folder, file))
    dfs.append((file, df))
    lengths.append(len(df))

min_len = min(lengths)

ethogram_rows = []

for file, df in dfs:
    df = df.iloc[:min_len]
    row_values = np.zeros(min_len)

    for i, row in df.iterrows():
        if any(b in df.columns and row[b] == 1 for b in red_behaviors):
            row_values[i] = 3
        elif any(b in df.columns and row[b] == 1 for b in orange_behaviors):
            row_values[i] = 2
        elif any(b in df.columns and row[b] == 1 for b in green_behaviors):
            row_values[i] = 1

    orange_count = np.sum(row_values == 2)
    ethogram_rows.append((file, row_values, orange_count))

# sort: most orange frames at the top
ethogram_rows = sorted(ethogram_rows, key=lambda x: x[2], reverse=True)

ethogram = np.vstack([row_values for file, row_values, orange_count in ethogram_rows])
sorted_files = [file for file, row_values, orange_count in ethogram_rows]
orange_counts = [orange_count for file, row_values, orange_count in ethogram_rows]

cmap = ListedColormap(["white", "green", "orange", "red"])

fig, ax = plt.subplots(figsize=(14, max(2, len(sorted_files)*0.3)))

thick = ethogram.copy()
rows, cols = thick.shape

spread = 2  # adjust as needed

for r in range(rows):

    # --- ORANGE (value = 2) ---
    orange_indices = np.where(ethogram[r] == 2)[0]
    for c in orange_indices:
        start = max(0, c - spread)
        end = min(cols, c + spread + 1)

        # only fill where it's NOT already red
        for cc in range(start, end):
            if thick[r, cc] != 3:
                thick[r, cc] = 2

    # --- RED (value = 3) ---
    red_indices = np.where(ethogram[r] == 3)[0]
    for c in red_indices:
        start = max(0, c - spread)
        end = min(cols, c + spread + 1)

        thick[r, start:end] = 3  # red overrides everything

ax.imshow(
    thick,
    aspect="auto",
    interpolation="nearest",
    cmap=cmap
)

ax.set_xlabel("Frame")
ax.set_ylabel("Recording")

ax.set_yticks(range(len(sorted_files)))
ax.set_yticklabels(
    [f"{file}  ({count} orange frames)" for file, count in zip(sorted_files, orange_counts)],
    fontsize=7
)

ax.set_ylim(len(sorted_files)-0.5, -0.5)

plt.tight_layout()
plt.savefig(out_pdf, dpi=300)
plt.close()

print("Saved:", out_pdf)