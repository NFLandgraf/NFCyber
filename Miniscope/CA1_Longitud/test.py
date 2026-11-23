#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


file = Path(r"E:\CA1Dopa_Miniscope\casc\out\66_Week8_YM_cascade_scale3.csv")

cell = "C019"
start = 0
end = 646  # seconds
fps = 10

# Load data
df = pd.read_csv(file)

# Find corresponding columns
col_zscore = next(c for c in df.columns if f"{cell}_" in c and c.endswith("_zscore"))
col_spikeprob = next(c for c in df.columns if f"{cell}_" in c and c.endswith("_spikeprob"))

# Time axis
time = np.arange(len(df)) / fps

# Select time window
mask = (time >= start) & (time <= end)

# Plot
fig, ax1 = plt.subplots(figsize=(14, 5))

ax1.plot(time[mask],df.loc[mask, col_zscore],color="black",linewidth=1,label="Z-score")
ax1.set_ylabel("Z-score", color="black")
ax1.set_xlabel("Time (s)")

ax2 = ax1.twinx()
ax2.plot(time[mask],
df.loc[mask, col_spikeprob],color="red",linewidth=1,label="CASCADE")
ax2.set_ylabel("Expected spikes/frame", color="red")

ax1.set_title(f"{cell} | CASCADE vs. Z-score")

ax1.spines["top"].set_visible(False)
ax2.spines["top"].set_visible(False)
ax2.set_xlim(50,150)

plt.tight_layout()
plt.show()

# Diagnostics
spikes = df[col_spikeprob].to_numpy()

print(f"Cell: {cell}")
print(f"Total inferred spikes: {np.nansum(spikes):.2f}")
print(f"Maximum CASCADE prediction: {np.nanmax(spikes):.3f}")





# Sum CASCADE output between 90 and 100 seconds
start_spikes = 90
end_spikes = 100

mask_spikes = (time >= start_spikes) & (time < end_spikes)

spike_sum = df.loc[mask_spikes, col_spikeprob].sum()

print(f"Cell: {cell}")
print(f"Inferred spikes between {start_spikes}-{end_spikes}s: {spike_sum:.3f}")