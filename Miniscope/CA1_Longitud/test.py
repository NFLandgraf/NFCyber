#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

scaling = 3

file = Path(rf"E:\CA1Dopa_Miniscope\2026-11-23_NewRound_InvestigatingSmallSpikeprobValues\D_Master_10_merge_scale{scaling}\A53T_66_Week8_YM_final_scale{scaling}.csv")

cell = "C019"
start = 0
end = 646  # seconds
fps = 10

# Load data
df = pd.read_csv(file)

# Find corresponding columns
col_dff = next(c for c in df.columns if f"{cell}_" in c and c.endswith("_dff"))
col_spikeprob = next(c for c in df.columns if f"{cell}_" in c and c.endswith("_spikeprob"))

# Time axis
time = np.arange(len(df)) / fps

# Select time window
mask = (time >= start) & (time <= end)

# Plot
fig, ax1 = plt.subplots(figsize=(14, 5))

ax1.plot(time[mask],df.loc[mask, col_dff]*100*scaling,color="black",linewidth=1,label="df/f [%]")
ax1.set_ylabel("df/f [%]", color="black")
ax1.set_xlabel("Time (s)")

ax2 = ax1.twinx()
ax2.plot(time[mask],
df.loc[mask, col_spikeprob]*10,color="red",linewidth=1,label="CASCADE")
ax2.set_ylabel("Spikes/s", color="red")

ax1.set_title(f"Spikes vs Df/f - scaling {scaling}")

ax1.spines["top"].set_visible(False)
ax2.spines["top"].set_visible(False)
ax2.set_xlim(90,190)

plt.tight_layout()
plt.show()

# Diagnostics
spikes = df[col_spikeprob].to_numpy()

#print(f"Cell: {cell}")
#print(f"Total inferred spikes: {np.nansum(spikes):.2f}")
#print(f"Maximum CASCADE prediction: {np.nanmax(spikes):.3f}")



#%%

# Sum CASCADE output between 90 and 100 seconds
start_spikes = 90
end_spikes = 100

mask_spikes = (time >= start_spikes) & (time < end_spikes)

spike_sum = df.loc[mask_spikes, col_spikeprob].sum()

print(f"Cell: {cell}")
print(f"Inferred spikes between {start_spikes}-{end_spikes}s: {spike_sum:.3f}")