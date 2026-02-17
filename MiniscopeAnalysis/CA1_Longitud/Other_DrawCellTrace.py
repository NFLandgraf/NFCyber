#%%
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import stats

csv_file = r"D:\11_ALL_MERGE_trim\66_Zost2_YM_final_trim(15m).csv"
df = pd.read_csv(csv_file)

fig, ax_left = plt.subplots()

# left y-axis
ax_left.plot(df['66_Zost2_YM_C041_dfovernoise'].values, linewidth=1.0, color='blue')
ax_left.set_xlabel("frame")
ax_left.set_ylabel("df/noise", color='blue')
ax_left.set_ylim(-1, 2.1)

# right y-axis
ax_right = ax_left.twinx()
ax_right.plot(df['66_Zost2_YM_C041_spikeprob'].values, linewidth=1.5, alpha=0.8, color='orange')
ax_right.set_ylabel("spike probability", color='orange')
ax_right.set_ylim(0, 3)

plt.tight_layout()
plt.show()

