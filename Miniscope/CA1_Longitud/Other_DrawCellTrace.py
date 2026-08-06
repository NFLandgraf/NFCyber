#%%
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import stats


fig, ax_left = plt.subplots()

csv_file = r"E:\CA1Dopa_Miniscope\test\Out\66_Zost2_YM_motcor_trim_TracesDff.csv"
df = pd.read_csv(str(csv_file), skiprows=[1], index_col='Unnamed: 0')
trace1 = pd.to_numeric(df[' C062'], errors='coerce')

csv_file = r"E:\CA1Dopa_Miniscope\test\Out\66_Zost2_YM_motcor_trim_TracesDff_cascade.csv"
df = pd.read_csv(str(csv_file), index_col='Unnamed: 0')
trace2 = pd.to_numeric(df['66_Zost2_YM_motcor_trim_TracesDff_C062_spikeprob'], errors='coerce')
trace2.index = trace1.index

# left y-axis
ax_left.plot(trace1, linewidth=1.0, color='black')
ax_left.set_xlabel("frame")
ax_left.set_ylabel("df/noise", color='black')
#ax_left.set_ylim(-0.01, 0.07)

# right y-axis
ax_right = ax_left.twinx()
ax_right.plot(trace2, linewidth=1, color='orange')
ax_right.set_ylabel("spike probability", color='orange')
#ax_right.set_ylim(0, 3)

x = 540
plt.xlim(x,x+20)

plt.tight_layout()
plt.show()


#%%


csv_file = r"E:\CA1Dopa_Miniscope\test\Out\66_Zost2_YM_motcor_trim_TracesDff.csv"


   
# Skip the "Time(s)/Cell Status" row
df = pd.read_csv(str(csv_file), skiprows=[1], index_col='Unnamed: 0')

time_col = df.columns[0]
time = df[time_col].to_numpy(dtype=np.float32)

cell_names = list(df.columns[1:])

neurons = df.iloc[:, 1:].to_numpy(dtype=np.float32)
flipped = neurons# / 10.0
#np.save(r"D:\new\Neuro_7_CNMFe\flip.npy", flipped)


column_maxima = df.max()

max_column = column_maxima.idxmax()
max_value = column_maxima.max()

print("Column:", max_column)
print("Maximum value:", max_value)



#%%

csv_file = r"E:\CA1Dopa_Miniscope\test\Out\66_Zost2_YM_motcor_trim_TracesDff.csv"
df = pd.read_csv(str(csv_file), skiprows=[1], index_col='Unnamed: 0')
df = df[1:]

trace = pd.to_numeric(df[' C000'], errors='coerce')

plt.plot(trace, linewidth=1.0, color='black')
plt.xlim(5,150)
plt.show()
