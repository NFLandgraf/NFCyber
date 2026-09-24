#%%
import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
#import joypy
import seaborn as sns



def joyplot(df, y_scale=5, y_offset_rate=1.6):

    # create joyplot with many traces on top of each other
    plt.figure(figsize=(10, 5))
    colors = sns.color_palette("tab10", n_colors=df.shape[1])

    y_offset = 0
    for i, col in enumerate(df.columns):
        plt.plot(df.index/10/60, df[col]/y_scale + y_offset, color='black', linewidth=1)
        y_offset += y_offset_rate

    #plt.ylim(-0.3, 7.5)
    #plt.xlim(0, 600)
    plt.xlabel('Time [min]')
    plt.title('Y-Maze_m151')
    #plt.savefig('Exemplcolor.pdf', dpi=300)
    #plt.close()
    plt.show()

file = r"E:\Minicortex\cascade_out\Minicortex_YM_m150_Neuro_trim_traces_cascade.csv"

df = pd.read_csv(file)
df = df[['Minicortex_YM_m150_Neuro_trim_traces_C0_spikeprob', 'Minicortex_YM_m150_Neuro_trim_traces_C0_zscore']]

#df = df.loc[:, 'Minicortex_YM_m150_Neuro_trim_traces_C0_spikeprob', ''Minicortex_YM_m150_Neuro_trim_traces_C0_zscore'']

df = df.rolling(window=5).mean()

joyplot(df)