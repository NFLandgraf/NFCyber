#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore
import numpy as np

df = pd.read_excel('C:\\Users\\nicol\\Desktop\\GG.xlsx', sheet_name='Tabelle1')

df = df.set_index('ID')


df_AT8 = df.filter(like="AT8")
df_Iba = df.filter(like="Iba1")
df_GFAP = df.filter(like="GFAP")



#%%

AT8_mean = np.mean(df_AT8.values)
AT8_sd = np.std(df_AT8.values)
AT8_z = (df_AT8 - AT8_mean) / AT8_sd

Iba_mean = np.mean(df_Iba.values)
Iba_sd = np.std(df_Iba.values)
Iba_z = (df_Iba - Iba_mean) / Iba_sd

GFAP_mean = np.mean(df_GFAP.values)
GFAP_sd = np.std(df_GFAP.values)
GFAP_z = (df_GFAP - GFAP_mean) / GFAP_sd



#%%

plt.figure(figsize=(6, 4))
sns.heatmap(AT8_z, cmap="coolwarm", vmin=-1.1, vmax=3.2, annot=False)
plt.title("AT8_z")
plt.show()

plt.figure(figsize=(6, 4))
sns.heatmap(Iba_z, cmap="coolwarm", vmin=-2, vmax=2.8, annot=False)
plt.title("Iba_z")
plt.show()

plt.figure(figsize=(6, 4))
sns.heatmap(GFAP_z, cmap="coolwarm", vmin=-1.5, vmax=3, annot=False)
plt.title("GFAP_z")
plt.show()