#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore

df = pd.read_excel('C:\\Users\\landgrafn\\Desktop\\aa.xlsx', sheet_name='Tabelle1')  # Passe den Pfad und Sheet-Namen an

df = df.set_index('ID')


df_AT8 = df.filter(like="AT8")
df_Iba = df.filter(like="Iba1")
df_GFAP = df.filter(like="GFAP")


print(df_AT8)
# print(df_Iba)
# print(df_GFAP)

#%%

print(type(df_AT8.loc[200, 'AT8 POA %']))




#%%
df_AT8 = df_AT8.apply(pd.to_numeric, errors='coerce')

df_AT8_z = df_AT8.apply(zscore)
print(df_AT8_z.var())
