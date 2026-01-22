#%%

import pandas as pd
import matplotlib.pyplot as plt





file = r"D:\48_Week8_OF_final.csv"

df = pd.read_csv(file)

dff = df['48_Week8_OF_C011_dfovernoise']
spike = df['48_Week8_OF_C011_spikeprob']

print(max(spike))
plt.plot(dff.index, dff)
plt.plot(spike.index, spike/5)
plt.show()
