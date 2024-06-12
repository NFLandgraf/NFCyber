#%%
import csv
import numpy as np
from matplotlib import pyplot as plt

def scale(liste):
    liste = np.array(liste)

    liste = liste - min(liste)
    liste = liste /max(liste)

    return liste

files = ['GRAB1.csv', 'GRAB2.csv']

data = []
for file in files:
    with open(file, 'r') as f:
        reader = csv.reader(f)
        data.append([float(value[0]) for value in list(reader)])

# delete useless data
del data[0][:4]
del data[0][data[0].index(max(data[0]))]
del data[1][:4]
del data[1][-1]

# scale data into 0-1 scale
for i, grab in enumerate(data):
    grab = scale(grab)
    data[i] = grab


x = np.arange(0, len(data[0])*5, 5) / 60
for grab in data:
    plt.plot(x, grab)

plt.axvline(x=2, color='grey', label='1uM NE', linestyle='--')
plt.axvline(x=22, color='grey', label='1uM NE+DA', linestyle='--')
plt.axvline(x=42, color='grey', label='washout', linestyle='--')

plt.text(8, 0, '1uM NE', fontsize = 10)
plt.text(26, 0, '1uM NE+DA', fontsize = 10)
plt.text(48, 0, 'washout', fontsize = 10)

plt.xlabel('time [min]')
plt.ylabel('scaled fluorescence')

plt.title('GRAB-DA2m HPC')
plt.show()
