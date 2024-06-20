#%%
import csv
import numpy as np
from matplotlib import pyplot as plt

files_1 = ['1-18.csv', '1-20.csv']
files_500 = ['500-19.csv', '500-21.csv']
files_100 = ['100-16.csv', '100-17.csv']
files = ['1-18.csv', '1-20.csv', '500-19.csv', '500-21.csv', '100-16.csv']

#files = ['1-18.csv']

x = list(np.arange(0, 840*5, 5) / 60)

def get_data(files):
    data = []
    for file in files:
        with open(file, 'r') as f:
            reader = csv.reader(f)
            data.append([float(value[0]) for value in list(reader)])
    
    return data

def clean_data(data):
    


    # trim lists
    reg_NE_start = 121
    starts = [25, 25, 121, 121, 74, 121]
    for i, rec in enumerate(data):

        # add np.nan to beginning of list if NE start before frame 121
        if starts[i] is not reg_NE_start:
            to_add = reg_NE_start - starts[i]
            nan_list = [np.nan] * to_add
            data[i] = nan_list + rec
        
        # if it's still too little, add more to the beginning
        while len(data[i]) < 840:
            data[i] = data[i] + [np.nan]

        # delete last values if rec too long
        while len(data[i]) > 840:
            del data[i][-1]
    
    
    return data

def scale_data(data):
    
    # scale data into 0-1 scale
    for i, rec in enumerate(data):
        
        liste = np.array(rec)
        liste = liste - np.nanmin(liste[:int(len(rec)/2)])
        liste = liste / np.nanmax(liste)
        
        data[i] = list(liste)

    return data


def nan_4_9999(data):
    for i, rec in enumerate(data):
        print(rec)
        data[i] = [np.nan if x == 9999.0 else x for x in data[i]]
    
    return data


def draw_scaled(data):
    
    for i in [0, 2, 4]:

        plt.plot(x, data[i], linewidth=1)
        plt.plot(x, data[i+1], linewidth=1)
        plt.ylim(-0.1, 1)
        plt.xlim(0,70)

        plt.xlabel('time [min]')
        plt.ylabel('scaled fluorescence')

        plt.axvline(x=10, color='grey', label='1uM NE', linestyle='--')
        plt.axvline(x=30, color='grey', label='1uM NE+DA', linestyle='--')
        plt.axvline(x=50, color='grey', label='washout', linestyle='--')

        plt.text(0, -0.1, 'baseline', fontsize = 10)
        plt.text(15, -0.1, 'NE', fontsize = 10)
        plt.text(35, -0.1, 'NE+DA', fontsize = 10)
        plt.text(55, -0.1, 'washout', fontsize = 10)

        plt.title('GRAB-DA2m HPC')
        plt.show()
        


def draw_zscored(data):

    plt.plot(x, data[0], linewidth=1, label='1uM')
    plt.plot(x, data[1], linewidth=1, label='500nM')
    plt.plot(x, data[2], linewidth=1, label='100uM')

    
        
    plt.xlim(0,70)

    plt.xlabel('time [min]')
    plt.ylabel('df/f0')

    plt.axvline(x=10, color='grey', linestyle='--')
    plt.axvline(x=30, color='grey', linestyle='--')
    plt.axvline(x=50, color='grey',  linestyle='--')

    txt = 0.965
    plt.text(0, txt, 'baseline', fontsize = 10)
    plt.text(15, txt, 'NE', fontsize = 10)
    plt.text(35, txt, 'NE+DA', fontsize = 10)
    plt.text(55, txt, 'washout', fontsize = 10)

    plt.title('GRAB-DA2m HPC')
    plt.legend()
    
    
    
    plt.show()
        

def zscore(data):
    for i, rec in enumerate(data):
        f0 = np.nanmean(rec[:121])
        data[i] = np.array(rec) / f0
    
    return data

    
def mean_traces(data):

    all_means = []
    for i in [0,2]:
        array1 = data[i]
        array2 = data[i+1]
        mean_array = []

        for i in range(len(array1)):
            mean_value = (array1[i] + array2[i]) / 2.0
            mean_array.append(mean_value)
        mean_array = list(mean_array)
        all_means.append(list(mean_array))
    all_means.append(data[4])
    return all_means


data = get_data(files)
del data[0][data[0].index(max(data[0]))]
for rec in data:
    del rec[:3]



data = clean_data(data)
#data = scale_data(data)
#draw_scaled(data)


# zscored
data = zscore(data)
data = mean_traces(data)


draw_zscored(data)


#%%


