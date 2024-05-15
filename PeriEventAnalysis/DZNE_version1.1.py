#%% 
# IMPORT & CLEAN

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats


file_traces = 'traces.csv'
file_GPIO = 'GPIO.csv'


def clean_df(file):
    # clean the csv file and return df

    df = pd.read_csv(file, low_memory=False)

    # drop rejected cells/columns
    rejected_columns = df.columns[df.isin([' rejected']).any()]
    df = df.drop(columns=rejected_columns)

    # Clean up df
    df = df.drop(0)
    df = df.astype(float)
    df = df.rename(columns = {' ': 'Time'})
    df['Time'] = df['Time'].round(1)
    df = df.set_index(['Time'])

    df.columns = df.columns.str.replace(' C', '')
    df.columns = df.columns.astype(int)

    # also return the df as the delta_f/f
    df_f = df.copy()

    # transform into zscores
    df_z = df.copy()
    for name, data in df_z.items():
        df_z[name] = (df_z[name] - df_z[name].mean()) / df_z[name].std()
    df_z.index.names = ['Time/zscore']

    return df_f, df_z



df_f, df_z = clean_df(file_traces)





#%%
# GET TTL EVENTS



#def events(file):
# returns a list with TTL lengths and a list with the time of their onset
value_threshold = 5000   # which value is definitely >value_low and <value_high
channel_name = ' GPIO-1'

TTL_duration_list = []
TTL_onset_list = []

df = pd.read_csv(file_GPIO, low_memory=False)



    
    

print(df)








#%% 
# PANEL A: EVENT-ALIGNED ACTIVITY HEATMAP

def event_align_df(df, stat_window=[-1, 1], vis_window=[-5, 5], color_border=[-4, 4]):
    # align the index to the event (time=0 at event)
    df.index = df.index - event

    # create new dfs for pre and post
    df_pre = df.loc[stat_window[0]:0, :]
    df_post = df.loc[0.1: stat_window[1], :]

    # calculate diff and store in dict
    peri_event_diff = {}
    for col in df.columns:
        diff = df_post[col].mean() - df_pre[col].mean()
        peri_event_diff[col] = diff

    # sort dictionary in ascending order and return list of cells
    sorted_peri_event_diff = [cell for cell, value in sorted(peri_event_diff.items(), key=lambda item: item[1], reverse=True)]
    df = df.reindex(columns=sorted_peri_event_diff)

    # drop every row outside of vis_window
    df = df.loc[vis_window[0] : vis_window[1], :]

    # adapt values 0-1 and clip at color_border
    scale_factor = 1/(color_border[1] - color_border[0])
    df = (df - color_border[0]) * scale_factor
    df = df.clip(0, 1)

    return df

def plot_event_aligned_activity_heatmap(df, color_border=[-4, 4]):
    # draw heatmap

    # define initial properties
    fig, axs = plt.subplots(1, 2, gridspec_kw={'width_ratios': [25, 1]})
    timeline = list(df.index.values)
    nb_cells = len(data.columns)
    data = data.transpose()
    gradient = np.vstack((np.linspace(1, 0, 256), np.linspace(1, 0, 256)))
    map_color = "seismic"

    im_left = axs[0].imshow(data, cmap=map_color, aspect="auto", extent=[timeline[0], timeline[-1], nb_cells, 0])
    axs[0].set_xlabel("Time from Event [s]")
    axs[0].set_ylabel("Neuron")
    axs[0].axvline(x=0, color="gray")

    im_right = axs[1].imshow(gradient.T, cmap=map_color, aspect="auto", extent=[0, 1, color_border[0], color_border[1]])
    axs[1].set_ylabel("z-score")
    axs[1].yaxis.set_label_position("right")
    axs[1].yaxis.tick_right()
    axs[1].set_yticks(np.arange(color_border[0], color_border[1]+1, 1)) 
    axs[1].xaxis.set_visible(False)
    
    fig.suptitle("Event-Aligned Activity Heatmap")
    plt.show()


df = df_z.copy(deep=True)
event = 130.5

df_event_aligned = event_align_df(df)
plot_event_aligned_activity_heatmap(df_event_aligned)







#%% PANEL B: EVENT-ALIGNED RESPONSES

df = df_z.copy(deep=True)
#print(df)
events = [130.5, 160.5, 1781.35, 1811.35, 1841.35, 1871.35, 1901.35]
stat_window = [-0.2, 0.2]
vis_window = [-1, 1]
rec_freq = 10   # how many data points per second

# create new df (index are cell numbers) for storing mean pre- and post-activites
df_diff = pd.DataFrame(index=df.columns)
df_diff.index.names = ['Cell/Event']

for i, event in enumerate(events):
    # gets rows with the wanted indexes according to stat_window
    event_pos = df.index.get_loc(round(event, 1))
    pre = df.iloc[int(event_pos+stat_window[0]*rec_freq) : event_pos]
    post = df.iloc[event_pos : int(event_pos+stat_window[1]*rec_freq)]

    # for Wilcoxon, we only need the post-pre-difference
    diff = (post.mean() - pre.mean())

    # insert a column for the difference in post_mean - pre_mean
    df_diff.insert(loc=len(df_diff.columns), column=i, value=diff)
#print(df_diff)

# Wilcoxon signed-rank test
responses = {'up' : [], 'down' : [], 'non' : []}
for i in range(len(df_diff)):
    row = df_diff.iloc[i]
    cell = df_diff.index[i]

    statistic, p_value = stats.wilcoxon(row, alternative='two-sided')

    if p_value <= 0.05:
        responses['up'].append(cell) if np.sum(row) > 0 else responses['down'].append(cell)
    else:
        responses['non'].append(cell)

cell = 0
df_cell = pd.DataFrame(columns=np.arange(vis_window[0], vis_window[1]+0.1, 0.1))
df_cell.index.names = ['Events/Time']

a = pd.Series(df.loc[events[0]+vis_window[0] : events[0]+vis_window[1], 0])


df_cell.loc[0] = a.values
print(df_cell)



