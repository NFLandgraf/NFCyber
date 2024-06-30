#%% 
# IMPORT & CLEAN

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import scipy.stats as stats


file_traces = 'traces.csv'
file_GPIO = 'GPIOs.csv'
rec_freq = 10   # how many data points per second
to_round = 1 if rec_freq <= 10 else 2


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
    df.index.names = ['Time/dff']
    df.index = df.index.round(to_round)

    df.columns = df.columns.str.replace(' C', '')
    df.columns = df.columns.astype(int)

    return df

def dff_2_zscore(df):
    # transform df/f into zscores
    df_z = df.copy()

    # transform into zscores
    df_z = df.copy()
    for name, data in df_z.items():
        df_z[name] = (df_z[name] - df_z[name].mean()) / df_z[name].std()
    df_z.index.names = ['Time/zscore']

    return df_z

df_f = clean_df(file_traces)
df_z = dff_2_zscore(df_f)



#%%
# TTL EVENTS

def event_list(file, channel_name=' GPIO-1', value_threshold=500, constant=False):
    # returns a list with the onset (and offset) timepoints, where certain channel is 'high'
    # choose constant=True fi you have e.g. camera frames to only get high onsets

    df = pd.read_csv(file, low_memory=False)
    df = df.set_index(['Time (s)'])
    
    # only take rows that show data for your channel name
    df = df[df[' Channel Name'] == channel_name]

    # iterate through rows and save on- and offsets
    high_times, curr_high_time = [], []
    value_low = True
    for idx in df.index:
        value = df.loc[idx, ' Value']

        if value_low and value >= value_threshold:
            curr_high_time.append(idx)
            value_low = False

        elif not value_low and value <= value_threshold:
            curr_high_time.append(idx)
            high_times.append(curr_high_time)
            curr_high_time = []
            value_low = True

    # if we have constant signal (e.g. for camera frames), let's just go with high onset tp
    if constant:
        high_times = [times[0] for times in high_times]
    
    print(f'{len(high_times)} total events in recording')
    return high_times

def events_length(high_times, length_rounding=2):

    # iterate through high_times and check each length, add info to respective list
    lengths_dict = {}
    for time in high_times:
        length = round(time[1] - time[0], length_rounding)
        
        if length not in lengths_dict:
            lengths_dict[length] = [time[0]]
        else:
            lengths_dict[length].append(time[0])
    
    # transform dict into df
    TTLs = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in lengths_dict.items()]))

    print(f'TTL durations: {list(TTLs.columns)}')
    #print(TTLs)
    return TTLs

def plot_TTLs(df, TTLs):
    # choose colormap for TTL lines
    cmap = matplotlib.colormaps['tab10']

    # iterate through TTL lengths and TTL length onsets and draw vertical line for each
    for i, col in enumerate(TTLs.columns):
        for idx in TTLs.index:
            value = TTLs.loc[idx, col]
            if idx == TTLs.index[0]:
                plt.axvline(x=value, label=col, linestyle='-', color=cmap(i))
            else:
                plt.axvline(x=value, linestyle='-', color=cmap(i))
    
    # take x axis range from the df_z
    plt.xlim(df.index[0], df.index[-1])
    plt.legend()
    plt.xlabel('Time [s]')
    plt.gca().yaxis.set_visible(False)
    plt.show()


high_times = event_list(file_GPIO)
TTLs = events_length(high_times)
plot_TTLs(df_z, TTLs)





#%% 
# PANEL A: EVENT-ALIGNED ACTIVITY HEATMAP

def heatmap_df(df, stat_window=[-1,1], vis_window=[-5,5], color_border=[-4,4]):
    # align the index to the event (time=0 at event)
    df = df.copy(deep=True)
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

def plot_event_aligned_activity_heatmap(df, color_border=[-4,4]):
    # draw heatmap

    # define initial properties
    fig, axs = plt.subplots(1, 2, gridspec_kw={'width_ratios': [25, 1]})
    timeline = list(df.index.values)
    nb_cells = len(df.columns)
    df = df.transpose()
    gradient = np.vstack((np.linspace(1, 0, 256), np.linspace(1, 0, 256)))
    map_color = "seismic"

    heatmap = axs[0].imshow(df, cmap=map_color, aspect="auto", extent=[timeline[0], timeline[-1], nb_cells, 0])
    axs[0].set_xlabel("Time from Event [s]")
    axs[0].set_ylabel("Neuron")
    axs[0].axvline(x=0, color="gray")

    colorscale = axs[1].imshow(gradient.T, cmap=map_color, aspect="auto", extent=[0, 1, color_border[0], color_border[1]])
    axs[1].set_ylabel("z-score")
    axs[1].yaxis.set_label_position("right")
    axs[1].yaxis.tick_right()
    axs[1].set_yticks(np.arange(color_border[0], color_border[1]+1, 1)) 
    axs[1].xaxis.set_visible(False)
    
    fig.suptitle("Event-Aligned Activity Heatmap")
    plt.show()


event = 130.5

df_event_aligned = heatmap_df(df_z)
plot_event_aligned_activity_heatmap(df_event_aligned)







#%% 
# PANEL B: EVENT-ALIGNED RESPONSES

def event_aligned_diff(df, events, stat_window=[-0.2,0.2]):
    # creates df with index=cells and columns=post-pre for each event
    # stat_window is the range pre and post event from which the mean is taken
    df = df.copy(deep=True)

    # create new df (index are cell numbers, columns will be post-pre for each event)
    df_diff = pd.DataFrame(index=df.columns)
    df_diff.index.names = ['Cell/Event(post-pre)']

    # go through events and add post-pre for every event for every cell as extra column
    for i, event in enumerate(events):
        # go to index of time of event, pre and post according to stat_window
        event_pos = df.index.get_loc(round(event, 1))
        pre = df.iloc[int(event_pos+stat_window[0]*rec_freq) : event_pos]
        post = df.iloc[event_pos : int(event_pos+stat_window[1]*rec_freq)]

        # for Wilcoxon, we only need the post-pre-difference
        diff = (post.mean() - pre.mean())

        # insert a column for the difference in post_mean - pre_mean
        df_diff.insert(loc=len(df_diff.columns), column=i, value=diff)
    
    return df_diff

def wilcoxon_test(df):
    # Wilcoxon signed-rank test: go through cells/index and calc p-value with event responses
    # input should be df with cells as index and post-pre for each event as columns
    responses = {'up' : [], 'down' : [], 'non' : []}

    for i in range(len(df)):
        row = df.iloc[i]
        cell = df.index[i]

        statistic, p_value = stats.wilcoxon(row, alternative='two-sided')

        # append cell to either up/down/non-modulated acc to p-value and event sum
        if p_value <= 0.05:
            responses['up'].append(cell) if np.sum(row) > 0 else responses['down'].append(cell)
        else:
            responses['non'].append(cell)

    return responses

def one_cell_mean_response_2_events(df, events, cell, vis_window=[-4, 4]):
    # returns the mean reponse of a cell to all events
    # If you get an error (e.g. length is different), there may be a problem with the timeseries,
    # e.g. [3.15001, 3.2501, 3.349999]
    
    # get the common [-1, -0.9, ..., 0.9, 1] index, for this, we need to round as well
    aligned_idx = (df.index - events[0]).round(to_round)
    common_idx = aligned_idx[(aligned_idx >= vis_window[0]) & (aligned_idx <= vis_window[1])]
    df_events = pd.DataFrame(index=common_idx)

    # takes each event and saves the zscores in vis_window to df_events
    for i, event in enumerate(events):
        # align the index to the event (time=0 at event)
        aligned_idx = (df.index - event).round(to_round)    # neccessary
        event_series = df.loc[(aligned_idx >= vis_window[0]) & (aligned_idx <= vis_window[1]), cell]
        df_events[f'Event_{i}'] = event_series.values
    
    # calc and return mean of all event responses of that cell
    df_events['cell_mean'] = df_events.mean(axis=1)

    return df_events['cell_mean']
    
def all_cells_mean_responses_2_events(df, events):
    # go through all cells/columns in df and collect their mean towards all events
    df_all_cells = pd.DataFrame()

    for cell in df.columns:
        mean_response = one_cell_mean_response_2_events(df, events, cell)
        mean_response.name = cell
        df_all_cells[f'Cell_{cell}'] = mean_response
    
    sem = df_all_cells.sem(axis='columns')
    df_all_cells['mean'] = df_all_cells.mean(axis='columns')
    df_all_cells['sem'] = sem

    return df_all_cells

def plot_event_aligned_pop_response(df):
    # plot all individual cell means plus the overall mean
    plt.plot(df, color='gray', alpha=0.1)
    plt.errorbar(df.index, df['mean'], yerr=df['sem'], color='red', alpha=0.2)
    plt.plot(df['mean'], color='black', alpha=1, label='mean')
    
    plt.legend()
    plt.ylabel('z-score')
    plt.xlabel('event-aligned time [s]')
    plt.title('Event-aligned population response')
    plt.show()


events = [130.5, 160.5, 1781.3, 1811.3, 1841.3, 1871.3, 1901.3]

df_diff = event_aligned_diff(df_z, events)
cell_responses2events = wilcoxon_test(df_diff)

cells_mean_2_events = all_cells_mean_responses_2_events(df_z, events)
plot_event_aligned_pop_response(cells_mean_2_events)






# show the mean zscore responses of all up/down/non-regulated cells in vis_window around event
