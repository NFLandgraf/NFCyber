#%%
# IMPORT and create DATAFRAMES
import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
import umap
from sklearn.cluster import DBSCAN
import itertools
import joypy
import seaborn as sns



data_prefix = 'C:\\Users\\landgrafn\\Desktop\\m90_data\\2024-10-31-16-52-47_CA1-m90_OF_'
video_dimensions = [664, 608]

file_TTL    =   data_prefix + 'GPIO.csv'
file_DLC    =   data_prefix + 'DLC.csv'
file_traces =   data_prefix + 'traces.csv'
file_events =   data_prefix + 'traces_events.csv'

def get_traces(file, round_after_comma=1):
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
    df.index = df.index.round(round_after_comma)

    df.columns = df.columns.str.replace(' C', '')
    df.columns = df.columns.astype(int)

    # normalize it via zscore
    df = dff_2_zscore(df)

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

def get_traces_events(file, df_traces, barsize=3):
    # takes the csv with the time of events/action potentials for each cell and returns df

    df_list = pd.read_csv(file, low_memory=False)

    # group by ' Cell Name' and collect the 'Time (s)' as lists before adding to dict
    events = df_list.groupby(' Cell Name')['Time (s)'].apply(list).to_dict()

    # get the maximum events that happen in a cell
    max_events = max(len(event_list) for event_list in events.values())

    # to have the same length, add nans to the event list of each cell 
    for key, event_list in events.items():
        events[key] = event_list + [np.nan] * (max_events - len(event_list))

    # make a df out of the dict and clean up the cell names
    df_list = pd.DataFrame(events)
    df_list.columns = sorted(df_list.columns.str.replace(' C', '').astype(int))
    df_list = df_list.round(1)


    # additionally, create a timeseries df with 0 and 1 for the events
    # copies df_traces but all values are 0
    df_timeseries = df_traces.astype('int64')
    for col in df_timeseries.columns:
        df_timeseries[col].values[:] = 0

    # for every event timepoint, put 1 in the respective row (+/- barsize)
    for col in df_list.columns:
        events = df_list[col].values
        events = events[~np.isnan(events)]
        for ev in events:
            df_timeseries.loc[ev-barsize : ev+barsize, col] = 1

    return df_list, df_timeseries
def get_TTLs(file, channel_name = ' GPIO-1', value_threshold=500, constant=True, print_csv=False):
    # returns list of TTL input times

    # expects nan values to be just a string: ' nan', convert these strings into proper nans
    df = pd.read_csv(file, na_values=[' nan'], low_memory=False)
    df = df.set_index('Time (s)')

    # only cares for the rows where the channel name is in the ' Channel Name' column
    df = df[df[' Channel Name'] == channel_name]
    
    # drops rows with nan and converts everything into ints
    df = df.dropna(subset=[' Value'])
    df[' Value'] = df[' Value'].astype(int)

    # creates series with True for values > threshold
    # checks at which index/time the row's value is high AND the row before's value is low
    # for that, it makes the whole series negative (True=False) and shifts the whole series one row further (.shift)
    # so, onset wherever row is > threshold and the opposite value of the shifted (1) row is positive as well
    is_high = df[' Value'] >= value_threshold
    onsets = df.index[is_high & ~is_high.shift(1, fill_value=False)]
    offsets = df.index[~is_high & is_high.shift(1, fill_value=False)]

    high_times = list(zip(onsets, offsets)) if not constant else list(onsets)
    #print(f'{len(high_times)} total events')

    if print_csv:
        with open(output, 'w') as f:
            wr = csv.writer(f, quoting=csv.QUOTE_ALL)
            wr.writerow(high_times)

    return high_times
def get_behav(file, df_TTLs, y_max):
    # cleans the DLC file and returns the df
    # idea: instead of picking some 'trustworthy' points to mean e.g. head, check the relations between 'trustworthy' points, so when one is missing, it can gues more or less where it is

    bps_all = ['nose', 'left_ear', 'right_ear', 'left_ear_tip', 'right_ear_tip', 'left_eye', 'right_eye', 'head_midpoint', 
                 'neck', 'mid_back', 'mouse_center', 'mid_backend', 'mid_backend2', 'mid_backend3', 
                 'tail_base', 'tail1', 'tail2', 'tail3', 'tail4', 'tail5', 'tail_end',
                 'left_shoulder', 'left_midside', 'left_hip', 'right_shoulder', 'right_midside', 'right_hip']
    bps_head = ['nose', 'left_eye', 'right_eye', 'head_midpoint', 'left_ear', 'right_ear', 'neck']
    bps_postbody = ['mid_backend', 'mid_backend2', 'mid_backend3', 'tail_base']

    df = pd.read_csv(file, header=None, low_memory=False)

    # combines the string of the two rows to create new header
    # df.iloc[0] takes all values of first row
    new_columns = [f"{col[0]}_{col[1]}" for col in zip(df.iloc[1], df.iloc[2])]     
    df.columns = new_columns
    df = df.drop(labels=[0, 1, 2], axis="index")

    # adapt index
    df.set_index('bodyparts_coords', inplace=True)
    df.index.names = ['Frame']

    # turn all the prior string values into floats and index into int
    df = df.astype(float)
    df.index = df.index.astype(int)

    # insert NaN if likelihood is below x
    for bp in bps_all:
        filter = df[f'{bp}_likelihood'] <= 0.9     
        df.loc[filter, f'{bp}_x'] = np.nan
        df.loc[filter, f'{bp}_y'] = np.nan
        df = df.drop(labels=f'{bp}_likelihood', axis="columns")
    
    # mean 'trustworthy' bodyparts (ears, ear tips, eyes, midpoint, neck) to 'head' column
    for c in ('_x', '_y'):
        df[f'head{c}'] = df[[bp+c for bp in bps_head]].mean(axis=1, skipna=True)
        df[f'postbody{c}'] = df[[bp+c for bp in bps_postbody]].mean(axis=1, skipna=True)

    # combine the x and y column into a single tuple column
    for bp in bps_all + ['head', 'postbody']:
        df[bp] = (list(zip(round(df[f'{bp}_x'], 1), round(y_max-df[f'{bp}_y'], 1))))
        df = df.drop(labels=f'{bp}_x', axis="columns")
        df = df.drop(labels=f'{bp}_y', axis="columns")   

    #df.to_csv('F:\\2024-10-31-16-52-47_CA1-m90_OF_DLC_clean.csv')
    df = df[['head', 'postbody']]

    # as an index for behavior, you have the frame number and the INSPX master time
    df['Time [TTL]'] = df_TTLs[:-1]
    df['Time [TTL]'] = df['Time [TTL]'].round(1)
    df.set_index([df.index, 'Time [TTL]'], inplace=True)

    return df

df_traces   =   get_traces(file_traces)
df_events_list, df_events_timeseries   =   get_traces_events(file_events, df_traces)
df_TTLs     =   get_TTLs(file_TTL)
df_behavior =   get_behav(file_DLC, df_TTLs, video_dimensions[1])

df_traces = df_traces.iloc[:6000, :]
df_behavior = df_behavior.iloc[:6000, :]

#%%
# Dimensionaly Reduction

def dimred_UMAP(df, n_neighbors=3, min_dist=0.1, n_components=4):

    # transposes df and reduces cell dimensions via UMAP
    df = df.T
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components)
    embedding = reducer.fit_transform(df)

    return embedding
embedding = dimred_UMAP(df_traces)

#%%
# Clustering

def cluster_DBSCAN(dim_red_data, eps=0.8, min_samples=9):

    # cluster via DBSCAN the dimred results
    dbscan_model = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan_model.fit(dim_red_data)
    clusters_nb = np.unique(clusters.labels_)
    #print(f"unique labels: {clusters_nb}")

    # create color map that includes gray for noise points
    unique_labels = np.unique(clusters.labels_)
    num_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    colors = plt.cm.tab10(np.linspace(0, 1, num_clusters)) if len(np.unique(clusters.labels_)) <= 10 else plt.cm.tab20(np.linspace(0, 1, num_clusters))
    colors = np.vstack((colors, np.array([0.5, 0.5, 0.5, 1])))
    label_colors = np.array([colors[label] if label != -1 else colors[-1] for label in clusters.labels_])

    # plot 1x 3D and 3x 2D plots
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    ax = fig.add_subplot(221, projection='3d')
    ax.scatter(dim_red_data[:, 0], dim_red_data[:, 1], dim_red_data[:, 2], c=label_colors, marker='o', s=50, alpha=0.6)
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_zlabel('UMAP 3')

    # legend for the clusters (including noise as a separate label)
    cluster_labels = [i for i in unique_labels if i != -1]
    cluster_colors = [colors[i] for i in unique_labels if i != -1]

    # Add legend to the 3D plot
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10) 
               for color in cluster_colors]
    ax.legend(handles=handles, labels=cluster_labels, title="Clusters", loc='upper left')

    axs[0, 1].scatter(dim_red_data[:, 0], dim_red_data[:, 1], c=label_colors, marker='o', s=50, alpha=0.6)
    axs[0, 1].set_xlabel('UMAP 1')
    axs[0, 1].set_ylabel('UMAP 2')
    axs[1, 0].scatter(dim_red_data[:, 1], dim_red_data[:, 2], c=label_colors, marker='o', s=50, alpha=0.6)
    axs[1, 0].set_xlabel('UMAP 2')
    axs[1, 0].set_ylabel('UMAP 3')
    axs[1, 1].scatter(dim_red_data[:, 0], dim_red_data[:, 2], c=label_colors, marker='o', s=50, alpha=0.6)
    axs[1, 1].set_xlabel('UMAP 1')
    axs[1, 1].set_ylabel('UMAP 3')

    axes = [ax] + [axs[i, j] for i in range(2) for j in range(2)]
    for ax in axes:
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    plt.tight_layout()
    #plt.show()
    plt.savefig('UMAP2.pdf', dpi=300)
    plt.close()

    return clusters.labels_, clusters_nb
def order_cells_UMAP(clusters, clusters_nb):

    # takes the cluster created from UMAP and sort the cells according to this for visuality
    df_UMAP_ordered = pd.DataFrame(index=df_traces.index)

    # for each cluster, append the cells included in the cluster
    # cell_clusters is a list of all cells that make up a  cluster
    cell_order, cell_clusters = [], []
    for cluster in clusters_nb[1:]:
        column_list = [index for index, value in enumerate(clusters) if value==cluster]
        for column in column_list:
            cell_order.append(df_traces.columns[column])
        cell_clusters.append(column_list)

    # add trash
    column_list = [index for index, value in enumerate(clusters) if value==clusters_nb[0]]
    for column in column_list:
        cell_order.append(df_traces.columns[column])
    cell_clusters.append(column_list)
        
    # create new df with UMAP-ordered cells
    df_UMAP_ordered = df_traces[cell_order]

    # create list with the lengths of the clusters
    cluster_lengths = list(itertools.accumulate([len(cluster) for cluster in cell_clusters]))

    return df_UMAP_ordered, cell_clusters, cluster_lengths
def activity_heatmap(df, mark_positions=None):
    df_t = df.T

    plt.imshow(df_t, aspect='auto', cmap='seismic', vmin=-3, vmax=3)
    plt.colorbar(label='z-score')

    tick_positions = range(0, len(df_t.columns), 500)
    plt.xticks(ticks=tick_positions, labels=df_t.columns[tick_positions], rotation=90)

    if mark_positions:
        for mark in mark_positions:
            plt.axhline(y=mark, color='black', linestyle='--', linewidth=0.5)

    plt.xlabel('Time [s]')
    plt.ylabel('Cells')
    plt.title('Activity Heatmap')
    #plt.show()
    plt.savefig('heatmap.pdf', dpi=300)
    plt.close()
def events_heatmap(df, mark_positions=None):
    df_t = df.T

    plt.imshow(df_t, aspect='auto', cmap='Reds')
    plt.colorbar(label='Event Presence')

    tick_positions = range(0, len(df_t.columns), 500)
    plt.xticks(ticks=tick_positions, labels=df_t.columns[tick_positions], rotation=90)

    if mark_positions:
        for mark in mark_positions:
            plt.axhline(y=mark, color='black', linestyle='--', linewidth=1.5)


    plt.xlabel('Time [s]')
    plt.ylabel('Cells')
    plt.title('Event Heatmap')
    plt.show()

# cluster the dimensionality reduced df_traces
clusters, clusters_nb = cluster_DBSCAN(embedding)
df_traces_UMAP, cell_clusters, cluster_lengths = order_cells_UMAP(clusters, clusters_nb)
activity_heatmap(df_traces_UMAP, cluster_lengths)

# sort the df_events according to the clustered df_traces
#df_events_heatmap = df_events_timeseries[df_traces_UMAP.columns]
#events_heatmap(df_events_heatmap, cluster_lengths)

#activity_heatmap(df_traces)





#%%
# print out jpg for every row in df_behav

def behavior_to_jpgs(df, width=50):
    folder = 'C:\\Users\\landgrafn\\Desktop\\m90_data\\DLC\\'

    for idx, row in df.iterrows():
       
        head_x, head_y = row['head']
        body_x, body_y = row['postbody']

        # Compute the direction vector from Point 1 to Point 2
        dx = body_x - head_x
        dy = body_y - head_y
        vector_length = np.sqrt(dx**2 + dy**2)
        
        # Normalize the direction vector
        dx /= vector_length
        dy /= vector_length
        ortho_dx = -dy
        ortho_dy = dx
        half_width = width / 2
        
        corner1 = (body_x + half_width * ortho_dx, body_y + half_width * ortho_dy)
        corner2 = (body_x - half_width * ortho_dx, body_y - half_width * ortho_dy)

        plt.figure(figsize=(6, 6))
        plt.fill([corner1[0], corner2[0], head_x], [corner1[1], corner2[1], head_y], color='gray')
        plt.scatter(head_x, head_y, color='black', s=100)
        plt.text(10, 580, f'{idx[0]}, {idx[1]}s', color='red')

        plt.xlim(0, 664)
        plt.ylim(0, 608)
        plt.xticks([])
        plt.yticks([])

        filename = f'{idx[0]}.jpg'
        plt.savefig(folder+filename, format='jpg')

        plt.close()

behavior_to_jpgs(df_behavior)




#%%

def joyplot(df, y_scale=5, y_offset_rate=1.6):

    # create joyplot with many traces on top of each other
    plt.figure(figsize=(10, 5))
    colors = sns.color_palette("tab10", n_colors=df.shape[1])

    y_offset = 0
    for i, col in enumerate(df.columns):
        plt.plot(df.index, df[col]/y_scale + y_offset, color=colors[i], linewidth=1)
        y_offset += y_offset_rate

    plt.ylim(-0.3, 7.5)
    plt.xlim(0, 600)
    plt.xlabel('Time [s]')
    plt.title('Thy1-GCaMP6s OpenField')
    plt.savefig('Exemplcolor.pdf', dpi=300)
    plt.close()
    #plt.show()

df = df_traces.rolling(window=5).mean()
#print(df_traces)
print(df.columns)
joyplot(df.loc[:, [19, 28, 178, 209]])





#%%



def calculate_speed_vector(old_df, x):
    df = old_df.copy()
    speed_vectors = []
    
    for i in range(len(df) - x):  # Iterate until we can compute a valid speed vector
        # Get the current point and the point x rows later (assuming 'head' is a tuple (x, y))
        current_point = df['head'].iloc[i]
        next_point = df['head'].iloc[i + x]
        
        # Check if any of the points is (NaN, NaN)
        if pd.isna(current_point[0]) or pd.isna(current_point[1]) or pd.isna(next_point[0]) or pd.isna(next_point[1]):
            speed_vectors.append(np.nan)
        else:
            # Calculate the differences in x and y coordinates
            dx = next_point[0] - current_point[0]
            dy = next_point[1] - current_point[1]
            
            # Calculate the speed vector (magnitude of change)
            magnitude = np.sqrt(dx**2 + dy**2)
            speed_vectors.append(magnitude)
    
    # The remaining rows should have NaN as there are no valid next points
    speed_vectors += [np.nan] * x  # Add NaN for rows without a valid next point

    # Store the calculated speed vectors in a new column
    df['speed'] = speed_vectors
    return df

#print(df_behavior)
x = 1
df = calculate_speed_vector(df_behavior, x)


speedo = df['speed'].astype(float)
print(speedo[100:200])
speedo.index = [x[1] for x in speedo.index]

#print(speedo)

#%%

x = 4.9
for i, row in enumerate(speedo.index):
    
    if i % 50 == 0:
        rowsum = speedo[row : row+x].sum()
        speedo[row : row+x] = rowsum

#print(speedo)
plt.figure(figsize=(10, 2))
plt.bar(speedo.index, speedo, color=(0,0,0))
plt.xlim(0,600)
plt.show()
# plt.savefig('speed.pdf', dpi=300)
# plt.close()