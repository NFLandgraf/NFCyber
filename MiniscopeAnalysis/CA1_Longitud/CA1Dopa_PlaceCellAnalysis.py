#%%
# IMPORT
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import csv
from shapely.geometry import Polygon, Point

#%%
'''
To keep the recording comparable between animals, we need to filter certain things
-Speed: drop rows that are below a certain speed, where the animal only sits around
-Movement: trim the recording in the end, when a certain distance has been travelled, so that every recording had the same distance 
-Events: Delete cells that didnt fire enough
'''
def filter_recording(df):

    def speed_filter(df, fps=10, step=1, speed_thresh_mm=15.0):
        # remove rows where the animals moves below threshold

        x_col = 'head_x'
        y_col = 'head_y'

        # Compute frame-to-frame differences with step (to reduce noise) and get the distance. Convert it to speed.
        dx = df[x_col].diff(step)
        dy = df[y_col].diff(step)
        dist = np.sqrt(dx**2 + dy**2)
        speed = dist / (step / fps)     # in mm/s

        # Fill NaNs in first rows (where speed is 0) and set threshold
        speed = speed.fillna(0)
        mask = speed >= speed_thresh_mm

        # Apply mask to dataframe
        df_filtered = df[mask].copy()
        df_filtered['speed'] = speed[mask]

        return df_filtered

    def movement_filter(df, step=1, move_threshold_m=20):
        # only use x meters travelled for each recording to compare between animals. For this you need to know the distances travelled for each animal and take the minimum.
        
        # Compute total distance travelled
        dx = df['head_x'].diff(step)
        dy = df['head_y'].diff(step)
        step_dist = np.sqrt(dx**2 + dy**2).fillna(0)
        total_dist = step_dist.iloc[step::step].sum().round()
        print(f'Dist moved {round(total_dist/1000, 1)}m')

        # Cumulative distance
        cum_dist = np.cumsum(step_dist)
        cum_dist = np.insert(cum_dist, 0, 0)  # align with df length

        # Find first frame that is above threshold and trim the recording there
        mask = cum_dist >= move_threshold_m*1000
        if mask.any():
            cutoff_pos = np.argmax(mask)  # first True position
            cutoff_index = df.index[cutoff_pos]
            print(f'Movement threshold reached at {cutoff_index}s / {df.index[-1]}s')
        else:
            cutoff_index = df.index[-1]
            print('!Movement threshold never reached! Animals not in sync')

        df_trimmed = df.loc[:cutoff_index]
        
        return df_trimmed

    def event_filter(df, event_thresh=5):
        # drop cells that have less events than event_thresh in the whole recording

        spike_cols = [c for c in df.columns if "spike" in c]
        total_events = df[spike_cols].sum(axis=0)

        # identifies cells that are below threshold and deletes everything of those cells (z, dff, spike)
        low_event_cols = total_events[total_events < event_thresh].index
        drop_cells = [col.replace("spike_", "") for col in low_event_cols]

        drop_cols = [c for c in df.columns if any(cell in c for cell in drop_cells)]
        df_filtered = df.drop(columns=drop_cols)

        return df_filtered

    df_speed = speed_filter(df)
    df_move = movement_filter(df_speed)
    df_event = event_filter(df_move)

    return df_event

main_df = pd.read_csv(r"D:\CA1Dopa_Miniscope\live\CA1Dopa_Longitud_f48_Post1_YMaze_main.csv", index_col='Time', low_memory=False)
main_df = filter_recording(main_df)


#%%

# PLACE CELLS
def bin_cellspikes(main_df, arena_is_OF=False):
    # split the area into spatial bins and for each cell, check how many spikes happened in each bin

    def get_areas_OF(dimensions=(460,460), n_bins=3):

            width, height = dimensions
            x_edges = np.linspace(0, width, n_bins+1, dtype=int)
            y_edges = np.linspace(0, height, n_bins+1, dtype=int)

            areas_OF = {}
            for col in range(n_bins):
                for row in range(n_bins):
                    # corners of the square bin
                    x0, x1 = x_edges[col], x_edges[col+1]
                    y0, y1 = y_edges[row], y_edges[row+1]

                    poly = Polygon([(x0,y0), (x1,y0), (x1,y1), (x0,y1)])
                    label = f'bin_{row}_{col}'  # row j, col i
                    areas_OF[label] = poly
            
            return areas_OF

    def get_areas_YMaze(dimensions=(580,485), centerpoint=(285, 310), c_rad=100, pointing_up=True):

        width, height       = dimensions
        c_x, c_y            = centerpoint

        # center
        start_angle = 90 if pointing_up else -90
        angles = np.deg2rad([start_angle, start_angle+120, start_angle+240])
        middle_corner, left_corner, right_corner = ((c_x + c_rad*np.cos(a), c_y + c_rad*np.sin(a)) for a in angles)

        #left arm
        left_arm_end_lefter     = (0, 400)
        left_arm_end            = (0, height)
        left_arm_end_righter    = (100, height)

        #middle arm
        middle_arm_end_lefter   = (c_x-c_rad, 0)
        middle_arm_end_righter  = (c_x+c_rad, 0)

        #right arm
        right_arm_end_lefter    = (500, height)
        right_arm_end           = (width, height)
        right_arm_end_righter   = (width, 400)

        # define Polygons
        left_arm = Polygon([left_arm_end_righter, left_arm_end, left_arm_end_lefter, left_corner, middle_corner, ])
        middle_arm = Polygon([middle_arm_end_lefter, middle_arm_end_righter, right_corner, left_corner])
        right_arm = Polygon([right_arm_end_righter, right_arm_end, right_arm_end_lefter, middle_corner, right_corner])
        center = Polygon([middle_corner, left_corner, right_corner])

        areas_YM = {'L':left_arm, 'M':middle_arm, 'R':right_arm, 'C':center}

        return areas_YM
    
    areas = get_areas_OF() if arena_is_OF else get_areas_YMaze()

    # puts head coords into spatial_bins and save the bin_labels in main_df
    labels = []
    for x, y in zip(main_df["head_x"], main_df["head_y"]):
        point = Point(x, y)
        label = None
        for bin_id, poly in areas.items():
            if poly.contains(point):
                label = bin_id
                break
        labels.append(label)
    main_df["head_bin"] = labels


    # creates df with the respective bin activity per cell
    spike_cells = [col for col in main_df.columns if 'spike' in col]
    place_map = {}
    for cell in spike_cells:
        bin_counts = main_df.groupby('head_bin')[cell].sum()
        place_map[cell] = bin_counts

    place_df = pd.DataFrame(place_map).fillna(0).astype(int)
    place_df = place_df.sort_index()

    print(place_df)
    return place_df, areas

def identify_place_cells(main_df, place_df, fps=10, n_shuffles=100, significance_level=0.05):
    '''
    Computing spatial information via Skaggs formula:
    SI = ∑i ⋅ pi ⋅ ri/r ⋅ log2(ri/r)
    i   = index of spatial bin
    pi  = probability of generelly being in bin i
    ri  = firing rate in bin i
    r   = overall mean firing rate (weighted by occupancy)

    If neuron fires uniformly everywhere    -> SI is low
    If neuron fires mostly in one bin       -> SI is high
    If neuron is inactive or noisy          -> SI is 0 or unreliable
    0 random, 0.3-0.5 weak place tuning, >0.5 likely a place cell, >1.0 strong, reliable place-specific firing

    We see the observed spatial info for each cell, but is it random/by chance?
    Here, we simulate null distribution, as if the cell spike had no relationship to location.
    So, for each cell, we take the n_spikes and the bin_occupancy by the animal and randomly put the spikes
    into the available bin, just dependent on the occupancy probility of the animal. This creates our shuffled/random
    distribution, which we test our real value against. If our value is higher than e.g. the 95th percentile of
    the shuffled values, it is significant.
    '''
    def spatial_information(firing_freq, occupancy):
        # how many spatial information (SI) is cell carrying
        
        # how likely is it that animal is in bin
        occupancy_probability = occupancy / occupancy.sum()

        # what is the overall firing rate of the cell in the whole area during the recording
        avrg_firing_rate = np.dot(firing_freq, occupancy_probability)

        # loop through each bin and executes the Skaggs formula (above)
        # the info is 'bits of information per spike', so the higher, the more spatial info the spike carries
        info = 0
        for r, p in zip(firing_freq, occupancy_probability):
            if r > 0 and p > 0:
                info += p * (r / avrg_firing_rate) * np.log2(r / avrg_firing_rate)
        return info
    
    def get_spatial_info_real(place_df, occupancy):

        # get the firing frequency of each cell per bin
        occupancy_sec = occupancy / fps
        df_firing_freq = place_df.div(occupancy_sec, axis=0)

        # stores the spatial infos (SI) value for each cell in dict
        spatial_info_real = {}
        for cell in df_firing_freq.columns:
            info = spatial_information(df_firing_freq[cell].values, occupancy.values)
            spatial_info_real[cell] = info

        return spatial_info_real

    def get_spatial_info_shuffle(place_df, occupancy):

        occupancy_probs = occupancy / occupancy.sum()
        spatial_info_shuffled = {cell: [] for cell in place_df.columns}

        for _ in tqdm(range(n_shuffles)):
            for cell in place_df.columns:
                
                # distribute all spikes of that cell weighted into the available bins
                n_spikes = int(place_df[cell].sum())
                shuffled_bins = np.random.choice(a=occupancy.index, size=n_spikes, p=occupancy_probs)

                # count how many spikes landed in each bin and calculate firing rate per bin
                shuffled_counts = pd.Series(shuffled_bins).value_counts().reindex(occupancy.index, fill_value=0)
                shuffled_firing_rate = shuffled_counts / occupancy
            
                # compute spatial info
                info = spatial_information(shuffled_firing_rate.values, occupancy.values)
                spatial_info_shuffled[cell].append(info)
        
        return spatial_info_shuffled

    def check_significance(spatial_info_real, spatial_info_shuffled):
        
        # compare real_si to shuffled_si
        place_cells = []
        p_values = {}
        for cell in spatial_info_real:
            real_si = spatial_info_real[cell]
            shuffled_si = spatial_info_shuffled[cell]

            # True=1, False=0, np.mean takes mean of True/False (returns array of booleans if a shuffled_si is bigger than real_si)
            p = np.mean(np.array(shuffled_si) >= real_si)
            p_values[cell] = p

            if p < significance_level:
                place_cells.append(cell)

        return p_values, place_cells

    occupancy = main_df.groupby('head_bin').size()
    spatial_info_real = get_spatial_info_real(place_df, occupancy)
    spatial_info_shuffled = get_spatial_info_shuffle(place_df, occupancy)
    place_cells, p_values = check_significance(spatial_info_real, spatial_info_shuffled)

    nb_place_cells, nb_all_cells = len(place_cells), len(spatial_info_real)
    print(f'Place cells {nb_place_cells} / {nb_all_cells} ({round((nb_place_cells/nb_all_cells)*100, 1)}%)')

    return spatial_info_real, spatial_info_shuffled, p_values, place_cells

place_df, areas = bin_cellspikes(main_df)
spatial_info_real, spatial_info_shuffled, p_values, place_cells = identify_place_cells(main_df, place_df)


#%%

# PLOT
def plot_spikes_polygons(main_df, cell_id, areas):
    
    plt.figure(figsize=(8,8))
    plt.plot(main_df['head_x'], main_df['head_y'], alpha=0.5, color='grey', zorder=1)

    # Plot spike locations for this cell
    spike_rows = main_df[main_df[cell_id] != 0]
    x = spike_rows['head_x']
    y = spike_rows['head_y']
    plt.scatter(x, y, s=80, alpha=0.8, label=f'{cell_id}', zorder=2)

    # Overlay polygons
    for label, poly in areas.items():
        px, py = poly.exterior.xy
        plt.plot(px, py, linewidth=2, label=label)

    plt.gca().set_aspect('equal')
    plt.legend()
    plt.show()
def plot_spikes_heatmap_polygons(main_df, cell, i, areas, arena_size=(580,485), bins=200, blur=4):
    # 2D histogram of spike locations
    spike_rows = main_df[main_df[cell] != 0]
    x = spike_rows['head_x']
    y = spike_rows['head_y']

    x_edges = np.linspace(0, arena_size[0], bins)
    y_edges = np.linspace(0, arena_size[1], bins)
    H, _, _ = np.histogram2d(y, x, bins=[y_edges, x_edges])
    blurred_heatmap = gaussian_filter(H, sigma=blur)

    plt.figure(figsize=(8,8))
    plt.plot(main_df['head_x'], main_df['head_y'], alpha=0.4, color='grey', zorder=2)
    plt.imshow(blurred_heatmap, origin='lower',
               extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
               cmap='hot', alpha=0.8, zorder=1)

    # Overlay polygons
    for label, poly in areas.items():
        px, py = poly.exterior.xy
        plt.plot(px, py, linewidth=2, label=label)

    plt.axis('off')
    plt.title(f'Cell {cell.replace("_spike","")}')
    plt.gca().set_aspect('equal')
    plt.tight_layout(pad=0)
    plt.savefig(f"{i}.png", bbox_inches='tight', pad_inches=0, dpi=300)
    plt.show()
def plot_histogram(cell, spatial_info_real, spatial_info_shuffled, bins=30):

    # plots histogram that shows the true SI and the shuffled data
    real_value = spatial_info_real[cell]
    shuffled_values = spatial_info_shuffled[cell]

    plt.figure(figsize=(8, 4))
    plt.hist(shuffled_values, bins=bins, alpha=0.7, color='gray', edgecolor='black', label='randomly assigned spikes')
    plt.axvline(real_value, color='orange', linewidth=5, label=f'Real SI = {real_value:.2f}')
    plt.title(f'Spatial Info Shuffle Distribution for Cell_{cell}')
    plt.xlabel('Spatial Information (bits/spike)')
    plt.ylabel('Count')
    plt.legend()
    plt.tight_layout()
    plt.show()
def draw_position_on_video(df, file_video):

    head_x = df["head_x"].values
    head_y = df["head_y"].values

    cap = cv2.VideoCapture(file_video)
    video_nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    video_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_duration = int(video_nframes / video_fps)
    print(f'VIDEO duration: {video_duration}s, frames: {video_nframes}, fps: {video_fps}')

    file_output = "position_overlay.mp4"
    out = cv2.VideoWriter(file_output, cv2.VideoWriter_fourcc(*'mp4v'), video_fps, (video_width, video_height))

    for frame_idx in tqdm(range(2000)):
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx < len(head_x):
            if not np.isnan(head_x[frame_idx]):
                x = int(head_x[frame_idx])
                y = int(head_y[frame_idx])
                cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)  # Green dot for position

        out.write(frame)

    cap.release()
    out.release()
    print("✅ Video with position overlay saved as 'position_overlay_first100.mp4'")


sorted_items = sorted(spatial_info_real.items(), key=lambda item: item[1], reverse=True)
for i, (cell, value) in enumerate(sorted_items):
    if i<1:
        print(cell)
        print(value)
        plot_spikes_heatmap_polygons(main_df, cell, i, areas)
        plot_spikes_polygons(main_df, cell, areas)
