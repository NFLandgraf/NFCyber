#%%
# check the TTLs in the GPIO file
import pandas as pd
import numpy as np

def get_Cam_TTLs(file_TTL, cam_name=' GPIO-1', cam_threshold=10000, constant=True):
    # returns list of TTL input times
    round_to = 3

    # expects nan values to be just a string: ' nan', convert these strings into proper nans
    df = pd.read_csv(file_TTL, na_values=[' nan'], low_memory=False)
    df = df.set_index('Time (s)')

    # only cares for the rows where the channel name is in the ' Channel Name' column
    df = df[df[' Channel Name'] == cam_name]
    
    # drops rows with nan and converts everything into ints
    df = df.dropna(subset=[' Value'])
    df[' Value'] = df[' Value'].astype(int)

    # creates series with True for values > threshold
    # checks at which index/time the row's value is high AND the row before's value is low
    # for that, it makes the whole series negative (True=False) and shifts the whole series one row further (.shift)
    # so, onset wherever row is > threshold and the opposite value of the shifted (1) row is positive as well
    is_high = df[' Value'] >= cam_threshold
    onsets = df.index[is_high & ~is_high.shift(1, fill_value=False)]
    offsets = df.index[~is_high & is_high.shift(1, fill_value=False)]

    # high times are all the times where signal goes from below to above threshold
    high_times = np.array(list(zip(onsets, offsets)) if not constant else list(onsets))

    # sanity check
    high_times_intervals = np.round(np.diff(high_times), round_to)
    unique_vals, counts = np.unique(high_times_intervals, return_counts=True)
    if len(unique_vals):# > 1:
        print('\nCam_TTL time intervals:')
        print(f'unique_valus: {unique_vals}')
        print(f'unique_count: {counts}')
        print(f'TTL_Borderss: {round(high_times[0], round_to)} - {round(high_times[-1], round_to)}')
    
    print(f'{len(high_times)} Cam_ttl')
    return high_times

file_TTL = r"E:\MiniPOA\relevant\119\MiniPOA_119_2026-19-27-13-41-15_B_GPIO.csv"
high_times = get_Cam_TTLs(file_TTL)



#%%
# print the time and frame number on every frame
import cv2
from pathlib import Path
from tqdm import tqdm

def print_time_on_video(video_path, output_path, start_time=5.0, sampling_rate=10):

    cap = cv2.VideoCapture(str(video_path))

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    for frame_number in tqdm(range(total_frames), desc="Processing video", unit="frame"):

        ret, frame = cap.read()

        if not ret:
            break

        time_s = start_time + frame_number / sampling_rate

        # 50% transparent black background (300 x 120 px)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (300, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        cv2.putText(frame, f"Time {time_s:.1f}s",
                    (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 255), 2)

        cv2.putText(frame, f"Frame {frame_number}",
                    (30, 90), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 255), 2)

        out.write(frame)

    cap.release()
    out.release()

    print(f"Saved: {output_path}")

paths = [r"E:\MiniPOA\relevant\108\MiniPOA_108_BWCW_Behav.mp4",
         r"E:\MiniPOA\relevant\109\MiniPOA_109_BWCW_Behav.mp4",
         r"E:\MiniPOA\relevant\118\MiniPOA_118_2026-10-23-14-19-45_BWCW_Behav.mp4",
         r"E:\MiniPOA\relevant\119\MiniPOA_119_2026-19-26-11-24-08_WCW_Behav.mp4"]

for video_path in paths:

    video_path = Path(video_path)
    output_path = video_path.with_name(video_path.stem + "_time.mp4")

    print_time_on_video(video_path, output_path)


#%%
# check the activity frequency 
import pandas as pd
import numpy as np

def get_Activity_TTLs(file_TTL, threshold=0):

    df = pd.read_csv(file_TTL)
    df = df.set_index('Uhrzeit')

    # Convert Activity to numeric
    df['Activity'] = pd.to_numeric(df['Activity'], errors='coerce')
    df = df.dropna(subset=['Activity'])

    # Detect rising edges (0 -> > threshold)
    is_high = df['Activity'] > threshold
    onsets = df.index[is_high & ~is_high.shift(1, fill_value=False)]

    high_times = np.array(onsets)

    # Convert timestamps to seconds relative to first timestamp
    time_seconds = (pd.to_timedelta(high_times).total_seconds())

    # Sanity check
    intervals = np.round(np.diff(time_seconds), 3)
    unique_vals, counts = np.unique(intervals, return_counts=True)

    print(f'Unique values: {unique_vals}')
    print(f'Unique counts: {counts}')
    print(f'TTL Borders: {high_times[0]} - {high_times[-1]}')

    print(f'{len(high_times)} Activity TTLs')

    return high_times

file_temp = r"E:\MiniPOA\relevant\108\MiniPOA_108_BWCW_Temp.csv"
high_times = get_Activity_TTLs(file_temp, threshold=0)

#%%
# clean temp file
import pandas as pd
from pathlib import Path

file = Path(r"C:\Data\activity.csv")

def clean_temp(file):

    df = pd.read_csv(file)

    # Remove Datum column
    df = df.drop(columns=['Datum'])

    # Convert Uhrzeit to elapsed seconds, starting at 0
    time = pd.to_timedelta(df['Uhrzeit'])
    df['Uhrzeit'] = (time - time.iloc[0]).dt.total_seconds()

    # Rename column
    df = df.rename(columns={'Uhrzeit': 'Time [s]'})

    # Downsample from 1000 Hz to 100 Hz (keep every 10th row)
    df = df.iloc[::10].reset_index(drop=True)

    df = df.rename(columns={'Uhrzeit': 'Time[s]'})
    df = df.rename(columns={'Temperatur (°C)': 'Temp'})

    # Save CSV
    output_file = file.with_name(file.stem + '_sync.csv')
    df.to_csv(output_file, index=False)

paths = [r"E:\MiniPOA\relevant\MiniPOA_108_2026-10-26-09-40-59_BWCW_Temp.csv",
         r"E:\MiniPOA\relevant\MiniPOA_109_2026-10-23-11-42-15_BWCW_Temp.csv",
         r"E:\MiniPOA\relevant\MiniPOA_118_2026-10-23-14-19-45_BWCW_Temp.csv",
         r"E:\MiniPOA\relevant\MiniPOA_119_2026-10-26-11-03-45_Base_Temp.csv",
         r"E:\MiniPOA\relevant\MiniPOA_119_2026-10-26-11-24-08_WCW_Temp.csv"]

for file in paths:
    file = Path(file)
    clean_temp(file)



#%%
# split tiff into 2
import tifffile
from pathlib import Path

file = Path(r"E:\MiniPOA\relevant\119\MiniPOA_119_WCWB_Neuro_prepro_spat_motcor.tiff")
split_frame = 32694

# Load TIFF
tiff = tifffile.imread(file)

print(f"Original TIFF: {tiff.shape[0]} frames")

#%%
# Split at specified frame
wcw = tiff[:split_frame]
b = tiff[split_frame:]

# Save both parts
tifffile.imwrite(file.with_name(file.stem + "_wvw.tif"), wcw, bigtiff=True)
tifffile.imwrite(file.with_name(file.stem + "_b.tif"), b, bigtiff=True)

print(f"Part 1: {wcw.shape[0]} frames")
print(f"Part 2: {b.shape[0]} frames")

#%%
# add 2 tiffs
import tifffile
import numpy as np
from pathlib import Path

file1 = Path(r"E:\MiniPOA\relevant\119\MiniPOA_119_WCWB_Neuro_prepro_spat_motcor_b.tif")
file2 = Path(r"E:\MiniPOA\relevant\119\MiniPOA_119_WCWB_Neuro_prepro_spat_motcor_wvw.tif")
output = Path(r"E:\MiniPOA\relevant\119\MiniPOA_119_BWCW_Neuro_prepro_spat_motcor.tif")

# Load TIFF stacks
tiff1 = tifffile.imread(file1)
tiff2 = tifffile.imread(file2)

# Print frame counts
print(f"TIFF 1: {tiff1.shape[0]} frames")
print(f"TIFF 2: {tiff2.shape[0]} frames")

# Concatenate along time axis
combined = np.concatenate([tiff1, tiff2], axis=0)

# Save combined TIFF
tifffile.imwrite(output, combined, bigtiff=True)

print(f"Combined: {combined.shape[0]} frames")
print(f"Saved: {output}")







#%%
# plot traces, temp and activity
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.ticker import FuncFormatter, MultipleLocator

file = Path(r"E:\MiniPOA\relevant\119\MiniPOA_119_BWCW_Neuro_traces.csv")
file_temp = Path(r"E:\MiniPOA\relevant\119\MiniPOA_119_BWCW_Temp_sync.csv")


# Load calcium traces
df = pd.read_csv(file, skiprows=[1])
df.columns = df.columns.str.strip()
time = pd.to_numeric(df.iloc[:, 0])
selected_cells = ['C00', 'C02', 'C13', 'C14', 'C17', 'C19', 'C20', 'C24', 'C27', ]
traces = df[selected_cells].apply(pd.to_numeric, errors='coerce')


# Load temperature and activity
df_temp = pd.read_csv(file_temp)
df_temp.columns = df_temp.columns.str.strip()
time_temp = pd.to_numeric(df_temp['Time [s]'])
temp = pd.to_numeric(df_temp['Temp'], errors='coerce')
activity = pd.to_numeric(df_temp['Activity'], errors='coerce')



# Plot settings
offset = 100
fig, ax = plt.subplots(figsize=(14, 8))

# Plot calcium traces
for i, cell in enumerate(traces.columns):
    signal = traces[cell].to_numpy()
    signal = signal - np.nanmean(signal)
    ax.plot(time, signal + i * offset, lw=0.8, color='black')


# Normalize Temp and Activity for stacked visualization
def normalize(signal, amplitude=60):
    signal = np.asarray(signal, dtype=float)
    min_val = np.nanmin(signal)
    max_val = np.nanmax(signal)
    if max_val == min_val:
        return np.zeros_like(signal)
    return (signal - min_val) / (max_val - min_val) * amplitude

# Position below calcium traces
temp_y = -150
activity_y = -250

# Periods
duration = 900
periods = [ (0,      duration, 'gray'),
            (959.2,  duration, 'red'),
            (2089.2, duration, 'blue'),
            (3169.2, duration, 'red')]
line_y = -330
for start, duration, color in periods:
    ax.hlines(y=line_y,xmin=start,xmax=start + duration,color=color,linewidth=5)


# Set temperature to NaN during experimental periods
mask = np.zeros(len(time_temp), dtype=bool)
for start, duration, color in periods:
    mask |= (time_temp >= start) & (time_temp <= start + duration)
temp.loc[~mask] = np.nan


# Remove sudden temperature spikes via rolling median
window = 1000          # Rolling median window (number of samples)
threshold = 0.000001      # Maximum allowed deviation from median (°C)
temp_median = temp.rolling(window=window,center=True,min_periods=1).median()
spikes = (temp - temp_median).abs() > threshold
temp.loc[spikes] = temp_median.loc[spikes]


ax.plot(time_temp, normalize(temp) + temp_y, color='darkorange', lw=1, label='Temp')
ax.plot(time_temp, normalize(activity) + activity_y, color='green', lw=1, label='Activity')


# Formatting
yticks = (list(np.arange(len(selected_cells)) * offset) + [temp_y, activity_y])
ylabels = selected_cells + ['Temp', 'Activity']
ax.set_yticks(yticks)
ax.set_yticklabels(ylabels)
ax.xaxis.set_major_locator(MultipleLocator(600))
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x / 60:g}'))
ax.set_xlabel("Time (min)")
ax.set_ylabel("Signal")
ax.set_ylim(bottom=-370)
ax.spines[['top', 'right']].set_visible(False)

# Temp scale
temp_min = np.nanmin(temp)
temp_max = np.nanmax(temp)
temp_ticks = np.arange(np.ceil(temp_min),np.floor(temp_max) + 1, 1)
temp_tick_positions = ((temp_ticks - temp_min) / (temp_max - temp_min) * 60 + temp_y)
ax_temp = ax.twinx()
ax_temp.set_ylim(ax.get_ylim())
ax_temp.set_yticks(temp_tick_positions)
ax_temp.set_yticklabels([f"{t:g}" for t in temp_ticks])
ax_temp.set_ylabel("Temperature (°C)", color="darkorange")
ax_temp.tick_params(axis="y", colors="darkorange")
ax_temp.spines["top"].set_visible(False)
ax_temp.spines["right"].set_color("darkorange")

# Neuro scale
scale_dff = 50       # Scale bar height in ΔF/F units
scale_x = time.max() * 1.1
scale_y = 10         # Bottom position of scale bar
ax.plot([scale_x, scale_x],[scale_y, scale_y + scale_dff], color='black',lw=2)
ax.text(scale_x - time.max() * 0.01,scale_y + scale_dff / 2,f'{scale_dff} ΔF/F',ha='right',va='center',fontsize=10)


plt.tight_layout()
fig.savefig("119_BWCW.pdf", format="pdf",bbox_inches="tight")
plt.show()
