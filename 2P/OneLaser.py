#%%


import tifffile as tiff
import numpy as np
import matplotlib.pyplot as plt

# ==== CONFIGURATION ====
file_path = r"D:\CA1Dopa_2pTube_2025-07-28_Day2_spont_airpuffs\CA1Dopa_2pTube_2025-07-28_A53T_972m_day2_spont_airpuffs\CA1Dopa_2pTube_2025-07-28_A53T_972m_day2_airpuffs.ome.tiff"
# ========================

# Open the TIFF file
with tiff.TiffFile(file_path) as tif:
    # Access data for channel 0 and z-slice 0
    data_channel_0 = tif.pages[0].asarray()  # This will give you the first channel data as a numpy array

# Now we want to access the first Z-slice (Z=0)
# Assuming the shape is (C, T, Z, X, Y), we can slice it as follows
selected_data = data_channel_0[0, :, 0, :, :]  # Shape: (T, X, Y)

print(f"Selected data shape: {selected_data.shape}")  # Should be (T, X, Y)

# Example: Show a mean projection of the selected data
mean_projection = np.mean(selected_data, axis=0)  # Mean across time (T)

# Plot the mean projection of the selected data
plt.figure(figsize=(8, 8))
plt.imshow(mean_projection, cmap='gray')
plt.title("Mean Projection of Channel 1, Z-slice 0")
plt.axis('off')
plt.colorbar()
plt.show()




#%%

# =================== mean projection across  ===================
mean_projections = np.mean(data, axis=(1, 2))  # shape: (C, X, Y)

# Plot each channel's mean projection
num_channels = mean_projections.shape[0]
fig, axes = plt.subplots(1, num_channels, figsize=(5 * num_channels, 5))

if num_channels == 1:
    axes = [axes]

for i in range(num_channels):
    ax = axes[i]
    im = ax.imshow(mean_projections[i], cmap='gray')
    ax.set_title(f'Channel {i}')
    ax.axis('off')
    fig.colorbar(im, ax=ax, shrink=0.7)

plt.tight_layout()
plt.show()


# === Interactive Viewer for Both Channels ===

print(f"Applying mean filter with size: {mean_filter_size} to both channels...")
filtered = np.empty_like(data, dtype=np.float32)

for c in range(C):
    for t in range(T):
        filtered[c, t] = uniform_filter(data[c, t], size=mean_filter_size)

# Sum across Z
z_summed = np.sum(filtered, axis=2)  # shape: (C, T, X, Y)



# =================== GUI ===================
print("Launching dual-channel viewer...")
fig, axes = plt.subplots(1, C, figsize=(6 * C, 6))
plt.subplots_adjust(bottom=0.15)

imgs = []
for i in range(C):
    ax = axes[i] if C > 1 else axes
    img = ax.imshow(z_summed[i, 0], cmap='gray')
    ax.set_title(f'Channel {i} - Time: 0')
    ax.axis('off')
    imgs.append(img)

# Slider
ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
slider = Slider(ax_slider, 'Time', 0, T - 1, valinit=0, valstep=1)

# Update both channels
def update(val):
    t = int(slider.val)
    for i in range(C):
        imgs[i].set_data(z_summed[i, t])
        axes[i].set_title(f'Channel {i} - Time: {t}')
    fig.canvas.draw_idle()

slider.on_changed(update)

plt.show()




# ==== STIMULUS-ALIGNED FLUORESCENCE TRACES ====
import matplotlib.pyplot as plt

# === USER INPUT ===
sampling_rate = 11.9818  # Hz
event_times_sec = [30.0, 60.0, 90.0, 120.0, 150.0]  # Event onset times in seconds
pre_event_window = 5.0  # Seconds before event
post_event_window = 10.0  # Seconds after event

# === Derived parameters ===
pre_frames = int(pre_event_window * sampling_rate)
post_frames = int(post_event_window * sampling_rate)
event_frames = [int(t * sampling_rate) for t in event_times_sec]
window_size = pre_frames + post_frames

# Check valid indices
valid_events = [t for t in event_frames if t - pre_frames >= 0 and t + post_frames < T]
if len(valid_events) < len(event_frames):
    print("Warning: Some events are too close to start/end and were skipped.")

# Compute flattened mean across XY pixels → shape: (C, T)
traces = np.mean(z_summed.reshape(C, T, -1), axis=2)

# Collect aligned traces
aligned_traces = np.empty((C, len(valid_events), window_size), dtype=np.float32)

for i, event in enumerate(valid_events):
    for c in range(C):
        aligned_traces[c, i] = traces[c, event - pre_frames : event + post_frames]

# Time vector for plotting
time_axis = np.linspace(-pre_event_window, post_event_window, window_size)

# Plotting: One subplot per channel
fig, axes = plt.subplots(C, 1, figsize=(8, 4 * C), sharex=True)

if C == 1:
    axes = [axes]  # Ensure axes is iterable

for c in range(C):
    ax = axes[c]
    mean_trace = aligned_traces[c].mean(axis=0)
    sem_trace = aligned_traces[c].std(axis=0) / np.sqrt(aligned_traces[c].shape[0])
    
    ax.plot(time_axis, mean_trace, label=f'Channel {c}')
    ax.fill_between(time_axis, mean_trace - sem_trace, mean_trace + sem_trace, alpha=0.3)
    ax.axvline(0, color='k', linestyle='--', label='Stimulus Onset')
    ax.set_ylabel('Mean Fluorescence (a.u.)')
    ax.set_title(f'Channel {c} - Stimulus-Aligned Fluorescence')

axes[-1].set_xlabel('Time (s)')

plt.tight_layout()
plt.show()