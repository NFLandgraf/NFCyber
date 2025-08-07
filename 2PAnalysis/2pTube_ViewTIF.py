#%%
# TIFF VIEWER
%matplotlib qt
import os
import tifffile
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.ndimage import gaussian_filter
from scipy.ndimage import uniform_filter1d


def tiff_viewer(img, title="TIFF Viewer"):
    """
    Interactive viewer for 3D/4D/5D image stacks.
    Supports shape: (T,Y,X), (T,Z,Y,X), or (C,T,Z,Y,X)

    Sliders for:
        Channel, Frame, Z-layer, vmin/vmax
    Keyboard: ← / → for time
    Button: 'Auto' to set vmin/vmax to current frame range
    """
    # Normalize to (C, T, Z, Y, X)
    print(img.ndim)
    if img.ndim == 3:
        img = img[np.newaxis, :, np.newaxis, :, :]
    elif img.ndim == 4:
        img = img[np.newaxis, :, :, :, :]
    elif img.ndim != 5:
        raise ValueError("Expected shape: (T,Y,X), (T,Z,Y,X), or (C,T,Z,Y,X)")

    num_c, num_t, num_z, height, width = img.shape
    global_min = float(np.min(img))
    global_max = float(np.max(img))
    current = {'c': 0, 't': 0, 'z': 0}

    # === Plot ===
    fig, ax = plt.subplots(figsize=(10, 10))  # make larger
    fig.canvas.manager.set_window_title(title)
    plt.subplots_adjust(bottom=0.25)

    frame_display = ax.imshow(img[0, 0, 0], cmap='gray', vmin=global_min, vmax=global_max)
    ax.axis('off')
    ax.set_title(f'Ch 0 / {num_c-1} | Frame 0 / {num_t-1} | Z 0 / {num_z-1}')

    # === Slider helpers ===
    def add_slider(y_pos, label, vmin, vmax, valinit, step=1):
        ax_slider = plt.axes([0.2, y_pos, 0.6, 0.03])
        return Slider(ax_slider, label, vmin, vmax, valinit=valinit, valstep=step)
    
    slider_c     = add_slider(0.2, 'Channel', 0, num_c - 1, 0)
    slider_frame = add_slider(0.15, 'Frame', 0, num_t - 1, 0)
    slider_z     = add_slider(0.1, 'Z', 0, num_z - 1, 0)
    slider_vmin  = add_slider(0.05, 'vmin', global_min, global_max/5, global_min, step=None)
    slider_vmax  = add_slider(0.0, 'vmax', global_min, global_max/5, global_max, step=None)

    # === Update ===
    def update_display():
        c = int(slider_c.val)
        t = int(slider_frame.val)
        z = int(slider_z.val)
        current.update({'c': c, 't': t, 'z': z})

        frame = img[c, t, z]
        frame_display.set_data(frame)
        frame_display.set_clim(vmin=slider_vmin.val, vmax=slider_vmax.val)
        ax.set_title(f'Ch {c} / {num_c-1} | Frame {t} / {num_t-1} | Z {z} / {num_z-1}')
        fig.canvas.draw_idle()

    # === Callbacks ===
    slider_c.on_changed(lambda val: update_display())
    slider_frame.on_changed(lambda val: update_display())
    slider_z.on_changed(lambda val: update_display())
    slider_vmin.on_changed(lambda val: (
        frame_display.set_clim(vmin=val, vmax=slider_vmax.val), fig.canvas.draw_idle()))
    slider_vmax.on_changed(lambda val: (
        frame_display.set_clim(vmin=slider_vmin.val, vmax=val), fig.canvas.draw_idle()))

    # === Keyboard support ===
    def on_key(event):
        t = current['t']
        if event.key == 'right' and t < num_t - 1:
            slider_frame.set_val(t + 1)
        elif event.key == 'left' and t > 0:
            slider_frame.set_val(t - 1)

    fig.canvas.mpl_connect('key_press_event', on_key)

    plt.show()

img = tifffile.imread(r"tret\suite2p\plane0\reg_tif.tif")  # works for 3D, 4D, or 5D
print(img.shape)

tiff_viewer(img)