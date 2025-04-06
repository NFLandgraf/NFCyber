import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import cv2
import numpy as np
from pathlib import Path
from matplotlib import image
from matplotlib import pyplot as plt 
import os
from shapely.geometry import Polygon
import json
from PIL import Image, ImageTk
import io
import threading

entries_file = 'entries.json'
def save_entries(entries):
    with open(entries_file, 'w') as f:
        json.dump(entries, f)
def initialize_gui():
    global dx_entry, dy_entry, dz_entry, CL_entry_x, CL_entry_y, CR_entry_x, CR_entry_y, CM_entry_x, CM_entry_y, LaL_entry_x, LaL_entry_y, LaR_entry_x, LaR_entry_y, RaR_entry_x, RaR_entry_y, RaL_entry_x, RaL_entry_y, MaL_entry_x, MaL_entry_y, MaR_entry_x, MaR_entry_y

    def load_entries():
        try:
            with open(entries_file, 'r') as f:
                entries = json.load(f)
        except FileNotFoundError:
            entries = {}  # Return an empty dictionary if file doesn't exist
        return entries

    entries = load_entries()

    dx_entry.insert(0, str(entries.get('dx', '')))
    dy_entry.insert(0, str(entries.get('dy', '')))
    dz_entry.insert(0, str(entries.get('dz', '')))

    CL_entry_x.insert(0, str(entries.get('CL_x', '')))
    CL_entry_y.insert(0, str(entries.get('CL_y', '')))
    CR_entry_x.insert(0, str(entries.get('CR_x', '')))
    CR_entry_y.insert(0, str(entries.get('CR_y', '')))
    CM_entry_x.insert(0, str(entries.get('CM_x', '')))
    CM_entry_y.insert(0, str(entries.get('CM_y', '')))

    LaL_entry_x.insert(0, str(entries.get('LaL_x', '')))
    LaL_entry_y.insert(0, str(entries.get('LaL_y', '')))
    LaR_entry_x.insert(0, str(entries.get('LaR_x', '')))
    LaR_entry_y.insert(0, str(entries.get('LaR_y', '')))

    RaR_entry_x.insert(0, str(entries.get('RaR_x', '')))
    RaR_entry_y.insert(0, str(entries.get('RaR_y', '')))
    RaL_entry_x.insert(0, str(entries.get('RaL_x', '')))
    RaL_entry_y.insert(0, str(entries.get('RaL_y', '')))

    MaL_entry_x.insert(0, str(entries.get('MaL_x', '')))
    MaL_entry_y.insert(0, str(entries.get('MaL_y', '')))
    MaR_entry_x.insert(0, str(entries.get('MaR_x', '')))
    MaR_entry_y.insert(0, str(entries.get('MaR_y', '')))

def open_gui():
    global dx_entry, dy_entry, dz_entry, CL_entry_x, CL_entry_y, CR_entry_x, CR_entry_y, CM_entry_x, CM_entry_y, LaL_entry_x, LaL_entry_y, LaR_entry_x, LaR_entry_y, RaR_entry_x, RaR_entry_y, RaL_entry_x, RaL_entry_y, MaL_entry_x, MaL_entry_y, MaR_entry_x, MaR_entry_y

    def get_files(path):
        files = [file for file in Path(path).iterdir() if file.is_file() and '.mp4' in file.name]
        return files

    def browse_folder():
        folder_selected = filedialog.askdirectory()
        path_entry.delete(0, tk.END)
        path_entry.insert(0, folder_selected)
    

    root = tk.Tk()
    root.title("Video Masking Tool")

    entry_width = 5
    spacing_x = 2
    spacing_y = 1

    # video paths
    path_label = ttk.Label(root, text="Video Folder")
    path_label.grid(row=0, column=0, padx=spacing_x, pady=spacing_y)
    path_entry = ttk.Entry(root, width=50)
    path_entry.grid(row=0, column=1, columnspan=3, padx=spacing_x, pady=spacing_y)
    browse_button = ttk.Button(root, text="Browse", command=browse_folder)
    browse_button.grid(row=0, column=4, padx=spacing_x, pady=spacing_y)
    
    # Parameter inputs for dx, dy, dz
    dx_label = ttk.Label(root, text="dx (horizontal shift)")
    dx_label.grid(row=2, column=3, padx=spacing_x, pady=spacing_y)
    dx_entry = ttk.Entry(root, width=entry_width)
    dx_entry.grid(row=2, column=4, padx=spacing_x, pady=spacing_y)
    
    dy_label = ttk.Label(root, text="dy (vertical shift)")
    dy_label.grid(row=3, column=3, padx=spacing_x, pady=spacing_y)
    dy_entry = ttk.Entry(root, width=entry_width)
    dy_entry.grid(row=3, column=4, padx=spacing_x, pady=spacing_y)
    
    dz_label = ttk.Label(root, text="dz (size scale)")
    dz_label.grid(row=4, column=3, padx=spacing_x, pady=spacing_y)
    dz_entry = ttk.Entry(root, width=entry_width)
    dz_entry.grid(row=4, column=4, padx=spacing_x, pady=spacing_y)

    # Corner Left
    CL_label = ttk.Label(root, text="left corner")
    CL_label.grid(row=2, column=0, padx=spacing_x, pady=spacing_y)
    CL_entry_x = ttk.Entry(root, width=entry_width)
    CL_entry_x.grid(row=2, column=1, padx=spacing_x, pady=spacing_y)
    CL_entry_y = ttk.Entry(root, width=entry_width)
    CL_entry_y.grid(row=2, column=2, padx=spacing_x, pady=spacing_y)

    # Corner Middle
    CM_label = ttk.Label(root, text="middle corner")
    CM_label.grid(row=3, column=0, padx=spacing_x, pady=spacing_y)
    CM_entry_x = ttk.Entry(root, width=entry_width)
    CM_entry_x.grid(row=3, column=1, padx=spacing_x, pady=spacing_y)
    CM_entry_y = ttk.Entry(root, width=entry_width)
    CM_entry_y.grid(row=3, column=2, padx=spacing_x, pady=spacing_y)

    # Corner Right
    CR_label = ttk.Label(root, text="right corner")
    CR_label.grid(row=4, column=0, padx=spacing_x, pady=spacing_y)
    CR_entry_x = ttk.Entry(root, width=entry_width)
    CR_entry_x.grid(row=4, column=1, padx=spacing_x, pady=spacing_y)
    CR_entry_y = ttk.Entry(root, width=entry_width)
    CR_entry_y.grid(row=4, column=2, padx=spacing_x, pady=spacing_y)

    # Left Arm End Lefter
    LaL_label = ttk.Label(root, text="left arm left")
    LaL_label.grid(row=5, column=0, padx=spacing_x, pady=spacing_y)
    LaL_entry_x = ttk.Entry(root, width=entry_width)
    LaL_entry_x.grid(row=5, column=1, padx=spacing_x, pady=spacing_y)
    LaL_entry_y = ttk.Entry(root, width=entry_width)
    LaL_entry_y.grid(row=5, column=2, padx=spacing_x, pady=spacing_y)

    # Left Arm End Righter
    LaR_label = ttk.Label(root, text="left arm right")
    LaR_label.grid(row=6, column=0, padx=spacing_x, pady=spacing_y)
    LaR_entry_x = ttk.Entry(root, width=entry_width)
    LaR_entry_x.grid(row=6, column=1, padx=spacing_x, pady=spacing_y)
    LaR_entry_y = ttk.Entry(root, width=entry_width)
    LaR_entry_y.grid(row=6, column=2, padx=spacing_x, pady=spacing_y)

    # Right Arm End Righter
    RaR_label = ttk.Label(root, text="right arm right")
    RaR_label.grid(row=7, column=0, padx=spacing_x, pady=spacing_y)
    RaR_entry_x = ttk.Entry(root, width=entry_width)
    RaR_entry_x.grid(row=7, column=1, padx=spacing_x, pady=spacing_y)
    RaR_entry_y = ttk.Entry(root, width=entry_width)
    RaR_entry_y.grid(row=7, column=2, padx=spacing_x, pady=spacing_y)

    # Right Arm End Lefter
    RaL_label = ttk.Label(root, text="right arm left")
    RaL_label.grid(row=8, column=0, padx=spacing_x, pady=spacing_y)
    RaL_entry_x = ttk.Entry(root, width=entry_width)
    RaL_entry_x.grid(row=8, column=1, padx=spacing_x, pady=spacing_y)
    RaL_entry_y = ttk.Entry(root, width=entry_width)
    RaL_entry_y.grid(row=8, column=2, padx=spacing_x, pady=spacing_y)

    # Middle Arm End Lefter
    MaL_label = ttk.Label(root, text="middle arm left")
    MaL_label.grid(row=9, column=0, padx=spacing_x, pady=spacing_y)
    MaL_entry_x = ttk.Entry(root, width=entry_width)
    MaL_entry_x.grid(row=9, column=1, padx=spacing_x, pady=spacing_y)
    MaL_entry_y = ttk.Entry(root, width=entry_width)
    MaL_entry_y.grid(row=9, column=2, padx=spacing_x, pady=spacing_y)

    # Middle Arm End Righter
    MaR_label = ttk.Label(root, text="middle arm right")
    MaR_label.grid(row=10, column=0, padx=spacing_x, pady=spacing_y)
    MaR_entry_x = ttk.Entry(root, width=entry_width)
    MaR_entry_x.grid(row=10, column=1, padx=spacing_x, pady=spacing_y)
    MaR_entry_y = ttk.Entry(root, width=entry_width)
    MaR_entry_y.grid(row=10, column=2, padx=10, pady=spacing_y)

    # Image
    image_frame = ttk.LabelFrame(root, text="Mask Preview")
    image_frame.grid(row=0, column=5, rowspan=12, padx=spacing_x, pady=spacing_y)
    image_label = ttk.Label(image_frame)
    image_label.pack()

    # Progress bar
    progress_var = tk.DoubleVar()
    progress_bar = ttk.Progressbar(root, variable=progress_var, maximum=100)
    progress_bar.grid(row=15, column=0, columnspan=5, padx=spacing_x, pady=spacing_y, sticky="we")

    # Create a Text widget to show the log of processed files
    file_log = tk.Text(root, height=10, width=60, wrap="word", state=tk.DISABLED)
    file_log.grid(row=13, column=5, rowspan=15, padx=spacing_x, pady=spacing_y)

    
    def get_coordinates(file):

        def get_xmax_ymax(file):
            vid = cv2.VideoCapture(str(file))
            width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return width, height
        def adapt(x, y):
            adapt_x = (x + dx) * dz
            adapt_y = (y + dy) * dz

            if adapt_x < 0:
                adapt_x = 0
            elif adapt_x > video_xmax:
                adapt_x = video_xmax

            if adapt_y < 0:
                adapt_y = 0
            elif adapt_y > video_ymax:
                adapt_y = video_ymax

            return (adapt_x, adapt_y)
        
        video_xmax, video_ymax = get_xmax_ymax(file)

        dx = int(dx_entry.get())
        dy = int(dy_entry.get())
        dz = int(dz_entry.get())
        CL =    adapt(int(CL_entry_x.get()), int(CL_entry_y.get()))
        CR =    adapt(int(CR_entry_x.get()), int(CR_entry_y.get()))
        CM =    adapt(int(CM_entry_x.get()), int(CM_entry_y.get()))
        LaL =   adapt(int(LaL_entry_x.get()), int(LaL_entry_y.get()))
        LaR =   adapt(int(LaR_entry_x.get()), int(LaR_entry_y.get()))
        RaR =   adapt(int(RaR_entry_x.get()), int(RaR_entry_y.get()))
        RaL =   adapt(int(RaL_entry_x.get()), int(RaL_entry_y.get()))
        MaL =   adapt(int(MaL_entry_x.get()), int(MaL_entry_y.get()))
        MaR =   adapt(int(MaR_entry_x.get()), int(MaR_entry_y.get()))
        
        # Return all coordinates as a dictionary or tuple
        return {
            "CL": CL,
            "CR": CR,
            "CM": CM,
            "LaL": LaL,
            "LaR": LaR,
            "RaR": RaR,
            "RaL": RaL,
            "MaL": MaL,
            "MaR": MaR
        }

    def get_areas(coords):
        # define all necessary areas
        left_arm = Polygon([coords["CL"], coords["CM"], coords["LaR"], coords["LaL"]])
        right_arm = Polygon([coords["CR"], coords["CM"], coords["RaL"], coords["RaR"]])
        middle_arm = Polygon([coords["CL"], coords["CR"], coords["MaR"], coords["MaL"]])
        center = Polygon([coords["CM"], coords["CL"], coords["CR"]])
        areas = [center, left_arm, middle_arm, right_arm]

        whole_area = Polygon([coords["CM"], coords["RaL"], coords["RaR"], 
                            coords["CR"], coords["MaR"], coords["MaL"], 
                            coords["CL"], coords["LaL"], coords["LaR"]])
        whole_area = np.array(list(whole_area.exterior.coords), dtype=np.int32).reshape((-1, 1, 2))

        return areas, whole_area

    # main functions
    def check_mask():

        entries = {
            'dx': dx_entry.get(),
            'dy': dy_entry.get(),
            'dz': dz_entry.get(),
            'CL_x': CL_entry_x.get(),
            'CL_y': CL_entry_y.get(),
            'CR_x': CR_entry_x.get(),
            'CR_y': CR_entry_y.get(),
            'CM_x': CM_entry_x.get(),
            'CM_y': CM_entry_y.get(),
            'LaL_x': LaL_entry_x.get(),
            'LaL_y': LaL_entry_y.get(),
            'LaR_x': LaR_entry_x.get(),
            'LaR_y': LaR_entry_y.get(),
            'RaR_x': RaR_entry_x.get(),
            'RaR_y': RaR_entry_y.get(),
            'RaL_x': RaL_entry_x.get(),
            'RaL_y': RaL_entry_y.get(),
            'MaL_x': MaL_entry_x.get(),
            'MaL_y': MaL_entry_y.get(),
            'MaR_x': MaR_entry_x.get(),
            'MaR_y': MaR_entry_y.get()
        }
        save_entries(entries)

        path = path_entry.get()
        if not path:
            messagebox.showerror("Error", "Please select a folder.")
            return
        
        files = get_files(path)
        if not files:
            messagebox.showerror("Error", "No MP4 files found in the selected folder.")
            return
        
        
        
        def write_and_draw(file, areas):

            cap = cv2.VideoCapture(str(file))
            ret, frame = cap.read()
            cap.release()

            if not ret:
                messagebox.showerror("Error", "Could not read frame.")
                return
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
            ax.imshow(frame_rgb)

            for arm in areas:
                    x, y = arm.exterior.xy
                    ax.fill(x, y, alpha=0.4)
                    
            fig.tight_layout(pad=0)

            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            plt.close(fig)

            # transform to tkinter-compatibel image
            img = Image.open(buf)
            img_tk = ImageTk.PhotoImage(img)

            # display image in gui
            image_label.config(image=img_tk)
            image_label.image = img_tk  # keep a reference!


        coords = get_coordinates(files[0])
        areas, whole_area = get_areas(coords)
        write_and_draw(files[0], areas)

    def start_processing():

        path = path_entry.get()
        if not path:
            messagebox.showerror("Error", "Please select a folder.")
            return
        
        files = get_files(path)
        if not files:
            messagebox.showerror("Error", "No MP4 files found in the selected folder.")
            return
        

        # Processing function to adjust and mask videos
        def adjust_video(input_file, output_file, coords, whole_area):

            # original stuff
            vid = cv2.VideoCapture(str(input_file))
            width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
            nframes = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(vid.get(cv2.CAP_PROP_FPS))
            print(f'Origin: {width}x{height}px, {nframes} frames, {fps} fps')

            # future dimensions after cropping
            x1 = coords['LaL'][0]
            y1 = min(coords['MaL'][1], coords['MaR'][1], coords['LaR'][1], coords['RaL'][1])
            x2 = coords['RaR'][0]
            y2 = max(coords['MaL'][1], coords['MaR'][1], coords['LaR'][1], coords['RaL'][1])
            new_width, new_height = x2-x1, y2-y1
            print(f'Wanted: {new_width}x{new_height}px, {nframes} frames, {fps} fps')
            

            # create new video
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video = cv2.VideoWriter(str(output_file), fourcc, fps, (new_width, new_height)) 

            for i in range(nframes):
                ret, frame = vid.read()
                if ret:    

                    # mask frame
                    if i == 0:
                        white_frame = np.ones_like(frame) * 255
                        mask = np.zeros_like(frame[:, :, 0])
                        cv2.fillPoly(mask, [whole_area], 255)
                    masked_frame = np.where(mask[:, :, None] == 255, frame, white_frame)

                    # crop frame
                    masked_cropped_frame = masked_frame[y1:y2, x1:x2]

                    video.write(masked_cropped_frame)

                    # Update progress bar
                    progress_var.set((i + 1) / nframes * 100)
                    root.update_idletasks()

                else:
                    break
            
            print(f'Edited: {np.shape(frame)[1]}x{np.shape(frame)[0]}px, {i+1} frames, {fps} fps\n\n')
            #vid.release()
            video.release()
            progress_var.set(0)

        coords = get_coordinates(files[0])
        areas, whole_area = get_areas(coords)

        print(coords)

        for idx, video_file in enumerate(files):
            
            # update the files processed in the gui
            filename = os.path.basename(video_file)
            file_log.config(state=tk.NORMAL)  # Enable the Text widget to update it
            file_log.insert(tk.END, f"{filename} - in process\n")  # Append the finished file

            output_video_file = video_file.with_name(video_file.stem + '_Masked.mp4')
            adjust_video(video_file, output_video_file, coords, whole_area)

            # update the files processed in the gui
            file_log.insert(tk.END, f"{filename} - finished\n")  # Append the finished file
            file_log.config(state=tk.DISABLED)  # Disable it again to prevent manual editing
            file_log.yview(tk.END)  # Scroll to the bottom of the text widget

        messagebox.showinfo(f"Processing Complete ({idx+1} videos)")

        
    # Do Buttons
    check_button = ttk.Button(root, text="Check Mask", command=check_mask)
    check_button.grid(row=5, column=4, padx=10, pady=20)

    process_button = tk.Button(root, text="Mask Videos", command=lambda: threading.Thread(target=start_processing).start())
    process_button.grid(row=6, column=4, padx=10, pady=20)

    initialize_gui()
    root.mainloop()

open_gui()
