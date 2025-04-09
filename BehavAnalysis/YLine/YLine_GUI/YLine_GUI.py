import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import cv2
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt 
import os
from shapely.geometry import Polygon, Point
import json
from PIL import Image, ImageTk
import io
import threading
import pandas as pd 

# for saving and retrieving the entries
entries_file = os.path.join(os.path.dirname(__file__), 'entries.json')
def save_entries(entries):
    with open(entries_file, 'w') as f:
        json.dump(entries, f)
def initialize_gui():
    global dx_entry, dy_entry, dz_entry, CL_entry_x, CL_entry_y, CR_entry_x, CR_entry_y, CM_entry_x, CM_entry_y, LaL_entry_x, LaL_entry_y, LaR_entry_x, LaR_entry_y, RaR_entry_x, RaR_entry_y, RaL_entry_x, RaL_entry_y, MaL_entry_x, MaL_entry_y, MaR_entry_x, MaR_entry_y
    global aCL_entry_x, aCL_entry_y, aCR_entry_x, aCR_entry_y, aCM_entry_x, aCM_entry_y, aLaL_entry_x, aLaL_entry_y, aLaR_entry_x, aLaR_entry_y, aRaR_entry_x, aRaR_entry_y, aRaL_entry_x, aRaL_entry_y, aMaL_entry_x, aMaL_entry_y, aMaR_entry_x, aMaR_entry_y
    
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

    aCL_entry_x.insert(0, str(entries.get('aCL_x', '')))
    aCL_entry_y.insert(0, str(entries.get('aCL_y', '')))
    aCR_entry_x.insert(0, str(entries.get('aCR_x', '')))
    aCR_entry_y.insert(0, str(entries.get('aCR_y', '')))
    aCM_entry_x.insert(0, str(entries.get('aCM_x', '')))
    aCM_entry_y.insert(0, str(entries.get('aCM_y', '')))
    aLaL_entry_x.insert(0, str(entries.get('aLaL_x', '')))
    aLaL_entry_y.insert(0, str(entries.get('aLaL_y', '')))
    aLaR_entry_x.insert(0, str(entries.get('aLaR_x', '')))
    aLaR_entry_y.insert(0, str(entries.get('aLaR_y', '')))
    aRaR_entry_x.insert(0, str(entries.get('aRaR_x', '')))
    aRaR_entry_y.insert(0, str(entries.get('aRaR_y', '')))
    aRaL_entry_x.insert(0, str(entries.get('aRaL_x', '')))
    aRaL_entry_y.insert(0, str(entries.get('aRaL_y', '')))
    aMaL_entry_x.insert(0, str(entries.get('aMaL_x', '')))
    aMaL_entry_y.insert(0, str(entries.get('aMaL_y', '')))
    aMaR_entry_x.insert(0, str(entries.get('aMaR_x', '')))
    aMaR_entry_y.insert(0, str(entries.get('aMaR_y', '')))

def open_gui():
    global dx_entry, dy_entry, dz_entry, CL_entry_x, CL_entry_y, CR_entry_x, CR_entry_y, CM_entry_x, CM_entry_y, LaL_entry_x, LaL_entry_y, LaR_entry_x, LaR_entry_y, RaR_entry_x, RaR_entry_y, RaL_entry_x, RaL_entry_y, MaL_entry_x, MaL_entry_y, MaR_entry_x, MaR_entry_y
    global aCL_entry_x, aCL_entry_y, aCR_entry_x, aCR_entry_y, aCM_entry_x, aCM_entry_y, aLaL_entry_x, aLaL_entry_y, aLaR_entry_x, aLaR_entry_y, aRaR_entry_x, aRaR_entry_y, aRaL_entry_x, aRaL_entry_y, aMaL_entry_x, aMaL_entry_y, aMaR_entry_x, aMaR_entry_y
    
    def get_files(path, common_name):
        files = [file for file in Path(path).iterdir() if file.is_file() and common_name in file.name]
        return files

    def browse_folder():
        folder_selected = filedialog.askdirectory()
        path_entry.delete(0, tk.END)
        path_entry.insert(0, folder_selected)

        # files
        files_mp4 = get_files(folder_selected, '.mp4')
        files_csv = get_files(folder_selected, '.csv')

        file_log.config(state=tk.NORMAL)
        file_log.delete('1.0', tk.END)  # Clear previous content
        if files_mp4 or files_csv:
            all_files = sorted(files_mp4 + files_csv)
            file_log.insert(tk.END, f"{len(files_mp4)} .mp4 and {len(files_csv)} .csv found\n")
            for file in all_files:
                file_log.insert(tk.END, f"{file.name}\n")
        file_log.config(state=tk.DISABLED)

    root = tk.Tk()
    root.title("Video Masking Tool")

    # video paths
    path_label = ttk.Label(root, text="Video Folder")
    path_label.grid(row=0, column=0, padx=5, pady=5)
    path_entry = ttk.Entry(root, width=50)
    path_entry.grid(row=0, column=1, columnspan=3, padx=5, pady=5)
    browse_button = ttk.Button(root, text="Browse", command=browse_folder)
    browse_button.grid(row=0, column=4, padx=5, pady=5)


    # MASK WINDOW
    box_left = tk.LabelFrame(root, text="MASKING", bg="#eef6ff", bd=2, relief="ridge", font=("Arial", 10, "bold"))
    box_left.grid(row=1, column=0, rowspan=15, columnspan=3, padx=10, pady=10, sticky="nsew")
    x_label = ttk.Label(box_left, text="x")
    x_label.grid(row=0, column=1, padx=5, pady=5)
    y_label = ttk.Label(box_left, text="y")
    y_label.grid(row=0, column=2, padx=5, pady=5)

    row_idx = 1
    entry_width = 5
    spacing_x = 2
    spacing_y = 1
    
    # Parameter inputs for dx, dy, dz
    dx_label = ttk.Label(box_left, text="dx (horizontal shift)")
    dx_label.grid(row=row_idx, column=0, padx=spacing_x, pady=spacing_y)
    dx_entry = ttk.Entry(box_left, width=entry_width)
    dx_entry.grid(row=row_idx, column=1, padx=spacing_x, pady=spacing_y)
    row_idx += 1
    
    dy_label = ttk.Label(box_left, text="dy (vertical shift)")
    dy_label.grid(row=row_idx, column=0, padx=spacing_x, pady=spacing_y)
    dy_entry = ttk.Entry(box_left, width=entry_width)
    dy_entry.grid(row=row_idx, column=1, padx=spacing_x, pady=spacing_y)
    row_idx += 1
    
    dz_label = ttk.Label(box_left, text="dz (size scale)")
    dz_label.grid(row=row_idx, column=0, padx=spacing_x, pady=spacing_y)
    dz_entry = ttk.Entry(box_left, width=entry_width)
    dz_entry.grid(row=row_idx, column=1, padx=spacing_x, pady=spacing_y)
    row_idx += 1

    # Corner Left
    CL_label = ttk.Label(box_left, text="left corner")
    CL_label.grid(row=row_idx, column=0, padx=spacing_x, pady=spacing_y)
    CL_entry_x = ttk.Entry(box_left, width=entry_width)
    CL_entry_x.grid(row=row_idx, column=1, padx=spacing_x, pady=spacing_y)
    CL_entry_y = ttk.Entry(box_left, width=entry_width)
    CL_entry_y.grid(row=row_idx, column=2, padx=spacing_x, pady=spacing_y)
    row_idx += 1

    # Corner Middle
    CM_label = ttk.Label(box_left, text="middle corner")
    CM_label.grid(row=row_idx, column=0, padx=spacing_x, pady=spacing_y)
    CM_entry_x = ttk.Entry(box_left, width=entry_width)
    CM_entry_x.grid(row=row_idx, column=1, padx=spacing_x, pady=spacing_y)
    CM_entry_y = ttk.Entry(box_left, width=entry_width)
    CM_entry_y.grid(row=row_idx, column=2, padx=spacing_x, pady=spacing_y)
    row_idx += 1

    # Corner Right
    CR_label = ttk.Label(box_left, text="right corner")
    CR_label.grid(row=row_idx, column=0, padx=spacing_x, pady=spacing_y)
    CR_entry_x = ttk.Entry(box_left, width=entry_width)
    CR_entry_x.grid(row=row_idx, column=1, padx=spacing_x, pady=spacing_y)
    CR_entry_y = ttk.Entry(box_left, width=entry_width)
    CR_entry_y.grid(row=row_idx, column=2, padx=spacing_x, pady=spacing_y)
    row_idx += 1

    # Left Arm End Lefter
    LaL_label = ttk.Label(box_left, text="left arm left")
    LaL_label.grid(row=row_idx, column=0, padx=spacing_x, pady=spacing_y)
    LaL_entry_x = ttk.Entry(box_left, width=entry_width)
    LaL_entry_x.grid(row=row_idx, column=1, padx=spacing_x, pady=spacing_y)
    LaL_entry_y = ttk.Entry(box_left, width=entry_width)
    LaL_entry_y.grid(row=row_idx, column=2, padx=spacing_x, pady=spacing_y)
    row_idx += 1

    # Left Arm End Righter
    LaR_label = ttk.Label(box_left, text="left arm right")
    LaR_label.grid(row=row_idx, column=0, padx=spacing_x, pady=spacing_y)
    LaR_entry_x = ttk.Entry(box_left, width=entry_width)
    LaR_entry_x.grid(row=row_idx, column=1, padx=spacing_x, pady=spacing_y)
    LaR_entry_y = ttk.Entry(box_left, width=entry_width)
    LaR_entry_y.grid(row=row_idx, column=2, padx=spacing_x, pady=spacing_y)
    row_idx += 1

    # Right Arm End Righter
    RaR_label = ttk.Label(box_left, text="right arm right")
    RaR_label.grid(row=row_idx, column=0, padx=spacing_x, pady=spacing_y)
    RaR_entry_x = ttk.Entry(box_left, width=entry_width)
    RaR_entry_x.grid(row=row_idx, column=1, padx=spacing_x, pady=spacing_y)
    RaR_entry_y = ttk.Entry(box_left, width=entry_width)
    RaR_entry_y.grid(row=row_idx, column=2, padx=spacing_x, pady=spacing_y)
    row_idx += 1

    # Right Arm End Lefter
    RaL_label = ttk.Label(box_left, text="right arm left")
    RaL_label.grid(row=row_idx, column=0, padx=spacing_x, pady=spacing_y)
    RaL_entry_x = ttk.Entry(box_left, width=entry_width)
    RaL_entry_x.grid(row=row_idx, column=1, padx=spacing_x, pady=spacing_y)
    RaL_entry_y = ttk.Entry(box_left, width=entry_width)
    RaL_entry_y.grid(row=row_idx, column=2, padx=spacing_x, pady=spacing_y)
    row_idx += 1

    # Middle Arm End Lefter
    MaL_label = ttk.Label(box_left, text="middle arm left")
    MaL_label.grid(row=row_idx, column=0, padx=spacing_x, pady=spacing_y)
    MaL_entry_x = ttk.Entry(box_left, width=entry_width)
    MaL_entry_x.grid(row=row_idx, column=1, padx=spacing_x, pady=spacing_y)
    MaL_entry_y = ttk.Entry(box_left, width=entry_width)
    MaL_entry_y.grid(row=row_idx, column=2, padx=spacing_x, pady=spacing_y)
    row_idx += 1

    # Middle Arm End Righter
    MaR_label = ttk.Label(box_left, text="middle arm right")
    MaR_label.grid(row=row_idx, column=0, padx=spacing_x, pady=spacing_y)
    MaR_entry_x = ttk.Entry(box_left, width=entry_width)
    MaR_entry_x.grid(row=row_idx, column=1, padx=spacing_x, pady=spacing_y)
    MaR_entry_y = ttk.Entry(box_left, width=entry_width)
    MaR_entry_y.grid(row=row_idx, column=2, padx=10, pady=spacing_y)
    row_idx += 1

    # ANALYSIS WINDOW
    box_right = tk.LabelFrame(root, text="ANALYSIS", bg="#f6ffe6", bd=2, relief="ridge", font=("Arial", 10, "bold"))    
    box_right.grid(row=1, column=3, rowspan=15, columnspan=3, padx=10, pady=10, sticky="nsew")
    x_label = ttk.Label(box_right, text="x")
    x_label.grid(row=0, column=1, padx=5, pady=5)
    y_label = ttk.Label(box_right, text="y")
    y_label.grid(row=0, column=2, padx=5, pady=5)

    row_idx = 1
    col_idx = 0

    # Corner Left
    aCL_label = ttk.Label(box_right, text="left corner")
    aCL_label.grid(row=row_idx, column=col_idx, padx=spacing_x, pady=spacing_y)
    aCL_entry_x = ttk.Entry(box_right, width=entry_width)
    aCL_entry_x.grid(row=row_idx, column=col_idx+1, padx=spacing_x, pady=spacing_y)
    aCL_entry_y = ttk.Entry(box_right, width=entry_width)
    aCL_entry_y.grid(row=row_idx, column=col_idx+2, padx=spacing_x, pady=spacing_y)
    row_idx += 1

    # Corner Middle
    aCM_label = ttk.Label(box_right, text="middle corner")
    aCM_label.grid(row=row_idx, column=col_idx, padx=spacing_x, pady=spacing_y)
    aCM_entry_x = ttk.Entry(box_right, width=entry_width)
    aCM_entry_x.grid(row=row_idx, column=col_idx+1, padx=spacing_x, pady=spacing_y)
    aCM_entry_y = ttk.Entry(box_right, width=entry_width)
    aCM_entry_y.grid(row=row_idx, column=col_idx+2, padx=spacing_x, pady=spacing_y)
    row_idx += 1

    # Corner Right
    aCR_label = ttk.Label(box_right, text="right corner")
    aCR_label.grid(row=row_idx, column=col_idx, padx=spacing_x, pady=spacing_y)
    aCR_entry_x = ttk.Entry(box_right, width=entry_width)
    aCR_entry_x.grid(row=row_idx, column=col_idx+1, padx=spacing_x, pady=spacing_y)
    aCR_entry_y = ttk.Entry(box_right, width=entry_width)
    aCR_entry_y.grid(row=row_idx, column=col_idx+2, padx=spacing_x, pady=spacing_y)
    row_idx += 1

    # Left Arm End Lefter
    aLaL_label = ttk.Label(box_right, text="left arm left")
    aLaL_label.grid(row=row_idx, column=col_idx, padx=spacing_x, pady=spacing_y)
    aLaL_entry_x = ttk.Entry(box_right, width=entry_width)
    aLaL_entry_x.grid(row=row_idx, column=col_idx+1, padx=spacing_x, pady=spacing_y)
    aLaL_entry_y = ttk.Entry(box_right, width=entry_width)
    aLaL_entry_y.grid(row=row_idx, column=col_idx+2, padx=spacing_x, pady=spacing_y)
    row_idx += 1

    # Left Arm End Righter
    aLaR_label = ttk.Label(box_right, text="left arm right")
    aLaR_label.grid(row=row_idx, column=col_idx, padx=spacing_x, pady=spacing_y)
    aLaR_entry_x = ttk.Entry(box_right, width=entry_width)
    aLaR_entry_x.grid(row=row_idx, column=col_idx+1, padx=spacing_x, pady=spacing_y)
    aLaR_entry_y = ttk.Entry(box_right, width=entry_width)
    aLaR_entry_y.grid(row=row_idx, column=col_idx+2, padx=spacing_x, pady=spacing_y)
    row_idx += 1

    # Right Arm End Righter
    aRaR_label = ttk.Label(box_right, text="right arm right")
    aRaR_label.grid(row=row_idx, column=col_idx, padx=spacing_x, pady=spacing_y)
    aRaR_entry_x = ttk.Entry(box_right, width=entry_width)
    aRaR_entry_x.grid(row=row_idx, column=col_idx+1, padx=spacing_x, pady=spacing_y)
    aRaR_entry_y = ttk.Entry(box_right, width=entry_width)
    aRaR_entry_y.grid(row=row_idx, column=col_idx+2, padx=spacing_x, pady=spacing_y)
    row_idx += 1

    # Right Arm End Lefter
    aRaL_label = ttk.Label(box_right, text="right arm left")
    aRaL_label.grid(row=row_idx, column=col_idx, padx=spacing_x, pady=spacing_y)
    aRaL_entry_x = ttk.Entry(box_right, width=entry_width)
    aRaL_entry_x.grid(row=row_idx, column=col_idx+1, padx=spacing_x, pady=spacing_y)
    aRaL_entry_y = ttk.Entry(box_right, width=entry_width)
    aRaL_entry_y.grid(row=row_idx, column=col_idx+2, padx=spacing_x, pady=spacing_y)
    row_idx += 1

    # Middle Arm End Lefter
    aMaL_label = ttk.Label(box_right, text="middle arm left")
    aMaL_label.grid(row=row_idx, column=col_idx, padx=spacing_x, pady=spacing_y)
    aMaL_entry_x = ttk.Entry(box_right, width=entry_width)
    aMaL_entry_x.grid(row=row_idx, column=col_idx+1, padx=spacing_x, pady=spacing_y)
    aMaL_entry_y = ttk.Entry(box_right, width=entry_width)
    aMaL_entry_y.grid(row=row_idx, column=col_idx+2, padx=spacing_x, pady=spacing_y)
    row_idx += 1

    # Middle Arm End Righter
    aMaR_label = ttk.Label(box_right, text="middle arm right")
    aMaR_label.grid(row=row_idx, column=col_idx, padx=spacing_x, pady=spacing_y)
    aMaR_entry_x = ttk.Entry(box_right, width=entry_width)
    aMaR_entry_x.grid(row=row_idx, column=col_idx+1, padx=spacing_x, pady=spacing_y)
    aMaR_entry_y = ttk.Entry(box_right, width=entry_width)
    aMaR_entry_y.grid(row=row_idx, column=col_idx+2, padx=10, pady=spacing_y)
    row_idx += 1

    # checkbox for video creation
    checkbox_var = tk.BooleanVar(value=False)  # Default is unchecked
    checkbox = ttk.Checkbutton(box_right, text="Create Analysed Video", variable=checkbox_var)
    checkbox.grid(row=14, column=2, padx=5, pady=5)

    # Image
    image_frame = ttk.LabelFrame(root, text="Mask Preview")
    image_frame.grid(row=0, column=6, rowspan=12, padx=spacing_x, pady=spacing_y)
    image_label = ttk.Label(image_frame)
    image_label.pack()

    # Progress bar
    progress_var = tk.DoubleVar()
    progress_bar = ttk.Progressbar(root, variable=progress_var, maximum=100)
    progress_bar.grid(row=17, column=0, columnspan=5, padx=spacing_x, pady=spacing_y, sticky="we")

    # Create a Text widget to show the log of processed files
    file_log = tk.Text(root, height=10, width=60, wrap="word", state=tk.DISABLED)
    file_log.grid(row=13, column=6, rowspan=15, padx=spacing_x, pady=spacing_y)

    

    
    def get_coordinates(file, analyse):

        def get_xmax_ymax(file):
            vid = cv2.VideoCapture(str(file))
            width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return width, height
        def adapt(x, y, dx, dy, dz):
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
        CL =    adapt(int(CL_entry_x.get()), int(CL_entry_y.get()), dx, dy, dz)
        CR =    adapt(int(CR_entry_x.get()), int(CR_entry_y.get()), dx, dy, dz)
        CM =    adapt(int(CM_entry_x.get()), int(CM_entry_y.get()), dx, dy, dz)
        LaL =   adapt(int(LaL_entry_x.get()), int(LaL_entry_y.get()), dx, dy, dz)
        LaR =   adapt(int(LaR_entry_x.get()), int(LaR_entry_y.get()), dx, dy, dz)
        RaR =   adapt(int(RaR_entry_x.get()), int(RaR_entry_y.get()), dx, dy, dz)
        RaL =   adapt(int(RaL_entry_x.get()), int(RaL_entry_y.get()), dx, dy, dz)
        MaL =   adapt(int(MaL_entry_x.get()), int(MaL_entry_y.get()), dx, dy, dz)
        MaR =   adapt(int(MaR_entry_x.get()), int(MaR_entry_y.get()), dx, dy, dz)

        aCL =    (int(aCL_entry_x.get()), int(aCL_entry_y.get()))
        aCR =    (int(aCR_entry_x.get()), int(aCR_entry_y.get()))
        aCM =    (int(aCM_entry_x.get()), int(aCM_entry_y.get()))
        aLaL =   (int(aLaL_entry_x.get()), int(aLaL_entry_y.get()))
        aLaR =   (int(aLaR_entry_x.get()), int(aLaR_entry_y.get()))
        aRaR =   (int(aRaR_entry_x.get()), int(aRaR_entry_y.get()))
        aRaL =   (int(aRaL_entry_x.get()), int(aRaL_entry_y.get()))
        aMaL =   (int(aMaL_entry_x.get()), int(aMaL_entry_y.get()))
        aMaR =   (int(aMaR_entry_x.get()), int(aMaR_entry_y.get()))
        
        # Return all coordinates as a dictionary or tuple
        coords = {"CL": CL,"CR": CR,"CM": CM,"LaL": LaL,"LaR": LaR,"RaR": RaR,"RaL": RaL,"MaL": MaL,"MaR": MaR}
        acoords = {"CL": aCL,"CR": aCR,"CM": aCM,"LaL": aLaL,"LaR": aLaR,"RaR": aRaR,"RaL": aRaL,"MaL": aMaL,"MaR": aMaR}

        if analyse==False:
            return coords
        elif analyse==True:
            return acoords

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
    def check_mask(analyse):

        entries = {k: v.get() for k, v in [('dx', dx_entry), ('dy', dy_entry), ('dz', dz_entry), 
                                           ('CL_x', CL_entry_x), ('CL_y', CL_entry_y), ('CR_x', CR_entry_x), ('CR_y', CR_entry_y), ('CM_x', CM_entry_x), ('CM_y', CM_entry_y), ('LaL_x', LaL_entry_x), ('LaL_y', LaL_entry_y), ('LaR_x', LaR_entry_x), ('LaR_y', LaR_entry_y), ('RaR_x', RaR_entry_x), ('RaR_y', RaR_entry_y), ('RaL_x', RaL_entry_x), ('RaL_y', RaL_entry_y), ('MaL_x', MaL_entry_x), ('MaL_y', MaL_entry_y), ('MaR_x', MaR_entry_x), ('MaR_y', MaR_entry_y), 
                                           ('aCL_x', aCL_entry_x), ('aCL_y', aCL_entry_y), ('aCR_x', aCR_entry_x), ('aCR_y', aCR_entry_y), ('aCM_x', aCM_entry_x), ('aCM_y', aCM_entry_y), ('aLaL_x', aLaL_entry_x), ('aLaL_y', aLaL_entry_y), ('aLaR_x', aLaR_entry_x), ('aLaR_y', aLaR_entry_y), ('aRaR_x', aRaR_entry_x), ('aRaR_y', aRaR_entry_y), ('aRaL_x', aRaL_entry_x), ('aRaL_y', aRaL_entry_y), ('aMaL_x', aMaL_entry_x), ('aMaL_y', aMaL_entry_y), ('aMaR_x', aMaR_entry_x), ('aMaR_y', aMaR_entry_y)]}
        save_entries(entries)

        path = path_entry.get()
        if not path:
            messagebox.showerror("Error", "Please select a folder.")
            return
        
        files = get_files(path, '.mp4')
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


        coords = get_coordinates(files[0], analyse)
        areas, whole_area = get_areas(coords)
        write_and_draw(files[0], areas)

    def start_processing():

        path = path_entry.get()
        if not path:
            messagebox.showerror("Error", "Please select a folder.")
            return
        
        files = get_files(path, '.mp4')
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

        coords = get_coordinates(files[0], analyse=False)
        areas, whole_area = get_areas(coords)

        for idx, video_file in enumerate(files):
            
            # update the files processed in the gui
            filename = os.path.basename(video_file)
            file_log.config(state=tk.NORMAL)  # Enable the Text widget to update it
            file_log.insert(tk.END, f"{filename} - masking\n")  # Append the finished file
            file_log.config(state=tk.DISABLED)  # Disable it again to prevent manual editing
            file_log.yview(tk.END)  # Scroll to the bottom of the text widget

            output_video_file = video_file.with_name(video_file.stem + '_Masked.mp4')
            adjust_video(video_file, output_video_file, coords, whole_area)

        messagebox.showinfo(f"Masking Complete ({idx+1} videos)")

    def analyse_videos():

        def define_bodyparts():

            bps_all = ['nose', 'left_ear', 'right_ear', 'left_ear_tip', 'right_ear_tip', 'left_eye', 'right_eye', 'head_midpoint', 
                        'neck', 'mid_back', 'mouse_center', 'mid_backend', 'mid_backend2', 'mid_backend3', 
                        'tail_base', 'tail1', 'tail2', 'tail3', 'tail4', 'tail5', 'tail_end',
                        'left_shoulder', 'left_midside', 'left_hip', 'right_shoulder', 'right_midside', 'right_hip']

            bps_entry = ['nose', 'left_ear', 'right_ear', 'left_ear_tip', 'right_ear_tip', 'left_eye', 'right_eye', 'head_midpoint', 
                        'neck', 'mid_back', 'mouse_center', 'mid_backend', 'mid_backend2', 'mid_backend3', 'tail_base', 
                        'left_shoulder', 'left_midside', 'left_hip', 'right_shoulder', 'right_midside', 'right_hip']
            
            bps_exit = ['nose', 'left_ear', 'right_ear', 'left_ear_tip', 'right_ear_tip', 'left_eye', 'right_eye', 'head_midpoint', 
                        'neck', 'mid_back', 'mouse_center', 'mid_backend',
                        'left_shoulder', 'left_midside', 'left_hip', 'right_shoulder', 'right_midside', 'right_hip']

            return bps_all, bps_entry, bps_exit

        def do_stuff(path, file, bps_all, bps_entry, bps_exit, areas, areas_int, create_video):

            def cleaning_raw_df(csv_file):
                df = pd.read_csv(csv_file, header=None, low_memory=False)

                # combines the string of the two rows to create new header
                new_columns = [f"{col[0]}_{col[1]}" for col in zip(df.iloc[1], df.iloc[2])]     # df.iloc[0] takes all values of first row
                df.columns = new_columns
                df = df.drop(labels=[0, 1, 2], axis="index")

                # adapt index
                df.set_index('bodyparts_coords', inplace=True)
                df.index.names = ['frames']

                # turn all the prior string values into floats and index into int
                df = df.astype(float)
                df.index = df.index.astype(int)
                
                # flip the video along the y axis
                # for column in df.columns:
                #     if '_y' in column:
                #         df[column] = y_pixel - df[column] 

                # whereever _likelihood <= x, change _x & _y to nan
                for bodypart in bps_all:
                    filter = df[f'{bodypart}_likelihood'] <= 0.9
                    df.loc[filter, f'{bodypart}_x'] = np.nan
                    df.loc[filter, f'{bodypart}_y'] = np.nan
                    df = df.drop(labels=f'{bodypart}_likelihood', axis="columns")
                
                return df

            def calc_all_pos(df, bps_entry, bps_exit, areas, areas_int):

                # position calculations
                def point_in_which_area(df, bp, frame, areas, areas_int):
                    nonlocal bp_not_in_any_area
                    # return area in which bodpart is located
                    x = df.loc[frame, f'{bp}_x']
                    y = df.loc[frame, f'{bp}_y']

                    if np.isnan(x):
                        return None
                    
                    bp_point = Point(x, y)
                    for i, area in enumerate(areas):
                        if area.contains(bp_point):
                            return areas_int[i]
                        
                    # if point is not nan and point is in non of the areas, return 99
                    bp_not_in_any_area += 1
                    return 99
                def point_in_spec_area(df, bp, frame, area):
                    # check if point is in area
                    # consider nan as not in area
                    x = df.loc[frame, f'{bp}_x']
                    if np.isnan(x):
                        return False

                    bp_point = Point(x, df.loc[frame, f'{bp}_y'])

                    return area.contains(bp_point)
                    
                def entry_into_arm(df, bps_list, frame, areas, areas_int):
                    # returns arm of entry if front 80% are in an ARM
                    # of all front 80%, none is allowed to be in another area and >= X bps need to be non-nan
                    nonnan_bps_in_new_area = 15
                    # entry can only happen if an exit happened before!

                    nonlocal frames_with_no_annot

                    # create list with all the areas in which the bodyparts are in
                    areas_of_bps = []
                    for bp in bps_list:
                        areas_of_bps.append(point_in_which_area(df, bp, frame, areas, areas_int))
                    
                    # only care about the ones that DLC has identified
                    if all(bp is None for bp in areas_of_bps):
                        frames_with_no_annot += 1
                        return False
                    areas_of_bps = [i for i in areas_of_bps if i is not None]

                    # if all bodyparts are in the area of the first bodypart and the first bodypart is not in center (=0)
                    if len(areas_of_bps) >= nonnan_bps_in_new_area:
                        first_value = areas_of_bps[0]
                        if all(elem == first_value for elem in areas_of_bps) and first_value != 0 and first_value != 99:
                            return first_value
                    return False
                def exit_outof_arm(df, bps_list, frame, curr_area, areas, areas_int):
                    # exit if front 80% not in current arm but rather parts of front 80% in other area (min 1 non-nan)
                    
                    # if a bp is in curr_arm, it is no exit -> return False; if no bp is in curr_arm, it may be an exit ->continue
                    for bp in bps_list:
                        
                        if point_in_spec_area(df, bp, frame, areas[curr_area]):
                            return False
                    
                    # check if any bp is in other area than curr_arm
                    areas_of_bps = []
                    for bp in bps_list:
                        areas_of_bps.append(point_in_which_area(df, bp, frame, areas, areas_int))
                    
                    # only care for bodyparts that DLC has identified
                    areas_of_bps = [i for i in areas_of_bps if i is not None]

                    # if now any bodypart is in a nother area, it is an exit
                    for area in areas_of_bps:
                        if area != curr_area:
                            return True
                    
                    return False

                # when animal is leaving an arm, position is automatically set as center
                all_positions = []
                curr_area = 0

                for frame in range(len(df.index)):
                    if curr_area == 0: 
                        entry = entry_into_arm(df, bps_entry, frame, areas, areas_int)
                        
                        if not entry:
                            all_positions.append(0)
                        elif entry:
                            all_positions.append(entry)
                            curr_area = entry

                    elif curr_area != 0:
                        exit = exit_outof_arm(df, bps_exit, frame, curr_area, areas, areas_int)
                        if not exit:
                            all_positions.append(curr_area)
                        elif exit:
                            all_positions.append(0)
                            curr_area = 0

                    # Update progress bar
                    progress_var.set((frame + 1) / len(df.index) * 100)
                    root.update_idletasks()

                return all_positions

            def calc_alteration_index(pos_changes):

                total_entries = len(pos_changes) - 2
                alterations = 0

                for i in range(len(pos_changes) - 2):
                    tripled = [pos_changes[i], pos_changes[i+1], pos_changes[i+2]]

                    if len(tripled) == len(set(tripled)):
                        alterations += 1
                        
                alt_index = alterations / total_entries

                return alterations, total_entries, alt_index

            def out_to_txt(file, pos_changes, alterations, total_entries, alt_index, alert):
                output_file = os.path.dirname(file) + '\\YLine_AI.txt'
                with open (output_file, 'a') as f:
                    f.write(f'\n\nfile_name: {file}'
                            f'\npos_chang: {pos_changes}'
                            f'\nalt_index: {alterations} / {total_entries} = {round(alt_index, 4)}'
                            f'\n{alert}')

            def do_video(csv_file, all_positions):

                # FIND VIDEO that corresponds to csv file !!needs to have the same beginning!!
                input_video_found = False
                csv_stem = Path(csv_file).stem
                endings = ['', 'YLineMasked', '_labeled', '_filtered_labeled', 'DLC']

                for i in range(len(csv_stem), 0, -1):       # start, stop, step

                    for ending in endings:
                        input_video_file = Path(path) / (csv_stem[:i] + ending + '.mp4')

                        if input_video_file.exists():
                            input_video_found = True
                            break

                    if input_video_found:
                        break
                        
                if not input_video_found:
                    print(f'No matching video file found for {csv_file}')
                    return
                

                # CREATE VIDEO
                # original stuff
                vid = cv2.VideoCapture(str(input_video_file))
                width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
                nframes = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = int(vid.get(cv2.CAP_PROP_FPS))
                if nframes != len(all_positions):
                    print('nframe of video different than len(all_positions)')
                
                # create new video
                output_video_file = input_video_file.with_name(input_video_file.stem + '_YLine.mp4')
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video = cv2.VideoWriter(str(output_video_file), fourcc, fps, (width, height)) 

                # update the files processed in the gui
                filename = os.path.basename(output_video_file)
                file_log.config(state=tk.NORMAL)  # Enable the Text widget to update it
                file_log.insert(tk.END, f"{filename} - creating video\n")  # Append the file
                file_log.config(state=tk.DISABLED)  # Disable it again to prevent manual editing
                file_log.yview(tk.END)  # Scroll to the bottom of the text widget

                for i in range(nframes):
                    ret, frame = vid.read()
                    if ret:    

                        # in which area is mouse in this frame
                        mouse_area_nb = all_positions[i]
                        mouse_area = areas[mouse_area_nb]
                        points_of_area = np.array(mouse_area.exterior.coords, dtype=np.int32)
                        
                        # draw on frame
                        cv2.polylines(frame, [points_of_area], isClosed=True, color=(0, 0, 255), thickness=2)
                        cv2.putText(frame, f'{mouse_area_nb}', (int(width/3), 50), cv2.FONT_HERSHEY_SIMPLEX , 1, (0, 0, 255), 2, cv2.LINE_AA)

                        video.write(frame)

                        # Update progress bar
                        progress_var.set((i + 1) / nframes * 100)
                        root.update_idletasks()

                    else:
                        break
                
                vid.release()
                video.release()

                return nframes


            df = cleaning_raw_df(file)
            all_positions = calc_all_pos(df, bps_entry, bps_exit, areas, areas_int)

            # only add areas that were entered after a change
            pos_changes = [pos for i, pos in enumerate(all_positions) if i == 0 or pos != all_positions[i-1]]

            # remove center (0) position
            pos_changes = [i for i in pos_changes if i != 0]

            alterations, total_entries, alt_index = calc_alteration_index(pos_changes)

            # alert = '!!!' if bp_not_in_any_area > 0 or frames_with_no_annot > 1*fps else ''
            # if bp_not_in_any_area > 0:
            #     alert = f'!!! {bp_not_in_any_area} frames where bps not in any area'
            # elif frames_with_no_annot > 1*fps:
            #     alert = alert + f' + {frames_with_no_annot} frames without annotations'
            alert = ''
            out_to_txt(file, pos_changes, alterations, total_entries, alt_index, alert)

            if create_video:
                do_video(csv_file, all_positions)
            
            return all_positions


        path = path_entry.get()
        files_csv = get_files(path, '.csv')
        files_mp4 = get_files(path, '.mp4')
      
        coords = get_coordinates(files_mp4[0], analyse=True)
        bps_all, bps_entry, bps_exit = define_bodyparts()
        areas, whole_area = get_areas(coords)
        areas_int = [0, 1, 2, 3]

        for idx, csv_file in enumerate(files_csv):
                    
            # update the files processed in the gui
            filename = os.path.basename(csv_file)
            file_log.config(state=tk.NORMAL)  # Enable the Text widget to update it
            file_log.insert(tk.END, f"{filename} - analyzing\n")  # Append the file
            file_log.config(state=tk.DISABLED)  # Disable it again to prevent manual editing
            file_log.yview(tk.END)  # Scroll to the bottom of the text widget

            frames_with_no_annot, bp_not_in_any_area = 0, 0
            do_stuff(path, csv_file, bps_all, bps_entry, bps_exit, areas, areas_int, create_video=checkbox_var.get())

        messagebox.showinfo(f"Masking Complete ({idx+1} videos)")



    # Do Buttons
    check_button = ttk.Button(box_left, text="Check Mask", command=lambda: check_mask(analyse=False))
    check_button.grid(row=13, column=1, padx=10, pady=20)

    process_button = tk.Button(box_left, text="Mask Videos", command=lambda: threading.Thread(target=start_processing).start())
    process_button.grid(row=13, column=2, padx=10, pady=20)

    check_button = ttk.Button(box_right, text="Check Mask", command=lambda: check_mask(analyse=True))
    check_button.grid(row=13, column=1, padx=10, pady=20)

    process_button = tk.Button(box_right, text="Analyse Videos", command=lambda: threading.Thread(target=analyse_videos).start())
    process_button.grid(row=13, column=2, padx=10, pady=20)

    

    initialize_gui()
    root.mainloop()

open_gui()
