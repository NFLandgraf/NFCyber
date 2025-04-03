import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import cv2
import numpy as np
from pathlib import Path
from matplotlib import image
import os

# Function to get video files
def get_files(path):
    files = [file for file in Path(path).iterdir() if file.is_file() and '.mp4' in file.name]
    return files

# Adapt function as in your code
def adapt(x, y, dx, dy, dz, width, height):
    adapt_x = (x + dx) * dz
    adapt_y = (y + dy) * dz

    if adapt_x < 0:
        adapt_x = 0
    elif adapt_x > width:
        adapt_x = width

    if adapt_y < 0:
        adapt_y = 0
    elif adapt_y > height:
        adapt_y = height

    return adapt_x, adapt_y

# Get video properties
def get_xmax_ymax(file):
    vid = cv2.VideoCapture(str(file))
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return width, height

# Processing function to adjust and mask videos
def mask_video(files, dx, dy, dz):
    def adjust_video(input_file, output_file):
        vid = cv2.VideoCapture(str(input_file))
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        nframes = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        
        x1, y1 = adapt(308, 250, dx, dy, dz, width, height)
        x2, y2 = adapt(385, 250, dx, dy, dz, width, height)

        # Create new video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(str(output_file), fourcc, fps, (x2 - x1, y2 - y1)) 
        
        for curr_frame in range(nframes):
            ret, frame = vid.read()
            if ret:
                frame = frame[y1:y2, x1:x2]  # Crop the frame
                video.write(frame)
            else:
                break

        vid.release()
        video.release()

    for video_file in files:
        output_video_file = video_file.with_name(video_file.stem + '_Masked.mp4')
        adjust_video(video_file, output_video_file)
        messagebox.showinfo("Processing Complete", f"Processed {video_file.name} successfully!")

# GUI for user input
def open_gui():
    root = tk.Tk()
    root.title("Video Masking Tool")

    entry_width = 5
    
    # Path selection
    def browse_folder():
        folder_selected = filedialog.askdirectory()
        path_entry.delete(0, tk.END)
        path_entry.insert(0, folder_selected)
    
    path_label = ttk.Label(root, text="Video Folder")
    path_label.grid(row=0, column=0, padx=10, pady=5)
    
    path_entry = ttk.Entry(root, width=50)
    path_entry.grid(row=0, column=1, padx=10, pady=5)
    
    browse_button = ttk.Button(root, text="Browse", command=browse_folder)
    browse_button.grid(row=0, column=3, padx=10, pady=5)
    
    # Parameter inputs for dx, dy, dz
    dx_label = ttk.Label(root, text="dx (horizontal shift)")
    dx_label.grid(row=1, column=3, padx=10, pady=5)
    dx_entry = ttk.Entry(root)
    dx_entry.grid(row=1, column=4, padx=10, pady=5)
    dx_entry.insert(0, "0")
    
    dy_label = ttk.Label(root, text="dy (vertical shift)")
    dy_label.grid(row=2, column=3, padx=10, pady=5)
    dy_entry = ttk.Entry(root)
    dy_entry.grid(row=2, column=4, padx=10, pady=5)
    dy_entry.insert(0, "0")
    
    dz_label = ttk.Label(root, text="dz (size scale)")
    dz_label.grid(row=3, column=3, padx=10, pady=5)
    dz_entry = ttk.Entry(root)
    dz_entry.grid(row=3, column=4, padx=10, pady=5)
    dz_entry.insert(0, "1")

    # Corner Left
    corner_left_label = ttk.Label(root, text="corner left")
    corner_left_label.grid(row=1, column=0, padx=10, pady=5)
    corner_left_entry_x = ttk.Entry(root, width=entry_width)
    corner_left_entry_x.grid(row=1, column=1, padx=10, pady=5)
    corner_left_entry_x.insert(0, "1")
    corner_left_entry_y = ttk.Entry(root, width=entry_width)
    corner_left_entry_y.grid(row=1, column=2, padx=10, pady=5)
    corner_left_entry_y.insert(0, "1")

    # Corner Right
    corner_right_label = ttk.Label(root, text="corner right")
    corner_right_label.grid(row=2, column=0, padx=10, pady=5)
    corner_right_entry_x = ttk.Entry(root, width=entry_width)
    corner_right_entry_x.grid(row=2, column=1, padx=10, pady=5)
    corner_right_entry_x.insert(0, "1")
    corner_right_entry_y = ttk.Entry(root, width=entry_width)
    corner_right_entry_y.grid(row=2, column=2, padx=10, pady=5)
    corner_right_entry_y.insert(0, "1")

    # Corner Middle
    corner_middle_label = ttk.Label(root, text="corner middle")
    corner_middle_label.grid(row=3, column=0, padx=10, pady=5)
    corner_middle_entry_x = ttk.Entry(root, width=entry_width)
    corner_middle_entry_x.grid(row=3, column=1, padx=10, pady=5)
    corner_middle_entry_x.insert(0, "1")
    corner_middle_entry_y = ttk.Entry(root, width=entry_width)
    corner_middle_entry_y.grid(row=3, column=2, padx=10, pady=5)
    corner_middle_entry_y.insert(0, "1")

    # Left Arm End Lefter
    left_arm_end_lefter_label = ttk.Label(root, text="left arm end lefter")
    left_arm_end_lefter_label.grid(row=4, column=0, padx=10, pady=5)
    left_arm_end_lefter_entry_x = ttk.Entry(root, width=entry_width)
    left_arm_end_lefter_entry_x.grid(row=4, column=1, padx=10, pady=5)
    left_arm_end_lefter_entry_x.insert(0, "1")
    left_arm_end_lefter_entry_y = ttk.Entry(root, width=entry_width)
    left_arm_end_lefter_entry_y.grid(row=4, column=2, padx=10, pady=5)
    left_arm_end_lefter_entry_y.insert(0, "1")

    # Left Arm End Righter
    left_arm_end_righter_label = ttk.Label(root, text="left arm end righter")
    left_arm_end_righter_label.grid(row=5, column=0, padx=10, pady=5)
    left_arm_end_righter_entry_x = ttk.Entry(root, width=entry_width)
    left_arm_end_righter_entry_x.grid(row=5, column=1, padx=10, pady=5)
    left_arm_end_righter_entry_x.insert(0, "1")
    left_arm_end_righter_entry_y = ttk.Entry(root, width=entry_width)
    left_arm_end_righter_entry_y.grid(row=5, column=2, padx=10, pady=5)
    left_arm_end_righter_entry_y.insert(0, "1")

    # Right Arm End Righter
    right_arm_end_righter_label = ttk.Label(root, text="right arm end righter")
    right_arm_end_righter_label.grid(row=6, column=0, padx=10, pady=5)
    right_arm_end_righter_entry_x = ttk.Entry(root, width=entry_width)
    right_arm_end_righter_entry_x.grid(row=6, column=1, padx=10, pady=5)
    right_arm_end_righter_entry_x.insert(0, "1")
    right_arm_end_righter_entry_y = ttk.Entry(root, width=entry_width)
    right_arm_end_righter_entry_y.grid(row=6, column=2, padx=10, pady=5)
    right_arm_end_righter_entry_y.insert(0, "1")

    # Right Arm End Lefter
    right_arm_end_lefter_label = ttk.Label(root, text="right arm end lefter")
    right_arm_end_lefter_label.grid(row=7, column=0, padx=10, pady=5)
    right_arm_end_lefter_entry_x = ttk.Entry(root, width=entry_width)
    right_arm_end_lefter_entry_x.grid(row=7, column=1, padx=10, pady=5)
    right_arm_end_lefter_entry_x.insert(0, "1")
    right_arm_end_lefter_entry_y = ttk.Entry(root, width=entry_width)
    right_arm_end_lefter_entry_y.grid(row=7, column=2, padx=10, pady=5)
    right_arm_end_lefter_entry_y.insert(0, "1")

    # Middle Arm End Lefter
    middle_arm_end_lefter_label = ttk.Label(root, text="middle arm end lefter")
    middle_arm_end_lefter_label.grid(row=8, column=0, padx=10, pady=5)
    middle_arm_end_lefter_entry_x = ttk.Entry(root, width=entry_width)
    middle_arm_end_lefter_entry_x.grid(row=8, column=1, padx=10, pady=5)
    middle_arm_end_lefter_entry_x.insert(0, "1")
    middle_arm_end_lefter_entry_y = ttk.Entry(root, width=entry_width)
    middle_arm_end_lefter_entry_y.grid(row=8, column=2, padx=10, pady=5)
    middle_arm_end_lefter_entry_y.insert(0, "1")

    # Middle Arm End Righter
    middle_arm_end_righter_label = ttk.Label(root, text="middle arm end righter")
    middle_arm_end_righter_label.grid(row=9, column=0, padx=10, pady=5)
    middle_arm_end_righter_entry_x = ttk.Entry(root, width=entry_width)
    middle_arm_end_righter_entry_x.grid(row=9, column=1, padx=10, pady=5)
    middle_arm_end_righter_entry_x.insert(0, "1")
    middle_arm_end_righter_entry_y = ttk.Entry(root, width=entry_width)
    middle_arm_end_righter_entry_y.grid(row=9, column=2, padx=10, pady=5)
    middle_arm_end_righter_entry_y.insert(0, "1")

    def get_coordinates():
        dx = dx_entry.get()
        dy = dy_entry.get()
        dz = dz_entry.get()

        corner_left =               (float(corner_left_entry_x.get()), float(corner_left_entry_y.get()))
        corner_right =              (float(corner_right_entry_x.get()), float(corner_right_entry_y.get()))
        corner_middle =             (float(corner_middle_entry_x.get()), float(corner_middle_entry_y.get()))
        
        left_arm_end_lefter =       (float(left_arm_end_lefter_entry_x.get()), float(left_arm_end_lefter_entry_y.get()))
        left_arm_end_righter =      (float(left_arm_end_righter_entry_x.get()), float(left_arm_end_righter_entry_y.get()))
        
        right_arm_end_righter =     (float(right_arm_end_righter_entry_x.get()), float(right_arm_end_righter_entry_y.get()))
        right_arm_end_lefter =      (float(right_arm_end_lefter_entry_x.get()), float(right_arm_end_lefter_entry_y.get()))
        
        middle_arm_end_lefter =     (float(middle_arm_end_lefter_entry_x.get()), float(middle_arm_end_lefter_entry_y.get()))
        middle_arm_end_righter =    (float(middle_arm_end_righter_entry_x.get()), float(middle_arm_end_righter_entry_y.get()))
        
        # Return all coordinates as a dictionary or tuple
        return {
            "dx": dx,
            "dy": dy,
            "dz": dz,
            "corner_left": corner_left,
            "corner_right": corner_right,
            "corner_middle": corner_middle,
            "left_arm_end_lefter": left_arm_end_lefter,
            "left_arm_end_righter": left_arm_end_righter,
            "right_arm_end_righter": right_arm_end_righter,
            "right_arm_end_lefter": right_arm_end_lefter,
            "middle_arm_end_lefter": middle_arm_end_lefter,
            "middle_arm_end_righter": middle_arm_end_righter
        }

    # Start processing
    def start_processing():
        path = path_entry.get()
        coordinates = get_coordinates()
        print(coordinates)
        
        if not path:
            messagebox.showerror("Error", "Please select a folder.")
            return
        
        files = get_files(path)
        
        if not files:
            messagebox.showerror("Error", "No MP4 files found in the selected folder.")
            return
        
        # Run the masking process
        mask_video(files, dx, dy, dz)

    process_button = ttk.Button(root, text="Start Processing", command=start_processing)
    process_button.grid(row=5, column=4, padx=10, pady=20)
    
    root.mainloop()

# Run the GUI
open_gui()
