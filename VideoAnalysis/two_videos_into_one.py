#%%
from moviepy.editor import VideoFileClip, clips_array

# Load the two video files
video1 = VideoFileClip("C:\\Users\landgrafn\\Desktop\\m90\\2024-10-31-16-52-47_CA1-m90_OF_OverlayTrained_morph.mp4")
video2 = VideoFileClip("C:\\Users\landgrafn\\Desktop\\m90\\2024-10-31-16-52-47_CA1-m90_OF_dff_trimmed_to_behav(firstPos).mp4")

# Resize videos if needed (optional, to make sure both videos are the same height)
# If both videos are already the same size, you can skip this step.
video1 = video1.resize(height=600)
video2 = video2.resize(height=600)

# Ensure both videos have the same duration (optional, if you want them to stop at the same time)
min_duration = min(video1.duration, video2.duration)
video1 = video1.subclip(0, min_duration)
video2 = video2.subclip(0, min_duration)

# Combine videos side by side
final_video = clips_array([[video1, video2]])

# Write the result to a new file
final_video.write_videofile("C:\\Users\landgrafn\\Desktop\\outputtt.mp4", codec="libx264", fps=30)
