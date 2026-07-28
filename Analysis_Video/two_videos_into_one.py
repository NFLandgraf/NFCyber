#%%
from moviepy.editor import VideoFileClip, clips_array, ColorClip

# Load the two video files
video1 = VideoFileClip(r"C:\Users\landgrafn\Desktop\ExampleVideos\RI_Example.mp4")
video2 = VideoFileClip(r"C:\Users\landgrafn\Desktop\ExampleVideos\RI_Example_sLEAP.mp4")
video3 = VideoFileClip(r"C:\Users\landgrafn\Desktop\ExampleVideos\RI_Example_sLEAP_SimBA.mp4")

# Resize videos if needed (optional, to make sure both videos are the same height)
# If both videos are already the same size, you can skip this step.
video1 = video1.resize(height=600)
video2 = video2.resize(height=600)
video3 = video3.resize(height=600)

# Ensure both videos have the same duration (optional, if you want them to stop at the same time)
min_duration = min(video1.duration, video2.duration, video3.duration)
video1 = video1.subclip(0, min_duration)
video2 = video2.subclip(0, min_duration)
video3 = video3.subclip(0, min_duration)

# Create white separator (width=10 px)
separator_width = 10
sep = ColorClip(size=(separator_width, 600), color=(255, 255, 255)).set_duration(min_duration)

# Combine with separators
final_video = clips_array([[video1, sep, video2, sep, video3]])

# force even final dimensions
even_w = final_video.w - (final_video.w % 2)
even_h = final_video.h - (final_video.h % 2)
final_video = final_video.crop(x1=0, y1=0, x2=even_w, y2=even_h)

# Write the result to a new file
final_video.write_videofile(
    r"C:\Users\landgrafn\Desktop\ExampleVideos\RI_Example_combine.mp4",
    codec="libx264",
    audio_codec="aac",
    fps=video1.fps,
    preset="medium",
    ffmpeg_params=["-pix_fmt", "yuv420p"]
)
