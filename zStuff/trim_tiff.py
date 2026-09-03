#%%
import tifffile

input_path = r"C:\Users\landgrafn\Desktop\Minicortex_m150_HC_F1000_P1.1.tiff"
output_path = r"C:\Users\landgrafn\Desktop\Minicortex_m150_HC_F1000_P1.1_trim.tiff"

last_frame = 1051

# Read, retain frames 0 through 9,999, and save
movie = tifffile.imread(input_path)
movie_trimmed = movie[:last_frame]

tifffile.imwrite(output_path,movie_trimmed,photometric="minisblack",bigtiff=True,)

print(f"Saved {len(movie_trimmed)} frames to:\n{output_path}")