#%%
import pandas as pd

# --- Settings ---
input_csv = "D:\\2025-02-12_hTauxAPP1(3m)_RI3_m221_Test_edit_fps.csv"  # your input file
output_csv = "D:\\2025-02-12_hTauxAPP1(3m)_RI3_m221_Test_edit_fps_frames.csv"
fps = 30
total_frames = 17908
# ----------------

# Load original time-based annotations
df = pd.read_csv(input_csv)

# Convert time to frame index
df['Frame'] = (df['Time'] * fps).round().astype(int)

# # Fill empty Behavior cells with forward fill
# df['Behaviour'] = df['Behaviour'].ffill()

# # Build full frame-indexed DataFrame
# framewise = pd.DataFrame({'Frame': range(total_frames)})
# framewise = framewise.merge(df[['Frame', 'Behaviour']], on='Frame', how='left')
# framewise['Behaviour'] = framewise['Behaviour'].fillna('')  # empty string for no behavior

# Save to CSV
df.to_csv(output_csv, index=False)
print(f"✅ Saved: {output_csv}")
