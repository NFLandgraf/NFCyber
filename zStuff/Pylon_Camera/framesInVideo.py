
import pandas as pd
import numpy as np

# Load the CSV file into a DataFrame
csv_file_path = 'C:\\Users\\Admin\\Desktop\jajajaj.csv'
df = pd.read_csv(csv_file_path)

# Filter rows where "Channel Name" is "GPIO-1"
filtered_values = df[df[' Channel Name'] == ' GPIO-1'][' Value'].to_numpy()

# Convert to array (if needed, it will already be an array-like object)
yoreal_array = filtered_values.tolist()  # or just use filtered_values directly

fail = 0
real_array = []
for i, value in enumerate(yoreal_array):
    print(value)

    try:
        real_array.append(int(value))
    
    except:
        print(value)
        fail += 1

#print(real_array)
print("fails")
print(fail)

def count_segments(values):
    segments = 0
    in_segment = False
    
    for value in values:
        if value > 5000:
            if not in_segment:
                segments += 1
                in_segment = True
        else:
            in_segment = False
            
    return segments



# Count segments
segment_count = count_segments(real_array)
print(f'Number of segments over 5000: {segment_count}')