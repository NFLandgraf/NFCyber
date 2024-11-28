#%%
import numpy as np
import matplotlib.pyplot as plt

# Parameters
x = np.linspace(0, 10, 1000)  # x-axis values
frequency = 1  # Frequency of the sine wave
max_amplitude = 0.1  # Maximum amplitude of the sine wave
transition_length = 500  # Number of points over which to transition to sine wave

# Generate a straight line
y = np.zeros_like(x)

# Gradually transition from straight line to sine wave
for i in range(transition_length, len(x)):
    amplitude = max_amplitude * (i - transition_length) / (len(x) - transition_length)
    y[i] = amplitude * np.sin(2 * np.pi * frequency * x[i])

# Plotting
fig, ax = plt.subplots(figsize=(30, 2))
ax.plot(x, y, color='gray', linewidth=8.0)  # Change color here

# Remove axes for a cleaner look
ax.axis('off')
plt.show()
# Save with transparent background
fig.savefig('sine_transition.png', transparent=True, dpi=300)
plt.close(fig)



#%%

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# Parameters
x = np.linspace(0, 10, 1000)  # x-axis values
frequency = 2  # Frequency of the sine wave
max_amplitude = 1  # Maximum amplitude of the sine wave
transition_length = 500  # Number of points over which to transition to sine wave

# Generate a straight line
y = np.zeros_like(x)

# Gradually transition from straight line to sine wave
for i in range(transition_length, len(x)):
    amplitude = max_amplitude * (i - transition_length) / (len(x) - transition_length)
    y[i] = amplitude * np.sin(2 * np.pi * frequency * x[i])

# Create fading effect
segments = np.array([[[x[i], y[i]], [x[i+1], y[i+1]]] for i in range(len(x)-1)])
alphas = np.linspace(1, 0, len(segments))  # Alpha values from opaque to transparent

# Create a line collection with fading
lc = LineCollection(segments, cmap='Grays', linewidth=8)
lc.set_array(alphas)  # Set alpha values

# Plotting
fig, ax = plt.subplots(figsize=(30, 2))
ax.add_collection(lc)
ax.autoscale()
ax.axis('off')

plt.show()
# Save with transparent background
fig.savefig('sine_transition_fade.png', transparent=True, dpi=300)
plt.close(fig)
