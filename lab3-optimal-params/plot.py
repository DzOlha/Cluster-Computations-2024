import matplotlib.pyplot as plt
import json
import os
import numpy as np

# Constants
MIN_PROCESSES = 1
MAX_PROCESSES = 10
MIN_GRANULARITY = 1
MAX_GRANULARITY = 20
MIN_BROADCAST_RATE = 1
MAX_BROADCAST_RATE = 50

# Load data from the results.json file
with open('results.json', 'r') as file:
    data = json.load(file)

# Extract the values from the JSON data
processes = []
granularity = []
broadcast_rate = []
time = []

# Iterate over the data and extract the values
for key in data:
    processes.append(data[key]['processes'])
    granularity.append(data[key]['granularity'])
    broadcast_rate.append(data[key]['broadcast_rate'])
    time.append(data[key]['time'])

# Create the scatter plot
plt.figure(figsize=(10, 6))

# Use time for the y-axis, and granularity for the size of the points
scatter = plt.scatter(processes,
                      time,        # Time for the y-axis
                      c=broadcast_rate,  # Broadcast rate for coloring
                      s=granularity,  # Granularity for size (no scaling)
                      cmap='winter',  # Choose your color map (e.g., 'winter')
                      edgecolors='k',
                      alpha=0.7)

# Add labels and title
plt.xlabel('Number of Processes')
plt.ylabel('Time')
plt.title('Scatter Plot of Time Dependency on Processes, Granularity, and Broadcast Rate')

# Add a color bar for broadcast rate
cbar = plt.colorbar(scatter)
cbar.set_label('Broadcast Rate')

# Create custom legend for granularity
# Create sample data for the legend
granularity_labels = [1, 10, 20]  # Example granularity values

# Add scatter points for the legend (same size as for granularity)
for g in granularity_labels:
    plt.scatter([], [], c='black', s=g,  # Use raw granularity values for the legend
                label=f'Granularity {g}', edgecolors='k', alpha=0.7)

# Add a legend for the granularity
plt.legend(title="Granularity")

# Save the plot to a file
plot_file_path = os.path.join(os.path.dirname(__file__), "scaling_results.png")
plt.savefig(plot_file_path)
print(f"Scaling plot saved to {plot_file_path}")

# Show the plot
plt.show()
