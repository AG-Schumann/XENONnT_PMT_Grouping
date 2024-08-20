#!/usr/bin/env python
# coding: utf-8

# ## Preparation & Setup

# **Imports**

# In[2]:


import os
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from tqdm import tqdm
import itertools
import concurrent.futures
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle
import random
import matplotlib.colors as mcolors

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model


# **Create hexagonal pattern**

# In[3]:


PMT_positions = []
rot_angle = -150/180 * np.pi   # rotate setup by 60°
#rot_angle = 0

# hexagonal top pattern
PMTs_top = 253  # number of PMTs in top array
row_nPMT = [0, 6, 9, 12, 13, 14, 15, 16, 17, 16, 17, 16, 17, 16, 15, 14, 13, 12, 9, 6]   # number of PMTs per row
row_nPMT_cumsum = [sum(row_nPMT[0:x + 1]) for x in range(0, len(row_nPMT))]   
y_start = 62.1863   # starting value for first PMT row

for pmt in range(0, PMTs_top):
    _row = [i for i,x in enumerate(row_nPMT_cumsum) if x <= pmt][-1]   # identify which row PMT is in
    _row_position = pmt - row_nPMT_cumsum[_row]   # identify position within row
    
    y = y_start - _row * 6.90959   # y coordinate of PMT
    if (row_nPMT[_row+1] % 2):   # x position for odd rows
        x = (int(row_nPMT[_row+1] / -2.) + _row_position) * 7.97850
    else:   # x position for even rows (including offset)
        x = (int(row_nPMT[_row+1] / -2.) + _row_position) * 7.97850 + 3.98925
    
    xp = x*np.cos(rot_angle) + y*np.sin(rot_angle)   # apply rotation
    yp = -x*np.sin(rot_angle) + y*np.cos(rot_angle)
    PMT_positions.append((xp,yp))   # store coordinates

# hexagonal bottom pattern, same as above, less PMTs, no rotation
PMTs_bottom = 241
row_nPMT = [0, 4, 9, 10, 13, 14, 15, 16, 15, 16, 17, 16, 15, 16, 15, 14, 13, 10, 9, 4]
row_nPMT_cumsum = [sum(row_nPMT[0:x + 1]) for x in range(0, len(row_nPMT))]
y_start = 62.1863

for pmt in range(0, PMTs_bottom):
    _row = [i for i,x in enumerate(row_nPMT_cumsum) if x <= pmt][-1]
    _row_position = pmt - row_nPMT_cumsum[_row]
    
    y = y_start - _row * 6.90959
    if (row_nPMT[_row+1] % 2):
        x = (int(row_nPMT[_row+1] / -2.) + _row_position) * 7.97850
    else:
        x = (int(row_nPMT[_row+1] / -2.) + _row_position) * 7.97850 + 3.98925
    
    PMT_positions.append((x,y,))

meta = {'tpc_radius': 66.4,
         'PMTs_top': PMTs_top,
         'PMTs_bottom': PMTs_bottom,
         'PMT_positions': PMT_positions,
         'PMTOuterRingRadius': 3.875, # cm
        }

pmt_xy = np.array(PMT_positions, dtype=[('x', np.float32),('y', np.float32)])
pmt_xy_top = pmt_xy[:PMTs_top]
pmt_xy_bottom = pmt_xy[PMTs_top:]


# **Data prep**

# In[29]:


# Constants
PATH = "/data/storage/posrec_hitpatterns"
PMTs_top = 253
Broken_runs = [3,18,32,51,69,88,105,124,170,171,172,173,174,175,176,177,178,179,190,191,192,193,194,195,196,197,198,199,239,257,268,272,298,328]

# Get list of all HDF5 files in the directory
file_names = [f for f in os.listdir(PATH) if f.endswith(".hdf5")]
file_names.sort()

# Filter out broken runs
file_names = [f for i, f in enumerate(file_names) if i not in Broken_runs]

# Initialize lists to hold data from all files
mc_x_list = []
mc_y_list = []
hitpatterns_list = []

# Load data from each file and append to lists
for file_name in file_names:
    with h5py.File(os.path.join(PATH, file_name), "r") as hf:
        mc_x_list.append(np.array(hf["mc_x"][:]))
        mc_y_list.append(np.array(hf["mc_y"][:]))
        hitpatterns_list.append(np.array(hf["hitpatterns"][:]))
        
# Concatenate data from all files
mc_x = np.concatenate(mc_x_list)
mc_y = np.concatenate(mc_y_list)
hitpatterns = np.concatenate(hitpatterns_list)

# Normalize hitpatterns by their maximum values
hitpatterns_max = np.max(hitpatterns, axis=1, keepdims=True)
intensity = hitpatterns / hitpatterns_max

# Combine mc_x and mc_y into one array for the target positions
mc_positions = np.vstack((mc_x, mc_y)).T

# Split the data into training, validation, and test sets
intensity_train, intensity_temp, mc_positions_train, mc_positions_temp = train_test_split(intensity, mc_positions, test_size=0.3, random_state=42)
intensity_val, intensity_test, mc_positions_val, mc_positions_test = train_test_split(intensity_temp, mc_positions_temp, test_size=0.5, random_state=42)


# In[42]:


print("MC runs used: ", len(file_names))
print("Hitpatterns used to train: ", len(intensity_train))
print("Hitpatterns used to test: ", len(intensity_test))


# In[46]:


data_array = intensity_test[:]
zero_count = np.count_nonzero(data_array == 0)
print(f"Number of zero entries: {zero_count}")


# In[47]:


data_array = intensity_test[:]

# Count the number of zeros in each hitpattern (row)
zero_counts = np.count_nonzero(data_array == 0, axis=1)

# Count how many hitpatterns contain more than X zero entries
hitpatterns_with_more_than_10_zeros = np.sum(zero_counts > 10)

print(f"Number of hitpatterns in total: {len(data_array)}")
print(f"Number of hitpatterns with more than 10 zero entries: {hitpatterns_with_more_than_10_zeros}")


# In[84]:


zero_entries_per_hitpattern = []
zero_entries_per_hitpattern_test = []

for idx, hitpattern in tqdm(enumerate(intensity)):
    zero_count = np.count_nonzero(hitpattern == 0)
    zero_entries_per_hitpattern.append(zero_count)
    #print(f"Number of zero entries: {zero_count}")
    

for idx, hitpattern in tqdm(enumerate(intensity_test)):
    zero_count = np.count_nonzero(hitpattern == 0)
    zero_entries_per_hitpattern_test.append(zero_count)
    #print(f"Number of zero entries: {zero_count}")


# In[67]:


# Plot the histogram
plt.figure(figsize=(10, 6))  # Optional: Set the figure size for better visibility
plt.hist(zero_entries_per_hitpattern, bins=253, color='blue', edgecolor='black')  # Adjust 'bins' as needed
plt.title('')
plt.xlabel('# of PMTs with no signal')
plt.ylabel('# of events')
plt.grid(True)
plt.show()


# ## Position reconstruction

# **Main model**

# In[30]:


# Create the neural network model
model = Sequential([
    Dense(512, activation='relu', input_shape=(PMTs_top,)),
    Dropout(0.3),   # randomly sets 30% of data to 0 to prevent overfitting
    Dense(256, activation='relu'),
    #Dropout(0.3),
    Dense(128, activation='relu'),
    #Dropout(0.3),
    Dense(2)  # Output layer for x and y coordinates
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Train the model
history = model.fit(intensity_train, mc_positions_train, 
                    epochs=100, 
                    batch_size=100, 
                    validation_data=(intensity_val, mc_positions_val))

# Evaluate the model on the test set
test_loss, test_mae = model.evaluate(intensity_test, mc_positions_test)

print(f'Test Mean Absolute Error: {test_mae}')


# **Analysis**

# In[32]:


# Plot training & validation loss values
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.show()

# Plot training & validation MAE values
plt.figure(figsize=(12, 6))
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Model Mean Absolute Error')
plt.ylabel('MAE')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.show()


# **Visualization**

# In[35]:


#visualize prediction with hitpatterns

event_idx = 42

#True Coordinates and hitpattern
test_event = intensity_test[event_idx]
true_coordinates = mc_positions_test[event_idx]

# Predict the position using the model
predicted_coordinates = model.predict(np.expand_dims(test_event, axis=0))[0]
print("True Coordinates (MC):", true_coordinates)
print("Predicted Coordinates:", predicted_coordinates)

# Extract x and y coordinates for the top PMTs
pmt_x_top = pmt_xy_top["x"]
pmt_y_top = pmt_xy_top["y"]

# Create the plot
plt.figure(figsize=(10, 8))

# Scatter plot with intensity as color
sc = plt.scatter(pmt_x_top, pmt_y_top, c=test_event, cmap='viridis', marker='o', s=800, norm=LogNorm())
plt.colorbar(sc, label='Intensity')  # Add colorbar
plt.scatter(true_coordinates[0], true_coordinates[1], c='red', marker='o', s=100, label='MC Event')
plt.scatter(predicted_coordinates[0], predicted_coordinates[1], c="black", marker="o", s=100, label="Prediction")

# Draw a circle
circle_radius = 67
circle = Circle((0, 0), circle_radius, color='k', fill=False, linewidth=1)
plt.gca().add_patch(circle)

plt.title('Top PMT Positions with Intensity')
plt.xlabel('X Coordinates')
plt.ylabel('Y Coordinates')
plt.grid(True)
plt.axis('equal')  # Ensures equal scaling on both axes
plt.legend()

plt.tight_layout()
#plt.show()

#plt.savefig("broken_posrec.png", dpi=300)


# In[38]:


# Loop over a range of indices

for event_idx in range(10):
    # True Coordinates and hitpattern
    test_event = intensity_test[event_idx]
    true_coordinates = mc_positions_test[event_idx]

    # Predict the position using the model
    predicted_coordinates = model.predict(np.expand_dims(test_event, axis=0))[0]
    print(f"Event {event_idx}:")
    print("True Coordinates (MC):", true_coordinates)
    print("Predicted Coordinates:", predicted_coordinates)

    # Extract x and y coordinates for the top PMTs
    pmt_x_top = pmt_xy_top["x"]
    pmt_y_top = pmt_xy_top["y"]

    # Create the plot for the current event
    plt.figure(figsize=(10, 8))

    # Scatter plot with intensity as color
    sc = plt.scatter(pmt_x_top, pmt_y_top, c=test_event, cmap='viridis', marker='o', s=800, norm=LogNorm())
    plt.colorbar(sc, label='Intensity')  # Add colorbar
    plt.scatter(true_coordinates[0], true_coordinates[1], c='red', marker='o', s=100, label='MC Event')
    plt.scatter(predicted_coordinates[0], predicted_coordinates[1], c="black", marker="o", s=100, label="Prediction")

    # Draw a circle
    circle_radius = 67
    circle = Circle((0, 0), circle_radius, color='k', fill=False, linewidth=1)
    plt.gca().add_patch(circle)

    plt.title(f'Top PMT Positions with Intensity (Event {event_idx})')
    plt.xlabel('X Coordinates')
    plt.ylabel('Y Coordinates')
    plt.grid(True)
    plt.axis('equal')  # Ensures equal scaling on both axes
    plt.legend()

    plt.tight_layout()
    
    # Save the plot for each event
    #plt.savefig(f"posrec_event_{event_idx}.png", dpi=300)
    
    # Display the plot (optional, comment out if not needed)
    plt.show()

    # Close the plot to avoid overlap in future plots
    plt.close()


# In[65]:


# Set up a 3x3 grid for subplots
fig, axes = plt.subplots(3, 3, figsize=(25, 20))  # 3 rows, 3 columns

for event_idx in range(9):  # Loop over 9 events
    row = event_idx // 3  # Determine the row (integer division)
    col = event_idx % 3   # Determine the column (modulus)

    ax = axes[row, col]  # Get the axis for the current event

    # True Coordinates and hitpattern
    test_event = intensity_test[event_idx]
    true_coordinates = mc_positions_test[event_idx]

    # Predict the position using the model
    predicted_coordinates = model.predict(np.expand_dims(test_event, axis=0))[0]
    #print(f"Event {event_idx}:")
    #print("True Coordinates (MC):", true_coordinates)
    #print("Predicted Coordinates:", predicted_coordinates)

    # Extract x and y coordinates for the top PMTs
    pmt_x_top = pmt_xy_top["x"]
    pmt_y_top = pmt_xy_top["y"]

    # Scatter plot with intensity as color
    sc = ax.scatter(pmt_x_top, pmt_y_top, c=test_event, cmap='viridis', marker='o', s=490, norm=LogNorm())
    ax.scatter(true_coordinates[0], true_coordinates[1], c='red', marker='o', s=50, label='MC Event')
    ax.scatter(predicted_coordinates[0], predicted_coordinates[1], c="black", marker="o", s=50, label="Prediction")

    # Draw a circle
    circle_radius = 67
    circle = Circle((0, 0), circle_radius, color='k', fill=False, linewidth=1)
    ax.add_patch(circle)

    ax.set_title(f'Event {event_idx}')
    ax.set_xlabel('X Coordinates')
    ax.set_ylabel('Y Coordinates')
    ax.grid(True)
    ax.axis('equal')  # Ensures equal scaling on both axes
    ax.legend()

# Adjust layout and show the figure
plt.tight_layout()
plt.colorbar(sc, ax=axes.ravel().tolist(), label='Intensity')  # Add colorbar to the whole figure

#plt.savefig("comparison_hitpatterns_zeroes3x3.png", dpi=300)
plt.show()


# **Dependency on signal**

# In[129]:


# Batch prediction for the entire intensity_test set at once
predicted_coordinates = model.predict(intensity_test, batch_size=32)

# Pre-allocate list to store the differences
r_differences = np.zeros(len(intensity_test))

# Compute radial distances for true and predicted coordinates in a vectorized way
r_true = np.sqrt(np.sum(mc_positions_test**2, axis=1))  # Radial distance for true coordinates
r_predicted = np.sqrt(np.sum(predicted_coordinates**2, axis=1))  # Radial distance for predicted coordinates

# Calculate the absolute difference between the true and predicted radial distances
r_differences = np.abs(r_true - r_predicted)

# Print the final list of r differences
#print("List of absolute differences in radial distances:", r_differences)


# In[101]:


plt.figure(figsize=(12, 6))
plt.plot(zero_entries_per_hitpattern_test, r_differences, "o", markersize=3)
plt.title('Relationship Between Zero Entries and Radial Differences')
plt.xlabel('Zero Entries Per Hitpattern')
plt.ylabel('True Radius - Predicted Radius')
plt.grid()
plt.axhline(y=3, color='red', linestyle='--', linewidth=1.5)
plt.show()


# **Dependency on radius**

# In[104]:


plt.figure(figsize=(12, 6))
plt.plot(r_true, r_differences, "o", markersize=3)
plt.title('Relationship Between Radial Distance and Radial Differences')
plt.xlabel('Radial distance of MC event from center')
plt.ylabel('True Radius - Predicted Radius')
plt.grid()
#plt.axhline(y=3, color='red', linestyle='--', linewidth=1.5)
plt.show()


# **Filter out events with >X zero-signals**

# In[164]:


zero_counts = np.sum(intensity_test == 0, axis=1)
indices_with_more_than_50_zeros = np.where(zero_counts > 5)[0]

# Create a mask to exclude these indices
mask = np.ones(len(r_true), dtype=bool)
mask[indices_with_more_than_50_zeros] = False

# Apply the mask to filter out the unwanted indices
r_true_filtered = r_true[mask]
r_predicted_filtered = r_predicted[mask]
r_differences_filtered = (r_true_filtered - r_predicted_filtered)
r_differences_filtered_abs = np.abs(r_true_filtered - r_predicted_filtered)

print("Hitpatterns used to test: ", len(intensity_test))
print(f"Number of events with more than 50 zero entries: {len(indices_with_more_than_50_zeros)}")
print(f"Number of events with less than 50 zero entries: {len(r_true_filtered)}")


# In[170]:


plt.figure(figsize=(12, 6))
plt.plot(r_true_filtered, r_differences_filtered_abs, "o", markersize=3)
plt.title('Relationship Between Radial Distance and Radial Differences')
plt.xlabel('Radial distance of MC event from center')
plt.ylabel('True Radius - Predicted Radius')
plt.grid()
plt.ylim(0,5)
#plt.axhline(y=3, color='red', linestyle='--', linewidth=1.5)
plt.show()


# In[171]:


# Define the threshold for radial difference
threshold = 1

# Create a boolean mask for radial differences greater than the threshold
mask_high_diff = r_differences_filtered > threshold

# Apply this mask to filter the true radii
r_true_high_diff = r_true_filtered[mask_high_diff]

# Plot the histogram
plt.figure(figsize=(12, 6))
plt.hist(r_true_high_diff, bins=500)#, edgecolor='black', alpha=0.7)
plt.yscale('log')
plt.title('Radius distribution of large error(>1.5cm) events')
plt.xlabel('True Radius [cm]')
plt.ylabel('')
plt.grid(True, which="both", ls="--")
plt.axvline(x=60, color='red', linestyle='--', linewidth=1.5)

plt.tight_layout()
plt.show()


# In[173]:


# Define the threshold for r_true and r_diff
r_true_threshold = 60
r_diff_threshold = 1

# Create a mask for events where r_true is less than the threshold
mask_r_true = r_true_filtered < r_true_threshold
mask_r_true_outside = r_true_filtered > r_true_threshold

# Filter r_true and r_diff based on the mask
r_true_filtered_by_r_true = r_true_filtered[mask_r_true]
r_diff_filtered_by_r_true = r_differences_filtered[mask_r_true]

r_true_filtered_by_r_true_outside = r_true_filtered[mask_r_true_outside]
r_diff_filtered_by_r_true_outside = r_differences_filtered[mask_r_true_outside]

# Create a mask for events where r_diff is less than the threshold
mask_r_diff = r_diff_filtered_by_r_true < r_diff_threshold
mask_r_diff_outside = r_diff_filtered_by_r_true_outside < r_diff_threshold

# Calculate the fraction of events where r_diff is less than the threshold
fraction_r_diff_smaller_than_threshold = np.mean(mask_r_diff)
fraction_r_diff_smaller_than_threshold_outside = np.mean(mask_r_diff_outside)


print(f"Fraction of events with r_true < {r_true_threshold} and r_diff < {r_diff_threshold}: {fraction_r_diff_smaller_than_threshold:.4f}")
print(f"Fraction of events with r_true > {r_true_threshold} and r_diff < {r_diff_threshold}: {fraction_r_diff_smaller_than_threshold_outside:.4f}")


# In[ ]:




