#!/usr/bin/env python
# coding: utf-8

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


# In[ ]:





# # Hitpatterns

# **Data**

# In[23]:


PATH = "/data/storage/posrec_hitpatterns"


# In[4]:


os.listdir(PATH)


# In[312]:


instructions = pd.read_csv(f"{PATH}/fuse_hitpatterns_run_00100_detector_instructions.csv")


# In[313]:


instructions.head()


# In[314]:


instructions.tail()


# In[322]:


hf = h5py.File(f"{PATH}/fuse_hitpatterns_run_00100.hdf5", "r")


# In[323]:


hf.keys()


# In[324]:


type(hf["hitpatterns"])


# In[325]:


hf["mc_electrons"]


# In[326]:


mc_electrons = np.array(hf["mc_electrons"][:])
print(mc_electrons)


# In[327]:


# Plot the histogram
plt.figure(figsize=(10, 6))  # Optional: Set the figure size for better visibility
plt.hist(mc_electrons, bins=2000, color='blue', edgecolor='black')  # Adjust 'bins' as needed
plt.title('hitpatterns_run_00103')
plt.xlabel('# of MC electrons')
plt.ylabel('# of events')
plt.grid(True)
plt.show()

#plt.savefig("hist_mc_electrons_run00100.png", dpi=300)


# In[328]:


# Create a Boolean mask where condition is true
mask = mc_electrons < 50

# Count the number of True values in the mask
count_smaller_than_200 = np.sum(mask)

print(f'Number of entries smaller than 200: {count_smaller_than_200}')


# In[329]:


hf["mc_x"].shape


# In[330]:


mc_x = np.array(hf["mc_x"][:])
print(mc_x)
print(len(mc_x))


# In[26]:


hitpatterns = np.array(hf["hitpatterns"][:])


# In[16]:


hitpatterns[:].shape


# In[17]:


print(len(hitpatterns[0]))
#print(hitpatterns[0])


# In[27]:


mc_x = np.array(hf["mc_x"][:]) 
mc_y = np.array(hf["mc_y"][:])
intensity = hitpatterns[0]/max(hitpatterns[0])

print(len(mc_x))
print(len(mc_y))
print(mc_x[0])
print(mc_y[0])
print(len(intensity))
print(sum(intensity))
print(max(intensity))
print((intensity))


# In[104]:


import numpy as np

# Load the .npy file
file_path = '3ch_mux.npy'
ch3_mux = np.load(file_path)

# Print the current array (optional)
print("Original Array:")
print(ch3_mux)

# Modify the array
# For example, add a new group [250, 251, 252]
new_group = [126,-1,-1]
ch3_mux = np.append(ch3_mux, [new_group], axis=0)

# Print the modified array (optional)
print("Modified Array:")
print(ch3_mux)

# Save the modified array to a new .npy file
new_file_path = '3ch_mux_mod_single_pmt.npy'
#np.save(new_file_path, ch3_mux)

print(f"Modified array saved to {new_file_path}")


# In[106]:


import numpy as np

# Load the .npy file
file_path = '7ch_mux.npy'
ch7_mux = np.load(file_path)

# Print the current array (optional)
print("Original Array:")
print(ch7_mux)

# Define multiple new groups
new_groups = [
    [0, -1, -1, -1, -1, -1, -1],
    [26, -1, -1, -1, -1, -1, -1],
    [167, -1, -1, -1, -1, -1, -1],
    [252, -1, -1, -1, -1, -1, -1],
    [226, -1, -1, -1, -1, -1, -1],
    [85, -1, -1, -1, -1, -1, -1]
]

# Convert the new groups to a numpy array
new_groups_array = np.array(new_groups)

# Append the new groups to the existing array
ch7_mux = np.append(ch7_mux, new_groups_array, axis=0)

# Print the modified array (optional)
print("Modified Array:")
print(ch7_mux)

# Save the modified array to a new .npy file
new_file_path = '7ch_mux_mod_single_pmt.npy'
#np.save(new_file_path, ch7_mux)

print(f"Modified array saved to {new_file_path}")


# In[102]:


# Load the .npy file
file_path = '7ch_mux_mod_single_pmt.npy'
ch7_mux = np.load(file_path)


# In[108]:


print(ch7_mux)


# In[160]:


import numpy as np

# Load the .npz file
data = np.load('old_stuff/muxing_patterns.npz')

# List all arrays in the .npz file
print("Keys in the .npz file:", data.files)

# Access and print each array
for key in data.files:
    print(f"{key}: {data[key]}")

# If you want to use a specific array
# example_array = data['array_name']  # Replace 'array_name' with the actual key


# **Create hexagonal pattern**

# In[28]:


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


# **Test**

# In[20]:


print(pmt_xy.shape)
print(len(pmt_xy_top["x"]))
#print(pmt_xy["x"])
#print(pmt_xy)


# **Plot Hexagonal Patterns**

# In[21]:


# Separate the positions for the top and bottom arrays
pmt_xy_top = pmt_xy[:PMTs_top]
pmt_xy_bottom = pmt_xy[PMTs_top:]

# Extract x and y coordinates for top and bottom arrays
pmt_x_top = pmt_xy_top['x']
pmt_y_top = pmt_xy_top['y']
pmt_x_bottom = pmt_xy_bottom['x']
pmt_y_bottom = pmt_xy_bottom['y']

# Create the plot
plt.figure(figsize=(16, 8))

plt.subplot(1, 2, 1)   # top array
plt.scatter(pmt_x_top, pmt_y_top, c='blue', marker='o')
plt.title('Top PMT Positions', fontsize=20)
plt.xlabel('X', fontsize=14)
plt.ylabel('Y', fontsize=14)
plt.grid(True)
plt.axis('equal')

plt.subplot(1, 2, 2)   # bottom array
plt.scatter(pmt_x_bottom, pmt_y_bottom, c='red', marker='o')
plt.title('Bottom PMT Positions', fontsize=20)
plt.xlabel('X', fontsize=14)
plt.ylabel('Y', fontsize=14)
plt.grid(True)
plt.axis('equal')

plt.tight_layout()
#plt.show()


# In[29]:


# PMT Numbering

# Extract x and y coordinates
pmt_x_top = pmt_xy_top["x"]
pmt_y_top = pmt_xy_top["y"]

# Create the plot
plt.figure(figsize=(10, 10))
plt.scatter(pmt_x_top, pmt_y_top, c='lightblue', marker='o', s=900)

# Annotate each PMT with its number
for i, (x, y) in enumerate(zip(pmt_x_top, pmt_y_top)):
    plt.text(x, y, str(i), fontsize=12, ha='center', va='center', color='black')

plt.title('PMT Positions')
plt.xlabel('X Coordinates')
plt.ylabel('Y Coordinates')
plt.grid(True)
plt.axis('equal')  # Ensures equal scaling on both axes
plt.show()

#plt.savefig("PMT_top_labelled.png", dpi=300)


# **Plot hitpattern**

# In[361]:


hf = h5py.File(f"{PATH}/fuse_hitpatterns_run_00162.hdf5", "r")
hitpatterns = np.array(hf["hitpatterns"][:])


# In[362]:


event_idx = 0

mc_x = np.array(hf["mc_x"][:]) 
mc_y = np.array(hf["mc_y"][:])
intensity = hitpatterns[event_idx]

# Extract x and y coordinates for the top PMTs
pmt_x_top = pmt_xy_top["x"]
pmt_y_top = pmt_xy_top["y"]

# Create the plot
plt.figure(figsize=(10, 8))

# Scatter plot with intensity as color
sc = plt.scatter(pmt_x_top, pmt_y_top, c=intensity, cmap='viridis', marker='o', s=800, norm=LogNorm())
plt.colorbar(sc, label='Intensity')  # Add colorbar
plt.scatter(mc_x[event_idx], mc_y[event_idx], c='red', marker='o', s=100, label='MC Point')

# Draw a circle
circle_radius = 67
circle = Circle((0, 0), circle_radius, color='k', fill=False, linewidth=1)
plt.gca().add_patch(circle)

plt.title('Top PMT Positions with Intensity')
plt.xlabel('X Coordinates')
plt.ylabel('Y Coordinates')
plt.grid(True)
plt.axis('equal')  # Ensures equal scaling on both axes

plt.tight_layout()
plt.show()

#plt.savefig("PMT_top_hitpattern.png", dpi=300)


# In[ ]:


mc_x = np.array(hf["mc_x"][:]) 
mc_y = np.array(hf["mc_y"][:])

# Extract x and y coordinates for the top PMTs
pmt_x_top = pmt_xy_top["x"]
pmt_y_top = pmt_xy_top["y"]

# Loop over the first 5 hit patterns
for i in range(5):
    intensity = hitpatterns[i]

    # Create the plot
    plt.figure(figsize=(10, 8))

    # Scatter plot with intensity as color
    sc = plt.scatter(pmt_x_top, pmt_y_top, c=intensity, cmap='viridis', marker='o', s=800, norm=LogNorm())
    plt.colorbar(sc, label='Intensity')  # Add colorbar
    plt.scatter(mc_x[i], mc_y[i], c='red', marker='o', s=100, label='MC Point')

    # Draw a circle
    circle_radius = 67
    circle = Circle((0, 0), circle_radius, color='k', fill=False, linewidth=1)
    plt.gca().add_patch(circle)

    plt.title(f'Top PMT Positions with Intensity (Hit Pattern {i+1})')
    plt.xlabel('X Coordinates')
    plt.ylabel('Y Coordinates')
    plt.grid(True)
    plt.axis('equal')  # Ensures equal scaling on both axes

    plt.tight_layout()
    plt.show()


# ## Grouping

# In[99]:


ch3_mux = np.load('3ch_mux.npy')
ch7_mux = np.load('7ch_mux.npy')
ch3_mux_mod = np.load('3ch_mux_mod_single_pmt.npy')
ch7_mux_mod = np.load('7ch_mux_mod_single_pmt.npy')


# **Test**

# In[83]:


print("Shape:", ch3_mux.shape)
print("Data type:", ch3_mux.dtype)
print(ch3_mux[:10])  # Print the first 10 elements
#print(ch3_mux)


# In[26]:


print("Shape:", ch7_mux.shape)
print("Data type:", ch7_mux.dtype)
#print(ch7_mux[:10])  # Print the first 10 elements


# **Find neighbouring PMTs & make groups**

# In[108]:


adjacent = np.load('pmt_adjacent.npy')   #GET FILE
"""
Shape: (253, 6)
Array containing all 6 neighbouring PMTs for each of the 253 top PMTs
example: [[ 1  7  8 -1 -1 -1], [ 0  2  8  9 -1 -1]] first two PMTs
entry [-1] if neighbouring PMT would be outside of setup
"""

# find PMTs within certain radius of event/thing/other pmt/group
def pmts_adjacent_to_thing(thing, radius, exclude_self=True):
    """
    Which PMTs are adjacent to the given thing?
    """
    adj = set(thing)   # set containing PMTs from "thing"
    for _ in range(radius):   # adjacency "hops", 1 means direct neighbours
        new_adj = set()   # empty set to initialize
        for ch in adj:   # loop over PMTs in set adj
            new_adj |= set(adjacent[ch])   # add PMT to new_adj if not already there
        adj |= new_adj-set([-1])   # include new_adj in adj, exclude placeholders [-1]
    if exclude_self:
        adj -= set(thing)   # remove original PMTs in thing from adj
    return adj

def groups_adjacent_to_pmt(pmt, groups, radius):
    """
    What groups are adjacent to this PMT?
    :param groups: current groups of assigned PMTs
    :param pmt: the PMT number you're looking to assign
    :param radius: how many PMTs away to consider "adjacent"
    """
    adj_channels = pmts_adjacent_to_thing([pmt], radius)   # call function defined previously, returns set of PMTs within specified radius
    # find what groups these PMTs belong to
    adj_groups = set()   # empty set to initialize
    for ch in adj_channels:   #loop through adjacent channels
        for i,group in enumerate(groups):
            if ch in group: # this channel has already been assigned
                adj_groups.add(i)
    return adj_groups   # return set of indices of groups adjacent to given PMT


def groups_adj_to_group(group, groups, radius):   #basically same as above, this time for groups
    """
    Which groups are adjacent to the given group?
    :returns: list of indexes of groups
    """
    adj_to_this_group = pmts_adjacent_to_thing(group, radius)   # call function defined previously
    results = set()   # empty set to initialize
    for pmt in adj_to_this_group:
        for i,other_group in enumerate(groups):
            if pmt in other_group:
                results.add(i)
    return results

def make_group(overlap_range=None, to_assign=None, max_group_size=None, name=None):
    # initialize groups
    groups = []   
    group_by='set'   # indicate grouping method
    # loop through PMTs to assign
    for j,phys_ch in enumerate(to_assign):
        #Determine adjacent groups
        if group_by == 'set':   # find which groups are adajcent to PMT
            adj_groups = groups_adjacent_to_pmt(phys_ch, groups, overlap_range)
        elif group_by == 'dist':
            pass
        else:
            raise ValueError('Invalid value for group_by')
        # find potential groups
        potential_groups = []   # groups current PMT could be added to
        for i,group in enumerate(groups):
            if max_group_size and len(group) >= max_group_size:
                continue   # skip group if already full
            if i in adj_groups:
                continue   # skip if group_i is adjacent to this PMT
            g = groups_adj_to_group(group, groups, 1)
            if adj_groups & g:
                continue   # skip if group_i is already adjacent to another group
            potential_groups.append(i)
        #Assign PMT to group    
        if len(potential_groups):   # multiple possible groups, pick the smallest
            min_len, min_i = 253, -1
            for group_i in potential_groups:
                # add this channel to the smallest existing group
                if (l := len(groups[group_i])) < min_len:
                    min_len = l
                    min_i = group_i
            groups[min_i].append(phys_ch)
        else: # no potential groups, make a new one
            groups.append([phys_ch])
    # Sort and format groups        
    for g in groups:
        g.sort()   # sort PMTs within group in ascending order
    groups.sort(key=lambda g:g[0])   # sort groups based on first element of each group
    g = np.zeros((len(groups), max(len(g) for g in groups)), dtype=np.int16)   # new empty array with dimensions "number of groups" x "length of largest group"
    for i,row in enumerate(groups):
        l = len(row)   # length of current group
        g[i][:l] = row   # fills first l elements of i-th row of g with elements of current group 
        g[i][l:] = -1   # fills remaining elements with -1
    if name is not None:
        return name, g
    return g


# In[28]:


print("Shape:", adjacent.shape)
print("Data type:", adjacent.dtype)
#print(adjacent[:10])  # Print the first 10 elements


# **Show groups: 3 and 7**

# In[126]:


results = {}
results['flower_7'] = np.load('7ch_mux.npy')
results['flower_3'] = np.load('3ch_mux.npy')
results['flower_7_mod'] = np.load('7ch_mux_mod_single_pmt.npy')
results['flower_3_mod'] = np.load('3ch_mux_mod_single_pmt.npy')


# In[122]:


#original

# Example function to generate pastel colors
def generate_pastel_colors(num_colors):
    pastel_colors = []
    for _ in range(num_colors):
        base_color = [random.random() for _ in range(3)]  # Random base color
        pastel_color = [(1 + c) / 2 for c in base_color]  # Blend with white
        pastel_colors.append(mcolors.to_hex(pastel_color))
    return pastel_colors

# Plotting function using pastel colors and ignoring -1 entries
def plot_pmt_groups(ax, pmt_positions, groups, title):
    num_groups = len(groups)
    pastel_colors = generate_pastel_colors(num_groups)
    
    # Create a dictionary to map PMT index to group number
    pmt_to_group = {}
    for group_number, group in enumerate(groups):
        for pmt in group:
            if pmt >= 0 and pmt < len(pmt_positions):  # Ensure valid PMT index
                pmt_to_group[pmt] = group_number + 1  # Group numbers start from 1

    # Plot the PMTs in groups
    for i, group in enumerate(groups):
        # Filter out -1 entries
        pmt_x = [pmt_positions[pmt][0] for pmt in group if pmt >= 0 and pmt < len(pmt_positions)]
        pmt_y = [pmt_positions[pmt][1] for pmt in group if pmt >= 0 and pmt < len(pmt_positions)]
        ax.scatter(pmt_x, pmt_y, c=[pastel_colors[i]], edgecolor='black', marker='o', label=f'Group {i+1}', s=1800)
        
        # Annotate each PMT with its group number, ignoring -1
        for pmt in group:
            if pmt >= 0 and pmt < len(pmt_positions):  # Ensure valid PMT index
                group_number = pmt_to_group.get(pmt, '?')  # Use '?' if the PMT isn't in the dictionary
                ax.text(pmt_positions[pmt][0], pmt_positions[pmt][1], str(group_number),
                         fontsize=18, ha='center', va='center', color='black')

    ax.set_title(title, fontsize=26)
    ax.set_xlabel('X', fontsize=20)
    ax.set_ylabel('Y', fontsize=20)
    ax.grid(True)
    ax.axis('equal')

# Include all PMTs in groups and ignore -1 entries
def include_all_pmts(groups, total_pmts):
    all_pmts = set(range(total_pmts))
    current_pmts = set(pmt for group in groups for pmt in group if pmt >= 0)  # Exclude -1
    missing_pmts = all_pmts - current_pmts
    # Append missing PMTs to their own groups
    for missing_pmt in missing_pmts:
        groups.append([missing_pmt])
    return groups, missing_pmts

# Get total number of PMTs
total_pmts = len(pmt_xy_top)

# Process groups
groups_with_all_pmts3, missing_pmts3 = include_all_pmts(results['flower_3'].tolist(), total_pmts)
groups_with_all_pmts7, missing_pmts7 = include_all_pmts(results['flower_7'].tolist(), total_pmts)

# Create a single figure with two subplots
fig, axs = plt.subplots(1, 2, figsize=(24, 12))

# Plot the groups
plot_pmt_groups(axs[0], pmt_xy_top, groups_with_all_pmts3, 'Top PMT Positions (3ch_mux)')
plot_pmt_groups(axs[1], pmt_xy_top, groups_with_all_pmts7, 'Top PMT Positions (7ch_mux)')

plt.tight_layout()
plt.show()


# In[126]:


#original, single pmts highlighted

# Example function to generate pastel colors
def generate_pastel_colors(num_colors):
    pastel_colors = []
    for _ in range(num_colors):
        base_color = [random.random() for _ in range(3)]  # Random base color
        pastel_color = [(1 + c) / 2 for c in base_color]  # Blend with white
        pastel_colors.append(mcolors.to_hex(pastel_color))
    return pastel_colors

# Plotting function using pastel colors and highlighting missing PMTs
def plot_pmt_groups(ax, pmt_positions, groups, missing_pmts, title):
    num_groups = len(groups)
    pastel_colors = generate_pastel_colors(num_groups)
    
    # Create a dictionary to map PMT index to group number
    pmt_to_group = {}
    for group_number, group in enumerate(groups):
        for pmt in group:
            if pmt >= 0 and pmt < len(pmt_positions):  # Ensure valid PMT index
                pmt_to_group[pmt] = group_number + 1  # Group numbers start from 1

    # Plot the PMTs in groups
    for i, group in enumerate(groups):
        # Filter out -1 entries
        pmt_x = [pmt_positions[pmt][0] for pmt in group if pmt >= 0 and pmt < len(pmt_positions)]
        pmt_y = [pmt_positions[pmt][1] for pmt in group if pmt >= 0 and pmt < len(pmt_positions)]
        ax.scatter(pmt_x, pmt_y, c=[pastel_colors[i]], edgecolor='black', marker='o', label=f'Group {i+1}', s=1800)
        
        # Annotate each PMT with its group number, ignoring -1
        for pmt in group:
            if pmt >= 0 and pmt < len(pmt_positions):  # Ensure valid PMT index
                group_number = pmt_to_group.get(pmt, '?')  # Use '?' if the PMT isn't in the dictionary
                ax.text(pmt_positions[pmt][0], pmt_positions[pmt][1], str(group_number),
                         fontsize=18, ha='center', va='center', color='black')

    # Highlight missing PMTs in bright red
    missing_x = [pmt_positions[pmt][0] for pmt in missing_pmts if pmt < len(pmt_positions)]
    missing_y = [pmt_positions[pmt][1] for pmt in missing_pmts if pmt < len(pmt_positions)]
    ax.scatter(missing_x, missing_y, c='red', edgecolor='black', marker='o', s=1800, label='Missing PMTs')

    ax.set_title(title, fontsize=26)
    ax.set_xlabel('X', fontsize=20)
    ax.set_ylabel('Y', fontsize=20)
    ax.grid(True)
    ax.axis('equal')

# Include all PMTs in groups and ignore -1 entries
def include_all_pmts(groups, total_pmts):
    all_pmts = set(range(total_pmts))
    current_pmts = set(pmt for group in groups for pmt in group if pmt >= 0)  # Exclude -1
    missing_pmts = all_pmts - current_pmts
    # Append missing PMTs to their own groups
    for missing_pmt in missing_pmts:
        groups.append([missing_pmt])
    return groups, missing_pmts

# Get total number of PMTs
total_pmts = len(pmt_xy_top)

# Process groups
groups_with_all_pmts3, missing_pmts3 = include_all_pmts(results['flower_3'].tolist(), total_pmts)
groups_with_all_pmts7, missing_pmts7 = include_all_pmts(results['flower_7'].tolist(), total_pmts)

# Create a single figure with two subplots
fig, axs = plt.subplots(1, 2, figsize=(24, 12))

# Plot the groups and highlight missing PMTs in each subplot
plot_pmt_groups(axs[0], pmt_xy_top, groups_with_all_pmts3, missing_pmts3, 'Top PMT Positions (3ch_mux)')
plot_pmt_groups(axs[1], pmt_xy_top, groups_with_all_pmts7, missing_pmts7, 'Top PMT Positions (7ch_mux)')

plt.tight_layout()
plt.show()

#plt.savefig("PMT_top_3_7_grouping_fixed.png", dpi=300)


# In[102]:


#modded npy file

# Example function to generate pastel colors
def generate_pastel_colors(num_colors):
    pastel_colors = []
    for _ in range(num_colors):
        base_color = [random.random() for _ in range(3)]  # Random base color
        pastel_color = [(1 + c) / 2 for c in base_color]  # Blend with white
        pastel_colors.append(mcolors.to_hex(pastel_color))
    return pastel_colors

# Plotting function using pastel colors and highlighting missing PMTs
def plot_pmt_groups(ax, pmt_positions, groups, missing_pmts, title):
    num_groups = len(groups)
    pastel_colors = generate_pastel_colors(num_groups)
    
    # Create a dictionary to map PMT index to group number
    pmt_to_group = {}
    for group_number, group in enumerate(groups):
        for pmt in group:
            if pmt < len(pmt_positions):
                pmt_to_group[pmt] = group_number + 1  # Group numbers start from 1

    # Plot the PMTs in groups
    for i, group in enumerate(groups):
        pmt_x = [pmt_positions[pmt][0] for pmt in group if pmt >= 0 and pmt < len(pmt_positions)]
        pmt_y = [pmt_positions[pmt][1] for pmt in group if pmt >= 0 and pmt < len(pmt_positions)]
        ax.scatter(pmt_x, pmt_y, c=[pastel_colors[i]], edgecolor='black', marker='o', label=f'Group {i+1}', s=1800)
        
        # Annotate each PMT with its group number
        for pmt in group:
            if pmt < len(pmt_positions):  # Ensure the PMT index is valid
                group_number = pmt_to_group.get(pmt, '?')  # Use '?' if the PMT isn't in the dictionary
                ax.text(pmt_positions[pmt][0], pmt_positions[pmt][1], str(group_number),
                         fontsize=18, ha='center', va='center', color='black')

    # Highlight missing PMTs in bright red
    if missing_pmts:
        missing_x = [pmt_positions[pmt][0] for pmt in missing_pmts if pmt < len(pmt_positions)]
        missing_y = [pmt_positions[pmt][1] for pmt in missing_pmts if pmt < len(pmt_positions)]
        ax.scatter(missing_x, missing_y, c='red', edgecolor='black', marker='o', s=1800, label='Missing PMTs')


    ax.set_title(title, fontsize=26)
    ax.set_xlabel('X', fontsize=20)
    ax.set_ylabel('Y', fontsize=20)
    ax.grid(True)
    ax.axis('equal')

# Include all PMTs in groups and identify missing PMTs
def include_all_pmts(groups, total_pmts):
    all_pmts = set(range(total_pmts))
    current_pmts = set(pmt for group in groups for pmt in group if pmt >= 0)  # Exclude -1
    missing_pmts = all_pmts - current_pmts
    if missing_pmts:
        # Assign each missing PMT to its own group for visualization
        for missing_pmt in missing_pmts:
            groups.append([missing_pmt])
    return groups, missing_pmts

total_pmts = len(pmt_xy_top)
groups_with_all_pmts3, missing_pmts3 = include_all_pmts(results['flower_3_mod'].tolist(), total_pmts)
groups_with_all_pmts7, missing_pmts7 = include_all_pmts(results['flower_7_mod'].tolist(), total_pmts)

# Create a single figure with two subplots
fig, axs = plt.subplots(1, 2, figsize=(24, 12))

# Plot the groups and highlight missing PMTs in each subplot
plot_pmt_groups(axs[0], pmt_xy_top, groups_with_all_pmts3, missing_pmts3, 'Top PMT Positions (3ch_mux)')
plot_pmt_groups(axs[1], pmt_xy_top, groups_with_all_pmts7, missing_pmts7, 'Top PMT Positions (7ch_mux)')

plt.tight_layout()
plt.show()


# In[31]:


def find_pmt_groups(pmt, groups):
    """
    Find the groups that a given PMT is part of.
    
    Parameters:
    pmt (int): The PMT number to find.
    groups (list of lists): The list of PMT groups.
    
    Returns:
    list: A list of group indices that the PMT is part of.
    """
    pmt_groups = []
    for group_index, group in enumerate(groups):
        if pmt in group:
            pmt_groups.append(group_index+1)
    return pmt_groups

# Example usage
pmt_to_find = 252
groups_with_all_pmts3 = results['flower_3'].tolist()
groups_with_all_pmts7 = results['flower_7'].tolist()

pmt_groups_3 = find_pmt_groups(pmt_to_find, groups_with_all_pmts3)
pmt_groups_7 = find_pmt_groups(pmt_to_find, groups_with_all_pmts7)

print(f"PMT {pmt_to_find} is part of the following groups in 3ch_mux: {pmt_groups_3}")
print(f"PMT {pmt_to_find} is part of the following groups in 7ch_mux: {pmt_groups_7}")


# In[32]:


print(groups_with_all_pmts7[36])


# In[33]:


# Find and print the groups a specific PMT is part of
pmt_to_find = 252
pmt_groups_3 = find_pmt_groups(pmt_to_find, groups_with_all_pmts3)
pmt_groups_7 = find_pmt_groups(pmt_to_find, groups_with_all_pmts7)

print(f"PMT {pmt_to_find} is part of the following groups in 3ch_mux: {pmt_groups_3}")
print(f"PMT {pmt_to_find} is part of the following groups in 7ch_mux: {pmt_groups_7}")


# In[42]:


import random
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

# Example function to generate pastel colors
def generate_pastel_colors(num_colors):
    pastel_colors = []
    for _ in range(num_colors):
        base_color = [random.random() for _ in range(3)]  # Random base color
        pastel_color = [(1 + c) / 2 for c in base_color]  # Blend with white
        pastel_colors.append(mcolors.to_hex(pastel_color))
    return pastel_colors

# Plotting function using pastel colors and highlighting missing PMTs
def plot_pmt_groups(ax, pmt_positions, groups, missing_pmts, title):
    num_groups = len(groups)
    pastel_colors = generate_pastel_colors(num_groups)
    
    # Create a dictionary to map PMT index to group number
    pmt_to_group = {}
    for group_number, group in enumerate(groups):
        for pmt in group:
            if pmt < len(pmt_positions):
                pmt_to_group[pmt] = group_number + 1  # Group numbers start from 1
                
    print("Group 37:", groups[36])
    

    # Plot the PMTs in groups
    for i, group in enumerate(groups):
        pmt_x = [pmt_positions[pmt][0] for pmt in group if pmt < len(pmt_positions)]
        pmt_y = [pmt_positions[pmt][1] for pmt in group if pmt < len(pmt_positions)]
        ax.scatter(pmt_x, pmt_y, c=[pastel_colors[i]], edgecolor='black', marker='o', label=f'Group {i+1}', s=1800)
        
        # Annotate each PMT with its group number
        for pmt in group:
            if pmt < len(pmt_positions):  # Ensure the PMT index is valid
                group_number = pmt_to_group.get(pmt, '?')  # Use '?' if the PMT isn't in the dictionary
                ax.text(pmt_positions[pmt][0], pmt_positions[pmt][1], str(group_number),
                         fontsize=18, ha='center', va='center', color='black')

    # Highlight missing PMTs in bright red
    if missing_pmts:
        missing_x = [pmt_positions[pmt][0] for pmt in missing_pmts if pmt < len(pmt_positions)]
        missing_y = [pmt_positions[pmt][1] for pmt in missing_pmts if pmt < len(pmt_positions)]
        ax.scatter(missing_x, missing_y, c='red', edgecolor='black', marker='o', s=1800, label='Missing PMTs')

    ax.set_title(title, fontsize=26)
    ax.set_xlabel('X', fontsize=20)
    ax.set_ylabel('Y', fontsize=20)
    ax.grid(True)
    ax.axis('equal')

# Include all PMTs in groups and identify missing PMTs
def include_all_pmts(groups, total_pmts):
    all_pmts = set(range(total_pmts))
    current_pmts = set(pmt for group in groups for pmt in group)
    missing_pmts = all_pmts - current_pmts
    if missing_pmts:
        # Assign each missing PMT to its own group for visualization
        for missing_pmt in missing_pmts:
            groups.append([missing_pmt])
    return groups, missing_pmts

# Function to find the groups a specific PMT is part of
def find_pmt_groups(pmt, groups):
    pmt_groups = []
    for group_index, group in enumerate(groups):
        if pmt in group:
            pmt_groups.append(group_index + 1)  # Group numbers start from 1
    return pmt_groups

# Load group information
results = {}
results['flower_3'] = np.load('3ch_mux.npy')
results['flower_7'] = np.load('7ch_mux.npy')

# Extract x and y coordinates for the top array
pmt_x_top = pmt_xy_top['x']
pmt_y_top = pmt_xy_top['y']

total_pmts = len(pmt_xy_top)
groups_with_all_pmts3, missing_pmts3 = include_all_pmts(results['flower_3'].tolist(), total_pmts)
groups_with_all_pmts7, missing_pmts7 = include_all_pmts(results['flower_7'].tolist(), total_pmts)

# Create a single figure with two subplots
fig, axs = plt.subplots(1, 2, figsize=(24, 12))

# Plot the groups and highlight missing PMTs in each subplot
plot_pmt_groups(axs[0], pmt_xy_top, groups_with_all_pmts3, missing_pmts3, 'Top PMT Positions (3ch_mux)')
plot_pmt_groups(axs[1], pmt_xy_top, groups_with_all_pmts7, missing_pmts7, 'Top PMT Positions (7ch_mux)')

plt.tight_layout()
plt.show()

# Find and print the groups a specific PMT is part of
pmt_to_find = 252
pmt_groups_3 = find_pmt_groups(pmt_to_find, groups_with_all_pmts3)
pmt_groups_7 = find_pmt_groups(pmt_to_find, groups_with_all_pmts7)

print(f"PMT {pmt_to_find} is part of the following groups in 3ch_mux: {pmt_groups_3}")
print(f"PMT {pmt_to_find} is part of the following groups in 7ch_mux: {pmt_groups_7}")


# In[ ]:





# **Random grouping**

# In[133]:


# true random

for n in [3,5]:
    n_trials = 1
    group_size = n
    n_groups = int(np.ceil(253/group_size))
    for t in range(n_trials):
        x = np.arange(n_groups*group_size, dtype=np.int16)
        x[253:] = -1
        np.random.shuffle(x)
        results[f'rand_{n}_0_{t}'] = x.reshape((-1, group_size))


# In[127]:


# deterministic, not true random

tasks = []
for max_group_size in tqdm([3,5], desc='Group size'):
    mgs = str(max_group_size)
    for overlap_range in tqdm([1,2], desc='Overlap range', leave=False):
        rg = str(overlap_range)
        #for trial in tqdm(253 if deterministic else 100, desc='Trial', leave=False):
        for trial in tqdm(range(1), desc='Trial', leave=False):
            t = f'{trial:03d}'
            #for step_by in range(1,5):
            for step_by in [1]:
                to_assign = np.zeros(253, dtype=np.int16)
                to_assign[0] = trial
                for i in range(1,253):
                    to_assign[i] = (to_assign[i-1]+step_by)%253
                tasks.append((overlap_range, to_assign, max_group_size, f'{det}_{mgs}_{rg}_{step_by}_{t}'))
                arr = make_group(overlap_range=overlap_range, to_assign=to_assign, max_group_size=max_group_size)
                results[f'det_{mgs}_{rg}_{step_by}_{t}'] = arr


# In[137]:


results.keys()


# In[130]:


keys = list(results.keys())
det_keys = [key for key in keys if key.startswith('det')]
print(det_keys)


# In[131]:


results["det_5_2_1_000"]


# In[134]:


#random grouping

# Example function to generate pastel colors
def generate_pastel_colors(num_colors):
    pastel_colors = []
    for _ in range(num_colors):
        base_color = [random.random() for _ in range(3)]  # Random base color
        pastel_color = [(1 + c) / 2 for c in base_color]  # Blend with white
        pastel_colors.append(mcolors.to_hex(pastel_color))
    return pastel_colors

# Plotting function using pastel colors and ignoring -1 entries
def plot_pmt_groups(ax, pmt_positions, groups, title):
    num_groups = len(groups)
    pastel_colors = generate_pastel_colors(num_groups)
    
    # Create a dictionary to map PMT index to group number
    pmt_to_group = {}
    for group_number, group in enumerate(groups):
        for pmt in group:
            if pmt >= 0 and pmt < len(pmt_positions):  # Ensure valid PMT index
                pmt_to_group[pmt] = group_number + 1  # Group numbers start from 1

    # Plot the PMTs in groups
    for i, group in enumerate(groups):
        # Filter out -1 entries
        pmt_x = [pmt_positions[pmt][0] for pmt in group if pmt >= 0 and pmt < len(pmt_positions)]
        pmt_y = [pmt_positions[pmt][1] for pmt in group if pmt >= 0 and pmt < len(pmt_positions)]
        ax.scatter(pmt_x, pmt_y, c=[pastel_colors[i]], edgecolor='black', marker='o', label=f'Group {i+1}', s=1800)
        
        # Annotate each PMT with its group number, ignoring -1
        for pmt in group:
            if pmt >= 0 and pmt < len(pmt_positions):  # Ensure valid PMT index
                group_number = pmt_to_group.get(pmt, '?')  # Use '?' if the PMT isn't in the dictionary
                ax.text(pmt_positions[pmt][0], pmt_positions[pmt][1], str(group_number),
                         fontsize=18, ha='center', va='center', color='black')

    ax.set_title(title, fontsize=26)
    ax.set_xlabel('X', fontsize=20)
    ax.set_ylabel('Y', fontsize=20)
    ax.grid(True)
    ax.axis('equal')

# Include all PMTs in groups and ignore -1 entries
def include_all_pmts(groups, total_pmts):
    all_pmts = set(range(total_pmts))
    current_pmts = set(pmt for group in groups for pmt in group if pmt >= 0)  # Exclude -1
    missing_pmts = all_pmts - current_pmts
    # Append missing PMTs to their own groups
    for missing_pmt in missing_pmts:
        groups.append([missing_pmt])
    return groups, missing_pmts

# Get total number of PMTs
total_pmts = len(pmt_xy_top)

# Process groups
groups_with_all_pmts3, missing_pmts3 = include_all_pmts(results['rand_3_0_0'].tolist(), total_pmts)
groups_with_all_pmts5, missing_pmts5 = include_all_pmts(results['rand_5_0_0'].tolist(), total_pmts)

# Create a single figure with two subplots
fig, axs = plt.subplots(1, 2, figsize=(24, 12))

# Plot the groups
plot_pmt_groups(axs[0], pmt_xy_top, groups_with_all_pmts3, 'PMT Grouping (rand_3ch_mux)')
plot_pmt_groups(axs[1], pmt_xy_top, groups_with_all_pmts5, 'PMT Grouping (rand_7ch_mux)')

plt.tight_layout()
plt.show()


# In[135]:


# Process groups
groups_with_all_pmts3, missing_pmts3 = include_all_pmts(results['det_3_1_1_000'].tolist(), total_pmts)
groups_with_all_pmts5, missing_pmts5 = include_all_pmts(results['det_3_2_1_000'].tolist(), total_pmts)

# Create a single figure with two subplots
fig, axs = plt.subplots(1, 2, figsize=(24, 12))

# Plot the groups
plot_pmt_groups(axs[0], pmt_xy_top, groups_with_all_pmts3, 'PMT Grouping (det_3ch_mux_1exc)')
plot_pmt_groups(axs[1], pmt_xy_top, groups_with_all_pmts5, 'PMT Grouping (det_3ch_mux_2exc)')

plt.tight_layout()
plt.show()


# In[136]:


# Process groups
groups_with_all_pmts3, missing_pmts3 = include_all_pmts(results['det_5_1_1_000'].tolist(), total_pmts)
groups_with_all_pmts5, missing_pmts5 = include_all_pmts(results['det_5_2_1_000'].tolist(), total_pmts)

# Create a single figure with two subplots
fig, axs = plt.subplots(1, 2, figsize=(24, 12))

# Plot the groups
plot_pmt_groups(axs[0], pmt_xy_top, groups_with_all_pmts3, 'PMT Grouping (det_5ch_mux_1exc)')
plot_pmt_groups(axs[1], pmt_xy_top, groups_with_all_pmts5, 'PMT Grouping (det_5ch_mux_2exc)')

plt.tight_layout()
plt.show()


# # Position reconstruction: Ungrouped

# **Main model**

# In[369]:


# Constants
PATH = "/data/storage/posrec_hitpatterns"  
FILE_NAME = "fuse_hitpatterns_run_00103.hdf5"
TOTAL_EVENTS = 9995
PMTs_top = 253

# Load data from HDF5 file
hf = h5py.File(f"{PATH}/{FILE_NAME}", "r")
mc_x = np.array(hf["mc_x"][:])
mc_y = np.array(hf["mc_y"][:])
hitpatterns = np.array(hf["hitpatterns"][:])
hitpatterns_max = np.max(hitpatterns, axis=1, keepdims=True)
intensity = hitpatterns / hitpatterns_max


# Combine mc_x and mc_y into one array for the target positions
mc_positions = np.vstack((mc_x, mc_y)).T

# Split the data into training, validation, and test sets
intensity_train, intensity_temp, mc_positions_train, mc_positions_temp = train_test_split(intensity, mc_positions, test_size=0.3, random_state=42)
intensity_val, intensity_test, mc_positions_val, mc_positions_test = train_test_split(intensity_temp, mc_positions_temp, test_size=0.5, random_state=42)


# Create the neural network model
model = Sequential([
    Dense(512, activation='relu', input_shape=(PMTs_top,)),
    #Dropout(0.3),   # randomly sets 30% of data to 0 to prevent overfitting
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
                    epochs=50, 
                    batch_size=100, 
                    validation_data=(intensity_val, mc_positions_val))

# Evaluate the model on the test set
test_loss, test_mae = model.evaluate(intensity_test, mc_positions_test)

print(f'Test Mean Absolute Error: {test_mae}')

# Save the model
#model.save('pmt_position_reconstruction_model.h5')


# In[333]:


PATH = "/data/storage/posrec_hitpatterns"  
FILE_NAME = "fuse_hitpatterns_run_00100.hdf5"
hf = h5py.File(f"{PATH}/{FILE_NAME}", "r")
mc_x = np.array(hf["mc_x"][:])
mc_y = np.array(hf["mc_y"][:])
hitpatterns = np.array(hf["hitpatterns"][:])
hitpatterns_max = np.max(hitpatterns, axis=1, keepdims=True)
intensity = hitpatterns / hitpatterns_max

print(hitpatterns[0])


# In[364]:


model.summary()


# In[ ]:


# Save the model
#model.save('pmt_position_reconstruction_model.keras')


# In[371]:


import matplotlib.pyplot as plt

# Plot training & validation loss values
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss (MSE)')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
#plt.ylim(0,50)
plt.show()

# Plot training & validation MAE values
plt.figure(figsize=(12, 6))
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Model Mean Absolute Error')
plt.ylabel('MAE')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
#plt.ylim(0,100)
plt.show()


# In[205]:


# Step 1: Make predictions on the test set
predictions = model.predict(intensity_test)

# Step 2: Calculate the MAE for each event
errors = np.abs(predictions - mc_positions_test)
mae_per_event = np.mean(errors, axis=1)

# Step 3: Calculate the radius for each event
radii = np.sqrt(np.sum(np.square(mc_positions_test), axis=1))

# Step 4: Plot the MAE against the radius
plt.figure(figsize=(10, 6))
plt.scatter(radii, mae_per_event, alpha=0.5, marker='o')
plt.xlabel('Radius (sqrt(x^2 + y^2))')
plt.ylabel('Mean Absolute Error')
plt.title('MAE vs Radius for Test Data')
plt.grid(True)
plt.show()


# In[206]:


# Step 1: Make predictions on the test set
predictions = model.predict(intensity_test)

# Step 2: Calculate the MAE for each event
errors = np.abs(predictions - mc_positions_test)
mae_per_event = np.mean(errors, axis=1)

# Step 3: Filter events with MAE > 1
filter_mask = mae_per_event > 1
filtered_mae = mae_per_event[filter_mask]
filtered_positions = mc_positions_test[filter_mask]

# Step 4: Calculate the radius for the filtered events
filtered_radii = np.sqrt(np.sum(np.square(filtered_positions), axis=1))

# Step 5: Plot the MAE against the radius
plt.figure(figsize=(10, 6))
plt.scatter(filtered_radii, filtered_mae, alpha=0.5, marker='o')
plt.xlabel('Radius (sqrt(x^2 + y^2))')
plt.ylabel('Mean Absolute Error')
plt.title('MAE vs Radius for Test Data (MAE > 1)')
plt.grid(True)
plt.show()


# In[207]:


# Step 1: Make predictions on the test set
predictions = model.predict(intensity_test)

# Step 2: Calculate the MAE for each event
errors = np.abs(predictions - mc_positions_test)
mae_per_event = np.mean(errors, axis=1)

# Step 3: Filter events with MAE > 1
filter_mask = mae_per_event > 1
filtered_mae = mae_per_event[filter_mask]
filtered_positions = mc_positions_test[filter_mask]

# Step 4: Extract x and y coordinates for the filtered events
filtered_x = filtered_positions[:, 0]
filtered_y = filtered_positions[:, 1]

# Step 5: Plot the events in their true positions
plt.figure(figsize=(10, 8))
plt.scatter(filtered_x, filtered_y, c=filtered_mae, cmap='viridis', alpha=0.5, marker='o')
plt.colorbar(label='Mean Absolute Error')
plt.xlabel('X Coordinates')
plt.ylabel('Y Coordinates')
plt.title('True Positions of Events with MAE > 1')
plt.grid(True)
plt.axis('equal')  # Ensures equal scaling on both axes

# Draw a circle
circle_radius = 67
circle = Circle((0, 0), circle_radius, color='k', fill=False, linewidth=1)
plt.gca().add_patch(circle)

inner_circle_radius = 60
inner_circle = Circle((0, 0), inner_circle_radius, color='k', fill=False, linewidth=1, linestyle='dotted')
plt.gca().add_patch(inner_circle)


plt.show()


# **Prediction**

# In[208]:


#model = tf.keras.models.load_model('pmt_position_reconstruction_model.h5')

# Predicting on a random test event
np.random.seed(42)  # For reproducibility
random_index = np.random.randint(len(intensity_test))
test_event = intensity_test[random_index]
true_coordinates = mc_positions_test[random_index]

# Predict the position using the model
predicted_coordinates = model.predict(np.expand_dims(test_event, axis=0))

print("True Coordinates (MC):", true_coordinates)
print("Predicted Coordinates:", predicted_coordinates[0])


# In[373]:


#visualize prediction with hitpatterns

event_idx = 5

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

plt.savefig("broken_posrec.png", dpi=300)


# **Test**

# In[334]:


data_array = hitpatterns[:]  # Replace this with your actual array

# Count the number of zeros in the array
zero_count = np.count_nonzero(data_array == 0)

print(f"Number of zero entries: {zero_count}")


# In[260]:


file_names[2]


# In[335]:


import numpy as np

# Assuming hitpatterns is a 2D array where each row is a hitpattern
hitpatterns = np.array(hf["hitpatterns"][:])

# Count the number of zeros in each hitpattern (row)
zero_counts = np.count_nonzero(hitpatterns == 0, axis=1)

# Count how many hitpatterns contain more than 10 zero entries
hitpatterns_with_more_than_10_zeros = np.sum(zero_counts > 10)

print(f"Number of hitpatterns in total: {len(hitpatterns)}")
print(f"Number of hitpatterns with more than 10 zero entries: {hitpatterns_with_more_than_10_zeros}")


# In[ ]:


from tensorflow.keras.models import load_model

# Load the model
model = load_model('pmt_position_reconstruction_model.keras')


# In[ ]:


# Print the results
#print("Test Event Hitpatterns:", test_event)
print("True Coordinates (MC):", true_coordinates)
print("Predicted Coordinates:", predicted_coordinates[0])


# **Larger model, more data**

# In[53]:


# Constants
PATH = "/data/storage/posrec_hitpatterns"
FILE_NAMES = [
    "fuse_hitpatterns_run_00100.hdf5",
    "fuse_hitpatterns_run_00334.hdf5",
    "fuse_hitpatterns_run_00409.hdf5",
    "fuse_hitpatterns_run_00449.hdf5",
    "fuse_hitpatterns_run_00483.hdf5",
    "fuse_hitpatterns_run_00357.hdf5",
    "fuse_hitpatterns_run_00362.hdf5",
    "fuse_hitpatterns_run_00261.hdf5",
    "fuse_hitpatterns_run_00478.hdf5",
    "fuse_hitpatterns_run_00367.hdf5"
]
PMTs_top = 253

# Initialize lists to hold data from all files
mc_x_list = []
mc_y_list = []
hitpatterns_list = []

# Load data from each file and append to lists
for file_name in FILE_NAMES:
    with h5py.File(f"{PATH}/{file_name}", "r") as hf:
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

# Create the neural network model
model_large = Sequential([
    Dense(512, activation='relu', input_shape=(PMTs_top,)),
    Dropout(0.3),   # randomly sets 30% of data to 0 to prevent overfitting
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(2)  # Output layer for x and y coordinates
])

# Compile the model
model_large.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model
history_large = model_large.fit(intensity_train, mc_positions_train, 
                    epochs=100, 
                    batch_size=32, 
                    validation_data=(intensity_val, mc_positions_val))

# Evaluate the model on the test set
test_loss_large, test_mae_large = model_large.evaluate(intensity_test, mc_positions_test)

print(f'Test Mean Absolute Error: {test_mae_large}')

# Save the model
model_large.save('pmt_position_reconstruction_model_large.keras')


# In[ ]:


# Plot training & validation loss values
plt.figure(figsize=(12, 6))
plt.plot(history_large.history['loss'], label='Train Loss')
plt.plot(history_large.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.show()

# Plot training & validation MAE values
plt.figure(figsize=(12, 6))
plt.plot(history_large.history['mae'], label='Train MAE')
plt.plot(history_large.history['val_mae'], label='Validation MAE')
plt.title('Model Mean Absolute Error')
plt.ylabel('MAE')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.show()


# In[ ]:


#model = tf.keras.models.load_model('pmt_position_reconstruction_model_large.keras')

# Predicting on a random test event
np.random.seed(42)  # For reproducibility
random_index = np.random.randint(len(intensity_test))
test_event = intensity_test[random_index]
true_coordinates = mc_positions_test[random_index]

# Predict the position using the model
predicted_coordinates = model_large.predict(np.expand_dims(test_event, axis=0))

print("True Coordinates (MC):", true_coordinates)
print("Predicted Coordinates:", predicted_coordinates[0])


# **ALL DATA**

# In[52]:


PATH = "/data/storage/posrec_hitpatterns"
file_names = [f for f in os.listdir(PATH) if f.endswith(".hdf5")]

len(file_names)


# In[473]:


file_names[2:3]


# In[309]:


data_array = hitpatterns[:]  # Replace this with your actual array

# Count the number of zeros in the array
zero_count = np.count_nonzero(data_array == 0)

print(f"Number of zero entries: {zero_count}")


# In[735]:


# Constants
PATH = "/data/storage/posrec_hitpatterns"
PMTs_top = 253

# Get list of all HDF5 files in the directory
file_names = [f for f in os.listdir(PATH) if f.endswith(".hdf5")]

# Initialize lists to hold data from all files
mc_x_list = []
mc_y_list = []
hitpatterns_list = []

# Load data from each file and append to lists
for file_name in file_names[340:341]:
    with h5py.File(os.path.join(PATH, file_name), "r") as hf:
        mc_x_list.append(np.array(hf["mc_x"][:]))
        mc_y_list.append(np.array(hf["mc_y"][:]))
        hitpatterns_list.append(np.array(hf["hitpatterns"][:]))
        
# Concatenate data from all files
mc_x = np.concatenate(mc_x_list)
mc_y = np.concatenate(mc_y_list)
hitpatterns = np.concatenate(hitpatterns_list)


# In[736]:


#340-350, 350-360, 360-370  

#3, 36, 38, 71, 73, 145, 147, 150, 154, 156, 157, 158, 163, 225, 229,
#235, 287, 311, 312, 313


# In[737]:


# Normalize hitpatterns by their maximum values
hitpatterns_max = np.max(hitpatterns, axis=1, keepdims=True)
intensity = hitpatterns / hitpatterns_max

# Combine mc_x and mc_y into one array for the target positions
mc_positions = np.vstack((mc_x, mc_y)).T

# Split the data into training, validation, and test sets
intensity_train, intensity_temp, mc_positions_train, mc_positions_temp = train_test_split(intensity, mc_positions, test_size=0.3, random_state=42)
intensity_val, intensity_test, mc_positions_val, mc_positions_test = train_test_split(intensity_temp, mc_positions_temp, test_size=0.5, random_state=42)

# Create the neural network model
model_all = Sequential([
    Dense(512, activation='relu', input_shape=(PMTs_top,)),
    #Dropout(0.3),   # randomly sets 30% of data to 0 to prevent overfitting
    Dense(256, activation='relu'),
    #Dropout(0.3),
    Dense(128, activation='relu'),
    #Dropout(0.3),
    Dense(2)  # Output layer for x and y coordinates
])

# Compile the model
model_all.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Train the model
history_all = model_all.fit(intensity_train, mc_positions_train, 
                    epochs=5, 
                    batch_size=100, 
                    validation_data=(intensity_val, mc_positions_val))

# Evaluate the model on the test set
test_loss_all, test_mae_all = model_all.evaluate(intensity_test, mc_positions_test)

print(f'Test Mean Absolute Error: {test_mae_all}')

# Save the model
#model.save('pmt_position_reconstruction_model_ALL_DATA.h5')


# In[57]:


# Plot training & validation loss values
plt.figure(figsize=(12, 6))
plt.plot(history_all.history['loss'], label='Train Loss')
plt.plot(history_all.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.show()

# Plot training & validation MAE values
plt.figure(figsize=(12, 6))
plt.plot(history_all.history['mae'], label='Train MAE')
plt.plot(history_all.history['val_mae'], label='Validation MAE')
plt.title('Model Mean Absolute Error')
plt.ylabel('MAE')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.show()


# In[ ]:





# In[58]:


#model = tf.keras.models.load_model('pmt_position_reconstruction_model_large.keras')

# Predicting on a random test event
np.random.seed(42)  # For reproducibility
random_index = np.random.randint(len(intensity_test))
test_event = intensity_test[random_index]
true_coordinates = mc_positions_test[random_index]

# Predict the position using the model
predicted_coordinates = model.predict(np.expand_dims(test_event, axis=0))

print("True Coordinates (MC):", true_coordinates)
print("Predicted Coordinates:", predicted_coordinates[0])


# In[ ]:





# # Position reconstruction: Grouped in 3

# In[45]:


# Constants
PATH = "/data/storage/posrec_hitpatterns"
FILE_NAME = "fuse_hitpatterns_run_00100.hdf5"
PMTs_top = 253
event_idx = 42

# Load data from HDF5 file
hf = h5py.File(f"{PATH}/{FILE_NAME}", "r")
mc_x = np.array(hf["mc_x"][:])
mc_y = np.array(hf["mc_y"][:])
hitpatterns = np.array(hf["hitpatterns"][:])
intensity = hitpatterns[event_idx]

# Load the 3ch_mux.npy file
ch3_mux_mod = np.load('3ch_mux_mod_single_pmt.npy')

# Create a new intensity array with the same shape as the original
combined_intensity = np.zeros_like(intensity)

# Sum the intensities within each group and assign to the new intensity array
#for group in ch3_mux_mod:
 #   group_intensity = np.sum(intensity[group])
  #  for pmt in group:
   #     combined_intensity[pmt] = group_intensity
        
# Sum the intensities within each group and normalize by the number of PMTs in the group
for group in ch3_mux_mod:
    valid_pmts = [pmt for pmt in group if (pmt < len(intensity) and pmt >= 0)]  # Ensure valid PMT indices
    if valid_pmts:  # Check if there are any valid PMTs
        group_intensity = np.sum(intensity[valid_pmts])
        normalized_intensity = group_intensity / len(valid_pmts)
        for pmt in valid_pmts:
            combined_intensity[pmt] = normalized_intensity

# Extract x and y coordinates for the top PMTs
pmt_x_top = pmt_xy_top["x"]
pmt_y_top = pmt_xy_top["y"]

# Create the plot
plt.figure(figsize=(10, 8))

# Scatter plot with combined intensity as color
sc = plt.scatter(pmt_x_top, pmt_y_top, c=combined_intensity, cmap='viridis', marker='o', s=800, norm=LogNorm())
plt.colorbar(sc, label='Intensity')  # Add colorbar
plt.scatter(mc_x[event_idx], mc_y[event_idx], c='red', marker='o', s=100, label='MC Point')

# Draw a circle
circle_radius = 67
circle = Circle((0, 0), circle_radius, color='k', fill=False, linewidth=1)
plt.gca().add_patch(circle)

plt.title('Top PMT Positions with Intensity (normed to group size)')
plt.xlabel('X Coordinates')
plt.ylabel('Y Coordinates')
plt.grid(True)
plt.axis('equal')  # Ensures equal scaling on both axes

plt.tight_layout()
plt.show()


# In[13]:


#Normalize Data

# Constants
PATH = "/data/storage/posrec_hitpatterns"  
FILE_NAME = "fuse_hitpatterns_run_00100.hdf5"
TOTAL_EVENTS = 9995
PMTs_top = 253

# Load the 3ch_mux.npy file
ch3_mux_mod = np.load('3ch_mux_mod_single_pmt.npy')

# Load data from HDF5 file
hf = h5py.File(f"{PATH}/{FILE_NAME}", "r")
mc_x = np.array(hf["mc_x"][:])
mc_y = np.array(hf["mc_y"][:])

hitpatterns = np.array(hf["hitpatterns"][:])
hitpatterns_norm = np.zeros_like(hitpatterns)



for idx, hitpattern in enumerate(tqdm(hitpatterns, desc="Processing Hitpatterns")):    
    # Create a new intensity array with the same shape as the original
    combined_intensity = np.zeros_like(hitpattern)
    # Sum the intensities within each group and normalize by the number of PMTs in the group
    for group in ch3_mux_mod:
        valid_pmts = [pmt for pmt in group if (pmt < len(hitpattern) and pmt >= 0)]  # Ensure valid PMT indices
        if valid_pmts:  # Check if there are any valid PMTs
            group_intensity = np.sum(hitpattern[valid_pmts])
            normalized_intensity = group_intensity / len(valid_pmts)
            for pmt in valid_pmts:
                combined_intensity[pmt] = normalized_intensity

    intensity_max = np.max(combined_intensity, axis=0, keepdims=True)
    intensity = combined_intensity / intensity_max
    hitpatterns_norm[idx] = intensity


# In[14]:


print(hitpatterns.shape)
print(hitpatterns_norm.shape)
print(combined_intensity.shape)
print(hitpatterns[5:])
print(hitpatterns_norm[5:])


# In[167]:


# Constants
PATH = "/data/storage/posrec_hitpatterns"  
FILE_NAME = "fuse_hitpatterns_run_00100.hdf5"
TOTAL_EVENTS = 9995
PMTs_top = 253

# Load the 3ch_mux.npy file
ch3_mux_mod = np.load('3ch_mux_mod_single_pmt.npy')

# Load data from HDF5 file
hf = h5py.File(f"{PATH}/{FILE_NAME}", "r")
mc_x = np.array(hf["mc_x"][:])
mc_y = np.array(hf["mc_y"][:])

hitpatterns = np.array(hf["hitpatterns"][:])
hitpatterns_norm = np.zeros_like(hitpatterns)


#Normalize data and combine intensities
for idx, hitpattern in enumerate(tqdm(hitpatterns, desc="Processing Hitpatterns")):    
    # Create a new intensity array with the same shape as the original
    combined_intensity = np.zeros_like(hitpattern)
    # Sum the intensities within each group and normalize by the number of PMTs in the group
    for group in ch3_mux_mod:
        valid_pmts = [pmt for pmt in group if (pmt < len(hitpattern) and pmt >= 0)]  # Ensure valid PMT indices
        if valid_pmts:  # Check if there are any valid PMTs
            group_intensity = np.sum(hitpattern[valid_pmts])
            normalized_intensity = group_intensity / len(valid_pmts)
            for pmt in valid_pmts:
                combined_intensity[pmt] = normalized_intensity

    intensity_max = np.max(combined_intensity, axis=0, keepdims=True)
    intensity = combined_intensity / intensity_max
    hitpatterns_norm[idx] = intensity

# Combine mc_x and mc_y into one array for the target positions
mc_positions = np.vstack((mc_x, mc_y)).T

# Split the data into training, validation, and test sets
intensity_train, intensity_temp, mc_positions_train, mc_positions_temp = train_test_split(hitpatterns_norm, mc_positions, test_size=0.3, random_state=42)
intensity_val, intensity_test, mc_positions_val, mc_positions_test = train_test_split(intensity_temp, mc_positions_temp, test_size=0.5, random_state=42)


# Create the neural network model
model = Sequential([
    Dense(512, activation='relu', input_shape=(PMTs_top,)),
    #Dropout(0.3),   # randomly sets 30% of data to 0 to prevent overfitting
    Dense(256, activation='relu'),
    #Dropout(0.3),
    Dense(128, activation='relu'),
    #Dropout(0.3),
    Dense(2)  # Output layer for x and y coordinates
])

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model
history = model.fit(intensity_train, mc_positions_train, 
                    epochs=50, 
                    batch_size=100, 
                    validation_data=(intensity_val, mc_positions_val))

# Evaluate the model on the test set
test_loss, test_mae = model.evaluate(intensity_test, mc_positions_test)

print(f'Test Mean Absolute Error: {test_mae}')

# Save the model
#model.save('pmt_position_reconstruction_model.h5')


# In[168]:


model.summary()


# In[169]:


# Plot training & validation loss values
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss (MSE)')
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


# In[170]:


#model = tf.keras.models.load_model('pmt_position_reconstruction_model.h5')

# Predicting on a random test event
np.random.seed(42)  # For reproducibility
random_index = np.random.randint(len(intensity_test))
test_event = intensity_test[random_index]
true_coordinates = mc_positions_test[random_index]

# Predict the position using the model
predicted_coordinates = model.predict(np.expand_dims(test_event, axis=0))

print("True Coordinates (MC):", true_coordinates)
print("Predicted Coordinates:", predicted_coordinates[0])


# In[171]:


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
plt.show()

#plt.savefig("nn_first_model_visualization_3.png", dpi=300)


# # Position reconstruction: Grouped in 7

# In[48]:


# Constants
PATH = "/data/storage/posrec_hitpatterns"
FILE_NAME = "fuse_hitpatterns_run_00100.hdf5"
PMTs_top = 253
event_idx = 42

# Load data from HDF5 file
hf = h5py.File(f"{PATH}/{FILE_NAME}", "r")
mc_x = np.array(hf["mc_x"][:])
mc_y = np.array(hf["mc_y"][:])
hitpatterns = np.array(hf["hitpatterns"][:])
hitpatterns_max = np.max(hitpatterns, axis=1, keepdims=True)
intensity = hitpatterns[event_idx]

# Load the 7ch_mux.npy file
ch7_mux_mod = np.load('7ch_mux_mod_single_pmt.npy')

# Create a new intensity array with the same shape as the original
combined_intensity = np.zeros_like(intensity)

# Sum the intensities within each group and assign to the new intensity array
#for group in ch7_mux_mod:
 #   group_intensity = np.sum(intensity[group])
  #  for pmt in group:
   #     combined_intensity[pmt] = group_intensity
        
# Sum the intensities within each group and normalize by the number of PMTs in the group
for group in ch7_mux_mod:
    valid_pmts = [pmt for pmt in group if (pmt < len(intensity) and pmt >= 0)]  # Ensure valid PMT indices
    if valid_pmts:  # Check if there are any valid PMTs
        group_intensity = np.sum(intensity[valid_pmts])
        normalized_intensity = group_intensity / len(valid_pmts)
        for pmt in valid_pmts:
            combined_intensity[pmt] = normalized_intensity

# Extract x and y coordinates for the top PMTs
pmt_x_top = pmt_xy_top["x"]
pmt_y_top = pmt_xy_top["y"]

# Create the plot
plt.figure(figsize=(10, 8))

# Scatter plot with combined intensity as color
sc = plt.scatter(pmt_x_top, pmt_y_top, c=combined_intensity, cmap='viridis', marker='o', s=800, norm=LogNorm())
plt.colorbar(sc, label='Intensity')  # Add colorbar
plt.scatter(mc_x[event_idx], mc_y[event_idx], c='red', marker='o', s=100, label='MC Point')

# Draw a circle
circle_radius = 67
circle = Circle((0, 0), circle_radius, color='k', fill=False, linewidth=1)
plt.gca().add_patch(circle)

plt.title('Top PMT Positions with Intensity (normed to group size)')
plt.xlabel('X Coordinates')
plt.ylabel('Y Coordinates')
plt.grid(True)
plt.axis('equal')  # Ensures equal scaling on both axes

plt.tight_layout()
plt.show()


# In[49]:


# Constants
PATH = "/data/storage/posrec_hitpatterns"  
FILE_NAME = "fuse_hitpatterns_run_00100.hdf5"
TOTAL_EVENTS = 9995
PMTs_top = 253

# Load the 3ch_mux.npy file
ch7_mux_mod = np.load('7ch_mux_mod_single_pmt.npy')

# Load data from HDF5 file
hf = h5py.File(f"{PATH}/{FILE_NAME}", "r")
mc_x = np.array(hf["mc_x"][:])
mc_y = np.array(hf["mc_y"][:])

hitpatterns = np.array(hf["hitpatterns"][:])
hitpatterns_norm = np.zeros_like(hitpatterns)


#Normalize data and combine intensities
for idx, hitpattern in enumerate(tqdm(hitpatterns, desc="Processing Hitpatterns")):    
    # Create a new intensity array with the same shape as the original
    combined_intensity = np.zeros_like(hitpattern)
    # Sum the intensities within each group and normalize by the number of PMTs in the group
    for group in ch7_mux_mod:
        valid_pmts = [pmt for pmt in group if (pmt < len(hitpattern) and pmt >= 0)]  # Ensure valid PMT indices
        if valid_pmts:  # Check if there are any valid PMTs
            group_intensity = np.sum(hitpattern[valid_pmts])
            normalized_intensity = group_intensity / len(valid_pmts)
            for pmt in valid_pmts:
                combined_intensity[pmt] = normalized_intensity

    intensity_max = np.max(combined_intensity, axis=0, keepdims=True)
    intensity = combined_intensity / intensity_max
    hitpatterns_norm[idx] = intensity

# Combine mc_x and mc_y into one array for the target positions
mc_positions = np.vstack((mc_x, mc_y)).T

# Split the data into training, validation, and test sets
intensity_train, intensity_temp, mc_positions_train, mc_positions_temp = train_test_split(hitpatterns_norm, mc_positions, test_size=0.3, random_state=42)
intensity_val, intensity_test, mc_positions_val, mc_positions_test = train_test_split(intensity_temp, mc_positions_temp, test_size=0.5, random_state=42)


# Create the neural network model
model = Sequential([
    Dense(512, activation='relu', input_shape=(PMTs_top,)),
    Dropout(0.3),   # randomly sets 30% of data to 0 to prevent overfitting
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(2)  # Output layer for x and y coordinates
])

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model
history = model.fit(intensity_train, mc_positions_train, 
                    epochs=50, 
                    batch_size=32, 
                    validation_data=(intensity_val, mc_positions_val))

# Evaluate the model on the test set
test_loss, test_mae = model.evaluate(intensity_test, mc_positions_test)

print(f'Test Mean Absolute Error: {test_mae}')

# Save the model
#model.save('pmt_position_reconstruction_model.h5')


# In[32]:


model.summary()


# In[33]:


# Plot training & validation loss values
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss (MSE)')
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


# In[34]:


#model = tf.keras.models.load_model('pmt_position_reconstruction_model.h5')

# Predicting on a random test event
np.random.seed(42)  # For reproducibility
random_index = np.random.randint(len(intensity_test))
test_event = intensity_test[random_index]
true_coordinates = mc_positions_test[random_index]

# Predict the position using the model
predicted_coordinates = model.predict(np.expand_dims(test_event, axis=0))

print("True Coordinates (MC):", true_coordinates)
print("Predicted Coordinates:", predicted_coordinates[0])


# In[50]:


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
plt.show()

#plt.savefig("nn_first_model_visualization_7.png", dpi=300)


# # Position reconstruction: Grouped in 3 - Random

# In[138]:


# Constants
PATH = "/data/storage/posrec_hitpatterns"
FILE_NAME = "fuse_hitpatterns_run_00100.hdf5"
PMTs_top = 253
event_idx = 42

# Load data from HDF5 file
hf = h5py.File(f"{PATH}/{FILE_NAME}", "r")
mc_x = np.array(hf["mc_x"][:])
mc_y = np.array(hf["mc_y"][:])
hitpatterns = np.array(hf["hitpatterns"][:])
hitpatterns_max = np.max(hitpatterns, axis=1, keepdims=True)
intensity = hitpatterns[event_idx]

# Load the 7ch_mux.npy file
#ch7_mux_mod = np.load('7ch_mux_mod_single_pmt.npy')

rand_3 = results["rand_3_0_0"]

# Create a new intensity array with the same shape as the original
combined_intensity = np.zeros_like(intensity)

# Sum the intensities within each group and assign to the new intensity array
#for group in ch7_mux_mod:
 #   group_intensity = np.sum(intensity[group])
  #  for pmt in group:
   #     combined_intensity[pmt] = group_intensity
        
# Sum the intensities within each group and normalize by the number of PMTs in the group
for group in rand_3:
    valid_pmts = [pmt for pmt in group if (pmt < len(intensity) and pmt >= 0)]  # Ensure valid PMT indices
    if valid_pmts:  # Check if there are any valid PMTs
        group_intensity = np.sum(intensity[valid_pmts])
        normalized_intensity = group_intensity / len(valid_pmts)
        for pmt in valid_pmts:
            combined_intensity[pmt] = normalized_intensity

# Extract x and y coordinates for the top PMTs
pmt_x_top = pmt_xy_top["x"]
pmt_y_top = pmt_xy_top["y"]

# Create the plot
plt.figure(figsize=(10, 8))

# Scatter plot with combined intensity as color
sc = plt.scatter(pmt_x_top, pmt_y_top, c=combined_intensity, cmap='viridis', marker='o', s=800, norm=LogNorm())
plt.colorbar(sc, label='Intensity')  # Add colorbar
plt.scatter(mc_x[event_idx], mc_y[event_idx], c='red', marker='o', s=100, label='MC Point')

# Draw a circle
circle_radius = 67
circle = Circle((0, 0), circle_radius, color='k', fill=False, linewidth=1)
plt.gca().add_patch(circle)

plt.title('Top PMT Positions with Intensity (normed to group size)')
plt.xlabel('X Coordinates')
plt.ylabel('Y Coordinates')
plt.grid(True)
plt.axis('equal')  # Ensures equal scaling on both axes

plt.tight_layout()
plt.show()


# In[139]:


# Constants
PATH = "/data/storage/posrec_hitpatterns"  
FILE_NAME = "fuse_hitpatterns_run_00100.hdf5"
TOTAL_EVENTS = 9995
PMTs_top = 253

# Load the 3ch_mux.npy file
#ch7_mux_mod = np.load('7ch_mux_mod_single_pmt.npy')

rand_3 = results["rand_3_0_0"]

# Load data from HDF5 file
hf = h5py.File(f"{PATH}/{FILE_NAME}", "r")
mc_x = np.array(hf["mc_x"][:])
mc_y = np.array(hf["mc_y"][:])

hitpatterns = np.array(hf["hitpatterns"][:])
hitpatterns_norm = np.zeros_like(hitpatterns)


#Normalize data and combine intensities
for idx, hitpattern in enumerate(tqdm(hitpatterns, desc="Processing Hitpatterns")):    
    # Create a new intensity array with the same shape as the original
    combined_intensity = np.zeros_like(hitpattern)
    # Sum the intensities within each group and normalize by the number of PMTs in the group
    for group in rand_3:
        valid_pmts = [pmt for pmt in group if (pmt < len(hitpattern) and pmt >= 0)]  # Ensure valid PMT indices
        if valid_pmts:  # Check if there are any valid PMTs
            group_intensity = np.sum(hitpattern[valid_pmts])
            normalized_intensity = group_intensity / len(valid_pmts)
            for pmt in valid_pmts:
                combined_intensity[pmt] = normalized_intensity

    intensity_max = np.max(combined_intensity, axis=0, keepdims=True)
    intensity = combined_intensity / intensity_max
    hitpatterns_norm[idx] = intensity

# Combine mc_x and mc_y into one array for the target positions
mc_positions = np.vstack((mc_x, mc_y)).T

# Split the data into training, validation, and test sets
intensity_train, intensity_temp, mc_positions_train, mc_positions_temp = train_test_split(hitpatterns_norm, mc_positions, test_size=0.3, random_state=42)
intensity_val, intensity_test, mc_positions_val, mc_positions_test = train_test_split(intensity_temp, mc_positions_temp, test_size=0.5, random_state=42)


# Create the neural network model
model = Sequential([
    Dense(512, activation='relu', input_shape=(PMTs_top,)),
    Dropout(0.3),   # randomly sets 30% of data to 0 to prevent overfitting
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(2)  # Output layer for x and y coordinates
])

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model
history = model.fit(intensity_train, mc_positions_train, 
                    epochs=50, 
                    batch_size=32, 
                    validation_data=(intensity_val, mc_positions_val))

# Evaluate the model on the test set
test_loss, test_mae = model.evaluate(intensity_test, mc_positions_test)

print(f'Test Mean Absolute Error: {test_mae}')

# Save the model
#model.save('pmt_position_reconstruction_model.h5')


# In[140]:


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
plt.show()

#plt.savefig("nn_first_model_visualization_7.png", dpi=300)


# # Position reconstruction: Grouped in 5 - Random

# In[ ]:





# In[ ]:





# ## Deep dive into data

# In[ ]:




