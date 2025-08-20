# Imports
import os
import h5py
import numpy as np
from tqdm import tqdm
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split

# Define paths
model_dir = '/scratch/midway3/nkoebelin/mux_models_new/'
hist_dir = '/scratch/midway3/nkoebelin/mux_models_new/training_history'
os.makedirs(model_dir, exist_ok=True)
PATH = "/project2/lgrandi/svetter/simulations_for_posrec/s2_hitpattern_sims_for_posrec/"
SAVE_DIR = "/scratch/midway3/nkoebelin/mux_data_new/"

# Constants
pmt_diameter = 7.62
PMTs_top = 253

# Load muxing patterns
muxing_patterns = np.load('mux_patterns_all_new.npz', allow_pickle=True)
print("Available patterns:", muxing_patterns.files)

# Load PMT geometry
hex_data = np.load("hex_pattern_data.npz", allow_pickle=True)
meta = hex_data["meta"].item()
pmt_xy = hex_data["pmt_xy"]
pmt_xy_top = hex_data["pmt_xy_top"]
pmt_x_top = hex_data["pmt_x_top"]
pmt_y_top = hex_data["pmt_y_top"]

# Loop through each mux pattern
for mux_name in muxing_patterns.files:
    # Filter to only process patterns that match a condition
    if not mux_name.startswith("det"):
        continue

    save_path = os.path.join(SAVE_DIR, f"saved_data_{mux_name}.npz")

    # Skip if the file already exists
    if os.path.exists(save_path):
        print(f"Skipping {mux_name}, data already exists.")
        continue
        
    print(f"\nProcessing mux pattern: {mux_name}")
    mux_pattern = muxing_patterns[mux_name]
    valid_groups = [np.array(group) for group in mux_pattern]

    # Load HDF5 files
    file_names = [f for f in os.listdir(PATH) if f.endswith(".hdf5")]
    file_names.sort()

    interface_x_list, interface_y_list, interface_electrons_list, hitpatterns_list = [], [], [], []

    for file_name in tqdm(file_names[:100], desc=f"Loading MC runs for {mux_name}"):
        with h5py.File(os.path.join(PATH, file_name), "r") as hf:
            interface_x_list.append(np.array(hf["interface_x"][:]))
            interface_y_list.append(np.array(hf["interface_y"][:]))
            interface_electrons_list.append(np.array(hf["interface_electrons"][:]))
            hitpatterns_list.append(np.array(hf["hitpatterns"][:]))

    interface_x = np.concatenate(interface_x_list)
    interface_y = np.concatenate(interface_y_list)
    interface_electrons = np.concatenate(interface_electrons_list)
    hitpatterns = np.concatenate(hitpatterns_list)
    s2_top_area = np.sum(hitpatterns, axis=1)

    # Process Hitpatterns
    intensity = np.zeros_like(hitpatterns)
    group_intensity = np.zeros((hitpatterns.shape[0], len(valid_groups)))

    for idx in tqdm(range(hitpatterns.shape[0]), desc=f"Normalizing {mux_name}"):
        hitpattern = hitpatterns[idx]
        combined_intensity = np.zeros_like(hitpattern)
        group_values = np.zeros(len(valid_groups))

        for i, group in enumerate(valid_groups):
            if len(group) > 0:
                group_sum = np.sum(hitpattern[group])
                group_avg = group_sum / len(group)
                combined_intensity[group] = group_avg
                group_values[i] = group_sum

        intensity_max = np.max(combined_intensity)
        group_values_max = np.max(group_values)

        if intensity_max > 0:
            intensity[idx] = combined_intensity / intensity_max
            group_intensity[idx] = group_values / group_values_max
        else:
            intensity[idx] = combined_intensity
            group_intensity[idx] = group_values

    interface_positions = np.vstack((interface_x, interface_y)).T

    # Apply area cut: only keep events with 10 ≤ s2_top_area < 1e5
    valid_mask = (s2_top_area >= 10) & (s2_top_area < 1e5)

    group_intensity = group_intensity[valid_mask]
    interface_positions = interface_positions[valid_mask]
    interface_electrons = interface_electrons[valid_mask]
    s2_top_area = s2_top_area[valid_mask]


    # Split the data
    X_train, X_temp, y_train, y_temp, e_train, e_temp, area_train, area_temp = train_test_split(
        group_intensity, interface_positions, interface_electrons, s2_top_area, test_size=0.5, random_state=42)

    X_val, X_test, y_val, y_test, e_val, e_test, area_val, area_test = train_test_split(
        X_temp, y_temp, e_temp, area_temp, test_size=0.8, random_state=42)

    # Build model
    model = Sequential([
        Input(shape=(group_intensity.shape[1],)),
        Dense(180, activation='elu', kernel_regularizer=l2(0.01)),
        Dropout(0.1),
        Dense(60, activation='elu', kernel_regularizer=l2(0.01)),
        Dense(20, activation='elu', kernel_regularizer=l2(0.01)),
        Dense(2, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

    # Train model
    history = model.fit(X_train, y_train,
                        epochs=100,
                        batch_size=500,
                        validation_data=(X_val, y_val),
                        verbose=1)

    # Evaluate
    test_loss, test_mae = model.evaluate(X_test, y_test)
    print(f'Test Loss: {test_loss:.4f}, MAE: {test_mae:.4f}')

    # Save model and history
    model_path = os.path.join(model_dir, f"model_{mux_name}.keras")
    history_path = os.path.join(hist_dir, f"history_{mux_name}.pkl")
    model.save(model_path)
    with open(history_path, 'wb') as f:
        pickle.dump(history.history, f)


    # Predict and compute error
    predictions = model.predict(X_test, batch_size=32, verbose=0)
    abs_errors = np.linalg.norm(predictions - y_test, axis=1)
    radial_errors = np.linalg.norm(predictions, axis=1) - np.linalg.norm(y_test, axis=1)
    pos_test = y_test  # y_test contains the interface_positions (i.e. [x, y])

   

    save_path = os.path.join(SAVE_DIR, f"saved_data_{mux_name}.npz")

    np.savez_compressed(save_path,
        e_test=e_test,
        area_test=area_test,
        pos_test=y_test,
        pos_pred=predictions,
        abs_errors=abs_errors,
        radial_errors=radial_errors
                       )

    print(f"Saved: {save_path}")


