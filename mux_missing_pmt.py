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

# Paths and constants
model_dir = '/scratch/midway3/nkoebelin/mux_models_new/'
hist_dir = '/scratch/midway3/nkoebelin/mux_models_new/training_history'
os.makedirs(model_dir, exist_ok=True)
os.makedirs(hist_dir, exist_ok=True)

PATH = "/project2/lgrandi/svetter/simulations_for_posrec/s2_hitpattern_sims_for_posrec/"
SAVE_DIR = "/scratch/midway3/nkoebelin/mux_data_new/"
muxing_patterns = np.load('mux_patterns_all_new.npz', allow_pickle=True)

# Loop over selected patterns
for mux_name in muxing_patterns.files:
    if not mux_name.startswith("ungrouped"):
        continue

    print(f"\nProcessing: {mux_name}")
    valid_groups = [np.array(group) for group in muxing_patterns[mux_name]]

    # Load data
    file_names = sorted([f for f in os.listdir(PATH) if f.endswith(".hdf5")])
    interface_x_list, interface_y_list, interface_electrons_list, hitpatterns_list = [], [], [], []

    for file_name in tqdm(file_names[:100], desc=f"Loading {mux_name}"):
        with h5py.File(os.path.join(PATH, file_name), "r") as hf:
            interface_x_list.append(np.array(hf["interface_x"]))
            interface_y_list.append(np.array(hf["interface_y"]))
            interface_electrons_list.append(np.array(hf["interface_electrons"]))
            hitpatterns_list.append(np.array(hf["hitpatterns"]))

    interface_x = np.concatenate(interface_x_list)
    interface_y = np.concatenate(interface_y_list)
    interface_electrons = np.concatenate(interface_electrons_list)
    hitpatterns = np.concatenate(hitpatterns_list)
    s2_top_area = np.sum(hitpatterns, axis=1)
    interface_positions = np.vstack((interface_x, interface_y)).T

    # Apply area cut
    valid_mask = (s2_top_area >= 10) & (s2_top_area < 1e5)
    hitpatterns = hitpatterns[valid_mask]
    interface_positions = interface_positions[valid_mask]
    interface_electrons = interface_electrons[valid_mask]
    s2_top_area = s2_top_area[valid_mask]

    # Define missing PMT configs
    for n_off in [1, 5, 10]:
        print(f"\n--- Training with {n_off} PMTs turned off ---")
        save_path = os.path.join(SAVE_DIR, f"saved_data_{mux_name}_off{n_off}.npz")
        if os.path.exists(save_path):
            print(f"Skipping {mux_name}_off{n_off}, already exists.")
            continue

        # Randomly zero out PMTs
        np.random.seed(n_off)
        hit_mod = hitpatterns.copy()
        pmts_off = np.random.choice(hit_mod.shape[1], size=n_off, replace=False)
        hit_mod[:, pmts_off] = 0

        # Group and normalize
        grouped = np.zeros((hit_mod.shape[0], len(valid_groups)))
        for i, group in enumerate(valid_groups):
            grouped[:, i] = np.sum(hit_mod[:, group], axis=1)
        grouped /= np.max(grouped, axis=1, keepdims=True)

        # Split
        X_train, X_temp, y_train, y_temp, e_train, e_temp, area_train, area_temp = train_test_split(
            grouped, interface_positions, interface_electrons, s2_top_area, test_size=0.5, random_state=42)

        X_val, X_test, y_val, y_test, e_val, e_test, area_val, area_test = train_test_split(
            X_temp, y_temp, e_temp, area_temp, test_size=0.8, random_state=42)

        # Build model
        model = Sequential([
            Input(shape=(grouped.shape[1],)),
            Dense(180, activation='elu', kernel_regularizer=l2(0.01)),
            Dropout(0.1),
            Dense(60, activation='elu', kernel_regularizer=l2(0.01)),
            Dense(20, activation='elu', kernel_regularizer=l2(0.01)),
            Dense(2, activation='linear')
        ])
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

        # Train
        history = model.fit(X_train, y_train,
                            epochs=100,
                            batch_size=500,
                            validation_data=(X_val, y_val),
                            verbose=1)

        # Evaluate
        loss, mae = model.evaluate(X_test, y_test)
        print(f"Test Loss: {loss:.4f}, MAE: {mae:.4f}")

        # Predict and compute errors
        predictions = model.predict(X_test, batch_size=32)
        abs_errors = np.linalg.norm(predictions - y_test, axis=1)
        radial_errors = np.linalg.norm(predictions, axis=1) - np.linalg.norm(y_test, axis=1)

        # Save
        model_path = os.path.join(model_dir, f"model_{mux_name}_off{n_off}.keras")
        hist_path = os.path.join(hist_dir, f"history_{mux_name}_off{n_off}.pkl")
        model.save(model_path)
        with open(hist_path, 'wb') as f:
            pickle.dump(history.history, f)

        np.savez_compressed(save_path,
            e_test=e_test,
            area_test=area_test,
            pos_test=y_test,
            pos_pred=predictions,
            abs_errors=abs_errors,
            radial_errors=radial_errors,
            pmts_off=pmts_off
        )
        print(f"Saved: {save_path}")
