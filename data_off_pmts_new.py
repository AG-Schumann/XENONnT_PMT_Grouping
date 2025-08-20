# Imports
import os
import h5py
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.models import load_model

# --- Configuration ---
mux_name = "adj_2"  # CHANGE THIS for the desired model
data_dir = '/scratch/midway3/nkoebelin/mux_data_new/'
model_dir = '/scratch/midway3/nkoebelin/mux_models_new/'
model_path = os.path.join(model_dir, f"model_{mux_name}.keras")
save_path = os.path.join(data_dir, f"saved_data_{mux_name}.npz")
analysis_save_path = os.path.join(data_dir, f"saved_data_{mux_name}_analysis.npz")

# --- Load model and mux pattern ---
model = load_model(model_path)
muxing_patterns = np.load('mux_patterns_all_new.npz', allow_pickle=True)
valid_groups = [np.array(group) for group in muxing_patterns[mux_name]]

# --- Load raw hitpattern data ---
PATH = "/project2/lgrandi/svetter/simulations_for_posrec/s2_hitpattern_sims_for_posrec/"
file_names = sorted([f for f in os.listdir(PATH) if f.endswith(".hdf5")])
interface_x_list, interface_y_list, interface_electrons_list, hitpatterns_list = [], [], [], []

for file_name in tqdm(file_names[:100], desc="Loading MC for noise application"):
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
interface_positions = np.vstack((interface_x, interface_y)).T

# --- Area cut and test split ---
valid_mask = (s2_top_area >= 10) & (s2_top_area < 1e5)
hitpatterns = hitpatterns[valid_mask]
interface_positions = interface_positions[valid_mask]
interface_electrons = interface_electrons[valid_mask]
s2_top_area = s2_top_area[valid_mask]

from sklearn.model_selection import train_test_split
_, X_test, _, y_test, _, e_test, _, area_test = train_test_split(
    hitpatterns, interface_positions, interface_electrons, s2_top_area,
    test_size=0.4, random_state=42
)

# --- Grouping helper ---
def apply_grouping(hp, groups):
    grouped = np.zeros((hp.shape[0], len(groups)))
    for i, g in enumerate(groups):
        grouped[:, i] = np.sum(hp[:, g], axis=1)
    grouped /= np.max(grouped, axis=1, keepdims=True)
    return grouped

# --- Standard prediction on unmodified test data ---
X_test_grouped = apply_grouping(X_test, valid_groups)
pos_pred = model.predict(X_test_grouped, batch_size=32)
abs_errors = np.linalg.norm(pos_pred - y_test, axis=1)
radial_errors = np.linalg.norm(pos_pred, axis=1) - np.linalg.norm(y_test, axis=1)

# --- 1. Gain fluctuations (2% Gaussian noise) ---
np.random.seed(42)
noisy_hp = X_test * np.random.normal(loc=1.0, scale=0.02, size=X_test.shape)
X_test_gain = apply_grouping(noisy_hp, valid_groups)
pos_pred_gain = model.predict(X_test_gain, batch_size=32)
abs_errors_gain = np.linalg.norm(pos_pred_gain - y_test, axis=1)
radial_errors_gain = np.linalg.norm(pos_pred_gain, axis=1) - np.linalg.norm(y_test, axis=1)

# --- 2. Missing PMTs ---
off_results = {}
off_pmt_indices = {}

for n_off in [1, 2, 5, 10]:
    np.random.seed(n_off)
    pmts_to_off = np.random.choice(X_test.shape[1], size=n_off, replace=False)
    X_mod = np.copy(X_test)
    X_mod[:, pmts_to_off] = 0
    X_test_off = apply_grouping(X_mod, valid_groups)

    pred = model.predict(X_test_off, batch_size=32)
    off_results[n_off] = {
        "pos": pred,
        "abs": np.linalg.norm(pred - y_test, axis=1),
        "rad": np.linalg.norm(pred, axis=1) - np.linalg.norm(y_test, axis=1)
    }
    off_pmt_indices[n_off] = pmts_to_off

# --- Save all results ---
save_dict = dict(
    pos_test=y_test,
    area_test=area_test,
    e_test=e_test,
    pos_pred=pos_pred,
    abs_errors=abs_errors,
    radial_errors=radial_errors,
    pos_pred_gain=pos_pred_gain,
    abs_errors_gain=abs_errors_gain,
    radial_errors_gain=radial_errors_gain,
)

for n_off, r in off_results.items():
    save_dict[f"pos_pred_off_{n_off}"] = r["pos"]
    save_dict[f"abs_errors_off_{n_off}"] = r["abs"]
    save_dict[f"radial_errors_off_{n_off}"] = r["rad"]
    save_dict[f"pmts_off_{n_off}"] = off_pmt_indices[n_off]

np.savez_compressed(analysis_save_path, **save_dict)
print(f"Saved analysis results to: {analysis_save_path}")
