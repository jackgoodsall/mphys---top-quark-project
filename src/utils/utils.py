import yaml
from pydantic import BaseModel
from typing import Dict, Any
from pathlib import Path
import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import vector
import sys
import joblib
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.data_utls.scalers import *

## Define some config models so that we have a first layer file 
## on passing bricked configs
class DataConfig(BaseModel):
    train: Dict[str, Any]
    validation: Dict[str, Any]
    test: Dict[str, Any]
### Can add even further structued configs if want
class BaseConfig(BaseModel):
    data: DataConfig
    model_parameters: Dict[str, Any]
    train: Dict[str, Any]
    data_pipeline: Dict[str, Any]

def load_and_split_config(config_input_file: str) -> BaseConfig:

    with open(config_input_file, "r") as f:
        raw_config = yaml.safe_load(f)
    cfg = BaseConfig(**raw_config)
    return cfg

def load_any_config(config_input_file: str) -> Dict[str, Any]:
    with open(config_input_file, "r") as f:
        raw_config = yaml.safe_load(f)
    return raw_config

def reverse_transform_variables(
        X,
        reverse_transform_tuple
):
    """
    Takes in the variables and a reverse transformer, and returns the 
    reversered transformation of the variables.
    """
    parts = []
    off_set = 0
    for number, transformer in enumerate(reverse_transform_tuple):
        if isinstance(transformer, PhiTransformer):
            sin_phi = X[:, :, number]
            cos_phi = X[:, :, number+1]
            phi = np.arctan2(sin_phi, cos_phi).reshape(-1 , 2, 1)
            print(phi.shape)
            
            
            parts.append(phi)
            off_set += 1
        else:
            parts.append(transformer.inverse_transform(X[..., number + off_set].reshape(-1, 1)).reshape( -1, 2, 1))
    parts = np.concatenate(parts, axis = 2)
    return parts

import h5py
import numpy as np

def load_top_targets_with_event_selection(
    original_data_file_path,
    particles_dataset="targets",    # dataset name for per-particle info
    event_dataset="event",       # dataset name for event-level info
    pid_index=-1,                     # index of PID within the last axis
    fourvec_slice=slice(0, 4),        # where the 4-vector lives in features
    top_pids = 6,                 # PDG IDs for t and t̄
    all_W_matched_index=2,            # index of "all W matched" flag in event_data
):
    """
    Load top-quark four-vectors for events where 'all W matched' == 1.

    Returns
    -------
    targets : np.ndarray, shape (n_sel_events, 2, 4)
        Four-momenta of the two tops per selected event.
    event_mask : np.ndarray, shape (n_events,)
        Boolean mask: True for events with all_W_matched == 1.
    """
    with h5py.File(original_data_file_path, "r") as f:
        particles = f[particles_dataset][()]   # (N_events, N_particles, N_features)
        event_data = f[event_dataset][()]      # (N_events, N_event_features)

    # Build event mask: all_W_matched == 1 at index 2
    event_mask = (event_data[:, all_W_matched_index] == 1)

    # Apply event selection to particle data
    particles_sel = particles[event_mask]      # (N_sel, N_particles, N_features)

    # Split out PID and four-vectors
    pid     = particles_sel[..., pid_index].astype(int)   # (N_sel, N_particles)
    fourvec = particles_sel[..., fourvec_slice]           # (N_sel, N_particles, 4)

    # Mask for top quarks
    top_quark_mask = (abs(pid) == 6)                   # (N_sel, N_particles)

    # We assume exactly 2 tops per *selected* event:
    n_sel_events = particles_sel.shape[0]
    tops = fourvec[top_quark_mask].reshape(n_sel_events, 2, 4)  # (N_sel, 2, 4)
    # fourvec → vector array (Cartesian)
    v = vector.array({
        "pt": fourvec[..., 0],
        "eta": fourvec[..., 1],
        "phi": fourvec[..., 2],
        "mass":  fourvec[..., 3],
    })

    # Convert to pt, eta, phi, E
    pt  = v.pt
    eta = v.eta
    phi = v.phi
    E   = v.E

    # Pack back into same layout
    fourvec_polar = np.stack((pt, eta, phi, E), axis=-1)

    return fourvec_polar




def add_four_momenta(p1, p2, cord_sys="polar"):
    """
    Add two 4-vectors using the scikit-hep 'vector' library.
    Input is (..., 4): either (pt, eta, phi, E) or (E, px, py, pz).
    Returns (..., 5): (pt, eta, phi, E, inv_mass).
    """

    p1 = np.asarray(p1)
    p2 = np.asarray(p2)

    if cord_sys == "polar":
        # vector likes named arguments, so we spread the columns
        v1 = vector.array({
            "pt":  p1[..., 0],
            "eta": p1[..., 1],
            "phi": p1[..., 2],
            "E":   p1[..., 3],
        })
        v2 = vector.array({
            "pt":  p2[..., 0],
            "eta": p2[..., 1],
            "phi": p2[..., 2],
            "E":   p2[..., 3],
        })

    else:  # Cartesian
        v1 = vector.array({
            "E":  p1[..., 0],
            "px": p1[..., 1],
            "py": p1[..., 2],
            "pz": p1[..., 3],
        })
        v2 = vector.array({
            "E":  p2[..., 0],
            "px": p2[..., 1],
            "py": p2[..., 2],
            "pz": p2[..., 3],
        })

    v = v1 + v2

    # Extract back to your preferred representation
    pt  = v.pt
    eta = v.eta
    phi = v.phi
    E   = v.E

    # vector handles m² exactly as you'd expect
    inv_mass = v.mass

    return np.stack((pt, eta, phi, E, inv_mass), axis=-1)

def plot_2d_histogram(
        X,
        Y,
        n_bins,
        X_label,
        Y_label,
        title
):  
    fig  = plt.figure()
    X_plot_range = (np.min(X), np.max(X))
    Y_plot_range = (np.min(Y), np.max(Y))
    h = plt.hist2d(
        X,
        Y,
        bins=n_bins,
        range=[X_plot_range, Y_plot_range],
        cmap="viridis",
        norm=LogNorm(),
    )
    plt.colorbar(h[3])
    plt.xlabel(X_label)
    plt.ylabel(Y_label)
    plt.title(title)
    plt.plot(X_plot_range, X_plot_range, "r--", linewidth=1)
    rmse = np.sqrt(np.mean((X - Y)**2))
    mae = np.mean(np.abs(X - Y))
    perc_error = np.mean((X - Y)/ Y)
    fig.text(  -0.05,   0.5, f"RMSE: {rmse}", 
        va="center", ha="right")
    fig.text(  -0.05,    0.55, f"MAE: {mae}", 
        va="center", ha="right")
    fig.text(  -0.05,    0.6, f"% Error: {perc_error * 100}", 
        va="center", ha="right")
    plt.tight_layout()
    plt.show()

def cartesian_to_polar(fourvec, eps=1e-9):
    px = fourvec[:, 1]
    py = fourvec[:, 2]
    pz = fourvec[:, 3]
    E  = fourvec[:, 0]

    pt  = np.sqrt(px**2 + py**2)
    phi = np.arctan2(py, px)

    p   = np.sqrt(px**2 + py**2 + pz**2)
    # avoid division-by-zero / log(0) nastiness
    eta = 0.5 * np.log((p + pz ) / (p - pz))

    return np.column_stack((pt, eta, phi, E))

def generate_reconstruction_report(
        test_output_file_path,
        report_file_dir,
        target_reverse_transformer_path,
        report_file_name = "reconstruction_report",
        coordinate_system = "polar"):
    
    if not Path(test_output_file_path).exists():
        raise FileNotFoundError(f"Test output file {test_output_file_path} does not exist")
    Path(report_file_dir).mkdir(exist_ok = True)

    with h5py.File(test_output_file_path, "r") as file_object:
        targets = file_object["targets"][()]
        predicted = file_object["predicted"][()]
    
    reverse_transformers = joblib.load(target_reverse_transformer_path)

    targets_transformed = load_top_targets_with_event_selection("../data/topquarkreconstruction/h5py_data/ttbar_h5py_raw_test.h5")
    predicted_transformed = reverse_transform_variables(predicted, reverse_transformers)
   
    combined_targets = add_four_momenta(targets_transformed[:, 0, :], targets_transformed[:, 1, :], cord_sys= coordinate_system)
    combined_predicted = add_four_momenta(predicted_transformed[:, 0, :], predicted_transformed[:, 1, :], cord_sys= coordinate_system)

    if coordinate_system == "cart":
        
        targets_transformed = cartesian_to_polar(targets_transformed.reshape(-1, 4)).reshape(-1,2,4)
        predicted_transformed = cartesian_to_polar(predicted_transformed.reshape(-1, 4)).reshape(-1,2,4)

    plot_2d_histogram(targets_transformed[:, 0, 0], predicted_transformed[:, 0, 0],
                        200, "Targets (GeV)", "Predicted (GeV)", "Top 1 Pt Histogram")
    plot_2d_histogram(targets_transformed[:, 0, 1], predicted_transformed[:, 0, 1],
                        200, "Targets (GeV)", "Predicted (GeV)", "Top 1 Eta Histogram")
    plot_2d_histogram(targets_transformed[:, 0, 2], predicted_transformed[:, 0, 2],
                        200, "Targets (GeV)", "Predicted (GeV)", "Top 1 Phi Histogram")
    plot_2d_histogram(targets_transformed[:, 0, 3], predicted_transformed[:, 0, 3],
                        200, "Targets (GeV)", "Predicted (GeV)", "Top 1 E Histogram")
    plot_2d_histogram(targets_transformed[:, 1, 0], predicted_transformed[:, 1, 0],
                        200, "Targets (GeV)", "Predicted (GeV)", "Top 2 Pt Histogram")
    plot_2d_histogram(targets_transformed[:, 1, 1], predicted_transformed[:, 1, 1],
                        200, "Targets (GeV)", "Predicted (GeV)", "Top 2 Eta Histogram")
    plot_2d_histogram(targets_transformed[:, 1, 2], predicted_transformed[:, 1, 2],
                        200, "Targets (GeV)", "Predicted (GeV)", "Top 2 Phi Histogram")
    plot_2d_histogram(targets_transformed[:, 1, 3], predicted_transformed[:, 1, 3],
                        200, "Targets (GeV)", "Predicted (GeV)", "Top 2 E Histogram")
    
    plot_2d_histogram(combined_targets[..., 0], combined_predicted[..., 0],
                        200, "Targets (GeV)", "Predicted (GeV)", "Combined Pt")
    plot_2d_histogram(combined_targets[..., 1], combined_predicted[..., 1],
                        200, "Targets (GeV)", "Predicted (GeV)", "Combined Et")
    plot_2d_histogram(combined_targets[..., 2], combined_predicted[..., 2],
                        200, "Targets (GeV)", "Predicted (GeV)", "Combined Phi")
    plot_2d_histogram(combined_targets[..., 3], combined_predicted[..., 3],
                        200, "Targets (GeV)", "Predicted (GeV)", "Combined E")
    
    plot_2d_histogram(combined_targets[..., 4], combined_predicted[..., 4],
                          200, "Targets (GeV)", "Predicted (GeV)", "Combined Invariant mass")

        
        
  