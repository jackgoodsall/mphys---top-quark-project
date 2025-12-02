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

def apply_invariant_flip(predicted: np.ndarray, targets: np.ndarray):
    """
    Apply the same set/permutation-invariant criterion as set_invariant_loss,
    but in NumPy, to decide per event whether to flip the *predicted* particles.

    Args:
        predicted: (N, P, F) array
        targets:   (N, P, F) array

    Returns:
        predicted_aligned: (N, P, F) predicted, with some events flipped along axis=1
        flip_mask:         (N,) boolean array, True where we flipped
    """
    # Squared errors for original pairing
    diff_orig = (predicted - targets) ** 2                          # (N, P, F)
    diff_flip = (predicted - targets[:, ::-1, :]) ** 2              # (N, P, F)

    # Reduce over non-batch dims (particles + features) → per-event loss
    per_event_orig = diff_orig.mean(axis=(1, 2))                    # (N,)
    per_event_flip = diff_flip.mean(axis=(1, 2))                    # (N,)

    # If flipping targets gives lower loss, equivalently we should flip predicted
    flip_mask = per_event_flip < per_event_orig                     # (N,)

    predicted_aligned = predicted.copy()
    predicted_aligned[flip_mask] = predicted_aligned[flip_mask, ::-1, :]

    return predicted_aligned, flip_mask

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
    top_quark_mask = (abs(pid) == top_pids)                   # (N_sel, N_particles)

    # We assume exactly 2 tops per *selected* event:
    n_sel_events = particles_sel.shape[0]
    fourvec = fourvec[top_quark_mask].reshape(n_sel_events, 2, 4)  # (N_sel, 2, 4)
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
        title,
        fig_save_path
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
    rsquared = 1 - np.sum((X- Y)**2)/np.sum((Y - np.mean(Y))**2)
    fig.text(  -0.05,    0.65, f" R^2: {rsquared}", 
        va="center", ha="right")
    plt.tight_layout()
    fig.savefig(fig_save_path / f"{title}.png")

def plot_difference_hist(
    X,
    Y,
    n_bins,
    X_label,
    title,
    fig_save_path,
    **kwargs
):
    """
    Plot histogram of (Y - X), i.e. reco - truth.
    """
    X = X.reshape(-1)
    Y = Y.reshape(-1)

    diff = Y - X  # reco - truth

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.hist(
        diff,
        bins=n_bins,
        histtype="step",
    )

    ax.set_xlabel(X_label)
    ax.set_ylabel("Counts")
    ax.set_title(title)
    ax.grid(True, linestyle=":", alpha=0.7)

    plt.tight_layout()

    fig_save_path = Path(fig_save_path)
    fig_save_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_save_path / f"{title}.png", dpi = 1000)

    return fig, ax

def plot_reco_truth(
    X,
    Y,
    n_bins,
    X_label,
    Y_label,
    title,
    fig_save_path,
    fig_save_name,
    truth_label="Truth",
    reco_label="Reco",
):
    X = X.reshape(-1)
    Y = Y.reshape(-1)
    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1,
        figsize=(8, 6),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]}
    )

    # Consistent binning for both
    data_min = min(np.min(X), np.min(Y))
    data_max = max(np.max(X), np.max(Y))
    bins = np.linspace(data_min, data_max, n_bins + 1)

    # --- Top: histograms ---
    ax_top.hist(
        X,
        bins=bins,
        histtype="step",
        label=truth_label,
    )
    ax_top.hist(
        Y,
        bins=bins,
        histtype="step",
        label=reco_label,
    )

    ax_top.set_ylabel(Y_label)
    ax_top.set_title(title)
    ax_top.legend()
    ax_top.grid(True, which="both", linestyle=":", alpha=0.7)

    # --- Bottom: reco / truth ratio ---
    truth_counts, _ = np.histogram(X, bins=bins)
    reco_counts, _ = np.histogram(Y, bins=bins)

    # Avoid divide-by-zero: put NaN where truth is 0
    ratio = np.divide(
        reco_counts,
        truth_counts,
        out=np.full_like(reco_counts, np.nan, dtype=float),
        where=truth_counts != 0,
    )

    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    ax_bottom.step(bin_centers, ratio, where="mid")
    ax_bottom.axhline(1.0, linestyle="--", linewidth=1)
    ax_bottom.set_xlabel(X_label)
    ax_bottom.set_ylabel("Reco / Truth")
    ax_bottom.grid(True, which="both", linestyle=":", alpha=0.7)

    plt.tight_layout()

    fig_save_path = Path(fig_save_path)
    fig_save_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_save_path / f"{fig_save_name}.png", dpi = 1000)

    return fig, (ax_top, ax_bottom)

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
        coordinate_system = "polar",
        raw_predict_file_path= "data/topquarkreconstruction/h5py_data/ttbar_h5py_raw_test.h5",
        pid = 6,
        use_raw_targets_for_truth = True):
    
    if not Path(test_output_file_path).exists():
        raise FileNotFoundError(f"Test output file {test_output_file_path} does not exist")
    Path(report_file_dir).mkdir(exist_ok = True)

    with h5py.File(test_output_file_path, "r") as file_object:
        targets = file_object["targets"][()]

        predicted = file_object["predicted"][()]
    
    reverse_transformers = joblib.load(target_reverse_transformer_path)

    if use_raw_targets_for_truth:
        # Load targets from the raw file and reverse-transform them. 
        # This gives a physically meaningful "truth" comparison.
        targets_transformed = load_top_targets_with_event_selection(raw_predict_file_path, top_pids = pid)
    else:
        # Use targets directly from the model's test output file and reverse-transform them.
        # This compares the prediction to the *target* that went into the model before transformation.
        
        targets_transformed = reverse_transform_variables(targets, reverse_transformers)


    predicted_transformed = reverse_transform_variables(predicted, reverse_transformers)
    
    combined_targets = add_four_momenta(targets_transformed[:, 0, :], targets_transformed[:, 1, :], cord_sys= coordinate_system)
    combined_predicted = add_four_momenta(predicted_transformed[:, 0, :], predicted_transformed[:, 1, :], cord_sys= coordinate_system)

    if coordinate_system == "cart":
        
        targets_transformed = cartesian_to_polar(targets_transformed.reshape(-1, 4)).reshape(-1,2,4)
        predicted_transformed = cartesian_to_polar(predicted_transformed.reshape(-1, 4)).reshape(-1,2,4)


    predicted_transformed, _ = apply_invariant_flip(predicted_transformed, targets_transformed)

    plot_2d_histogram(targets_transformed[:, 0, 0], predicted_transformed[:, 0, 0],
                        200, "Targets (GeV)", "Predicted (GeV)", "Top 1 Pt Histogram", report_file_dir)
    
    plot_2d_histogram(targets_transformed[:, 0, 1], predicted_transformed[:, 0, 1],
                        200, "Targets ", "Predicted ", "Top 1 Eta Histogram", report_file_dir)
    plot_2d_histogram(targets_transformed[:, 0, 2], predicted_transformed[:, 0, 2],
                        200, "Targets", "Predicted", "Top 1 Phi Histogram", report_file_dir)
    plot_2d_histogram(targets_transformed[:, 0, 3], predicted_transformed[:, 0, 3],
                        200, "Targets (GeV)", "Predicted(GeV)", "Top 1 E Histogram", report_file_dir)
    plot_2d_histogram(targets_transformed[:, 1, 0], predicted_transformed[:, 1, 0],
                        200, "Targets (GeV)", "Predicted (GeV)", "Top 2 Pt Histogram", report_file_dir)
    plot_2d_histogram(targets_transformed[:, 1, 1], predicted_transformed[:, 1, 1],
                        200, "Targets ", "Predicted ", "Top 2 Eta Histogram", report_file_dir)
    plot_2d_histogram(targets_transformed[:, 1, 2], predicted_transformed[:, 1, 2],
                        200, "Targets ", "Predicted ", "Top 2 Phi Histogram", report_file_dir)
    plot_2d_histogram(targets_transformed[:, 1, 3], predicted_transformed[:, 1, 3],
                        200, "Targets (GeV)", "Predicted (GeV)", "Top 2 E Histogram", report_file_dir)
    
    plot_2d_histogram(combined_targets[..., 0], combined_predicted[..., 0],
                        200, "Targets (GeV)", "Predicted (GeV)", "Combined Pt", report_file_dir)
    plot_2d_histogram(combined_targets[..., 1], combined_predicted[..., 1],
                        200, "Targets ", "Predicted ", "Combined Eta", report_file_dir)
    plot_2d_histogram(combined_targets[..., 2], combined_predicted[..., 2],
                        200, "Targets ", "Predicted ", "Combined Phi", report_file_dir)
    plot_2d_histogram(combined_targets[..., 3], combined_predicted[..., 3],
                        200, "Targets (GeV)", "Predicted (GeV)", "Combined E", report_file_dir)
    
    plot_2d_histogram(combined_targets[..., 4], combined_predicted[..., 4],
                          200, "Targets (GeV)", "Predicted (GeV)", "Invariant mass", report_file_dir)

    plot_reco_truth(
    X=targets_transformed[..., 0,0],
    Y=predicted_transformed[..., 0,0],
    n_bins=200,
    X_label=r"$P_{t} (GeV)$",
    Y_label="Count",
    title=r"$P_{t}$",
    fig_save_path=report_file_dir,
    fig_save_name= "IndividualPt",
    truth_label="Truth",
    reco_label="Reco",
    )
    plot_reco_truth(
    X=targets_transformed[..., 1],
    Y=predicted_transformed[..., 1],
    n_bins=200,
    X_label=r"$/Eta$",
    Y_label="Count",
    title=r"$/Eta$",
    fig_save_path=report_file_dir,
    fig_save_name="InvidualEta",
    truth_label="Truth",
    reco_label="Reco",
    )
    plot_reco_truth(
    X=targets_transformed[..., 2],
    Y=predicted_transformed[..., 2],
    n_bins=200,
    X_label=r"$/Phi ^{\circ}$",
    Y_label="Count",
    title="Phi",
    fig_save_path=report_file_dir,
    fig_save_name="IndividualPhi",
    truth_label="Truth",
    reco_label="Reco",
    )

    plot_reco_truth(
    X=targets_transformed[..., 3],
    Y=predicted_transformed[..., 3],
    n_bins=200,
    X_label=r"Energy (GeV)",
    Y_label="Count",
    title= "Energy",
    fig_save_path=report_file_dir,
    fig_save_name="IndividualPt",
    truth_label="Truth",
    reco_label="Reco",
    )
    plot_difference_hist(
    X=targets_transformed[..., 0],
    Y=predicted_transformed[..., 0],
    n_bins=200,
    X_label=r"$P_t$ (GeV)",
    Y_label="Count",
    title=r"$P_t$ ",
    fig_save_path=report_file_dir,
    truth_label="Truth",
    reco_label="Reco",
    )
    plot_difference_hist(
    X=targets_transformed[..., 1],
    Y=predicted_transformed[..., 1],
    n_bins=200,
    X_label="Targets",
    Y_label="Count",
    title="Eta",
    fig_save_path=report_file_dir,
    truth_label="Truth",
    reco_label="Reco",
    )
    plot_difference_hist(
    X=targets_transformed[..., 2],
    Y=predicted_transformed[..., 2],
    n_bins=200,
    X_label="Targets",
    Y_label="Predicted",
    title="Phi",
    fig_save_path=report_file_dir,
    truth_label="Truth",
    reco_label="Reco",
    )

    plot_difference_hist(
    X=targets_transformed[..., 3],
    Y=predicted_transformed[..., 3],
    n_bins=200,
    X_label="Targets (GeV)",
    Y_label="Predicted (GeV)",
    title="E",
    fig_save_path=report_file_dir,
    truth_label="Truth",
    reco_label="Reco",
    )

    # Combined
    plot_reco_truth(
    X=combined_targets[..., 0],
    Y=combined_predicted[..., 0],
    n_bins=200,
    X_label=r"$P_{t}$ (GeV)",
    Y_label="Count",
    title=r"$P_{t}$",
    fig_save_path=report_file_dir,
    fig_save_name="Combinedpt",
    truth_label="Truth",
    reco_label="Reco",
    )
    plot_reco_truth(
    X=combined_targets[..., 1],
    Y=combined_predicted[..., 1],
    n_bins=200,
    X_label=r"$/Eta$",
    Y_label="Count",
    title=r"$/Eta$",
    fig_save_path=report_file_dir,
    fig_save_name="Combined eta",
    truth_label="Truth",
    reco_label="Reco",
    )
    
    plot_reco_truth(
    X=combined_targets[..., 2],
    Y=combined_predicted[..., 2],
    n_bins=200,
    X_label=r"$/Phi ^{\circ}$",
    Y_label="Count",
    title="Phi",
    fig_save_path=report_file_dir,
    fig_save_name="CombinedPhi",
    truth_label="Truth",
    reco_label="Reco",
    )
    plot_reco_truth(
    X=combined_targets[..., 3],
    Y=combined_predicted[..., 3],
    n_bins=200,
    X_label=r"$Energy (GeV)$",
    Y_label="Count",
    title= "Energy",
    fig_save_path=report_file_dir,
    fig_save_name="Combinedenergy",
    truth_label="Truth",
    reco_label="Reco",
    )
    plot_reco_truth(
    X=combined_targets[..., 4],
    Y=combined_predicted[..., 4],
    n_bins=200,
    X_label=r"$M_{t\bar{t}} (GeV)$",
    Y_label="Count",
    title="Invariant Mass",
    fig_save_path=report_file_dir,
    fig_save_name="invmass",
    truth_label="Truth",
    reco_label="Reco",
    )

  