import numpy as np

def apply_mask(arrays, mask):
    # Return a tuple, not a generator (easier to debug/use)
    return tuple(a[mask, ...] for a in arrays)


def calculate_energy_value( array ): 
    """
      Takes in an array of shape (pt, eta, phi, M) and returns M 
      """
    mass = array[... , 3] 
    p = array[... , 0] * np.cosh(array[..., 1]) 
    return np.sqrt(mass**2 + p**2)


# ---- Kinematics: from (pt, eta, phi, m) ----
def energy_from_pt_eta_m(X):
    """X[..., :] = (pt, eta, phi, m) -> E"""
    pt  = X[..., 0]
    eta = X[..., 1]
    m   = X[..., 3]
    p   = pt * np.cosh(eta)
    # E = sqrt(p^2 + m^2)
    return np.sqrt(np.maximum(p**2 + m**2, 0.0))

# ---- Kinematics: from (pt, eta, phi, E) ----
def mass_from_pt_eta_E(X):
    """X[..., :] = (pt, eta, phi, E) -> m"""
    pt  = X[..., 0]
    eta = X[..., 1]
    E   = X[..., 3]
    p   = pt * np.cosh(eta)
    # m = sqrt(E^2 - p^2) (guard negatives)
    return np.sqrt(np.maximum(E**2 - p**2, 0.0))

# ---- Cartesian components ----
def px_py_pz_from_pt_eta_phi(X):
    """X[..., :] = (pt, eta, phi, *) -> (px, py, pz)"""
    pt  = X[..., 0]
    eta = X[..., 1]
    phi = X[..., 2]
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    return px, py, pz


def convert_polar_to_cartesian(
    X,
    *,
    include_mass: bool = False,
    eta_clip: float = 100,
    pt_clip = None,
    pad_value: float = np.nan,
):
    """
    X[..., :] = (pt, eta, phi, m)
    Returns (..., 4) if include_mass=False: (E, px, py, pz)
            (..., 5) if include_mass=True : (E, px, py, pz, m)
    """
   
    pt  = X[..., 0]
    eta = np.clip(X[..., 1], -eta_clip, eta_clip)  # tame sinh/cosh
    phi = X[..., 2]
    m   = X[..., 3]

    if pt_clip is not None:
        pt = np.clip(pt, 0.0, pt_clip)

    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    P  = pt * np.cosh(eta)
    E  = np.hypot(P, m)  # stable sqrt(P**2 + m**2)

    if include_mass:
        Y = np.stack([E, px, py, pz, m], axis=-1)   # (..., 5)
    else:
        Y = np.stack([E, px, py, pz], axis=-1)      # (..., 4)

    # Replace any non-finite leftovers
    bad = ~np.isfinite(Y)
    if bad.any():
        Y = Y.copy()
        Y[bad] = pad_value
    return Y

import numpy as np

import numpy as np


def delta_phi(phi1, phi2):
    """
    Compute Δφ in range [-π, π] with broadcasting.
    phi1, phi2 can be arrays with any broadcastable shape.
    """
    dphi = phi1 - phi2
    return (dphi + np.pi) % (2 * np.pi) - np.pi


def create_interaction_matrix(jet_data, number_features=4):
    """
    jet_data: (N, P, 4) array with features [pt, eta, phi, E]
    src_mask: (N, P) boolean mask, True = masked/invalid jet
    returns: (N, P, P, 4) interaction features [ΔR, kT, z, m²]
    """
    N, P, F = jet_data.shape
    assert F >= 4, "jet_data last dim must contain at least [pt, eta, phi, E]"
    src_mask = np.isnan(jet_data).any(axis=-1)   # (N, P)

    # Unpack features
    pt = jet_data[..., 0]
    eta = jet_data[..., 1]
    phi = jet_data[..., 2]
    E = jet_data[..., 3]

    # Expand dims for pairwise broadcasting: (N, P, 1) vs (N, 1, P)
    pt_i = pt[:, :, None]    # (N, P, 1)
    pt_j = pt[:, None, :]    # (N, 1, P)

    eta_i = eta[:, :, None]
    eta_j = eta[:, None, :]

    phi_i = phi[:, :, None]
    phi_j = phi[:, None, :]

    E_i = E[:, :, None]
    E_j = E[:, None, :]

    # Pairwise distances
    d_eta = eta_i - eta_j
    d_phi = delta_phi(phi_i, phi_j)
    delta_r = np.sqrt(d_eta**2 + d_phi**2)    # (N, P, P)

    # kT and z
    pt_min = np.minimum(pt_i, pt_j)
    kT = pt_min * delta_r
    z = pt_min / (pt_i + pt_j + 1e-12)        # tiny epsilon to avoid 0/0

    # Build 3-momenta from (pt, eta, phi)
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)

    px_i = px[:, :, None]
    py_i = py[:, :, None]
    pz_i = pz[:, :, None]

    px_j = px[:, None, :]
    py_j = py[:, None, :]
    pz_j = pz[:, None, :]

    # Pairwise sums
    px_pair = px_i + px_j
    py_pair = py_i + py_j
    pz_pair = pz_i + pz_j
    E_pair = E_i + E_j

    p2_pair = px_pair**2 + py_pair**2 + pz_pair**2
    m2 = E_pair**2 - p2_pair

    # Stack into (N, P, P, 4)
    interaction_features = np.stack([delta_r, kT, z, m2], axis=-1)

    # Apply mask: any pair where jet i or j is masked -> set to NaN
    # src_mask: (N, P)
    mask_i = src_mask[:, :, None]   # (N, P, 1)
    mask_j = src_mask[:, None, :]   # (N, 1, P)
    pair_mask = mask_i | mask_j     # (N, P, P)

    interaction_features[pair_mask] = np.nan

    return interaction_features

