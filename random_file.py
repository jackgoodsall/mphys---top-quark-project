import uproot
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import vector

data_file_path = "/mnt/iusers01/fse-ugpgt01/phy01/b58521jg/masters_project/introduction_work/transformers/Nov13/4tops_withtruth_13Nov.root"

vector.register_awkward()
file = uproot.open(data_file_path)


def make_train_val_test_indices(N, seed=123):
    rng = np.random.default_rng(seed)

    # shuffled event indices
    idx = rng.permutation(N)

    # split sizes
    n_train = int(0.90 * N)
    n_val   = int(0.05 * N)
    n_test  = N - n_train - n_val   # avoids rounding drift

    train_idx = idx[:n_train]
    val_idx   = idx[n_train:n_train + n_val]
    test_idx  = idx[n_train + n_val:]

    return train_idx, val_idx, test_idx


# ---------------------------------------------------
# Reco: nel_pt filter (same as you had)
# ---------------------------------------------------
reco_tree  = file["Reco;1"]
truth_tree = file["Truth;1"]

nel_pt_array = reco_tree["nel_pt"].array()
good_mask_global = nel_pt_array == 0

print("Reco branches:", reco_tree.keys())
print("Truth branches:", truth_tree.keys())

# Reco jet branches and Truth top branches
reco_keys  = ["jet_pt", "jet_eta", "jet_phi", "jet_mass", "jet_btag"]
truth_keys = ["top_pt", "top_eta", "top_phi", "top_mass"]

all_interaction_features = []
all_jet_masses = []

# store truth tops too
truth_top_pt_list, truth_top_eta_list, truth_top_phi_list,  truth_top_E_list, truth_top_PID_list = [], [], [], [], []
# store jets
jet_pt_list, jet_eta_list, jet_phi_list, jet_mass_list, jet_E_list, jet_btag_list = [], [], [], [], [], []


event_index = 0

reco_iter  = reco_tree.iterate(reco_keys,  how="zip", step_size=1000_000)
truth_iter = truth_tree.iterate(truth_keys, how="zip", step_size=1000_000)

for reco_batch, truth_batch in zip(reco_iter, truth_iter):
    # ----------------------------------------
    # Reco jets
    # ----------------------------------------
    reco_batch  =reco_batch["jet"]
    jet_pt   = reco_batch["pt"]
    jet_eta  = reco_batch["eta"]
    jet_phi  = reco_batch["phi"]
    jet_mass = reco_batch["mass"]
    jet_btag = reco_batch["btag"]

    batch_size = len(jet_pt)
    batch_mask = good_mask_global[event_index:event_index + batch_size]
    event_index += batch_size

    # Filter by nel_pt == 0
    jet_pt   = jet_pt[batch_mask]
    jet_eta  = jet_eta[batch_mask]
    jet_phi  = jet_phi[batch_mask]
    jet_mass = jet_mass[batch_mask]
    jet_btag = jet_btag[batch_mask]

    if len(jet_pt) == 0:
        continue

    # Pad jagged arrays to fixed size (assume max 25 jets per event)
    jet_pt   = ak.pad_none(jet_pt,   25, axis=1, clip=True)
    jet_eta  = ak.pad_none(jet_eta,  25, axis=1, clip=True)
    jet_phi  = ak.pad_none(jet_phi,  25, axis=1, clip=True)
    jet_mass = ak.pad_none(jet_mass, 25, axis=1, clip=True)
    jet_btag = ak.pad_none(jet_mass, 25 , axis = 1 , clip = True)

    # Fill None values with NaN
    jet_pt   = ak.fill_none(jet_pt,   np.nan)
    jet_eta  = ak.fill_none(jet_eta,  np.nan)
    jet_phi  = ak.fill_none(jet_phi,  np.nan)
    jet_mass = ak.fill_none(jet_mass, np.nan)
    jet_btag = ak.fill_none(jet_btag, np.nan)

    # Convert to numpy and ensure float64
    jet_pt   = np.array(ak.to_numpy(jet_pt),   dtype="float64")
    jet_eta  = np.array(ak.to_numpy(jet_eta),  dtype="float64")
    jet_phi  = np.array(ak.to_numpy(jet_phi),  dtype="float64")
    jet_mass = np.array(ak.to_numpy(jet_mass), dtype="float64")
    jet_btag = np.array(ak.to_numpy(jet_btag), dtype="float64")


    # Energy for jets: E = sqrt(p^2 + m^2), p = pt * cosh(eta)
    p = jet_pt * np.cosh(jet_eta)
    jet_E = np.sqrt(p**2 + jet_mass**2)

    jet_pt_list.append(jet_pt)
    jet_eta_list.append(jet_eta)
    jet_phi_list.append(jet_phi)
    jet_mass_list.append(jet_mass)
    jet_E_list.append(jet_E)
    jet_btag_list.append(jet_btag)
    # ----------------------------------------
    # Truth tops (same mask to keep alignment)
    # ----------------------------------------
    truth_batch = truth_batch["top"]
    top_pt   = truth_batch["pt"][batch_mask]
    top_eta  = truth_batch["eta"][batch_mask]
    top_phi  = truth_batch["phi"][batch_mask]
    top_mass = truth_batch["mass"][batch_mask]
    top_pid = ak.full_like(top_mass, 6)


    # Compute top energy the same way
    top_p = top_pt * np.cosh(top_eta)
    top_E = np.sqrt(top_p**2 + top_mass**2)

    truth_top_pt_list.append(top_pt)
    truth_top_eta_list.append(top_eta)
    truth_top_phi_list.append(top_phi)
    truth_top_E_list.append(top_E)
    truth_top_PID_list.append(top_pid)

# ---------------------------------------------------
# Truth tops final awkward arrays
# ---------------------------------------------------
truth_top_pt   = ak.concatenate(truth_top_pt_list)   if truth_top_pt_list   else ak.Array([])
truth_top_eta  = ak.concatenate(truth_top_eta_list)  if truth_top_eta_list  else ak.Array([])
truth_top_phi  = ak.concatenate(truth_top_phi_list)  if truth_top_phi_list  else ak.Array([])
truth_top_E    = ak.concatenate(truth_top_E_list)    if truth_top_E_list    else ak.Array([])
truth_top_PID    = ak.concatenate(truth_top_PID_list)    if truth_top_E_list    else ak.Array([])


truth_top_pt   = truth_top_pt[..., np.newaxis]
truth_top_eta  = truth_top_eta[..., np.newaxis]
truth_top_phi  = truth_top_phi[..., np.newaxis]
truth_top_E    = truth_top_E[..., np.newaxis]
truth_top_PID  = truth_top_PID[..., np.newaxis]

truth_top_info = ak.concatenate([truth_top_pt, truth_top_eta, truth_top_phi, truth_top_E, truth_top_PID], axis = -1 )


# ---------------------------------------------------
# Jets final awkward arrays
# ---------------------------------------------------
jet_pt_all   = ak.concatenate(jet_pt_list)   if jet_pt_list   else ak.Array([])
jet_eta_all  = ak.concatenate(jet_eta_list)  if jet_eta_list  else ak.Array([])
jet_phi_all  = ak.concatenate(jet_phi_list)  if jet_phi_list  else ak.Array([])
jet_mass_all = ak.concatenate(jet_mass_list) if jet_mass_list else ak.Array([])
jet_E_all    = ak.concatenate(jet_E_list)    if jet_E_list    else ak.Array([])
jet_Btag_all    = ak.concatenate(jet_btag_list)    if jet_E_list    else ak.Array([])

# optional: replace None with NaN after concatenation
jet_pt_all   = ak.fill_none(jet_pt_all,   np.nan)
jet_eta_all  = ak.fill_none(jet_eta_all,  np.nan)
jet_phi_all  = ak.fill_none(jet_phi_all,  np.nan)
jet_mass_all = ak.fill_none(jet_mass_all, np.nan)
jet_E_all    = ak.fill_none(jet_E_all,    np.nan)
jet_Btag_all    = ak.fill_none(jet_Btag_all,    np.nan)



jet_pt_all   = jet_pt_all[...,   np.newaxis]
jet_eta_all  = jet_eta_all[...,  np.newaxis]
jet_phi_all  = jet_phi_all[...,  np.newaxis]
jet_mass_all = jet_mass_all[..., np.newaxis]
jet_E_all    = jet_E_all[...,    np.newaxis]
jet_Btag_all    = jet_Btag_all[...,    np.newaxis]

jet_info = ak.concatenate(
    [jet_pt_all, jet_eta_all, jet_phi_all, jet_mass_all, jet_E_all, jet_Btag_all],
    axis=-1
)
event_data = ak.ones_like(jet_info)[: , 0, :3]

print("\nTruth tops (awkward types):")
print("  pt  :", truth_top_pt.type)
print("  eta :", truth_top_eta.type)
print("  phi :", truth_top_phi.type)
print("  E   :", truth_top_E.type)

print(np.asarray(truth_top_info).shape)
print(np.asarray(jet_info).shape)
print(np.asarray(event_data).shape)


import h5py

# ---------------------------------------------------
# Convert to NumPy
# ---------------------------------------------------
jets_np    = np.asarray(jet_info)         # shape: (N, n_jets, jet_features)
targets_np = np.asarray(truth_top_info)   # shape: (N, n_tops, top_features)
event_np   = np.asarray(event_data)       # shape: (N, event_features)

# Sanity check: same number of events
assert jets_np.shape[0] == targets_np.shape[0] == event_np.shape[0]
N = jets_np.shape[0]
print("Total events:", N)

# ---------------------------------------------------
# Make train / val / test indices (90 / 5 / 5)
# ---------------------------------------------------
train_idx, val_idx, test_idx = make_train_val_test_indices(N, seed=123)

print("Train events:", len(train_idx))

print("Val events  :", len(val_idx))
print("Test events :", len(test_idx))

# ---------------------------------------------------
# Helper to save one split
# ---------------------------------------------------
def save_split(filename, idx):
    with h5py.File(filename, "w") as f:
        f.create_dataset("jets",    data=jets_np[idx],    compression="gzip")
        f.create_dataset("targets", data=targets_np[idx], compression="gzip")
        f.create_dataset("event",   data=event_np[idx],   compression="gzip")
    print(f"Saved {filename} with {len(idx)} events")

# ---------------------------------------------------
# Save the three splits
# ---------------------------------------------------
save_split("train.h5", train_idx)
save_split("val.h5",   val_idx)
save_split("test.h5",  test_idx)


import h5py

# ---------------------------------------------------
# Convert to NumPy
# ---------------------------------------------------
jets_np    = np.asarray(jet_info)
targets_np = np.asarray(truth_top_info)
event_np   = np.asarray(event_data)

assert jets_np.shape[0] == targets_np.shape[0] == event_np.shape[0]
N = jets_np.shape[0]

# ---------------------------------------------------
# Make split indices
# ---------------------------------------------------
train_idx, val_idx, test_idx = make_train_val_test_indices(N, seed=123)

# ---------------------------------------------------
# Helper to save a split with compression level 4
# ---------------------------------------------------
def save_split(filename, idx):
    with h5py.File(filename, "w") as f:
        f.create_dataset(
            "jet",
            data=jets_np[idx],
            compression="gzip",
            compression_opts=4,
        )
        f.create_dataset(
            "targets",
            data=targets_np[idx],
            compression="gzip",
            compression_opts=4,
        )
        f.create_dataset(
            "event",
            data=event_np[idx],
            compression="gzip",
            compression_opts=4,
        )
    print(f"Saved {filename} ({len(idx)} events, gzip level 4)")

# ---------------------------------------------------
# Save the three splits
# ---------------------------------------------------
save_split("train.h5", train_idx)
save_split("val.h5",   val_idx)
save_split("test.h5",  test_idx)
