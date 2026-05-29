# Shared constants used across data, model, and training modules.

# Object class labels (used in matching cost, loss, and HDF5 outputs)
CLASS_NULL = 0  # padding / unmatched query slot
CLASS_TOP = 1   # top quark
CLASS_W = 2     # W boson

# Key used to inject matched targets into per-layer output dicts
TARGETS_KEY = "__targets__"
