### Copied from https://github.com/els285/MancHEP-AI/blob/main/src/data/DataScaler.py

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, FunctionTransformer



def safe_transform(X, func):
    """Apply func elementwise, preserving NaNs."""
    X = np.asarray(X, dtype=float)
    out = np.empty_like(X)
    mask = np.isnan(X)
    out[~mask] = func(X[~mask])
    out[mask] = np.nan
    return out

class LogMinMax(BaseEstimator, TransformerMixin):
    """
    Logarithmic Min-Max Scaler with partial_fit support and NaN safety.
    Scales features as:
        X_scaled = (log1p(X) - log_min) / (log_max - log_min)
    Keeps NaN values as NaN.
    """

    def __init__(self, feature_range=(0, 1), clip=False):
        self.feature_range = feature_range
        self.clip = clip
        self.log_min_ = None
        self.log_max_ = None
        self.scale_ = None
        self.min_offset_ = None

    def _log_transform(self, X):
        # elementwise log1p, preserves NaN
        X_pos = np.clip(X, a_min=0, a_max=None)
        return np.log1p(X_pos)

    def fit(self, X, y=None):
        self.log_min_ = None
        self.log_max_ = None
        return self.partial_fit(X, y)

    def partial_fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        X_log = self._log_transform(X)

        # compute batch min/max ignoring NaNs
        batch_min = np.nanmin(X_log, axis=0)
        batch_max = np.nanmax(X_log, axis=0)

        if self.log_min_ is None:
            self.log_min_ = batch_min
            self.log_max_ = batch_max
        else:
            self.log_min_ = np.minimum(self.log_min_, batch_min)
            self.log_max_ = np.maximum(self.log_max_, batch_max)

        # handle potential zero-division
        scale = (self.feature_range[1] - self.feature_range[0]) / np.maximum(
            self.log_max_ - self.log_min_, 1e-12
        )
        min_offset = self.feature_range[0] - self.log_min_ * scale

        self.scale_ = scale
        self.min_offset_ = min_offset

        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        mask = np.isnan(X)

        X_log = self._log_transform(X)
        X_scaled = X_log * self.scale_ + self.min_offset_

        if self.clip:
            low, high = self.feature_range
            X_scaled = np.clip(X_scaled, low, high)

        X_scaled[mask] = np.nan  # reapply NaN mask
        return X_scaled

    def inverse_transform(self, X_scaled):
        X_scaled = np.asarray(X_scaled, dtype=np.float64)
        mask = np.isnan(X_scaled)

        X_log = (X_scaled - self.min_offset_) / self.scale_
        X = np.expm1(X_log)

        X[mask] = np.nan  # restore NaNs
        return X


class LogMinMaxScaler(BaseEstimator, TransformerMixin):
    """
    Logarithmic Min-Max Scaler with partial_fit support and NaN safety.
    Scales features as:
        X_scaled = (log1p(X) - log_min) / (log_max - log_min)
    Keeps NaN values as NaN.
    """

    def __init__(self, feature_range=(0, 1), clip=False):
        self.feature_range = feature_range
        self.clip = clip
        self.log_min_ = None
        self.log_max_ = None
        self.scale_ = None
        self.min_offset_ = None
        self.scalar = StandardScaler()
        self._bounds_frozen = False

    def _log_transform(self, X):
        # elementwise log1p, preserves NaN
        X_pos = np.clip(X, a_min=0, a_max=None)
        return np.log1p(X_pos)

    def fit(self, X, y=None):
        self.log_min_ = None
        self.log_max_ = None
        self.scalar = StandardScaler()
        self._bounds_frozen = False
        self.partial_fit_bounds(X)
        self.freeze_bounds()
        return self.partial_fit_standardization(X)

    def partial_fit(self, X, y=None):
        """Fit a complete single batch.

        Streaming callers must use ``partial_fit_bounds`` over every batch,
        freeze, then ``partial_fit_standardization`` over every batch.  Refuse
        the old order-dependent update instead of silently reproducing it.
        """
        if self.log_min_ is not None or hasattr(self.scalar, "n_samples_seen_"):
            raise RuntimeError(
                "streaming LogMinMaxScaler fitting requires the two-pass API: "
                "partial_fit_bounds -> freeze_bounds -> partial_fit_standardization"
            )
        return self.fit(X, y)

    def partial_fit_bounds(self, X, y=None):
        if self._bounds_frozen:
            raise RuntimeError("log-space bounds are already frozen")
        X = np.asarray(X, dtype=np.float64)
        X_log = self._log_transform(X)
        batch_min = np.nanmin(X_log, axis=0)
        batch_max = np.nanmax(X_log, axis=0)
        if self.log_min_ is None:
            self.log_min_ = batch_min
            self.log_max_ = batch_max
        else:
            self.log_min_ = np.minimum(self.log_min_, batch_min)
            self.log_max_ = np.maximum(self.log_max_, batch_max)

        # handle potential zero-division
        scale = (self.feature_range[1] - self.feature_range[0]) / np.maximum(
            self.log_max_ - self.log_min_, 1e-12
        )
        min_offset = self.feature_range[0] - self.log_min_ * scale

        self.scale_ = scale
        self.min_offset_ = min_offset
        return self

    def freeze_bounds(self):
        if self.log_min_ is None:
            raise RuntimeError("cannot freeze bounds before fitting them")
        self._bounds_frozen = True
        return self

    def partial_fit_standardization(self, X, y=None):
        if not self._bounds_frozen:
            raise RuntimeError("freeze log-space bounds before fitting moments")
        X = np.asarray(X, dtype=np.float64)
        X_log = self._log_transform(X)
        X_scaled = X_log * self.scale_ + self.min_offset_
        self.scalar.partial_fit(X_scaled)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        mask = np.isnan(X)

        X_log = self._log_transform(X)
        X_scaled = X_log * self.scale_ + self.min_offset_

        if self.clip:
            low, high = self.feature_range
            X_scaled = np.clip(X_scaled, low, high)
        X_scaled = self.scalar.transform(X_scaled)
        X_scaled[mask] = np.nan  # reapply NaN mask
        return X_scaled

    def inverse_transform(self, X_scaled):
        X_scaled = np.asarray(X_scaled, dtype=np.float64)
        mask = np.isnan(X_scaled)
        X_scaled = self.scalar.inverse_transform(X_scaled)

        X_log = (X_scaled - self.min_offset_) / self.scale_
        X = np.expm1(X_log)

        X[mask] = np.nan  # restore NaNs
        return X


class PerFeatureScaler(BaseEstimator, TransformerMixin):
    """
    Per-column scaler for the 4 interaction features [ΔR, kT, z, m²].

    Each column gets a physically appropriate transformer:
        ΔR → StandardScaler   (roughly Gaussian, can be 0)
        kT → LogMinMax        (heavy-tailed, >= 0)
        z  → StandardScaler   (bounded [0, 0.5])
        m² → LogMinMax        (heavy-tailed; LogMinMax clips at 0, absorbing tiny
                               negative m² from float round-off)

    Operates on flattened [N, 4] arrays. Supports partial_fit for streaming fits.
    Replaces the single shared LogMinMaxScaler (which log-scaled ΔR, physically wrong).
    """

    FEATURE_NAMES = ("dR", "kT", "z", "m2")

    def __init__(self):
        self.transformers = [
            StandardScaler(),  # dR
            LogMinMax(),       # kT
            StandardScaler(),  # z
            LogMinMax(),       # m2
        ]

    def partial_fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        for i, t in enumerate(self.transformers):
            t.partial_fit(X[:, i:i + 1])
        return self

    def partial_fit_bounds(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        for i, transformer in enumerate(self.transformers):
            if hasattr(transformer, "partial_fit_bounds"):
                transformer.partial_fit_bounds(X[:, i:i + 1])
        return self

    def freeze_bounds(self):
        for transformer in self.transformers:
            if hasattr(transformer, "freeze_bounds"):
                transformer.freeze_bounds()
        return self

    def partial_fit_standardization(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        for i, transformer in enumerate(self.transformers):
            column = X[:, i:i + 1]
            if hasattr(transformer, "partial_fit_standardization"):
                transformer.partial_fit_standardization(column)
            else:
                transformer.partial_fit(column)
        return self

    def fit(self, X, y=None):
        self.transformers = [StandardScaler(), LogMinMax(), StandardScaler(), LogMinMax()]
        return self.partial_fit(X, y)

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        cols = [self.transformers[i].transform(X[:, i:i + 1]) for i in range(X.shape[1])]
        return np.concatenate(cols, axis=1)

    def inverse_transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        cols = [self.transformers[i].inverse_transform(X[:, i:i + 1]) for i in range(X.shape[1])]
        return np.concatenate(cols, axis=1)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.FEATURE_NAMES)


class LogScaler(BaseEstimator, TransformerMixin):
    def __init__(self, offset=1e-3, add_mask = False):
        self.offset = offset
        self.add_mask = add_mask

    def fit(self, X, y=None):
        return self  # no fitting needed

    def transform(self, X):
        # Create a mask for nan values
        X = np.asarray(X, dtype=float)
        log = safe_transform(X + self.offset, np.log)
        mask = np.isnan(log)
        if self.add_mask:
            return np.hstack([mask, log])
        return log

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = [f"x{i}" for i in range(self.n_features_in_)]
        if self.add_mask:
            mask_features = [f"{feat}_mask" for feat in input_features]
            log_features = [f"log({feat})" for feat in input_features]
            return np.array(mask_features + log_features)
        else:
            return np.array([f"{feat}" for feat in input_features])

    def inverse_transform(self, X):
        if self.add_mask:
            # Assumes first half of features are masks, second half are log values
            X = X[:, X.shape[1] // 2:]
        return np.exp(X) - self.offset


class PhiTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        sin_phi = safe_transform(X ,np.sin)
        cos_phi = safe_transform(X, np.cos)
        return np.hstack([sin_phi, cos_phi])

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = [f"x{i}" for i in range(self.n_features_in_)]

        # Extract the raw feature name without namespace (e.g., 'phi__x14' -> 'x14')
        def clean_name(name):
            if '__' in name:
                prefix, var = name.split('__', 1)
                return f"{prefix}{var}"  # e.g., phi__x14 -> phix14
            return name  # fallback

        out = []
        for feat in input_features:
            base = clean_name(feat)
            out.append(f"sin({base})")
            out.append(f"cos({base})")
        return np.array(out)

    def partial_fit(self, X):
        return self

    def inverse_transform(self, X):
        sin_phi = X[:, 0]
        cos_phi = X[:, 1]
        phi = np.arctan2(sin_phi, cos_phi)
        return phi



class ArctanScaler(FunctionTransformer):

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return None
        return [f"{name}" for name in input_features]

    def partial_fit(self, X):
        return self

    def fit(self, X):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        out = np.empty_like(X)
        mask = np.isnan(X)
        out[~mask] = np.arctan(X[~mask]) * 2 / np.pi
        out[mask] = np.nan
        return out

    def inverse_transform(self, X):
        X = np.asarray(X, dtype=float)
        out = np.empty_like(X)
        mask = np.isnan(X)
        out[~mask] = 2 * np.tan(X[~mask] * np.pi /2 )
        out[mask] = np.nan
        return out




class TanhScaler(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        X = np.asarray(X)
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        return self

    def transform(self, X):
        return 0.5 * (np.tanh((X - self.mean_) / self.std_) + 1)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return None
        return [f"{name}" for name in input_features]

    def inverse_transform(self, X):
        return self.std_ * np.arctanh(2 * X - 1) + self.mean_


class NoOpScaler(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self  # No fitting needed

    def transform(self, X):
        return np.asarray(X)  # Return input unchanged

    def inverse_transform(self, X):
        return np.asarray(X)  # Also return unchanged

    def get_feature_names_out(self, input_features=None):
        return input_features
