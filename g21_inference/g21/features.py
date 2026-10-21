"""Target transform, dimensionality reduction and parameter scaling."""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA


class AsinhTransform:
    """y = asinh(T/scale). ~linear for |T|<<scale, ~log for |T|>>scale: tames -1e5..+40 mK."""

    def __init__(self, scale: float = 1.0):
        if scale <= 0:
            raise ValueError("scale must be positive")
        self.scale = float(scale)

    def forward(self, t):
        return np.arcsinh(np.asarray(t, dtype=float) / self.scale)

    def inverse(self, y):
        return self.scale * np.sinh(np.asarray(y, dtype=float))


class CurveReducer:
    """PCA fitted on (transformed) training curves only."""

    def __init__(self, variance_to_keep: float = 0.9999, max_components: int = 30):
        self.variance_to_keep = variance_to_keep
        self.max_components = max_components

    def fit(self, y):
        n_max = min(self.max_components, y.shape[0] - 1, y.shape[1])
        pca = PCA(n_components=n_max, svd_solver="full").fit(y)
        cum = np.cumsum(pca.explained_variance_ratio_)
        k = min(int(np.searchsorted(cum, self.variance_to_keep) + 1), n_max)
        self.mean_ = pca.mean_
        self.components_ = pca.components_[:k]
        self.n_components_ = k
        self.explained_ = float(cum[k - 1])
        return self

    def transform(self, y):
        return (np.atleast_2d(y) - self.mean_) @ self.components_.T

    def inverse_transform(self, c):
        return np.atleast_2d(c) @ self.components_ + self.mean_


class ParamScaler:
    """Maps (log10 sigma, log10 mass) to the unit square using training bounds."""

    def fit(self, p):
        self.lo_, self.hi_ = p.min(0), p.max(0)
        self.span_ = np.where(self.hi_ > self.lo_, self.hi_ - self.lo_, 1.0)
        return self

    def transform(self, p):
        return (np.atleast_2d(p) - self.lo_) / self.span_

    def inverse_transform(self, u):
        return np.atleast_2d(u) * self.span_ + self.lo_
