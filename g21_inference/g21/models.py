"""Forward emulator (params -> curve coefficients) and direct inverse regressors."""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor


class ForwardEmulator:
    """One Gaussian process per PCA coefficient: unit-square params -> coefficients."""

    def __init__(self, seed: int = 0, n_restarts: int = 2, nu: float = 2.5):
        self.seed, self.n_restarts, self.nu = seed, n_restarts, nu

    def fit(self, x, c):
        self.gps_ = []
        for k in range(c.shape[1]):
            kernel = (ConstantKernel(1.0, (1e-3, 1e3))
                      * Matern(length_scale=[0.3, 0.3], length_scale_bounds=(1e-2, 1e2), nu=self.nu)
                      + WhiteKernel(1e-3, (1e-9, 1e-1)))
            gp = GaussianProcessRegressor(kernel, normalize_y=True,
                                          n_restarts_optimizer=self.n_restarts,
                                          random_state=self.seed + k)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                gp.fit(x, c[:, k])
            self.gps_.append(gp)
        return self

    def predict(self, x, return_var: bool = False):
        if not return_var:
            return np.column_stack([gp.predict(x) for gp in self.gps_])
        out = [gp.predict(x, return_std=True) for gp in self.gps_]
        mean = np.column_stack([o[0] for o in out])
        var = np.column_stack([o[1] ** 2 for o in out])
        return mean, var


def make_inverse(name: str, kwargs: dict | None, seed: int):
    """Direct inverse regressor: standardized PCA coefficients -> unit-square params."""
    kw = dict(kwargs or {})
    if name == "rf":
        return RandomForestRegressor(n_estimators=kw.pop("n_estimators", 300),
                                     random_state=seed, n_jobs=-1, **kw)
    if name == "extra_trees":
        return ExtraTreesRegressor(n_estimators=kw.pop("n_estimators", 300),
                                   random_state=seed, n_jobs=-1, **kw)
    if name == "knn":
        return KNeighborsRegressor(**kw)
    if name == "mlp":
        hidden = tuple(kw.pop("hidden_layer_sizes", (128, 128)))
        return MLPRegressor(hidden_layer_sizes=hidden, max_iter=kw.pop("max_iter", 3000),
                            random_state=seed, **kw)
    raise ValueError(f"unknown inverse model {name!r}")
