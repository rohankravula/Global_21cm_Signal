"""Forward-fit inference: scan a dense parameter grid with the emulator and weight it."""
from __future__ import annotations

import numpy as np


def calibrate_inflation(c_true, mean, var, noise: float) -> np.ndarray:
    """Per-component factor so the emulator's predictive variance matches its real
    held-out squared error: infl_k = max(1, (E[r_k^2] - noise^2) / E[var_k])."""
    r2 = ((c_true - mean) ** 2).mean(0)
    return np.maximum(1.0, (r2 - noise ** 2) / np.maximum(var.mean(0), 1e-12))


class GridPosterior:
    """Pseudo-posterior over (log10 sigma, log10 mass) with a uniform prior.

    The emulator is evaluated once on a dense grid. An observed curve (PCA coefficients c)
    is compared with each cell using a diagonal Gaussian likelihood whose variance is
        v_k(cell) = inflation_k * emulator_variance_k(cell) + noise^2
    i.e. observation noise plus the emulator's own (calibrated) uncertainty. Where curves
    barely change with a parameter, the posterior stays broad: the data cannot pin it down.
    """

    def __init__(self, emulator, scaler, n_per_axis: int = 121, noise: float = 0.05,
                 inflation: np.ndarray | None = None):
        u = np.linspace(0.0, 1.0, n_per_axis)
        unit = np.stack(np.meshgrid(u, u, indexing="ij"), -1).reshape(-1, 2)
        self.noise = noise
        self.params_ = scaler.inverse_transform(unit)
        mean, var = emulator.predict(unit, return_var=True)
        infl = np.ones(mean.shape[1]) if inflation is None else inflation
        self.mean_, self.var_ = mean, infl[None, :] * var + noise ** 2
        self._inv = 1.0 / self.var_
        self._const = (mean ** 2 * self._inv).sum(1) + np.log(self.var_).sum(1)

    def nll(self, c) -> np.ndarray:
        c = np.atleast_2d(c)
        return 0.5 * ((c ** 2) @ self._inv.T - 2 * c @ (self.mean_ * self._inv).T + self._const[None, :])

    def infer(self, c_obs, chunk: int = 128) -> dict:
        c_obs = np.atleast_2d(c_obs)
        out = {k: [] for k in ("map", "mean", "std")}
        for i in range(0, len(c_obs), chunk):
            nll = self.nll(c_obs[i:i + chunk])
            w = np.exp(-(nll - nll.min(1, keepdims=True)))
            w /= w.sum(1, keepdims=True)
            mean = w @ self.params_
            var = w @ self.params_ ** 2 - mean ** 2
            out["map"].append(self.params_[nll.argmin(1)])
            out["mean"].append(mean)
            out["std"].append(np.sqrt(np.maximum(var, 0)))
        return {k: np.concatenate(v) for k, v in out.items()}
