"""Loading, cleaning and splitting the .npz curves."""
from __future__ import annotations

import glob
import logging
import os
import re
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)

_NUM = r"[0-9.]+(?:[eE][+-]?\d+)?"
_PAT = re.compile(rf"sigma_({_NUM})_mass_({_NUM})_dTb\.npz$")


@dataclass
class Dataset:
    params: np.ndarray   # (N, 2) = log10(sigma), log10(mass)
    curves: np.ndarray   # (N, G) dTb in mK on the common grid
    files: list          # N file paths
    u_grid: np.ndarray   # (G,) common grid, u = log(1+z)

    @property
    def z_grid(self):
        return np.expm1(self.u_grid)


def parse_params(path: str) -> tuple[float, float]:
    """(sigma, mass) from a filename like sigma_1.0e-39_mass_1.0e+00_dTb.npz."""
    m = _PAT.search(os.path.basename(path))
    if not m:
        raise ValueError(f"Cannot parse parameters from filename: {path}")
    return float(m.group(1)), float(m.group(2))


def load_curve(path: str):
    """Return (z ascending, dTb) with non-finite data rejected and duplicate z removed."""
    with np.load(path, allow_pickle=False) as d:
        z = np.asarray(d["z"], dtype=float)
        t = np.asarray(d["dTb"], dtype=float)
    if z.shape != t.shape or z.ndim != 1:
        raise ValueError("z and dTb must be 1-D arrays of equal length")
    if not (np.isfinite(z).all() and np.isfinite(t).all()):
        raise ValueError("non-finite values")
    z_u, idx = np.unique(z, return_index=True)  # sorts ascending, drops repeated z
    return z_u, t[idx]


def make_grid(z_min: float, z_max: float, n: int) -> np.ndarray:
    return np.linspace(np.log1p(z_min), np.log1p(z_max), n)


def resample(z: np.ndarray, t: np.ndarray, u_grid: np.ndarray) -> np.ndarray:
    """Interpolate a curve onto the common log(1+z) grid; refuse to extrapolate."""
    if z.min() > np.expm1(u_grid[0]) or z.max() < np.expm1(u_grid[-1]):
        raise ValueError(f"curve covers z=[{z.min():.3g},{z.max():.3g}], grid needs more")
    return np.interp(u_grid, np.log1p(z), t)


def load_dataset(data_dir: str, pattern: str, z_min: float, z_max: float, n_grid: int,
                 min_files: int = 10) -> Dataset:
    files = sorted(glob.glob(os.path.join(data_dir, pattern)))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern!r} in {data_dir!r}")
    u_grid = make_grid(z_min, z_max, n_grid)
    params, curves, kept, skipped = [], [], [], []
    for f in files:
        try:
            s, m = parse_params(f)
            z, t = load_curve(f)
            curves.append(resample(z, t, u_grid))
            params.append((np.log10(s), np.log10(m)))
            kept.append(f)
        except Exception as e:  # corrupt / unreadable / unparsable: skip but say so
            skipped.append((f, str(e)))
    if skipped:
        shown = "; ".join(f"{os.path.basename(f)}: {msg}" for f, msg in skipped[:3])
        log.warning("Skipped %d/%d files (first few: %s)", len(skipped), len(files), shown)
    if len(kept) < min_files:
        raise RuntimeError(f"Only {len(kept)} usable files out of {len(files)}")
    log.info("Loaded %d/%d files", len(kept), len(files))
    return Dataset(np.array(params), np.array(curves), kept, u_grid)


def split_indices(params: np.ndarray, train: float, val: float, test: float, seed: int,
                  keep_boundary_in_train: bool = True) -> dict:
    """Random train/val/test split by curve (never by redshift point).

    With keep_boundary_in_train, curves on the edge of the parameter box are forced
    into train so that val/test only require interpolation, not extrapolation.
    """
    if abs(train + val + test - 1.0) > 1e-9:
        raise ValueError("split fractions must sum to 1")
    n = len(params)
    n_val, n_test = int(round(val * n)), int(round(test * n))
    n_train = n - n_val - n_test
    rng = np.random.default_rng(seed)
    if keep_boundary_in_train:
        lo, hi = params.min(0), params.max(0)
        edge = ((params <= lo + 1e-9) | (params >= hi - 1e-9)).any(1)
    else:
        edge = np.zeros(n, bool)
    edge_idx = np.flatnonzero(edge)
    inner = rng.permutation(np.flatnonzero(~edge))
    if len(inner) < n_val + n_test:
        raise ValueError("not enough interior points for the requested val/test sizes")
    va, te = inner[:n_val], inner[n_val:n_val + n_test]
    tr = np.concatenate([edge_idx, inner[n_val + n_test:]])
    assert len(tr) == n_train
    return {"train": np.sort(tr), "val": np.sort(va), "test": np.sort(te)}
