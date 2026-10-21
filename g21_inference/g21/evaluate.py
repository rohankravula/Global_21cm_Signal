"""Metrics and plots."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PARAM_NAMES = ["log10 sigma", "log10 mass"]


def grid_steps(params: np.ndarray) -> np.ndarray:
    """Typical spacing of the training grid along each axis (dex)."""
    return np.array([np.median(np.diff(np.unique(np.round(params[:, j], 6)))) for j in range(2)])


def param_metrics(true: np.ndarray, pred: np.ndarray, steps: np.ndarray) -> dict:
    err = pred - true
    within = np.abs(err) <= steps * 1.0001
    return {
        "mae_dex": np.abs(err).mean(0).round(4).tolist(),
        "median_abs_err_dex": np.median(np.abs(err), 0).round(4).tolist(),
        "rmse_dex": np.sqrt((err ** 2).mean(0)).round(4).tolist(),
        "within_1_grid_step": within.mean(0).round(3).tolist(),
        "within_1_grid_step_both": float(within.all(1).mean().round(3)),
    }


def binned(true, pred, bins, extra=None) -> dict:
    """Mean abs error (dex) of [sigma, mass] within bins of true log10 sigma."""
    out = {}
    for lo, hi in zip(bins[:-1], bins[1:]):
        k = (true[:, 0] >= lo - 1e-6) & (true[:, 0] < hi - 1e-6 if hi < bins[-1] else true[:, 0] <= hi + 1e-6)
        if k.any():
            e = {"n": int(k.sum()), "mae_dex": np.abs(pred[k] - true[k]).mean(0).round(3).tolist()}
            if extra is not None:
                e["median_posterior_std_dex"] = np.median(extra[k], 0).round(3).tolist()
            out[f"log_sigma[{lo:g},{hi:g}]"] = e
    return out


def mean_mae(true, pred) -> float:
    return float(np.abs(pred - true).mean())


def reconstruction_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(((y_true - y_pred) ** 2).mean()))


def plot_pred_vs_true(true, preds: dict, path, title=""):
    n = len(preds)
    fig, axes = plt.subplots(2, n, figsize=(3.2 * n, 6.2), squeeze=False)
    for j, (name, pred) in enumerate(preds.items()):
        for i in range(2):
            ax = axes[i, j]
            ax.scatter(true[:, i], pred[:, i], s=14, alpha=0.7)
            lim = [true[:, i].min() - 0.2, true[:, i].max() + 0.2]
            ax.plot(lim, lim, "k--", lw=1)
            ax.set_xlim(lim); ax.set_ylim(lim)
            ax.set_title(f"{name}: {PARAM_NAMES[i]}", fontsize=9)
            ax.set_xlabel("true"); ax.set_ylabel("predicted")
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_emulator_examples(u_grid, y_true, y_pred, labels, path):
    z = np.expm1(u_grid)
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5), sharex=True)
    for ax, yt, yp, lab in zip(axes.ravel(), y_true, y_pred, labels):
        ax.plot(z, yt, label="simulated"); ax.plot(z, yp, "--", label="emulated")
        ax.set_xscale("log"); ax.set_title(lab, fontsize=9); ax.set_xlabel("z")
        ax.set_ylabel("asinh(dTb / 1 mK)")
    axes[0, 0].legend(); fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
