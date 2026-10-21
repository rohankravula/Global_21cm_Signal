"""End-to-end training / evaluation. Importable (used by train.py and the tests)."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import joblib
import numpy as np
import yaml
from sklearn.preprocessing import StandardScaler

from .data import Dataset, split_indices
from .evaluate import (binned, grid_steps, mean_mae, param_metrics, plot_emulator_examples,
                       plot_pred_vs_true, reconstruction_rmse)
from .features import AsinhTransform, CurveReducer, ParamScaler
from .inference import GridPosterior, calibrate_inflation
from .models import ForwardEmulator, make_inverse

log = logging.getLogger(__name__)


def run(cfg: dict, ds: Dataset, out_dir: str) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    seed, inf = cfg["seed"], cfg["inference"]

    # ---- split (by curve) ---------------------------------------------------------
    idx = split_indices(ds.params, seed=seed, **cfg["split"])
    tr, va, te = idx["train"], idx["val"], idx["test"]
    log.info("Split: train=%d val=%d test=%d", len(tr), len(va), len(te))

    # ---- preprocessing: everything is fitted on the training curves only ------------
    tf = AsinhTransform(cfg["transform"]["asinh_scale_mK"])
    y = tf.forward(ds.curves)
    red = CurveReducer(**cfg["reducer"]).fit(y[tr])
    c = red.transform(y)
    sc = ParamScaler().fit(ds.params[tr])
    x = sc.transform(ds.params)
    csc = StandardScaler().fit(c[tr])
    steps = grid_steps(ds.params[tr])
    log.info("PCA keeps %d components (%.5f of variance)", red.n_components_, red.explained_)

    # ---- model selection on the validation set ---------------------------------------
    emu_tr = ForwardEmulator(seed, **cfg["forward"]).fit(x[tr], c[tr])
    chat_va = emu_tr.predict(x[va])
    emulator_val = {
        "emulator_rmse_asinh": round(reconstruction_rmse(y[va], red.inverse_transform(chat_va)), 4),
        "pca_floor_rmse_asinh": round(reconstruction_rmse(y[va], red.inverse_transform(c[va])), 4),
        "predict_mean_curve_rmse_asinh": round(reconstruction_rmse(y[va], y[tr].mean(0)[None]), 4),
    }
    selection, best_cfg = {}, {}
    for name, cands in inf["candidates"].items():
        scores = []
        for kw in cands:
            m = make_inverse(name, kw, seed).fit(csc.transform(c[tr]), x[tr])
            pred = sc.inverse_transform(m.predict(csc.transform(c[va])))
            scores.append(mean_mae(ds.params[va], pred))
        j = int(np.argmin(scores))
        best_cfg[name] = cands[j]
        selection[name] = {"best_config": cands[j], "val_mae_dex": round(scores[j], 4),
                           "all_val_mae_dex": [round(s, 4) for s in scores]}
        log.info("val  %-12s best=%s  MAE=%.3f dex", name, cands[j], scores[j])
    noise = inf["posterior_noise"]
    mu_va, var_va = emu_tr.predict(x[va], return_var=True)
    infl = calibrate_inflation(c[va], mu_va, var_va, noise)   # calibrated on held-out val
    post_tr = GridPosterior(emu_tr, sc, inf["grid_per_axis"], noise, infl)
    rv = post_tr.infer(c[va])
    for key in ("map", "mean"):
        selection[f"forward_fit_{key}"] = {"val_mae_dex": round(mean_mae(ds.params[va], rv[key]), 4)}
        log.info("val  forward_fit_%s MAE=%.3f dex", key, selection[f"forward_fit_{key}"]["val_mae_dex"])

    # ---- final fit (optionally on train+val), then ONE evaluation on the test set ----
    fit = np.concatenate([tr, va]) if inf["refit_on_train_val"] else tr
    emu = ForwardEmulator(seed, **cfg["forward"]).fit(x[fit], c[fit])
    post = GridPosterior(emu, sc, inf["grid_per_axis"], noise, infl)
    inv = {n: make_inverse(n, best_cfg[n], seed).fit(csc.transform(c[fit]), x[fit]) for n in best_cfg}

    p_true = ds.params[te]
    y_hat_te = red.inverse_transform(emu.predict(x[te]))
    emulator_test = {
        "emulator_rmse_asinh": round(reconstruction_rmse(y[te], y_hat_te), 4),
        "pca_floor_rmse_asinh": round(reconstruction_rmse(y[te], red.inverse_transform(c[te])), 4),
        "predict_mean_curve_rmse_asinh": round(reconstruction_rmse(y[te], y[tr].mean(0)[None]), 4),
    }
    baseline = np.repeat(ds.params[tr].mean(0)[None], len(te), 0)
    results, preds0 = {}, None
    rng = np.random.default_rng(seed + 1)
    for nl in cfg["evaluation"]["noise_levels"]:
        yn = y[te] + nl * rng.standard_normal(y[te].shape) if nl > 0 else y[te]
        cn = red.transform(yn)
        preds = {"baseline_train_mean": baseline}
        for n, m in inv.items():
            preds[n] = sc.inverse_transform(m.predict(csc.transform(cn)))
        r = post.infer(cn)
        preds["forward_fit_MAP"], preds["forward_fit_mean"] = r["map"], r["mean"]
        res = {n: param_metrics(p_true, p, steps) for n, p in preds.items()}
        bins = cfg["evaluation"]["sigma_bins"]
        res["by_sigma_bin"] = {n: binned(p_true, p, bins, r["std"] if n.startswith("forward_fit") else None)
                               for n, p in preds.items() if n != "baseline_train_mean"}
        res["forward_fit_median_posterior_std_dex"] = np.median(r["std"], 0).round(3).tolist()
        results[f"noise_{nl}"] = res
        if preds0 is None:
            preds0 = preds
    plot_pred_vs_true(p_true, {k: v for k, v in preds0.items() if k != "baseline_train_mean"},
                      out / "test_pred_vs_true.png", "Test set, noise-free curves")
    err = np.sqrt(((y[te] - y_hat_te) ** 2).mean(1))
    pick = np.argsort(err)[np.linspace(0, len(err) - 1, 6).astype(int)]
    labels = [f"logσ={p_true[i,0]:.2f}, logm={p_true[i,1]:.2f}, rmse={err[i]:.2f}" for i in pick]
    plot_emulator_examples(ds.u_grid, y[te][pick], y_hat_te[pick], labels, out / "emulator_examples.png")

    # ---- persist ----------------------------------------------------------------------
    metrics = {
        "n_curves": len(ds.params), "split_sizes": {k: len(v) for k, v in idx.items()},
        "n_pca_components": red.n_components_, "pca_variance_kept": round(red.explained_, 6),
        "grid_step_dex": steps.round(3).tolist(),
        "emulator_validation": emulator_val, "emulator_test": emulator_test,
        "validation_selection": selection, "test_results": results,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out / "split.json").write_text(json.dumps(
        {k: [os.path.basename(ds.files[i]) for i in v] for k, v in idx.items()}, indent=1))
    (out / "config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    joblib.dump({"cfg": cfg, "tf": tf, "red": red, "sc": sc, "csc": csc, "emu": emu, "post": post,
                 "inv": inv, "u_grid": ds.u_grid, "steps": steps}, out / "bundle.joblib")
    return metrics


def format_summary(m: dict) -> str:
    lines = [f"curves={m['n_curves']} split={m['split_sizes']} PCA comps={m['n_pca_components']} "
             f"grid step (dex)={m['grid_step_dex']}",
             f"emulator test RMSE (asinh units): {m['emulator_test']['emulator_rmse_asinh']}  "
             f"[PCA floor {m['emulator_test']['pca_floor_rmse_asinh']}, "
             f"mean-curve baseline {m['emulator_test']['predict_mean_curve_rmse_asinh']}]", ""]
    for nl, res in m["test_results"].items():
        lines.append(f"TEST {nl}   (MAE in dex: [sigma, mass] | fraction within 1 grid step: both)")
        for name, r in res.items():
            if isinstance(r, dict) and "mae_dex" in r:
                lines.append(f"  {name:22s} MAE={r['mae_dex']}  within1step={r['within_1_grid_step_both']}")
        lines.append(f"  forward-fit median posterior std (dex): {res['forward_fit_median_posterior_std_dex']}")
        lines.append("  mass MAE (dex) by true log10 sigma bin  [forward_fit_mean: median posterior std of mass]:")
        for name in ("extra_trees", "forward_fit_mean"):
            if name in res["by_sigma_bin"]:
                cells = []
                for b, e in res["by_sigma_bin"][name].items():
                    sd = f"/±{e['median_posterior_std_dex'][1]}" if "median_posterior_std_dex" in e else ""
                    cells.append(f"{b}: {e['mae_dex'][1]}{sd} (n={e['n']})")
                lines.append(f"    {name:18s} " + " | ".join(cells))
    return "\n".join(lines)
