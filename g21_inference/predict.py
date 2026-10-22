"""Estimate (sigma, mass) from 21cm curves using a trained run.
 
Easiest usage (from anywhere):
    python predict.py                          # latest run, 5 held-out test curves
    python predict.py 20261021_223500          # a specific run, by name
    python predict.py 20261021_223500 my_curve_dTb.npz other_dTb.npz   # your own curves
    python predict.py --n 10                   # more test curves
 
Test curves are ones the model never trained on, so the comparison with the true
values (read from the filename) is an honest check. Curves need arrays 'z' and 'dTb'.
"""
import argparse
import json
import sys
from pathlib import Path
 
import joblib
import numpy as np
 
from g21.data import load_curve, parse_params, resample
 
HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
 
 
def list_runs():
    return sorted(p for p in RUNS.glob("*") if (p / "bundle.joblib").exists()) if RUNS.exists() else []
 
 
def resolve_run(name):
    """Run folder from a name (e.g. 20261021_223500), a path, or None for the latest run."""
    if name is None:
        runs = list_runs()
        if not runs:
            sys.exit(f"No trained runs found in {RUNS}.\nRun  python train.py  first.")
        return runs[-1]
    for cand in (Path(name), RUNS / name, HERE / name):
        if (cand / "bundle.joblib").exists():
            return cand.resolve()
    available = ", ".join(p.name for p in list_runs()) or "none yet - run  python train.py  first"
    sys.exit(f"Can't find a trained run called '{name}'.\nAvailable runs: {available}")
 
 
def default_curves(run, bundle, n):
    """Evenly spaced curves from the run's held-out test set."""
    split = json.loads((run / "split.json").read_text())["test"]
    data_dir = Path(bundle["cfg"]["data"]["dir"])
    if not data_dir.is_absolute():
        data_dir = (HERE / data_dir).resolve()
    picks = np.linspace(0, len(split) - 1, min(n, len(split))).round().astype(int)
    paths = [data_dir / split[i] for i in picks]
    missing = [p for p in paths if not p.exists()]
    if missing:
        sys.exit(f"Test curves not found, e.g. {missing[0]}\n"
                 "Check data.dir in the run's config_used.yaml, or pass curve files explicitly.")
    return paths
 
 
def describe(p, sd):
    return (f"sigma={10 ** p[0]:.3g}  mass={10 ** p[1]:.3g}   "
            f"[log10 sigma={p[0]:.2f}±{sd[0]:.2f}, log10 mass={p[1]:.2f}±{sd[1]:.2f}]")
 
 
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("items", nargs="*", help="[run name] [curve .npz files ...]")
    ap.add_argument("--run", help="run name or path (same as giving it first)")
    ap.add_argument("--n", type=int, default=5, help="how many test curves to use by default")
    ap.add_argument("--all-models", action="store_true", help="also print the baseline models")
    args = ap.parse_args()
 
    items = list(args.items)
    run_name = args.run
    if run_name is None and items and not items[0].lower().endswith(".npz"):
        run_name = items.pop(0)
    run = resolve_run(run_name)
    b = joblib.load(run / "bundle.joblib")
    print(f"Using run: {run.name}")
 
    curves = [Path(c) for c in items] or default_curves(run, b, args.n)
    using_test = not items
    if using_test:
        print(f"No curves given: using {len(curves)} held-out test curves (never seen in training).")
 
    errs = []
    for path in curves:
        if not path.exists():
            print(f"\n{path}\n  file not found, skipping")
            continue
        z, t = load_curve(str(path))
        y = b["tf"].forward(resample(z, t, b["u_grid"]))
        mean, std = b["forest"].predict(y, return_std=True)
        p = b["sc"].inverse_transform(mean)[0]
        sd = (std * b["sc"].span_)[0]
        print(f"\n{path.name}")
        try:
            s_true, m_true = parse_params(str(path))
            truth = np.log10([s_true, m_true])
            print(f"  true      : sigma={s_true:.3g}  mass={m_true:.3g}")
            errs.append(np.abs(p - truth))
        except ValueError:
            truth = None
        print(f"  predicted : {describe(p, sd)}")
        if truth is not None:
            print(f"  error     : {abs(p[0] - truth[0]):.3f} dex in sigma, {abs(p[1] - truth[1]):.3f} dex in mass")
        if sd[1] > 0.5:
            print("  note: large tree spread for mass -> mass is poorly determined for this curve")
        if args.all_models:
            c = b["red"].transform(y)
            r = b["post"].infer(c)
            print(f"  forward-fit posterior: log10 sigma={r['mean'][0][0]:.2f}±{r['std'][0][0]:.2f}  "
                  f"log10 mass={r['mean'][0][1]:.2f}±{r['std'][0][1]:.2f}")
            for name, m in b["inv"].items():
                q = b["sc"].inverse_transform(m.predict(b["csc"].transform(c)))[0]
                print(f"  baseline {name:12s}: sigma={10 ** q[0]:.3g}  mass={10 ** q[1]:.3g}")
 
    if len(errs) > 1:
        e = np.mean(errs, 0)
        print(f"\nAverage error over {len(errs)} curves: {e[0]:.3f} dex in sigma, {e[1]:.3f} dex in mass "
              "(1 dex = a factor of 10)")
 
 
if __name__ == "__main__":
    main()
 
