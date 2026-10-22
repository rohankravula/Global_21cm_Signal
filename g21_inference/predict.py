#!/usr/bin/env python
"""Infer (sigma, mass) from one or more curves.
   python predict.py --run runs/<id> path/to/sigma_*_dTb.npz [...]
Curves need arrays 'z' and 'dTb' (same format as the training files)."""
import argparse

import joblib
import numpy as np

from g21.data import load_curve, resample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("curves", nargs="+")
    ap.add_argument("--all-models", action="store_true", help="also print the baseline models")
    args = ap.parse_args()
    b = joblib.load(f"{args.run}/bundle.joblib")
    for path in args.curves:
        z, t = load_curve(path)
        y = b["tf"].forward(resample(z, t, b["u_grid"]))
        c = b["red"].transform(y)
        print(f"\n{path}")
        mean, std = b["forest"].predict(y, return_std=True)
        p = b["sc"].inverse_transform(mean)[0]
        sd = (std * b["sc"].span_)[0]
        print(f"  best estimate (curve_forest): sigma={10**p[0]:.3g}  mass={10**p[1]:.3g}   "
              f"[log10 sigma={p[0]:.2f}±{sd[0]:.2f}, log10 mass={p[1]:.2f}±{sd[1]:.2f}]")
        if sd[1] > 0.5:
            print("  note: large tree spread for mass -> mass is poorly determined for this curve")
        if args.all_models:
            r = b["post"].infer(c)
            s_mean, s_std = r["mean"][0], r["std"][0]
            print(f"  forward-fit posterior       : log10 sigma={s_mean[0]:.2f}±{s_std[0]:.2f}  "
                  f"log10 mass={s_mean[1]:.2f}±{s_std[1]:.2f}")
            for name, m in b["inv"].items():
                q = b["sc"].inverse_transform(m.predict(b["csc"].transform(c)))[0]
                print(f"  baseline {name:12s}     : sigma={10**q[0]:.3g}  mass={10**q[1]:.3g}")


if __name__ == "__main__":
    main()
