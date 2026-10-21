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
    args = ap.parse_args()
    b = joblib.load(f"{args.run}/bundle.joblib")
    for path in args.curves:
        z, t = load_curve(path)
        c = b["red"].transform(b["tf"].forward(resample(z, t, b["u_grid"])))
        print(f"\n{path}")
        r = b["post"].infer(c)
        s_map, s_mean, s_std = r["map"][0], r["mean"][0], r["std"][0]
        print(f"  forward-fit best match : sigma={10**s_map[0]:.3g}  mass={10**s_map[1]:.3g}")
        print(f"  forward-fit posterior  : log10 sigma={s_mean[0]:.2f}±{s_std[0]:.2f}  "
              f"log10 mass={s_mean[1]:.2f}±{s_std[1]:.2f}")
        if s_std[1] > 0.5:
            print("  note: wide mass posterior -> the curve barely depends on mass here (degenerate region)")
        for name, m in b["inv"].items():
            p = b["sc"].inverse_transform(m.predict(b["csc"].transform(c)))[0]
            print(f"  direct {name:12s}    : sigma={10**p[0]:.3g}  mass={10**p[1]:.3g}")


if __name__ == "__main__":
    main()
