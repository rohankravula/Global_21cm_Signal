#!/usr/bin/env python
"""Train + evaluate the inference pipeline.   python train.py --data-dir ../Global_21cm_Signal/new_data"""
import argparse
import logging
from datetime import datetime

import yaml

from g21.data import load_dataset
from g21.pipeline import format_summary, run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--data-dir", help="overrides data.dir in the config")
    ap.add_argument("--out-dir", help="default: runs/<timestamp>")
    ap.add_argument("--z-min", type=float)
    ap.add_argument("--z-max", type=float)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cfg = yaml.safe_load(open(args.config))
    if args.data_dir: cfg["data"]["dir"] = args.data_dir
    if args.z_min: cfg["data"]["z_min"] = args.z_min
    if args.z_max: cfg["data"]["z_max"] = args.z_max
    d = cfg["data"]
    ds = load_dataset(d["dir"], d["glob"], d["z_min"], d["z_max"], d["n_grid"])
    out = args.out_dir or f"runs/{datetime.now():%Y%m%d_%H%M%S}"
    metrics = run(cfg, ds, out)
    print("\n" + format_summary(metrics) + f"\n\nSaved to {out}/")


if __name__ == "__main__":
    main()
