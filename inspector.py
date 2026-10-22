import argparse
import os
import sys
 
import numpy as np
import pandas as pd
 
 
def load(path):
    """Return a DataFrame with columns nu, z, dTb (z/nu only if present)."""
    with np.load(path) as d:
        cols = {k: d[k] for k in ("nu", "z", "dTb") if k in d.files}
    if "dTb" not in cols:
        sys.exit(f"{path}: no 'dTb' array (found: {list(cols) or 'nothing usable'})")
    return pd.DataFrame(cols)
 
 
def summarize(df):
    print(f"points: {len(df)}")
    if "z" in df:
        print(f"z range: {df['z'].min():.4g} .. {df['z'].max():.4g}  "
              f"(duplicate z values: {df['z'].duplicated().sum()})")
    i = df["dTb"].idxmin()
    where = f" at z={df.loc[i, 'z']:.4g}" if "z" in df else ""
    print(f"dTb range: {df['dTb'].min():.4g} .. {df['dTb'].max():.4g} mK "
          f"(minimum{where})")
 
 
def save_plot(df, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
 
    x_col = "z" if "z" in df else "nu"
    d = df.sort_values(x_col)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(d[x_col], d["dTb"])
    ax.set_xscale("log")
    ax.set_xlabel(x_col)
    ax.set_ylabel("dTb (mK)")
    ax.set_title(os.path.basename(path))
    out = os.path.splitext(path)[0] + "_plot.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"saved plot: {out}")
 
 
def show_gui(df, max_rows=5000):
    try:
        import tkinter as tk
        from tkinter import ttk
    except ImportError:
        sys.exit("tkinter is not installed; run without --gui.")
    root = tk.Tk()
    root.title("Raw data")
    tree = ttk.Treeview(root, columns=list(df.columns), show="headings")
    for c in df.columns:
        tree.heading(c, text=c)
        tree.column(c, width=180, anchor="center")
    for row in df.head(max_rows).itertuples(index=False):  # cap: 58k rows is slow
        tree.insert("", tk.END, values=[f"{v:.6g}" for v in row])
    bar = ttk.Scrollbar(root, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=bar.set)
    bar.pack(side="right", fill="y")
    tree.pack(expand=True, fill="both")
    if len(df) > max_rows:
        print(f"GUI shows the first {max_rows} of {len(df)} rows.")
    root.mainloop()
 
 
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", help="path to a .npz file")
    ap.add_argument("--rows", type=int, default=15, help="rows to print (default 15)")
    ap.add_argument("--plot", action="store_true", help="save a dTb plot as PNG")
    ap.add_argument("--gui", action="store_true", help="open a scrollable table window")
    args = ap.parse_args()
 
    if not os.path.isfile(args.file):
        sys.exit(f"File not found: {args.file}")
 
    df = load(args.file)
    print(f"{os.path.basename(args.file)}")
    summarize(df)
    print()
    print(df.head(args.rows).to_string(index=False))
    if len(df) > args.rows:
        print(f"... ({len(df) - args.rows} more rows)")
    if args.plot:
        save_plot(df, args.file)
    if args.gui:
        show_gui(df)
 
 
if __name__ == "__main__":
    main()
