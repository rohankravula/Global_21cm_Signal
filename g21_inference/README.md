# g21_inference — recover (σ, mass) from a global 21cm signal

Given a brightness-temperature curve `dTb(z)`, estimate the dark-matter–baryon cross-section σ and
dark-matter mass that produced it. Trained on `Data files/new_data/*.npz` (17 σ × 22 mass = 374 curves).

## The idea in one paragraph
Each file in `Data files/new_data` is one simulated 21cm curve made with a known σ (dark-matter–baryon
cross-section) and dark-matter mass, both written in the filename. The model is trained to go backwards: given a
curve, estimate the σ and mass that produced it. 262 curves train it, 56 tune it (validation), and 56 are held back as
a final test it never sees.

## Quick start
Run everything from inside `g21_inference/` (the default data path in `config.yaml` is `../Data files/new_data`):
```bash
python -m pip install -r requirements.txt
python train.py                      # several minutes; creates a new folder runs/<YYYYMMDD_HHMMSS>/
python predict.py                    # latest run, 5 held-out test curves: true vs predicted
python -m pytest                     # 10 tests
```

### Training: `train.py`
Prints the data split, the validation tuning, and a final TEST table (error in dex, plus the share of curves within one
grid step) at several noise levels, then `Saved to runs/<timestamp>/`. Every run gets its own folder, so earlier runs
are never overwritten. Options: `--config`, `--data-dir`, `--out-dir`, `--z-min`, `--z-max`. Most settings live in
`config.yaml`; each run saves the settings it used as `config_used.yaml`.

### Predicting: `predict.py`
```bash
python predict.py                                    # latest run, 5 held-out test curves
python predict.py 20261021_223357                    # a specific run, by name (ls runs)
python predict.py 20261021_223357 --n 10             # more test curves
python predict.py 20261021_223357 path/to/curve.npz  # your own curve(s)
python predict.py --all-models                       # also print the baseline models
```
With no curve files it uses evenly spaced curves from the run's test set (never seen in training). For each curve it
prints the true σ and mass (read from the filename), the prediction with ± tree spread, and the error in dex
(1 dex = a factor of 10), then the average error. A "large tree spread" note means mass is poorly determined for that
curve. A wrong run name lists the runs that exist. Curves need arrays `z` and `dTb`; files without σ and mass in the
name just get a prediction.

## What it does
1. **Load/clean** (`g21/data.py`): sort by z, drop duplicate z points, reject NaN/corrupt files (reported, not silent),
   resample every curve onto one common grid, uniform in log(1+z), z = 5.1–1000, 400 points. Parameters come from the
   filename and are used as log10.
2. **Split 70/15/15 by curve** (262 / 56 / 56), seeded. Edge-of-grid curves stay in train so val/test measure
   *interpolation*. Split lists are saved to `split.json`. Validation picks hyper-parameters; the test set is evaluated once.
3. **Transform**: `asinh(dTb / 1 mK)` (dTb spans −1.1e5 … +38 mK).
4. **Main model — `curve_forest`**: extra-trees on the *full* 400-point asinh curve, one forest for σ and one for mass,
   each with its own `max_features` chosen on the validation set. The spread of the trees' predictions is reported as an
   uncertainty. (Refit on train+val before the test evaluation.)
5. **Baselines, for comparison**: PCA(30)+random forest / extra-trees / kNN / MLP, and a Gaussian-process emulator with
   forward-fit posterior (gives principled uncertainty when the data are noisy).
6. **Evaluate once on test**, at noise 0 / 0.05 / 0.2 (Gaussian noise in asinh units), overall and per σ regime.

Outputs per run (in `runs/<timestamp>/`):
- `metrics.json`: all scores (validation tuning and test results at each noise level)
- `split.json`: which files are in train / val / test
- `config_used.yaml`: the exact settings used
- `test_pred_vs_true.png`: predicted vs true σ and mass on the test set
- `emulator_examples.png`: example curves from the forward-fit baseline
- `bundle.joblib`: the trained models (what `predict.py` loads)

`example_run/` holds a sample of these files from the seed-42 run.

## Results
Grid spacing is 0.25 dex in σ and 0.34 dex in mass. Errors are mean absolute error in dex on **noise-free** held-out curves.

| model | σ error | mass error | curves within 1 grid step (both) |
|---|---|---|---|
| **curve_forest (main)** | **0.022** | **0.050** | **98%** |
| extra-trees on PCA(30) | 0.040 | 0.199 | 79% |
| forward-fit (GP emulator) | 0.052 | 0.649 | 59% |
| predict the training mean | 0.987 | 1.693 | 0% |

Robustness of `curve_forest` over five *fresh* random 70/15/15 splits (seeds 1–5, everything re-tuned each time):
σ error 0.028 ± 0.006 dex, mass error 0.064 ± 0.005 dex, 96–100% of test curves within one grid step.

**Why it improved.** Compressing each curve to 30 PCA components discards the subtle differences that identify mass
(validation error 0.23 → 0.087 dex just from using the full curve). Splitting σ and mass into separate forests with their
own settings took cross-validated error from 0.073 to 0.059 dex. Neural nets, kNN, boosting, RBF/GP emulators and data
augmentation were all tried and were worse on 374 curves.

## Known limitations
- **Noise.** The models are trained on noise-free simulations. With Gaussian noise of 0.05 (asinh units) added to the test
  curves, `curve_forest` drops to 0.16 / 0.78 dex (σ / mass) and the forward-fit (0.05 / 0.67) becomes competitive,
  especially for σ < 1e-42 where the curves barely depend on mass (neighbouring masses differ by < 0.001 asinh units). Noise-free
  scores in that regime reflect exploiting tiny simulation differences. For real/noisy data, train with noise augmentation
  or use the forward-fit posterior. Tree spread is a good warning flag (rank-corr with error 0.8–0.96 for σ, 0.35–0.82 for mass).
- Only 374 curves; held-out points are interior grid points. Extrapolating outside σ∈[1e-43,1e-39], mass∈[1e-6,10] is not supported.
- The seed-42 test set was inspected during development; the five-seed repeat above is the cleaner estimate.
- Astrophysical parameters (Tmin, fX, f★) are fixed in these files, so they are not inferred.
- `Data files/data/` is not used: it overlaps `new_data` (identical to ~1e-8 mK) and has two unreadable files.
