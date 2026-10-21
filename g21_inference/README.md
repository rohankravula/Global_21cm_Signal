# g21_inference — recover (σ, mass) from a global 21cm signal

Given a brightness-temperature curve `dTb(z)`, estimate the dark-matter–baryon cross-section σ and
dark-matter mass that produced it. Trained on the `new_data/*.npz` simulations (17 σ × 22 mass = 374 curves).

## Quick start
```bash
git clone https://github.com/rohankravula/Global_21cm_Signal.git      # next to this folder
pip install -r requirements.txt
python train.py --data-dir ../Global_21cm_Signal/new_data              # ~2 min; writes runs/<timestamp>/
python predict.py --run runs/<timestamp> some_curve_dTb.npz            # curve needs arrays `z` and `dTb`
pytest                                                                 # 9 tests
```
Everything tunable is in `config.yaml` (z-range, split, transform, models, noise levels).

## What it does
1. **Load/clean** (`g21/data.py`): sort by z, drop duplicate z points, reject NaN/corrupt files (reported, not silent),
   resample every curve onto one common grid, uniform in log(1+z), z = 5.1–1000, 400 points. Parameters are parsed
   from the filename and used as log10.
2. **Split 70/15/15 by curve** (262 / 56 / 56), seeded. Edge-of-grid curves stay in train so val/test measure
   *interpolation*. Split lists are saved to `split.json`.
3. **Transform**: `asinh(dTb / 1 mK)` (dTb spans −1.1e5 … +38 mK), then PCA (30 components) — both fitted on train only.
4. **Two ways to do inference**, compared on the same held-out curves:
   - *Direct inverse*: curve → (σ, mass) with random forest, extra-trees, kNN, MLP (settings picked on val).
   - *Forward-fit*: a Gaussian-process emulator (σ, mass → curve) scanned over a dense parameter grid. Each cell is weighted
     by a Gaussian likelihood that includes observation noise **plus the emulator's own calibrated uncertainty**, giving a best
     match, a posterior mean and a posterior width.
5. **Evaluate once on test**, at noise 0 / 0.05 / 0.2 (Gaussian noise in asinh units), overall and per σ regime.

Outputs per run: `metrics.json`, `split.json`, `config_used.yaml`, `test_pred_vs_true.png`, `emulator_examples.png`, `bundle.joblib`.

## Results (one seed, 56 test curves — treat as indicative)
Grid spacing is 0.25 dex in σ and 0.34 dex in mass.

| true log σ | n | forward-fit mass error (dex) | posterior width it reports | extra-trees mass error, noise 0 → 0.05 |
|---|---|---|---|---|
| ≥ −41 (strong coupling) | 30 | **0.08** | ±0.03 | 0.13 → 0.13 |
| −42 … −41 | 12 | 1.28 | ±0.75 | 0.30 → 0.98 |
| −43 … −42 (weak coupling) | 14 | 1.32 | ±1.30 | 0.27 → 1.37 |

σ itself is recovered to ~0.05 dex overall. Emulator test RMSE is 0.22 (asinh units) versus 2.02 for "predict the mean curve".

**Key finding — mass is physically degenerate at low σ.** For σ ≲ 1e-42, curves at neighbouring masses differ by < 0.001
(asinh units), i.e. the signal contains essentially no mass information. In that region the forward-fit correctly reports a wide
posterior instead of a confident wrong answer. The direct models look good there *only on noise-free curves* (they exploit
sub-0.001 differences) and collapse as soon as small noise is added — so noise-free scores in that regime are misleading.

## Known limitations
- Forward-fit posteriors are somewhat overconfident in the strong-coupling regime (std ≈ 0.03 vs actual error ≈ 0.08 dex).
- Only 374 curves and a single split; use several seeds / k-fold before quoting numbers. Held-out points are interior grid
  points; extrapolating outside σ∈[1e-43,1e-39], mass∈[1e-6,10] is not supported.
- Astrophysical parameters (Tmin, fX, f★) are fixed in these files, so they are not inferred.
- Noise model is simple white noise in asinh space, not a real instrument model.
- MLP underperforms with this little data (kept for comparison). PCA is capped at 30 components (99.96% of variance).
- `data/` is not used: it overlaps `new_data` (identical to ~1e-8 mK) and has two unreadable files.
