# Global_21cm_Signal

Rohan Ravula and Aryan Rahimeh's project on the global 21cm signal and dark matter.

## What this project is about
The global 21cm signal is the average brightness temperature of hydrogen's 21cm line (`dTb`) as a function of
redshift `z` (equivalently, frequency `nu = 1420.4058 / (1 + z)` MHz). If dark matter interacts with ordinary matter,
the signal changes, so the curve carries information about the dark-matter–baryon cross-section (σ) and the
dark-matter mass.

We have simulated curves for a grid of σ and mass values. The machine-learning part of the project goes backwards:
given a curve, estimate the σ and mass that produced it.

## Repository layout
| Folder / file | What it is |
|---|---|
| [`g21_inference/`](g21_inference/) | The main code: trains a model to recover σ and mass from a curve, plus tests. **Start here.** See its README. |
| `Data files/new_data/` | 374 simulated curves (17 σ values × 22 mass values), one `.npz` per curve. This is the training data. |
| `Data files/data/` | An older, overlapping set of curves. Not used by `g21_inference`. |
| `meeting notes/` | Notes and early practice scripts from meetings, by date. |
| `inspector.py` | Small tool to look inside a `.npz` data file. |

## The data
Each file is named `sigma_<value>_mass_<value>_dTb.npz`, so the σ and mass used for the simulation are in the filename.
The file holds three arrays: `nu`, `z` and `dTb`. Other physics parameters are fixed in these files (Tmin = 500,
fX = 1, f★ = 1e-2), so only σ and mass vary.

To look at a file:
```bash
python inspector.py "Data files/new_data/sigma_1.0e-40_mass_1.0e+00_dTb.npz"
```
(needs `numpy` and `pandas`.)

## Quick start
```bash
cd g21_inference
python -m pip install -r requirements.txt
python train.py        # trains on new_data with a 70/15/15 train/validation/test split
python predict.py      # predicts σ and mass for held-out test curves and compares with the truth
```
Results on curves the model never trained on: σ is off by about 0.03 dex and mass by about 0.06 dex on average
(1 dex = a factor of 10) for noise-free curves. The model is much less accurate when noise is added to the curves.
Details, limitations and how to read the outputs are in [`g21_inference/README.md`](g21_inference/README.md).

## Meeting notes
- [2024-12-09](meeting%20notes/2024-12-09/)
- [2025-01-25](meeting%20notes/2025-01-25/)
- [2025-03-21](meeting%20notes/2025-03-21/)
- [2025-04-21](meeting%20notes/2025-04-21/)
