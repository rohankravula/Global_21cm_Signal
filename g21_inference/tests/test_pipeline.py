import numpy as np
import pytest

from g21.data import (Dataset, load_curve, make_grid, parse_params, resample, split_indices)
from g21.features import AsinhTransform, ParamScaler
from g21.pipeline import run


def test_parse_params_handles_exponent_formats():
    assert parse_params("x/sigma_1.0e-39_mass_1.0e+00_dTb.npz") == (1e-39, 1.0)
    assert parse_params("sigma_5.6e-42_mass_4.6e-06_dTb.npz") == (5.6e-42, 4.6e-06)
    with pytest.raises(ValueError):
        parse_params("something_else.npz")


def test_load_curve_sorts_and_drops_duplicate_z(tmp_path):
    z = np.array([10.0, 9.0, 8.0, 8.0, 8.0, 7.0])       # descending, with repeated z (as in the real files)
    t = np.array([1.0, 2.0, 3.0, 3.0, 3.0, 4.0])
    f = tmp_path / "c.npz"
    np.savez(f, z=z, dTb=t, nu=1420.4 / (1 + z))
    z_u, t_u = load_curve(str(f))
    assert list(z_u) == [7.0, 8.0, 9.0, 10.0] and list(t_u) == [4.0, 3.0, 2.0, 1.0]


def test_load_curve_rejects_nan(tmp_path):
    f = tmp_path / "c.npz"
    np.savez(f, z=np.array([1.0, 2.0]), dTb=np.array([0.0, np.nan]))
    with pytest.raises(ValueError):
        load_curve(str(f))


def test_resample_refuses_to_extrapolate():
    u = make_grid(5, 1000, 50)
    with pytest.raises(ValueError):
        resample(np.array([20.0, 500.0]), np.array([0.0, 0.0]), u)


def _grid_params(ns=17, nm=22):
    s, m = np.meshgrid(np.linspace(-43, -39, ns), np.linspace(-6, 1, nm), indexing="ij")
    return np.column_stack([s.ravel(), m.ravel()])


def test_split_is_70_15_15_disjoint_reproducible_and_keeps_edges_in_train():
    p = _grid_params()
    a = split_indices(p, 0.7, 0.15, 0.15, seed=1)
    b = split_indices(p, 0.7, 0.15, 0.15, seed=1)
    assert {k: len(v) for k, v in a.items()} == {"train": 262, "val": 56, "test": 56}
    assert all(np.array_equal(a[k], b[k]) for k in a)
    assert len(set(a["train"]) | set(a["val"]) | set(a["test"])) == len(p)
    edge = (p[:, 0] == p[:, 0].min()) | (p[:, 0] == p[:, 0].max()) | (p[:, 1] == p[:, 1].min()) | (p[:, 1] == p[:, 1].max())
    assert set(np.flatnonzero(edge)) <= set(a["train"])


def test_split_validates_fractions():
    with pytest.raises(ValueError):
        split_indices(_grid_params(), 0.7, 0.2, 0.2, seed=0)


def test_asinh_roundtrip_over_huge_dynamic_range():
    tf = AsinhTransform(1.0)
    t = np.array([-1.1e5, -300.0, -0.04, 0.0, 38.0])
    assert np.allclose(tf.inverse(tf.forward(t)), t)


def test_param_scaler_roundtrip():
    p = _grid_params()
    sc = ParamScaler().fit(p)
    u = sc.transform(p)
    assert u.min() == 0 and u.max() == 1 and np.allclose(sc.inverse_transform(u), p)


def _synthetic_dataset():
    """Smooth toy 'signals' whose shape depends on both parameters (no real physics)."""
    p = _grid_params(12, 12)
    u = make_grid(5.1, 1000, 120)
    a = (p[:, 0:1] + 43) / 4          # 0..1
    b = (p[:, 1:2] + 6) / 7           # 0..1
    curves = -100 * (0.3 + a) * np.exp(-(u[None] - (2.5 + 2 * b)) ** 2 / (2 * 0.4 ** 2)) \
             + 20 * b * np.exp(-(u[None] - 4.5) ** 2 / (2 * 0.8 ** 2))
    return Dataset(p, curves, [f"f{i}.npz" for i in range(len(p))], u)


def test_pipeline_recovers_parameters_on_synthetic_data(tmp_path):
    cfg = {
        "seed": 0,
        "split": {"train": 0.7, "val": 0.15, "test": 0.15, "keep_boundary_in_train": True},
        "transform": {"asinh_scale_mK": 1.0},
        "reducer": {"variance_to_keep": 0.9999, "max_components": 8},
        "curve_forest": {"n_estimators": 60, "selection_trees": 30,
                         "max_features": {"sigma": [0.1, 0.5], "mass": [0.1, 0.5]}},
        "forward": {"n_restarts": 0, "nu": 2.5},
        "inference": {"grid_per_axis": 41, "posterior_noise": 0.05, "refit_on_train_val": True,
                      "candidates": {"extra_trees": [{"min_samples_leaf": 1}], "knn": [{"n_neighbors": 1}]}},
        "evaluation": {"sigma_bins": [-43, -41, -39], "noise_levels": [0.0]},
    }
    m = run(cfg, _synthetic_dataset(), str(tmp_path))
    res = m["test_results"]["noise_0.0"]
    # every method should beat "always predict the training mean" by a wide margin
    base = np.mean(res["baseline_train_mean"]["mae_dex"])
    for name in ("curve_forest", "extra_trees", "forward_fit_mean"):
        assert np.mean(res[name]["mae_dex"]) < 0.25 * base, name
    assert res["forward_fit_mean"]["within_1_grid_step_both"] >= 0.8
    for f in ("metrics.json", "split.json", "bundle.joblib", "test_pred_vs_true.png"):
        assert (tmp_path / f).exists()


def test_curve_forest_returns_mean_and_spread_and_is_deterministic():
    from g21.models import CurveForest
    rng = np.random.default_rng(0)
    y = rng.normal(size=(80, 30))
    u = np.column_stack([y[:, 0] * 0.5 + 0.5, y[:, 1] ** 2])       # learnable from the curve
    a = CurveForest((0.3, 0.3), 40, seed=3).fit(y, u)
    b = CurveForest((0.3, 0.3), 40, seed=3).fit(y, u)
    m, sd = a.predict(y[:5], return_std=True)
    assert m.shape == (5, 2) and sd.shape == (5, 2) and (sd >= 0).all()
    assert np.allclose(m, b.predict(y[:5]))
    assert np.abs(m - u[:5]).mean() < 0.3
