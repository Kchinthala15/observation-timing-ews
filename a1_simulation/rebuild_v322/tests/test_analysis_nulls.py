"""Window statistics, Kendall tau-a, fixed-time endpoints, binning rules, evaluability,
surrogate reproducibility and the plus-one Monte Carlo p-value."""
import numpy as np
from scipy.stats import kendalltau

from ewsv322 import analysis as AN
from ewsv322 import nulls as NL
from ewsv322.rng import gen


def test_kendall_tau_a_matches_scipy_without_ties_and_nan_rule():
    g = gen(21, "test")
    x = np.arange(20.0); y = g.normal(size=20)
    assert np.isclose(AN.kendall_tau_a(x, y), kendalltau(x, y)[0])
    y5 = y.copy(); y5[5:] = np.nan
    assert np.isnan(AN.kendall_tau_a(x, y5))             # only 5 valid statistics (< 6)
    y6 = y.copy(); y6[6:] = np.nan
    assert np.isfinite(AN.kendall_tau_a(x, y6))
    Y = np.vstack([y, -y])
    assert np.allclose(AN.kendall_tau_a(x, Y), [kendalltau(x, y)[0], -kendalltau(x, y)[0]])


def test_index_and_elapsed_are_identical():
    g = gen(22, "test")
    for _ in range(20):
        n = int(g.integers(20, 90))
        t = np.sort(g.uniform(0, 100, n)); y = g.normal(size=n)
        a = AN.taus("index", t, y); b = AN.taus("elapsed", t, y)
        assert (a[0] == b[0] or (np.isnan(a[0]) and np.isnan(b[0])))
        assert (a[1] == b[1] or (np.isnan(a[1]) and np.isnan(b[1])))


def test_fixed_time_start_positions():
    for W in (20, 30, 40):
        st = AN.window_starts(W, 43)
        assert len(st) == 43 and st[0] == 0 and np.isclose(st[-1], 100 - W)
        assert np.allclose(np.diff(st), (100 - W) / 42)


def test_fixed_time_min_obs():
    t = np.array([1.0, 2.0, 3.0, 4.0] + list(np.linspace(30, 99, 40)))
    y = gen(23, "test").normal(size=len(t))
    var, ac, st, _ = AN.window_stats("fixed_time|W20", t, y)
    assert np.isnan(var[0])                              # window [0,20) holds 4 observations (< 5)


def test_binning_rules():
    t = np.array([5.0, 35.0, 95.0]); y = np.array([1.0, 2.0, 4.0])
    assert np.allclose(AN._bin(t, y, 10, "drop"), [1.0, 2.0, 4.0])
    lo = AN._bin(t, y, 10, "locf")
    assert np.allclose(lo, [1, 1, 1, 2, 2, 2, 2, 2, 2, 4])
    li = AN._bin(t, y, 10, "linear_interp")
    assert np.isclose(li[0], 1) and np.isclose(li[3], 2) and np.isclose(li[9], 4)
    assert np.isclose(li[6], 2 + (4 - 2) * (65 - 35) / (95 - 35))


def test_short_series_not_evaluable():
    t = np.linspace(1, 99, 8); y = gen(24, "test").normal(size=8)
    tv, ta, nvv, nva, _ = AN.taus("index", t, y)       # window 5 -> 4 windows < 6
    assert np.isnan(tv) and np.isnan(ta) and nvv == 4


def test_plus_one_pvalue_ties_invalid_and_minimum():
    B = 200
    assert NL.mc_pvalue(0.5, np.full(B, 0.4))[0] == 1 / (B + 1)       # minimum attainable p
    assert NL.mc_pvalue(0.5, np.full(B, 0.5))[0] == 1.0               # ties count as exceedances
    s = np.full(B, 0.4); s[:100] = np.nan
    p, inv = NL.mc_pvalue(0.5, s)
    assert p == 101 / 201 and inv == 100                                # invalid count as exceedances
    assert NL.mc_pvalue(0.5, np.full(B, np.nan))[0] == 1.0             # all invalid -> p = 1
    assert np.isnan(NL.mc_pvalue(np.nan, np.full(B, 0.1))[0])
    assert 1 / (B + 1) < 0.05


def test_detection_requires_positive_tau_and_p_below_005():
    assert NL.detected(0.3, 0.01)
    assert not NL.detected(0.0, 0.004) and not NL.detected(-0.2, 0.004)
    assert not NL.detected(0.3, 0.05) and not NL.detected(np.nan, 0.01)


def test_surrogates_reproducible_and_nulls_behave():
    g = gen(25, "test")
    t = np.sort(g.uniform(0, 100, 60)); y = g.normal(size=60)
    from ewsv322 import kalman
    f = kalman.fit(t, y)
    ar = NL.ar1_params(y)
    for null in ("ou_ss", "ar1", "shuffle"):
        a = NL.surrogate(null, t, y, gen(1, "surrogates", "regular_N|index|" + null, 3), fit=f, ar=ar)
        b = NL.surrogate(null, t, y, gen(1, "surrogates", "regular_N|index|" + null, 3), fit=f, ar=ar)
        assert np.array_equal(a, b) and len(a) == 60
    sh = NL.surrogate("shuffle", t, y, gen(1, "x"))
    assert np.allclose(np.sort(sh), np.sort(y))
    assert NL.surrogate("ou_ss", t, y, gen(1, "x"), fit=None) is None    # invalid fit -> invalid surrogate


def test_run_arm_row_fields():
    g = gen(26, "test")
    t = np.sort(g.uniform(0, 100, 60)); y = g.normal(size=60)
    rows = NL.run_arm(3_000_001, "regular_N", t, y, ["index", "elapsed"], ["ou_ss", "shuffle"], n_surr=10)
    assert len(rows) == 4
    for r in rows:
        assert 1 / 11 - 1e-12 <= r["p_var"] <= 1 and r["n_obs"] == 60
    idx = [r for r in rows if r["analysis"] == "index"]
    el = [r for r in rows if r["analysis"] == "elapsed"]
    for a, b in zip(idx, el):
        assert a["tau_var"] == b["tau_var"] and a["p_var"] == b["p_var"]   # identity check incl. surrogates
