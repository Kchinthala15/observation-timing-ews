"""Latent-process tests: Euler update, initial state, time-varying coefficients, fold stop/continuation, noise-only."""
import numpy as np

from ewsv322 import spec as S
from ewsv322 import latent as L
from ewsv322 import schedule as SC
from ewsv322.rng import gen


def test_coefficients_at_key_times():
    assert np.isclose(L.theta_of("ou_csd", 0.0), 1.2) and np.isclose(L.theta_of("ou_csd", 15.0), 1.2)
    assert np.isclose(L.theta_of("ou_csd", 45.0), 1.2 + (0.05 - 1.2) * 0.5)
    assert np.isclose(L.theta_of("ou_csd", 75.0), 0.05) and np.isclose(L.theta_of("ou_csd", 100.0), 0.05)
    assert np.isclose(L.mu_of("ou_combined", 75.0), -2.0) and np.isclose(L.mu_of("ou_combined", 10.0), 0.0)
    assert np.isclose(L.theta_of("level_drop_only", 60.0), 1.2) and np.isclose(L.mu_of("level_drop_only", 45.0), -1.0)
    assert np.isclose(L.theta_of("stable", 90.0), 1.2) and np.isclose(L.mu_of("stable", 90.0), 0.0)
    assert np.isclose(L.r_of(0.0), 1.0) and np.isclose(L.r_of(45.0), 0.525)
    assert np.isclose(L.r_of(75.0), 0.05) and np.isclose(L.r_of(80.0), 0.05) and np.isclose(L.r_of(100.0), 0.05)


def test_euler_update_left_endpoint():
    Z = gen(1, "test").standard_normal(S.NGRID - 1)
    X = L.ou_path("ou_combined", 0.1, Z)
    x = 0.1
    for j in range(S.NGRID - 1):
        t = j * S.DT
        q = min(max((t - 15) / 60, 0), 1)
        th = 1.2 + (0.05 - 1.2) * q
        mu = -2.0 * q
        x = x + th * (mu - x) * 0.02 + 0.3 * np.sqrt(0.02) * Z[j]
        if j in (0, 749, 750, 3000, 4999):
            assert np.isclose(X[j + 1], x, rtol=0, atol=1e-10)


def test_initial_state_is_stationary_normal():
    x0 = np.array([L.simulate("ou_csd", 1_000_000 + i)["X"][0] for i in range(600)])
    assert abs(x0.mean()) < 0.03                          # ~3.8 standard errors
    assert abs(x0.var() - 0.3 ** 2 / (2 * 1.2)) < 0.008  # ~3.6 standard errors


def test_simulate_draw_order_matches_documentation():
    g = gen(77, "latent")
    x0 = g.normal(0.0, np.sqrt(0.09 / 2.4)); Z = g.standard_normal(S.NGRID - 1)
    assert np.allclose(L.simulate("stable", 77)["X"], L.ou_path("stable", x0, Z))


def test_fold_stops_immediately_after_tipping():
    Z = np.full(S.NGRID - 1, -50.0)
    X, tipped, t_tip = L.fold_path(Z)
    assert tipped and t_tip > 0
    j = int(round(t_tip / S.DT))
    assert np.isfinite(X[: j + 1]).all() and np.isnan(X[j + 1:]).all()
    assert X[j] < -np.sqrt(L.r_of(t_tip))


def test_fold_continues_after_75_with_r_held():
    X, tipped, _ = L.fold_path(np.zeros(S.NGRID - 1))
    assert not tipped and np.isfinite(X).all() and len(X) == S.NGRID
    assert abs(X[-1] - np.sqrt(0.05)) < 0.02          # tracks the upper branch with r = 0.05 after t = 75


def test_fold_tipping_detected_after_75():
    Z = np.zeros(S.NGRID - 1)
    j80 = int(round(80 / S.DT))
    Z[j80] = -50.0                                    # step of 0.1*sqrt(0.02)*(-50) = -0.707
    X, tipped, t_tip = L.fold_path(Z)
    assert tipped and np.isclose(t_tip, (j80 + 1) * S.DT)
    assert t_tip > 75


def test_noise_only_truth_is_zero_and_values_are_pure_error():
    X = L.simulate("noise_only", 123)["X"]
    assert np.all(X == 0)
    x, e, y = SC.measure(X, [3.3, 20.01, 20.015], gen(9, "err_inf|step|2.00|primary"))
    assert np.all(x == 0) and np.array_equal(y, e)


def test_value_at_linear_interpolation():
    X = np.arange(S.NGRID, dtype=float) * 2.0        # X(t_j) = 2j  ->  X(t) = 2 t / dt
    for t in (0.0, 0.013, 13.337, 99.999):
        assert np.isclose(L.value_at(X, t), 2 * t / S.DT)
