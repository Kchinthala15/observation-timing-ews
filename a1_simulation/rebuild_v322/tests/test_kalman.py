"""State-space OU surrogate: exact likelihood, profiling of Q, rho definition, exact transitions."""
import numpy as np
from scipy.stats import multivariate_normal

from ewsv322 import kalman as K
from ewsv322.rng import gen


def _cov(t, theta, rho, V):
    D = np.abs(t[:, None] - t[None, :])
    return V * np.exp(-theta * D) + rho * V * np.eye(len(t))


def test_kalman_likelihood_equals_dense_gaussian():
    g = gen(11, "test")
    t = np.sort(g.uniform(0, 30, 15))
    y = g.normal(0.2, 1.0, 15)
    for theta, rho, V in ((0.7, 0.3, 1.3), (5.0, 0.0, 0.4), (0.02, 3.2, 2.0)):
        dense = multivariate_normal(mean=np.full(15, 0.2), cov=_cov(t, theta, rho, V)).logpdf(y)
        assert np.isclose(K.loglik_at(t, y, 0.2, theta, rho, V), dense, rtol=0, atol=1e-8)


def test_profiled_variance_is_the_maximiser():
    g = gen(12, "test")
    t = np.sort(g.uniform(0, 30, 25))
    y = g.normal(0.0, 0.5, 25)
    f = K.fit(t, y)
    assert f is not None and np.isclose(f["mu"], y.mean())
    assert np.isclose(K.loglik_at(t, y, f["mu"], f["theta"], f["rho"], f["V"]), f["loglik"], atol=1e-8)
    for s in (0.95, 1.05):
        assert K.loglik_at(t, y, f["mu"], f["theta"], f["rho"], f["V"] * s) < f["loglik"]
    # the selected grid point beats every other grid point (each at its own profiled V)
    for th in K.THETA_GRID[::7]:
        for rh in K.RHO_GRID:
            d = y - f["mu"]
            slog, sv2, _, _ = K._kalman_unit(t, d, np.array([th]), np.array([rh]))
            V = sv2[0] / len(t)
            assert K.loglik_at(t, y, f["mu"], th, rh, V) <= f["loglik"] + 1e-9
    assert np.isclose(f["Q"], 2 * f["theta"] * f["V"]) and np.isclose(f["R"], f["rho"] * f["V"])


def test_surrogate_rho_definition_and_exact_transition():
    t = np.array([10.0, 10.5, 12.0])
    f = {"mu": 1.0, "theta": 0.8, "rho": 0.5, "V": 2.0, "R": 1.0}
    Y = np.array([K.simulate(t, f, gen(13, "test", i)) for i in range(20000)])
    C = np.cov(Y.T)
    target = _cov(t, 0.8, 0.5, 2.0)                  # V exp(-theta|dt|) + rho*V on the diagonal
    assert np.allclose(Y.mean(axis=0), 1.0, atol=0.05)
    assert np.allclose(C, target, rtol=0.06, atol=0.05)


def test_invalid_fit_returns_none():
    t = np.arange(10.0)
    assert K.fit(t, np.ones(10)) is None                 # zero variance -> profiled Q not positive
    assert K.fit(np.array([1.0, 1.0, 2.0]), np.array([0.1, 0.2, 0.3])) is None   # non-increasing times


def test_edge_flags_are_reported():
    g = gen(14, "test")
    t = np.sort(g.uniform(0, 30, 30))
    f = K.fit(t, g.normal(0, 1, 30))
    assert isinstance(f["theta_edge"], bool) and isinstance(f["rho_edge"], bool)
