"""Stationary OU + measurement-error model — config keys nulls.state_space_ou_surrogate,
analyses.continuous_time.

Model at observation times t_1 < ... < t_n (constant mean mu):
    X_1 ~ N(mu, V),  X_{i+1} = mu + exp(-theta*D)(X_i - mu) + eta,  eta ~ N(0, V(1 - exp(-2 theta D)))
    Y_i = X_i + eps_i,  eps_i ~ N(0, R),  R = rho * V,  V = V_inf = Q / (2 theta)
For fixed (theta, rho) every covariance is proportional to V, so the exact Gaussian likelihood is
maximised over V (equivalently over Q = 2 theta V > 0) in closed form:
    V_hat = (1/n) sum v_i^2 / F_i  (Kalman run with V = 1),
    loglik = -0.5 * (sum log F_i + n log V_hat + n + n log 2 pi).
The grid point with the largest profiled log-likelihood is selected (first maximum in grid order).
"""
import numpy as np
from . import spec as S

THETA_GRID = np.geomspace(S.THETA_GRID_LO, S.THETA_GRID_HI, S.THETA_GRID_N)
RHO_GRID = np.array(S.RHO_GRID)
TH, RH = [a.ravel() for a in np.meshgrid(THETA_GRID, RHO_GRID, indexing="ij")]  # theta-major, 32*8 points
LOG2PI = np.log(2 * np.pi)


def _kalman_unit(t, d, th, rh):
    """Kalman filter with V = 1, vectorised over grid points. Returns (sum_logF, sum_v2F, v, F)."""
    n = len(t)
    G = len(th)
    m = np.zeros(G)
    P = np.ones(G)
    slog = np.zeros(G)
    sv2 = np.zeros(G)
    vs = np.empty((n, G)); Fs = np.empty((n, G))
    for i in range(n):
        if i > 0:
            a = np.exp(-th * (t[i] - t[i - 1]))
            m = a * m
            P = a * a * P + (1.0 - a * a)
        F = P + rh
        v = d[i] - m
        slog += np.log(F)
        sv2 += v * v / F
        K = P / F
        m = m + K * v
        P = P * rh / F            # = P(1-K), exact also for rho = 0
        vs[i] = v; Fs[i] = F
    return slog, sv2, vs, Fs


def fit(t, y, mu=None):
    """Grid MLE. Returns dict or None if no grid point gives a finite likelihood (fit invalid)."""
    t = np.asarray(t, float); y = np.asarray(y, float)
    n = len(t)
    if n < 2 or not np.all(np.isfinite(y)) or np.any(np.diff(t) <= 0):
        return None
    mu = float(np.mean(y)) if mu is None else float(mu)
    d = y - mu
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        slog, sv2, vs, Fs = _kalman_unit(t, d, TH, RH)
        V = sv2 / n
        ll = -0.5 * (slog + n * np.log(V) + n + n * LOG2PI)
    ok = np.isfinite(ll) & (V > 0) & np.isfinite(V)
    if not ok.any():
        return None
    ll = np.where(ok, ll, -np.inf)
    b = int(np.argmax(ll))
    it, ir = divmod(b, len(RHO_GRID))
    th, rh, Vb = float(TH[b]), float(RH[b]), float(V[b])
    innov = vs[:, b] / np.sqrt(Fs[:, b] * Vb)       # standardised innovations
    return {"mu": mu, "theta": th, "rho": rh, "V": Vb, "Q": 2 * th * Vb, "R": rh * Vb, "loglik": float(ll[b]),
            "theta_edge": it in (0, len(THETA_GRID) - 1), "rho_edge": ir in (0, len(RHO_GRID) - 1),
            "innovations": innov}


def loglik_at(t, y, mu, theta, rho, V):
    """Exact log-likelihood at given parameters (used by the dense-covariance unit test)."""
    t = np.asarray(t, float); d = np.asarray(y, float) - mu
    slog, sv2, _, _ = _kalman_unit(t, d, np.array([theta]), np.array([rho]))
    n = len(t)
    return float(-0.5 * (slog[0] + n * np.log(V) + sv2[0] / V + n * LOG2PI))


def simulate(t, f, rng):
    """One stationary OU path at the exact timestamps + independent N(0, R) errors.
    Draw order: n standard normals for the latent path, then n for the errors."""
    t = np.asarray(t, float)
    n = len(t)
    z1 = rng.standard_normal(n)
    z2 = rng.standard_normal(n)
    th, V = f["theta"], f["V"]
    x = np.empty(n)
    x[0] = np.sqrt(V) * z1[0]
    for i in range(1, n):
        a = np.exp(-th * (t[i] - t[i - 1]))
        x[i] = a * x[i - 1] + np.sqrt(V * (1.0 - a * a)) * z1[i]
    return f["mu"] + x + np.sqrt(f["R"]) * z2


def ljung_box(z, lags=5):
    """Ljung-Box Q on standardised innovations; returns (Q, p) with chi-square(lags) reference."""
    from scipy.stats import chi2
    z = np.asarray(z, float)
    n = len(z)
    if n <= lags + 1:
        return np.nan, np.nan
    zc = z - z.mean()
    den = np.sum(zc * zc)
    if den <= 0:
        return np.nan, np.nan
    r = np.array([np.sum(zc[k:] * zc[:-k]) / den for k in range(1, lags + 1)])
    Qs = n * (n + 2) * np.sum(r * r / (n - np.arange(1, lags + 1)))
    return float(Qs), float(chi2.sf(Qs, lags))
