"""Latent processes — config keys latent.*, latent.ou_equations.*, nulls.controls.noise_only_control.

Grid: t_j = j*dt, j = 0..NGRID-1 (t_0 = 0, t_last = T).
OU models (Euler-Maruyama, left-endpoint coefficients):
    X[j+1] = X[j] + theta(t_j)*(mu(t_j) - X[j])*dt + sigma*sqrt(dt)*Z[j]
Fold: X[j+1] = X[j] + (r(t_j) - X[j]^2)*dt + sigma_f*sqrt(dt)*Z[j], X(0)=1;
    tipping checked at every grid time in [0, T] (latent.fold.tipping_rule; the stale
    't<=75' sentence in latent.ou_equations.fold.tipping is superseded — IMPL-NOTE (see IMPLEMENTATION_NOTES.md)).
Noise-only control: X(t) = 0 for all t.
"""
import numpy as np
from . import spec as S
from .rng import patient_stream

T_GRID = np.arange(S.NGRID) * S.DT


def q(t):
    return np.clip((np.asarray(t, float) - S.T_START) / (S.T_END - S.T_START), 0.0, 1.0)


def theta_of(model, t):
    if model in ("ou_csd", "ou_combined"):
        return S.THETA0 + (S.THETA_MIN - S.THETA0) * q(t)
    if model in ("stable", "level_drop_only"):
        return np.full_like(np.asarray(t, float), S.THETA0)
    raise ValueError(model)


def mu_of(model, t):
    if model in ("ou_csd", "stable"):
        return np.full_like(np.asarray(t, float), S.MU0)
    if model in ("ou_combined", "level_drop_only"):
        return -S.LEVEL_DROP * q(t)
    raise ValueError(model)


def r_of(t):
    """r(t) = r0 + (r_end - r0) q(t) for t <= 75; r_end = 0.05 for t > 75 (q is clipped at 1)."""
    t = np.asarray(t, float)
    return np.where(t > S.T_END, S.FOLD_R_AFTER, S.FOLD_R0 + (S.FOLD_REND - S.FOLD_R0) * q(t))


def reference_level(model, t):
    """schedules.smooth.reference_level."""
    if model in ("ou_csd", "stable", "ou_combined", "level_drop_only"):
        return mu_of(model, t)
    if model == "fold":
        return np.sqrt(r_of(t))
    raise ValueError(f"smooth schedule has no reference level for model {model!r}")


def ou_path(model, x0, Z):
    th = theta_of(model, T_GRID[:-1])
    mu = mu_of(model, T_GRID[:-1])
    sd = S.SIGMA * np.sqrt(S.DT)
    X = np.empty(S.NGRID)
    X[0] = x0
    for j in range(S.NGRID - 1):
        X[j + 1] = X[j] + th[j] * (mu[j] - X[j]) * S.DT + sd * Z[j]
    return X


def fold_path(Z, x0=None):
    """Returns (X, tipped, t_tip). After tipping, X is NaN (no subsequent values computed)."""
    x0 = np.sqrt(S.FOLD_R0) if x0 is None else x0
    r = r_of(T_GRID)
    thr = -np.sqrt(r)
    sd = S.FOLD_SIGMA * np.sqrt(S.DT)
    X = np.full(S.NGRID, np.nan)
    X[0] = x0
    if X[0] < thr[0]:
        return X, True, 0.0
    for j in range(S.NGRID - 1):
        X[j + 1] = X[j] + (r[j] - X[j] ** 2) * S.DT + sd * Z[j]
        if X[j + 1] < thr[j + 1]:
            return X, True, float(T_GRID[j + 1])
    return X, False, np.nan


def simulate(model, seed):
    """dict(X, tipped, t_tip). Draw order on stream 'latent': OU -> X0 then Z[0..NGRID-2]; fold -> Z only."""
    if model == "noise_only":
        return {"X": np.zeros(S.NGRID), "tipped": False, "t_tip": np.nan}
    rng = patient_stream(seed, "latent")
    if model == "fold":
        Z = rng.standard_normal(S.NGRID - 1)
        X, tipped, t_tip = fold_path(Z)
        return {"X": X, "tipped": tipped, "t_tip": t_tip}
    x0 = rng.normal(0.0, np.sqrt(S.SIGMA ** 2 / (2 * S.THETA0)))
    Z = rng.standard_normal(S.NGRID - 1)
    return {"X": ou_path(model, x0, Z), "tipped": False, "t_tip": np.nan}


def value_at(X, t):
    """observation_mapping: linear interpolation between neighbouring grid values; t never rounded."""
    t = np.asarray(t, float)
    u = t / S.DT
    j = np.clip(np.floor(u).astype(int), 0, S.NGRID - 2)
    w = u - j
    return X[j] * (1.0 - w) + X[j + 1] * w
