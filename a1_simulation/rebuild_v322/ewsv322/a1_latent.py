"""A1 latent models: stable_slow_<theta> and fold_v2 grid variants. Frozen models are delegated
unchanged to latent.simulate.

stable_slow: stationary OU with theta fixed (0.2 or 0.05), mu = 0, sigma = 0.3,
    X(0) ~ N(0, sigma^2/(2 theta)); Euler-Maruyama on the frozen grid, same draw order on stream
    'latent' as the frozen OU models (X0 first, then Z[0..NGRID-2]).
fold_v2|sf=<sigma_f>|re=<r_end>: the frozen fold equations and tipping rule with sigma_f and r_end
    replaced; r0 = 1 and r(t) = r_end for t > 75 (as frozen, where r_after_transition = r_end).
"""
import numpy as np
from . import spec as S
from . import latent
from .rng import patient_stream
from .a1_spec import STABLE_SLOW, parse_fold

T_GRID = latent.T_GRID


def _stable_slow(theta, seed):
    rng = patient_stream(seed, "latent")
    x0 = rng.normal(0.0, np.sqrt(S.SIGMA ** 2 / (2 * theta)))
    Z = rng.standard_normal(S.NGRID - 1)
    sd = S.SIGMA * np.sqrt(S.DT)
    X = np.empty(S.NGRID)
    X[0] = x0
    for j in range(S.NGRID - 1):
        X[j + 1] = X[j] + theta * (S.MU0 - X[j]) * S.DT + sd * Z[j]
    return {"X": X, "tipped": False, "t_tip": np.nan}


def r_of_v2(t, r_end):
    t = np.asarray(t, float)
    return np.where(t > S.T_END, r_end, S.FOLD_R0 + (r_end - S.FOLD_R0) * latent.q(t))


def _fold_v2(sf, re, seed):
    rng = patient_stream(seed, "latent")
    Z = rng.standard_normal(S.NGRID - 1)
    r = r_of_v2(T_GRID, re)
    thr = -np.sqrt(r)
    sd = sf * np.sqrt(S.DT)
    X = np.full(S.NGRID, np.nan)
    X[0] = np.sqrt(S.FOLD_R0)
    for j in range(S.NGRID - 1):
        X[j + 1] = X[j] + (r[j] - X[j] ** 2) * S.DT + sd * Z[j]
        if X[j + 1] < thr[j + 1]:
            return {"X": X, "tipped": True, "t_tip": float(T_GRID[j + 1])}
    return {"X": X, "tipped": False, "t_tip": np.nan}


def simulate(model, seed):
    if model in STABLE_SLOW:
        return _stable_slow(STABLE_SLOW[model], seed)
    if model.startswith("fold_v2|"):
        sf, re = parse_fold(model)
        return _fold_v2(sf, re, seed)
    return latent.simulate(model, seed)
