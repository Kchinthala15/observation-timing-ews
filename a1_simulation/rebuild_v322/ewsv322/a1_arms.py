"""A1 arms: the frozen arms from arms.build, plus poisson_matched.

poisson_matched: the same 8 baseline observations; then k_post = N_i - 8 observation times drawn
i.i.d. Uniform[15, 100) from stream 'pm_times' and sorted (homogeneous Poisson times conditional on
their count); measured with independent errors from stream 'err_pm'. Exactly count-matched to
regular_N and informative_reduced_N, and independent of the patient's state.
"""
import numpy as np
from . import spec as S
from . import arms
from .schedule import measure
from .rng import patient_stream


def poisson_matched(X, seed, base, N_i):
    k_post = int(N_i) - S.N_BASE
    t = np.sort(patient_stream(seed, "pm_times").uniform(S.T_START, S.T, size=max(k_post, 0)))
    _, _, y = measure(X, t, patient_stream(seed, "err_pm"))
    return arms._arm(base, t, y)


def build(X, seed, base, inf, schedule, m, trigger, wanted, n_variability_draws=0):
    frozen_wanted = set(wanted) - {"poisson_matched"}
    out = arms.build(X, seed, base, inf, schedule, m, trigger, frozen_wanted, n_variability_draws)
    if "poisson_matched" in wanted:
        out["poisson_matched"] = poisson_matched(X, seed, base, out["_meta"]["N_i"])
    return out
