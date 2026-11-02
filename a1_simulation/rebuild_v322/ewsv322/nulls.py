"""Candidate nulls and the Monte Carlo p-value — config keys nulls.*, surrogates.*.

Surrogates are generated from the arm's complete analysed series (t, y) at its own timestamps,
then passed through the identical analysis. Surrogate k of (arm, analysis, null) uses the stream
spawn_key = (K('surrogates'), K('<arm>|<analysis>|<null>'), k) with the patient seed as entropy.
"""
import numpy as np
from . import spec as S
from . import kalman
from .analysis import taus
from .rng import surrogate_stream


def ar1_params(y):
    """IMPL-NOTE (see IMPLEMENTATION_NOTES.md): AR(1) in observation order: phi = lag-1 least-squares coefficient of the demeaned
    series, clipped to [-0.99, 0.99]; innovation SD = SD(y, ddof=0)*sqrt(1-phi^2); start ~ N(0, SD^2)."""
    yc = y - y.mean()
    den = np.sum(yc[:-1] ** 2)
    phi = float(np.clip(np.sum(yc[1:] * yc[:-1]) / den, -0.99, 0.99)) if den > 0 else 0.0
    sd = float(np.std(yc))
    return phi, sd


def surrogate(null, t, y, rng, fit=None, ar=None):
    if null == "shuffle":
        return rng.permutation(y)
    if null == "ar1":
        phi, sd = ar
        n = len(y)
        z = rng.standard_normal(n)
        s = np.empty(n)
        s[0] = sd * z[0]
        e_sd = sd * np.sqrt(1 - phi * phi)
        for i in range(1, n):
            s[i] = phi * s[i - 1] + e_sd * z[i]
        return y.mean() + s
    if null == "ou_ss":
        if fit is None:
            return None          # fit could not be evaluated: surrogate statistic invalid
        return kalman.simulate(t, fit, rng)
    raise ValueError(null)


def mc_pvalue(t_obs, t_surr):
    """p = (1 + #{surrogate >= observed, or surrogate invalid}) / (B + 1). NaN if observed is NaN."""
    t_surr = np.asarray(t_surr, float)
    if not np.isfinite(t_obs):
        return np.nan, int((~np.isfinite(t_surr)).sum())
    invalid = ~np.isfinite(t_surr)
    exceed = invalid | (np.where(invalid, -np.inf, t_surr) >= t_obs)
    return float((1 + exceed.sum()) / (len(t_surr) + 1)), int(invalid.sum())


def detected(tau, p):
    return bool(np.isfinite(tau) and np.isfinite(p) and tau > 0 and p < S.ALPHA_DET)


def run_arm(seed, arm_name, t, y, analyses, nulls, n_surr=S.N_SURR):
    """Run every (analysis, null) for one arm. Returns list of result dicts (one per analysis x null)."""
    t = np.asarray(t, float); y = np.asarray(y, float)
    fit = kalman.fit(t, y) if "ou_ss" in nulls else None
    ar = ar1_params(y) if "ar1" in nulls and len(y) > 1 else (0.0, float(np.std(y)) if len(y) else 0.0)
    lb_q, lb_p = kalman.ljung_box(fit["innovations"]) if fit is not None else (np.nan, np.nan)
    obs = {a: taus(a, t, y) for a in analyses}
    out = []
    for null in nulls:
        # IMPL-NOTE (see IMPLEMENTATION_NOTES.md): 'elapsed' is an identity check, so it re-uses the 'index' surrogate series
        # (same stream key '<arm>|index|<null>'); 'index' is always computed alongside it.
        groups = {}
        for a in analyses:
            groups.setdefault("index" if a == "elapsed" else a, []).append(a)
        for key_analysis, members in groups.items():
            sv = {a: np.full(n_surr, np.nan) for a in members}
            sa = {a: np.full(n_surr, np.nan) for a in members}
            for k in range(n_surr):
                rng = surrogate_stream(seed, arm_name, key_analysis, null, k)
                ys = surrogate(null, t, y, rng, fit=fit, ar=ar)
                if ys is None or not np.all(np.isfinite(ys)):
                    continue
                for a in members:
                    tv, ta, _, _, _ = taus(a, t, ys)
                    sv[a][k] = tv; sa[a][k] = ta
            for a in members:
                tv, ta, nvv, nva, info = obs[a]
                pv, inv_v = mc_pvalue(tv, sv[a])
                pa, inv_a = mc_pvalue(ta, sa[a])
                row = {"arm": arm_name, "analysis": a, "null": null, "n_obs": len(y),
                       "n_valid_windows_var": nvv, "n_valid_windows_ac": nva,
                       "evaluable_var": bool(np.isfinite(tv)), "evaluable_ac": bool(np.isfinite(ta)),
                       "tau_var": tv, "p_var": pv, "det_var": detected(tv, pv), "n_invalid_surr_var": inv_v,
                       "tau_ac": ta, "p_ac": pa, "det_ac": detected(ta, pa), "n_invalid_surr_ac": inv_a}
                row.update(info)
                if null == "ou_ss":
                    row.update({"fit_valid": fit is not None,
                                "fit_theta": fit["theta"] if fit else np.nan, "fit_rho": fit["rho"] if fit else np.nan,
                                "fit_V": fit["V"] if fit else np.nan,
                                "fit_theta_edge": bool(fit["theta_edge"]) if fit else False,
                                "fit_rho_edge": bool(fit["rho_edge"]) if fit else False,
                                "ljung_box_p": lb_p})
                out.append(row)
    return out
