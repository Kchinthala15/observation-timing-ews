"""A1 tests: frozen ou_ss / ar1 / shuffle (delegated unchanged to nulls.run_arm) plus ar1_rep and
shuffle_rep (A1 new_nulls).

ar1_rep / shuffle_rep - the analyst pipeline 'regularise first, then use a standard test':
  * binned|nb|rule and interp|npts: z = R(t, y) is the representation series (frozen definitions in
    analysis.py); fit the frozen AR(1) rule to z (or permute z); window statistics are computed
    directly on the surrogate z* (z* is not re-binned).
  * index and fixed_time|W: the representation is the raw series, so the surrogate is the frozen
    AR(1) (or shuffle) of y in observation order, assigned to the observed timestamps, analysed by
    the frozen analysis.
Surrogate k uses stream (K('surrogates'), K('<arm>|<analysis>|<test>'), k) with the patient seed.
"""
import numpy as np
from . import spec as S
from . import nulls
from .analysis import taus, _bin, _index_stats, kendall_tau_a
from .rng import surrogate_stream
from .a1_spec import FROZEN_NULLS, REP_NULLS


def representation(analysis, t, y):
    """The regular series an analyst would build; None if the analysis works on the raw series."""
    if analysis.startswith("binned|"):
        _, nb, rule = analysis.split("|")
        return _bin(np.asarray(t, float), np.asarray(y, float), int(nb), rule)
    if analysis.startswith("interp|"):
        if len(t) < 2:
            return np.empty(0)
        grid = np.linspace(0.0, S.T, int(analysis.split("|")[1]))
        return np.interp(grid, t, y)
    return None


def _rep_taus(z):
    var, ac, _ = _index_stats(z)
    x = np.arange(len(var), dtype=float)
    tv = kendall_tau_a(x, var) if len(var) else np.nan
    ta = kendall_tau_a(x, ac) if len(ac) else np.nan
    return tv, ta


def _rep_surrogate(base_null, series, rng, ar=None):
    if base_null == "shuffle":
        return rng.permutation(series)
    if len(series) < 2:
        return None
    return nulls.surrogate("ar1", None, series, rng, ar=ar if ar is not None else nulls.ar1_params(series))


def run_rep(seed, arm_name, t, y, analyses, rep_nulls, n_surr):
    t = np.asarray(t, float); y = np.asarray(y, float)
    out = []
    for a in analyses:
        tv_o, ta_o, nvv, nva, info = taus(a, t, y)          # observed statistic: frozen analysis
        z = representation(a, t, y)
        src = z if z is not None else y
        ar = nulls.ar1_params(src) if len(src) > 1 else None
        for null in rep_nulls:
            base_null = null.replace("_rep", "")
            sv = np.full(n_surr, np.nan); sa = np.full(n_surr, np.nan)
            for k in range(n_surr):
                rng = surrogate_stream(seed, arm_name, a, null, k)
                if z is not None:
                    zs = _rep_surrogate(base_null, z, rng, ar)
                    if zs is None or not np.all(np.isfinite(zs)):
                        continue
                    sv[k], sa[k] = _rep_taus(zs)
                else:
                    ys = _rep_surrogate(base_null, y, rng, ar)
                    if ys is None or not np.all(np.isfinite(ys)):
                        continue
                    tv, ta, _, _, _ = taus(a, t, ys)
                    sv[k], sa[k] = tv, ta
            pv, inv_v = nulls.mc_pvalue(tv_o, sv)
            pa, inv_a = nulls.mc_pvalue(ta_o, sa)
            row = {"arm": arm_name, "analysis": a, "null": null, "n_obs": len(y),
                   "n_valid_windows_var": nvv, "n_valid_windows_ac": nva,
                   "evaluable_var": bool(np.isfinite(tv_o)), "evaluable_ac": bool(np.isfinite(ta_o)),
                   "tau_var": tv_o, "p_var": pv, "det_var": nulls.detected(tv_o, pv), "n_invalid_surr_var": inv_v,
                   "tau_ac": ta_o, "p_ac": pa, "det_ac": nulls.detected(ta_o, pa), "n_invalid_surr_ac": inv_a,
                   "rep_length": int(len(z)) if z is not None else int(len(y))}
            row.update(info)
            out.append(row)
    return out


def run_arm(seed, arm_name, t, y, analyses, test_list, n_surr=S.N_SURR):
    frozen = [n for n in test_list if n in FROZEN_NULLS]
    rep = [n for n in test_list if n in REP_NULLS]
    unknown = set(test_list) - set(frozen) - set(rep)
    if unknown:
        raise ValueError(f"unknown tests {unknown}")
    rows = nulls.run_arm(seed, arm_name, t, y, analyses, frozen, n_surr) if frozen else []
    if rep:
        rows += run_rep(seed, arm_name, t, y, analyses, rep, n_surr)
    return rows
