"""Comparison arms — config keys arms.*, analyses.pre_escalation, analyses.thinning.

Every arm = the same 8 baseline events + a set of post-baseline events, sorted by time.
"""
import numpy as np
from . import spec as S
from .schedule import measure, regular_times
from .rng import patient_stream, tag as stream_tag


def _arm(base, t, y, x=None):
    tt = np.concatenate([base["t"], t])
    yy = np.concatenate([base["y"], y])
    o = np.argsort(tt, kind="stable")
    return {"t": tt[o], "y": yy[o]}


def build(X, seed, base, inf, schedule, m, trigger, wanted, n_variability_draws=0):
    """Return dict arm_name -> {'t','y'} plus bookkeeping in key '_meta'.

    Count convention: informative total = 8 + number of post-baseline events (always >= 8);
    N_i = min(60, informative total), so 8 <= N_i <= 60.
    """
    n_post = len(inf["t"])
    n_total = S.N_BASE + n_post
    N_i = min(S.N_CAP, n_total)
    k_post = N_i - S.N_BASE
    t_esc = inf["t_esc"]
    never = not np.isfinite(t_esc)
    out = {}
    meta = {"n_inf_total": n_total, "n_post": n_post, "N_i": N_i, "t_esc": t_esc,
            "escalated": not never, "never_escalated": never}

    def reg(name, n, stream, t_hi=None):
        tr = regular_times(n, t_hi)
        _, _, yr = measure(X, tr, patient_stream(seed, stream))
        return _arm(base, tr, yr)

    if any(w in wanted for w in ("regular_N", "informative_pre", "regular_pre_trunc", "regular_pre_matched")):
        out["regular_N"] = reg("regular_N", k_post, "err_reg_N")
    if "informative_all" in wanted or "informative_pre" in wanted or "thinned" in wanted:
        out["informative_all"] = _arm(base, inf["t"], inf["y"])
    if "informative_reduced_N" in wanted:
        idx = np.sort(patient_stream(seed, "subsample").choice(n_post, size=k_post, replace=False))
        out["informative_reduced_N"] = _arm(base, inf["t"][idx], inf["y"][idx])
    for d in range(1, n_variability_draws + 1):
        idx = np.sort(patient_stream(seed, f"subsample|draw{d}").choice(n_post, size=k_post, replace=False))
        out[f"informative_reduced_N|draw{d}"] = _arm(base, inf["t"][idx], inf["y"][idx])
    if "regular_matched" in wanted:
        out["regular_matched"] = reg("regular_matched", n_total - S.N_BASE, "err_reg_matched")
    if "thinned" in wanted and schedule != "smooth":
        # analyses.thinning: post-escalation events kept with probability 1/m, stream thin|<tag>.
        # IMPL-NOTE (see IMPLEMENTATION_NOTES.md): 'post-escalation' = events strictly after t_esc (the triggering event itself was
        # sampled at the base rate and is kept). One uniform per post-escalation event, in time order.
        post = inf["t"] > t_esc
        keep = np.ones(n_post, bool)
        u = patient_stream(seed, f"thin|{stream_tag(schedule, m, trigger)}").random(int(post.sum()))
        keep[np.where(post)[0]] = u < 1.0 / float(m)
        out["thinned"] = _arm(base, inf["t"][keep], inf["y"][keep])
    if any(w in wanted for w in ("informative_pre", "regular_pre_trunc", "regular_pre_matched")) and schedule != "smooth":
        if never:
            # analyses.pre_escalation: never-escalated patients use the full series, flagged.
            out["informative_pre"] = out.get("informative_all") or _arm(base, inf["t"], inf["y"])
            out["regular_pre_trunc"] = out.get("regular_N") or reg("regular_N", k_post, "err_reg_N")
            out["regular_pre_matched"] = reg("regular_matched", n_total - S.N_BASE, "err_reg_matched")
        else:
            pre = inf["t"] < t_esc
            out["informative_pre"] = _arm(base, inf["t"][pre], inf["y"][pre])
            rN = out["regular_N"]
            out["regular_pre_trunc"] = {"t": rN["t"][rN["t"] < t_esc], "y": rN["y"][rN["t"] < t_esc]}
            n_pre_post = int(pre.sum())
            # IMPL-NOTE (see IMPLEMENTATION_NOTES.md): matched-count pre-escalation regular arm = 8 baseline + n_pre_post evenly spaced
            # over [15, t_esc), with its own error stream 'err_reg_pre_matched' (not named in the JSON).
            out["regular_pre_matched"] = reg("regular_pre_matched", n_pre_post, "err_reg_pre_matched", t_hi=t_esc)
    out = {k: v for k, v in out.items() if k in wanted or k.startswith("informative_reduced_N|draw")}
    out["_meta"] = meta
    return out
