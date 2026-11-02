"""One simulated patient through the full pipeline: latent -> baseline -> schedule -> arms -> analyses."""
import numpy as np
from . import spec as S
from . import latent, schedule, arms, nulls


def primary_outcome(model, seed, sched="step", m=S.M_PRIMARY):
    """Primary-trigger escalation time (np.inf = never) — used to build the independent-trigger
    source distribution for the same model and seed block. Tipped fold patients return NaN."""
    lat = latent.simulate(model, seed)
    if lat["tipped"]:
        return np.nan
    base = schedule.baseline(lat["X"], seed)
    inf = schedule.informative(lat["X"], seed, base, sched, m, "primary", model=model)
    return inf["t_esc"]


def run(model, seed, design, n_surr=S.N_SURR):
    """design keys: schedule, m, trigger, arms, analyses, nulls, n_variability_draws, indep_outcomes,
    and optionally variability_analyses / variability_nulls (used only for the extra subsampling-draw arms)."""
    common = {"model": model, "seed": int(seed), "schedule": design["schedule"],
              "m": design.get("m"), "trigger": design["trigger"], "config_hash": S.CONFIG_HASH}
    lat = latent.simulate(model, seed)
    if lat["tipped"]:
        # latent.fold.tipping_rule: tipped runs are counted, not analysed.
        return [dict(common, tipped=True, t_tip=lat["t_tip"], arm=None)], None
    base = schedule.baseline(lat["X"], seed)
    t_ind = None
    if design["trigger"] == "indep":
        t_ind = schedule.draw_indep(seed, design["indep_outcomes"])
    inf = schedule.informative(lat["X"], seed, base, design["schedule"], design.get("m"),
                               design["trigger"], t_esc_indep=t_ind, model=model)
    A = arms.build(lat["X"], seed, base, inf, design["schedule"], design.get("m"), design["trigger"],
                   set(design["arms"]), design.get("n_variability_draws", 0))
    meta = A.pop("_meta")
    common.update(tipped=False, t_tip=np.nan, **meta)
    rows = []
    for arm_name, ser in A.items():
        is_pre = arm_name in ("informative_pre", "regular_pre_trunc", "regular_pre_matched")
        if arm_name.startswith("informative_reduced_N|draw"):
            # phase2_matrix P1.variability_draws: extra subsampling draws use only the listed analysis and null
            an = design.get("variability_analyses", design["analyses"])
            nl = design.get("variability_nulls", design["nulls"])
        else:
            an, nl = design["analyses"], design["nulls"]
        res = nulls.run_arm(seed, arm_name, ser["t"], ser["y"], an, nl, n_surr)
        for r in res:
            r.update(common)
            r["n_baseline_in_arm"] = int(np.sum(ser["t"] < S.T_START))
            if is_pre and r["n_obs"] < S.PRE_MIN_OBS:
                # analyses.pre_min_obs: fewer than 10 observations -> not evaluable (non-detection)
                for ind in ("var", "ac"):
                    r[f"evaluable_{ind}"] = False; r[f"det_{ind}"] = False
                r["pre_min_obs_fail"] = True
            rows.append(r)
    debug = {"base": base, "inf": inf, "arms": A, "X": lat["X"]}
    return rows, debug
