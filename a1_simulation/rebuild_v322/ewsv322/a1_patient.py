"""One patient through the A1 pipeline. Identical to patient.run except that the latent model,
arms and tests come from the A1 modules (which delegate to the frozen modules for everything frozen),
so frozen arms/tests reproduce the Phase 2 results exactly."""
import numpy as np
from . import spec as S
from . import schedule
from . import a1_latent, a1_arms, a1_nulls


def run(model, seed, design, n_surr=S.N_SURR):
    common = {"model": model, "seed": int(seed), "schedule": design["schedule"],
              "m": design.get("m"), "trigger": design["trigger"], "config_hash": S.CONFIG_HASH}
    pre = [a for a in design["arms"] if a.endswith("_pre") or "_pre_" in a]
    if pre or design.get("n_variability_draws", 0):
        raise ValueError(f"A1 designs do not use pre-escalation arms or variability draws: {pre}")
    lat = a1_latent.simulate(model, seed)
    if lat["tipped"]:
        return [dict(common, tipped=True, t_tip=lat["t_tip"], arm=None)], None
    base = schedule.baseline(lat["X"], seed)
    t_ind = None
    if design["trigger"] == "indep":
        t_ind = schedule.draw_indep(seed, design["indep_outcomes"])
    inf = schedule.informative(lat["X"], seed, base, design["schedule"], design.get("m"),
                               design["trigger"], t_esc_indep=t_ind, model=model)
    A = a1_arms.build(lat["X"], seed, base, inf, design["schedule"], design.get("m"), design["trigger"],
                      set(design["arms"]), 0)
    meta = A.pop("_meta")
    common.update(tipped=False, t_tip=np.nan, **meta)
    rows = []
    for arm_name in design["arms"]:                      # fixed order = design order
        if arm_name not in A:
            if arm_name == "thinned" and design["schedule"] == "smooth":
                continue                                 # thinning is not defined for the smooth schedule
            raise ValueError(f"arm {arm_name} was requested but not built")
        ser = A[arm_name]
        for r in a1_nulls.run_arm(seed, arm_name, ser["t"], ser["y"], design["analyses"], design["nulls"], n_surr):
            r.update(common)
            r["n_baseline_in_arm"] = int(np.sum(ser["t"] < S.T_START))
            rows.append(r)
    return rows, {"base": base, "inf": inf, "arms": A, "X": lat["X"]}
