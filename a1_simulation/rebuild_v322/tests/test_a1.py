"""Protocol Addendum A1 v1.2: hash lock, new models, new arm, new tests, frozen identity, stream planning."""
import json

import numpy as np
import pytest

from ewsv322 import spec as S
from ewsv322 import latent, patient, schedule, analysis, nulls
from ewsv322 import phase2 as P2
from ewsv322 import a1_spec as A
from ewsv322 import a1_latent, a1_arms, a1_nulls, a1_patient, a1_run


# ------------------------------------------------------------------ frozen inputs
def test_a1_hash_lock_and_lineage():
    assert A._sha(A.A1_PATH) == A.A1_SHA256
    assert A.A1["config_lineage"]["parent_config_sha256"] == S.CONFIG_HASH
    assert A.A1["config_lineage"]["grandparent_config_sha256"] == S.PARENT_SHA256
    assert A.A1["version"] == "1.2-FROZEN"


def test_block_list_complete():
    ids = a1_run.block_list()
    for b in ["G1a", "G1b", "G1c", "G1d", "F1", "XS1", "XS2", "XN1", "XN2", "P5", "P5FA", "E2", "E2FA", "FOLD2"] + \
             [f"XR{i}" for i in range(1, 11)]:
        assert b in ids
    assert sum(e.startswith("FOLDP|") for e in ids) == 6


def test_x_blocks_inherit_frozen_designs():
    for i in range(1, 11):
        r = a1_run.resolve(f"XR{i}")
        fb = P2.BLOCKS[f"R{i}"]
        assert (r["model"], r["design"]["schedule"], r["design"]["m"], r["design"]["trigger"]) == \
               (fb["model"], fb["schedule"], fb["m"], fb["trigger"])
        assert r["base"] == P2.BASE and r["ids"][:500] == list(range(500)) and len(r["ids"]) == 1000
    assert "thinned" not in a1_run.resolve("XR10")["design"]["arms"]


# ------------------------------------------------------------------ new models
@pytest.mark.parametrize("model,theta", [("stable_slow_0.2", 0.2), ("stable_slow_0.05", 0.05)])
def test_stable_slow_is_stationary(model, theta):
    v = S.SIGMA ** 2 / (2 * theta)
    x0 = np.array([a1_latent.simulate(model, 1_234_000 + s)["X"][0] for s in range(400)])
    xT = np.array([a1_latent.simulate(model, 1_234_000 + s)["X"][-1] for s in range(400)])
    assert abs(x0.var() / v - 1) < 0.2 and abs(xT.var() / v - 1) < 0.25


def test_stable_slow_draw_order_matches_frozen_ou():
    """Same 'latent' stream and draw order as the frozen OU: X0 is the same standard normal scaled by the
    stationary SD, and the recovered innovations Z are identical."""
    seed = 777
    a = a1_latent.simulate("stable_slow_0.2", seed)["X"]
    b = latent.simulate("stable", seed)["X"]
    sd_a, sd_b = np.sqrt(S.SIGMA ** 2 / (2 * 0.2)), np.sqrt(S.SIGMA ** 2 / (2 * S.THETA0))
    assert np.isclose(a[0] / sd_a, b[0] / sd_b, rtol=1e-12)
    za = (a[1:] - a[:-1] - 0.2 * (0 - a[:-1]) * S.DT) / (S.SIGMA * np.sqrt(S.DT))
    zb = (b[1:] - b[:-1] - S.THETA0 * (0 - b[:-1]) * S.DT) / (S.SIGMA * np.sqrt(S.DT))
    np.testing.assert_allclose(za, zb, rtol=0, atol=1e-9)


def test_fold_v2_reduces_to_frozen_fold_at_frozen_settings():
    for seed in (11, 12, 13):
        f = latent.simulate("fold", seed)
        g = a1_latent.simulate(A.fold_model_name(0.10, 0.05), seed)
        assert f["tipped"] == g["tipped"]
        np.testing.assert_array_equal(np.nan_to_num(f["X"], nan=-99), np.nan_to_num(g["X"], nan=-99))


def test_frozen_models_delegate_unchanged():
    for m in ("ou_csd", "stable", "level_drop_only", "ou_combined", "noise_only"):
        np.testing.assert_array_equal(a1_latent.simulate(m, 5)["X"], latent.simulate(m, 5)["X"])


# ------------------------------------------------------------------ new arm
def _setup(model="ou_csd", seed=3_000_007):
    lat = a1_latent.simulate(model, seed)
    base = schedule.baseline(lat["X"], seed)
    inf = schedule.informative(lat["X"], seed, base, "step", 2.0, "primary", model=model)
    return lat, base, inf


def test_poisson_matched_count_times_and_baseline():
    seed = 3_000_007
    lat, base, inf = _setup(seed=seed)
    A_ = a1_arms.build(lat["X"], seed, base, inf, "step", 2.0, "primary",
                       {"regular_N", "informative_reduced_N", "poisson_matched"})
    pm, rg, ir = A_["poisson_matched"], A_["regular_N"], A_["informative_reduced_N"]
    assert len(pm["t"]) == len(rg["t"]) == len(ir["t"]) == A_["_meta"]["N_i"]
    post = pm["t"][pm["t"] >= S.T_START]
    assert np.all(np.diff(pm["t"]) > 0) and np.all((post >= S.T_START) & (post < S.T))
    np.testing.assert_array_equal(pm["t"][:S.N_BASE], base["t"])


def test_poisson_matched_times_do_not_depend_on_state():
    # same seed, different latent model -> identical observation times (state-independent), N_i permitting
    seed = 3_000_011
    t = {}
    for model in ("ou_csd", "stable"):
        lat, base, inf = _setup(model, seed)
        k = 20 + S.N_BASE
        t[model] = a1_arms.poisson_matched(lat["X"], seed, base, k)["t"]
    np.testing.assert_array_equal(t["ou_csd"], t["stable"])


def test_poisson_matched_times_uniform():
    from scipy.stats import kstest
    lat, base, _ = _setup(seed=3_000_013)
    pooled = np.concatenate([a1_arms.poisson_matched(lat["X"], 4_000_000 + s, base, 60)["t"][S.N_BASE:]
                             for s in range(200)])
    assert len(pooled) == 200 * 52
    assert kstest(pooled, "uniform", args=(S.T_START, S.T - S.T_START)).pvalue > 0.001


def test_frozen_arms_unchanged_by_adding_poisson_matched():
    seed = 3_000_021
    lat, base, inf = _setup(seed=seed)
    from ewsv322 import arms
    w = {"regular_N", "informative_reduced_N", "thinned"}
    f = arms.build(lat["X"], seed, base, inf, "step", 2.0, "primary", w)
    g = a1_arms.build(lat["X"], seed, base, inf, "step", 2.0, "primary", w | {"poisson_matched"})
    for a in w:
        np.testing.assert_array_equal(f[a]["t"], g[a]["t"]); np.testing.assert_array_equal(f[a]["y"], g[a]["y"])


# ------------------------------------------------------------------ new tests (ar1_rep, shuffle_rep)
def _series(seed=3_000_031):
    lat, base, inf = _setup(seed=seed)
    A_ = a1_arms.build(lat["X"], seed, base, inf, "step", 2.0, "primary", {"informative_reduced_N"})
    return A_["informative_reduced_N"]["t"], A_["informative_reduced_N"]["y"]


def test_representation_matches_frozen_analysis():
    t, y = _series()
    for a in ("binned|60|linear_interp", "binned|30|drop", "binned|120|locf", "interp|60"):
        z = a1_nulls.representation(a, t, y)
        tv, ta = a1_nulls._rep_taus(z)
        tv0, ta0, _, _, _ = analysis.taus(a, t, y)
        assert (np.isnan(tv) and np.isnan(tv0)) or tv == tv0
        assert (np.isnan(ta) and np.isnan(ta0)) or ta == ta0
    assert a1_nulls.representation("fixed_time|W30", t, y) is None


def test_shuffle_rep_is_permutation_of_representation():
    t, y = _series()
    z = a1_nulls.representation("binned|60|linear_interp", t, y)
    zs = a1_nulls._rep_surrogate("shuffle", z, np.random.default_rng(1))
    np.testing.assert_array_equal(np.sort(zs), np.sort(z)); assert len(zs) == 60


def test_ar1_rep_surrogate_length_and_determinism():
    t, y = _series()
    from ewsv322.rng import surrogate_stream
    z = a1_nulls.representation("binned|30|drop", t, y)
    a = a1_nulls._rep_surrogate("ar1", z, surrogate_stream(5, "x", "binned|30|drop", "ar1_rep", 0))
    b = a1_nulls._rep_surrogate("ar1", z, surrogate_stream(5, "x", "binned|30|drop", "ar1_rep", 0))
    assert len(a) == len(z); np.testing.assert_array_equal(a, b)


def test_rep_tests_on_fixed_time_equal_frozen_ar1_and_shuffle_paths():
    from ewsv322.rng import surrogate_stream
    t, y = _series()
    a, seed, arm, n = "fixed_time|W30", 21, "informative_reduced_N", 6
    rows = {r["null"]: r for r in a1_nulls.run_arm(seed, arm, t, y, [a], ["ar1_rep", "shuffle_rep"], n_surr=n)}
    tv0, ta0, _, _, _ = analysis.taus(a, t, y)
    for null, frozen in (("ar1_rep", "ar1"), ("shuffle_rep", "shuffle")):
        sv = np.full(n, np.nan); sa = np.full(n, np.nan)
        for k in range(n):
            rng = surrogate_stream(seed, arm, a, null, k)
            ys = nulls.surrogate(frozen, t, y, rng, ar=nulls.ar1_params(y))
            sv[k], sa[k], _, _, _ = analysis.taus(a, t, ys)
        assert rows[null]["p_var"] == nulls.mc_pvalue(tv0, sv)[0]
        assert rows[null]["p_ac"] == nulls.mc_pvalue(ta0, sa)[0]


def test_rep_rows_complete():
    t, y = _series()
    rows = a1_nulls.run_arm(9, "informative_reduced_N", t, y, ["binned|60|linear_interp", "fixed_time|W30"],
                            ["ar1_rep", "shuffle_rep"], n_surr=5)
    assert len(rows) == 4 and {r["null"] for r in rows} == {"ar1_rep", "shuffle_rep"}
    for r in rows:
        for ind in ("var", "ac"):
            assert (not r[f"evaluable_{ind}"]) or 0 < r[f"p_{ind}"] <= 1


def test_unknown_test_rejected():
    t, y = _series()
    with pytest.raises(ValueError):
        a1_nulls.run_arm(9, "x", t, y, ["index"], ["bogus"], n_surr=2)


# ------------------------------------------------------------------ identity with the frozen pipeline
@pytest.mark.parametrize("model", ["ou_csd", "stable", "level_drop_only"])
def test_a1_patient_reproduces_frozen_patient_exactly(model):
    d = {"schedule": "step", "m": 2.0, "trigger": "primary", "arms": ["regular_N", "informative_reduced_N", "thinned"],
         "analyses": ["index"], "nulls": ["ou_ss", "ar1", "shuffle"], "n_variability_draws": 0}
    seed = 3_000_003
    f, _ = patient.run(model, seed, d, n_surr=5)
    g, _ = a1_patient.run(model, seed, d, n_surr=5)
    key = lambda r: (r["arm"], r["analysis"], r["null"])
    fd, gd = {key(r): r for r in f}, {key(r): r for r in g}
    assert fd.keys() == gd.keys()
    for k in fd:
        for fld in ("tau_var", "p_var", "det_var", "tau_ac", "p_ac", "det_ac", "n_obs"):
            a, b = fd[k][fld], gd[k][fld]
            assert (isinstance(a, float) and np.isnan(a) and np.isnan(b)) or a == b, (k, fld)


@pytest.mark.parametrize("schedule_,m,trigger", [("ramp", 2.0, "primary"), ("smooth", None, "none"),
                                                ("step", 2.0, "indep"), ("step", 6.0, "single")])
def test_a1_patient_identity_other_designs_with_extra_arms_and_tests(schedule_, m, trigger):
    arms_f = ["regular_N", "informative_reduced_N"] + (["thinned"] if schedule_ != "smooth" else [])
    d_f = {"schedule": schedule_, "m": m, "trigger": trigger, "arms": arms_f, "analyses": ["index"],
           "nulls": ["ou_ss"], "n_variability_draws": 0}
    if trigger == "indep":
        d_f["indep_outcomes"] = np.array([30.0, 45.0, 60.0, np.inf])
    d_a = dict(d_f, arms=arms_f + ["poisson_matched"], nulls=["ou_ss", "ar1", "shuffle"])
    seed = 3_000_005
    f, _ = patient.run("ou_csd", seed, d_f, n_surr=4)
    g, _ = a1_patient.run("ou_csd", seed, d_a, n_surr=4)
    gd = {(r["arm"], r["analysis"], r["null"]): r for r in g}
    for r in f:
        x = gd[(r["arm"], r["analysis"], r["null"])]
        for fld in ("tau_var", "p_var", "tau_ac", "p_ac", "n_obs"):
            assert (np.isnan(r[fld]) and np.isnan(x[fld])) or r[fld] == x[fld]


def test_a1_patient_rejects_unknown_arm():
    d = {"schedule": "step", "m": 2.0, "trigger": "primary", "arms": ["regular_N", "bogus"],
         "analyses": ["index"], "nulls": ["ou_ss"], "n_variability_draws": 0}
    with pytest.raises(ValueError):
        a1_patient.run("ou_csd", 3_000_006, d, n_surr=2)


def test_a1_patient_rejects_pre_arms():
    d = {"schedule": "step", "m": 2.0, "trigger": "primary", "arms": ["regular_N", "informative_pre"],
         "analyses": ["index"], "nulls": ["ou_ss"], "n_variability_draws": 0}
    with pytest.raises(ValueError):
        a1_patient.run("ou_csd", 3_000_006, d, n_surr=2)


def test_smooth_schedule_skips_thinned():
    d = {"schedule": "smooth", "m": None, "trigger": "none", "arms": ["regular_N", "informative_reduced_N", "thinned", "poisson_matched"],
         "analyses": ["index"], "nulls": ["ou_ss"], "n_variability_draws": 0}
    rows, _ = a1_patient.run("ou_csd", 3_000_004, d, n_surr=3)
    assert {r["arm"] for r in rows} == {"regular_N", "informative_reduced_N", "poisson_matched"}


# ------------------------------------------------------------------ stream planning
def test_planned_streams_cover_new_names():
    r = a1_run.resolve("F1")
    p = a1_run.planned(r["model"], r["design"])
    for n in ("pm_times", "err_pm", "poisson_matched|index|ar1", "thinned|index|shuffle", "latent"):
        assert n in p
    r5 = a1_run.resolve("P5")
    p5 = a1_run.planned(r5["model"], r5["design"])
    assert "informative_reduced_N|binned|60|linear_interp|ar1_rep" in p5


def test_one_patient_uses_only_planned_streams():
    from ewsv322 import rng as R
    for entry in ("F1", "P5", "G1a", "XR10", "XR9", "XS1", "E2FA", "FOLDP|" + A.fold_model_name(0.07, 0.10)):
        r = a1_run.resolve(entry)
        d = dict(r["design"])
        if d["trigger"] == "indep":
            d["indep_outcomes"] = np.array([30.0, 50.0, np.inf])
        R.registry_clear()
        a1_patient.run(r["model"], r["base"] + 1, d, n_surr=1)
        used = set(R.registry_snapshot())
        assert used <= set(a1_run.planned(r["model"], r["design"])), (entry, used - set(a1_run.planned(r["model"], r["design"])))


def test_fold_selection_rule_exact_ties():
    def res(model, sf, re, x, n=300, tipped=0):
        return {"model": model, "sigma_f": sf, "r_end": re, "n_enrolled": n + tipped, "tipped": tipped,
                "n_nontipped": n, "x": x}
    # 112/300 and 188/300 are equally far from 50% (float arithmetic would not tie them)
    r = [res("a", 0.07, 0.05, 188), res("b", 0.05, 0.05, 112), res("c", 0.05, 0.10, 112)]
    assert a1_run.select_fold(r) == "c"          # tie -> smaller sigma_f, then larger r_end
    assert a1_run.select_fold([res("d", 0.05, 0.05, 150, n=280, tipped=20)]) is None     # tipping 6.7% > 5%
    assert a1_run.select_fold([res("e", 0.05, 0.05, 100)]) is None                       # 33% < 35%
    assert a1_run.select_fold([res("f", 0.05, 0.05, 105)]) == "f"                        # exactly 35% included


def test_compare_to_frozen_flags_missing_rows():
    ref = [{"patient_id": 0.0, "arm": "regular_N", "analysis": "index", "null": "ou_ss", **{f: 0.1 for f in ("tau_var", "p_var", "tau_ac", "p_ac", "n_obs")}, "det_var": False, "det_ac": False},
           {"patient_id": 1.0, "arm": "regular_N", "analysis": "index", "null": "ou_ss", **{f: 0.1 for f in ("tau_var", "p_var", "tau_ac", "p_ac", "n_obs")}, "det_var": False, "det_ac": False}]
    rows = [dict(ref[0], patient_id=0)]
    n, bad = a1_run.compare_to_frozen(rows, ref, a1_run.COMPARE, [0, 1], ["regular_N"], ["ou_ss"])
    assert n == 1 and len(bad) == 1 and "missing" in bad[0]
    rows2 = [dict(ref[0], patient_id=0, tau_var=0.2), dict(ref[1], patient_id=1)]
    n, bad = a1_run.compare_to_frozen(rows2, ref, a1_run.COMPARE, [0, 1], ["regular_N"], ["ou_ss"])
    assert n == 2 and len(bad) == 1 and "tau_var" in bad[0]
