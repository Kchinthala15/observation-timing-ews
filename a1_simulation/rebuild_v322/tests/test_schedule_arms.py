"""Baseline sharing, event-level errors, trigger coupling, matched counts, independent trigger,
smooth schedule, never-escalated handling, thinning and subsampling streams."""
import numpy as np

from ewsv322 import spec as S
from ewsv322 import latent as L
from ewsv322 import schedule as SC
from ewsv322 import arms as AR
from ewsv322 import patient as P
from ewsv322.rng import gen

SEED = 3_000_017


def _setup(model="ou_csd", seed=SEED, schedule="step", m=2.0, trigger="primary", t_ind=None):
    X = L.simulate(model, seed)["X"]
    base = SC.baseline(X, seed)
    inf = SC.informative(X, seed, base, schedule, m, trigger, t_esc_indep=t_ind, model=model)
    return X, base, inf


def test_baseline_times_and_errors():
    X, base, _ = _setup()
    assert np.allclose(base["t"], [15 * i / 9 for i in range(1, 9)])
    e = gen(SEED, "err_baseline").normal(0.0, 0.1, size=8)
    assert np.array_equal(base["e"], e)
    assert np.allclose(base["y"], L.value_at(X, base["t"]) + e)
    assert base["t"].max() < 15


def test_baseline_identical_across_arms_and_matched_counts():
    for seed in range(3_000_100, 3_000_130):
        X, base, inf = _setup(seed=seed)
        A = AR.build(X, seed, base, inf, "step", 2.0, "primary",
                     {"regular_N", "informative_reduced_N", "informative_all", "regular_matched", "thinned",
                      "informative_pre", "regular_pre_trunc", "regular_pre_matched"}, 2)
        meta = A.pop("_meta")
        N = meta["N_i"]
        assert 8 <= N <= 60 and N == min(60, 8 + len(inf["t"]))
        assert len(A["regular_N"]["t"]) == N and len(A["informative_reduced_N"]["t"]) == N
        for name, ser in A.items():
            pre = ser["t"] < 15
            assert pre.sum() == 8, name
            assert np.array_equal(ser["t"][pre], base["t"]) and np.array_equal(ser["y"][pre], base["y"]), name
        # reduced arm keeps the original informative events and their errors
        all_pairs = set(zip(inf["t"].tolist(), inf["y"].tolist()))
        red = A["informative_reduced_N"]
        assert set(zip(red["t"][8:].tolist(), red["y"][8:].tolist())) <= all_pairs
        assert np.allclose(A["regular_N"]["t"][8:], SC.regular_times(N - 8))


def test_two_events_in_one_grid_interval_get_independent_errors():
    X = np.zeros(S.NGRID)
    times = [40.001, 40.004, 40.0199]                      # all inside grid interval [40.00, 40.02)
    _, e, _ = SC.measure(X, times, gen(SEED, "err_reg_N"))
    assert len(set(e.tolist())) == 3
    assert np.array_equal(e, gen(SEED, "err_reg_N").normal(0.0, 0.1, size=3))


def test_informative_errors_are_per_event_in_order():
    X, base, inf = _setup()
    g = gen(SEED, "err_inf|step|2.00|primary")
    expected = np.array([g.normal(0.0, 0.1) for _ in range(len(inf["t"]))])
    assert np.array_equal(inf["e"], expected)
    assert np.allclose(inf["y"], inf["x"] + inf["e"])
    assert np.allclose(inf["x"], L.value_at(X, inf["t"]))


def test_trigger_sees_the_measured_value_that_is_analysed():
    n_esc = 0
    for seed in range(3_000_200, 3_000_260):
        X, base, inf = _setup(seed=seed)
        k = len(inf["trig_inputs"])
        assert np.array_equal(inf["trig_inputs"], inf["y"][:k])
        thr = base["mean"] - 1.5 * base["sd"]
        if np.isfinite(inf["t_esc"]):
            n_esc += 1
            assert inf["t_esc"] == inf["t"][k - 1]
            assert inf["trig_inputs"][-1] < thr and inf["trig_inputs"][-2] < thr
            assert not np.any((inf["trig_inputs"][:-2] < thr) & (inf["trig_inputs"][1:-1] < thr))
        A = AR.build(X, seed, base, inf, "step", 2.0, "primary", {"informative_all"})
        assert np.isin(inf["y"], A["informative_all"]["y"]).all()
    assert n_esc > 0


def test_independent_trigger_ignores_patient_data():
    outcomes = np.array([30.0, 45.5, np.inf, 60.25, np.inf])
    d1 = SC.draw_indep(SEED, outcomes)
    assert d1 == outcomes[int(gen(SEED, "indep_trigger").integers(len(outcomes)))]
    # same seed -> same draw whatever the patient's latent model/trajectory (function never sees it)
    _, _, inf_a = _setup("stable", SEED, trigger="indep", t_ind=d1)
    _, _, inf_b = _setup("ou_csd", SEED, trigger="indep", t_ind=d1)
    assert inf_a["t_esc"] == inf_b["t_esc"] == d1
    assert len(inf_a["trig_inputs"]) == 0
    # rate switch happens exactly at t_esc with no event placed there
    _, _, inf = _setup("stable", SEED, trigger="indep", t_ind=50.0)
    assert inf["t_esc"] == 50.0 and not np.any(inf["t"] == 50.0)
    _, _, inf = _setup("stable", SEED, trigger="indep", t_ind=np.inf)
    assert not np.isfinite(inf["t_esc"])


def test_independent_trigger_rates_switch():
    pre, post = [], []
    for seed in range(3_000_300, 3_000_500):
        _, _, inf = _setup("stable", seed, trigger="indep", t_ind=50.0)
        pre.append(np.sum(inf["t"] < 50)); post.append(np.sum(inf["t"] >= 50))
    assert abs(np.mean(pre) - 0.6 * 35) < 1.5        # lambda0 over [15, 50)
    assert abs(np.mean(post) - 1.2 * 50) < 2.0       # m*lambda0 over [50, 100)


def test_smooth_schedule_reference_levels_and_expected_count():
    t = np.array([0.0, 45.0, 80.0])
    assert np.allclose(L.reference_level("ou_combined", t), -2 * L.q(t))
    assert np.allclose(L.reference_level("level_drop_only", t), -2 * L.q(t))
    assert np.allclose(L.reference_level("ou_csd", t), 0) and np.allclose(L.reference_level("stable", t), 0)
    assert np.allclose(L.reference_level("fold", t), np.sqrt(L.r_of(t)))
    counts = []
    for seed in range(3_000_600, 3_000_800):
        _, _, inf = _setup("ou_csd", seed, schedule="smooth", m=None, trigger="none")
        counts.append(len(inf["t"]))
        assert np.all((inf["t"] >= 15) & (inf["t"] < 100)) and np.all(np.diff(inf["t"]) > 0)
    assert abs(np.mean(counts) - 52) < 2.5


def test_never_escalated_pre_analysis_uses_full_series():
    X, base, inf = _setup("stable", SEED, trigger="indep", t_ind=np.inf)
    A = AR.build(X, SEED, base, inf, "step", 2.0, "indep",
                 {"informative_all", "regular_N", "regular_matched", "informative_pre", "regular_pre_trunc",
                  "regular_pre_matched", "thinned"})
    meta = A.pop("_meta")
    assert meta["never_escalated"]
    for a, b in (("informative_pre", "informative_all"), ("regular_pre_trunc", "regular_N"),
                 ("regular_pre_matched", "regular_matched"), ("thinned", "informative_all")):
        assert np.array_equal(A[a]["t"], A[b]["t"]) and np.array_equal(A[a]["y"], A[b]["y"])


def test_pre_escalation_arms_when_escalated():
    for seed in range(3_000_900, 3_000_960):
        X, base, inf = _setup(seed=seed)
        if not np.isfinite(inf["t_esc"]):
            continue
        A = AR.build(X, seed, base, inf, "step", 2.0, "primary",
                     {"informative_pre", "regular_pre_trunc", "regular_pre_matched"})
        A.pop("_meta")
        te = inf["t_esc"]
        assert np.all(A["informative_pre"]["t"] < te) and np.all(A["regular_pre_trunc"]["t"] < te)
        assert len(A["regular_pre_matched"]["t"]) == len(A["informative_pre"]["t"])
        return
    raise AssertionError("no escalated patient found")


def test_thinning_and_subsampling_streams():
    for seed in range(3_001_000, 3_001_040):
        X, base, inf = _setup(seed=seed)
        if np.isfinite(inf["t_esc"]) and len(inf["t"]) > 60:
            A = AR.build(X, seed, base, inf, "step", 2.0, "primary", {"thinned", "informative_reduced_N"}, 3)
            A.pop("_meta")
            post = inf["t"] > inf["t_esc"]
            u = gen(seed, "thin|step|2.00|primary").random(int(post.sum()))
            keep = np.ones(len(inf["t"]), bool); keep[np.where(post)[0]] = u < 0.5
            assert np.array_equal(A["thinned"]["t"][8:], inf["t"][keep])
            assert not np.array_equal(A["informative_reduced_N"]["t"], A["informative_reduced_N|draw1"]["t"])
            assert not np.array_equal(A["informative_reduced_N|draw1"]["t"], A["informative_reduced_N|draw2"]["t"])
            return
    raise AssertionError("no suitable patient found")


def test_patient_pipeline_runs_and_is_deterministic():
    design = {"schedule": "step", "m": 2.0, "trigger": "primary", "arms": ["regular_N", "informative_reduced_N"],
              "analyses": ["index"], "nulls": ["shuffle"], "n_variability_draws": 0}
    r1, _ = P.run("ou_csd", SEED, design, n_surr=10)
    r2, _ = P.run("ou_csd", SEED, design, n_surr=10)
    assert len(r1) == 2
    for a, b in zip(r1, r2):
        for k in ("tau_var", "p_var", "tau_ac", "p_ac"):
            assert (a[k] == b[k]) or (np.isnan(a[k]) and np.isnan(b[k]))
    assert r1[0]["n_obs"] == r1[1]["n_obs"] == r1[0]["N_i"]


def test_tipped_fold_patient_is_counted_not_analysed():
    design = {"schedule": "step", "m": 2.0, "trigger": "primary", "arms": ["regular_N"],
              "analyses": ["index"], "nulls": ["shuffle"]}
    real = L.simulate
    def forced(model, seed):                          # force a tip via an injected kick
        Z = np.zeros(S.NGRID - 1); Z[2000] = -200.0  # t=40: X~0.78, step -2.83 -> below -sqrt(r)~-0.78
        X, tipped, t_tip = L.fold_path(Z)
        return {"X": X, "tipped": tipped, "t_tip": t_tip}
    L.simulate = forced
    try:
        rows, _ = P.run("fold", SEED, design, n_surr=5)
    finally:
        L.simulate = real
    assert len(rows) == 1 and rows[0]["tipped"] and rows[0]["arm"] is None and rows[0]["t_tip"] > 0
