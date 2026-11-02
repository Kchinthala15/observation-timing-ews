"""Phase 2: config lineage, frozen matrix, gate, interpretation rule, variability design, summariser."""
import json
import os

import numpy as np
import pytest

from ewsv322 import spec as S
from ewsv322 import stats as ST
from ewsv322 import patient as P
from ewsv322 import run as RUN
from ewsv322 import phase2 as P2


# ------------------------------------------------------------------ config and matrix
def test_config_lineage_and_parent_unchanged():
    assert S.config_sha256(S.PARENT_CONFIG_PATH) == S.PARENT_SHA256
    assert S.CFG["config_lineage"]["parent_config_sha256"] == S.PARENT_SHA256
    with open(S.PARENT_CONFIG_PATH, encoding="utf-8") as f:
        parent = json.load(f)
    for k, v in parent.items():
        assert S.CFG[k] == v, k                                    # every parent value unchanged
    assert set(S.CFG) - set(parent) == {"phase2_matrix", "config_lineage", "status_phase2"}


def test_matrix_blocks_counts_and_roles():
    ids = [b["id"] for b in P2.MATRIX["blocks"]]
    assert ids == ["P1", "S1", "S2", "S3", "R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "R10",
                   "N1", "N2", "E1", "E2", "E3"]
    assert P2.OPTIONAL == ["E1", "E2", "E3"] and len(P2.REQUIRED) == 16
    p1 = P2.BLOCKS["P1"]
    assert (p1["model"], p1["schedule"], p1["m"], p1["trigger"], p1["n_patients"]) == ("ou_csd", "step", 2.0, "primary", 1000)
    assert p1["nulls"] == ["ou_ss", "ar1", "shuffle"]
    for b in P2.MATRIX["blocks"]:
        assert len(P2.block_ids(b)) == b["n_patients"]
        if b["id"] != "P1":
            assert b["nulls"] == ["ou_ss"]
    assert P2.PRIMARY_STREAM_COMPARISON == "regular_N-vs-informative_reduced_N"


# ------------------------------------------------------------------ interpretation rule
@pytest.mark.parametrize("L,U,cat,anyloss", [
    (-0.20, -0.06, "Material detection loss", True),
    (-0.04, -0.01, "No material difference", True),      # overlap case: not material loss
    (-0.05, 0.05, "No material difference", False),      # boundaries included in 'no material difference'
    (-0.06, -0.01, "Inconclusive", True),
    (-0.03, 0.07, "Inconclusive", False),
    (0.06, 0.20, "Material detection gain", False),
    (-0.051, 0.0, "Inconclusive", False),
])
def test_interpretation_categories_mutually_exclusive(L, U, cat, anyloss):
    r = ST.interpret(L, U)
    assert r["category"] == cat and r["upper_ci_below_zero_any_loss"] == anyloss


# ------------------------------------------------------------------ gate
def _rec(tmp, **over):
    rec = {"config_sha256": S.PARENT_SHA256, "selected_null_from_phase1a": "ou_ss", "decision": {"calibrated": True}}
    rec.update(over)
    (tmp / "phase1b_record.json").write_text(json.dumps(rec))


def test_gate(tmp_path):
    with pytest.raises(SystemExit, match="not found"):
        P2.check_gate(str(tmp_path))
    _rec(tmp_path, decision={"calibrated": False})
    with pytest.raises(SystemExit, match="did not pass"):
        P2.check_gate(str(tmp_path))
    _rec(tmp_path, selected_null_from_phase1a="ar1")
    with pytest.raises(SystemExit, match="not ou_ss"):
        P2.check_gate(str(tmp_path))
    _rec(tmp_path, config_sha256="0" * 64)
    with pytest.raises(SystemExit, match="other than the recorded parent"):
        P2.check_gate(str(tmp_path))
    _rec(tmp_path)
    assert P2.check_gate(str(tmp_path))["selected_null_from_phase1a"] == "ou_ss"


def test_completed_block_is_not_rerun(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(RUN, "OUT", str(tmp_path))
    _rec(tmp_path)
    (tmp_path / "phase2_P1_record.json").write_text("{}")
    P2.cmd_block("P1", 1)
    assert "already completed" in capsys.readouterr().out


# ------------------------------------------------------------------ variability design
def test_variability_draws_only_for_listed_ids_and_analysis():
    b = P2.BLOCKS["P1"]
    design = P2.block_design(b)
    jobs = P2._jobs(b, design, [0, 199, 200], S.PHASE_BASE["phase2"], 5)
    assert [j[3]["n_variability_draws"] for j in jobs] == [10, 10, 0]
    small = dict(jobs[0][3], arms=["regular_N", "informative_reduced_N"], analyses=["index"], nulls=["ou_ss", "shuffle"],
                 n_variability_draws=2)
    rows, _ = P.run("ou_csd", S.PHASE_BASE["phase2"] + 0, small, n_surr=5)
    draw_rows = [r for r in rows if r["arm"].startswith("informative_reduced_N|draw")]
    assert len(draw_rows) == 2 and all(r["analysis"] == "index" and r["null"] == "ou_ss" for r in draw_rows)
    assert len([r for r in rows if not r["arm"].startswith("informative_reduced_N|draw")]) == 4


# ------------------------------------------------------------------ summariser on synthetic rows
def _synthetic_p1(n=200, p_reg=0.6, p_inf=0.3, seed=1):
    g = np.random.default_rng(seed)
    rows = []
    det = {a: g.random(n) < (p_inf if a.startswith("informative") or a == "thinned" else p_reg)
           for a in RUN.ALL_ARMS}
    det2 = {a: g.random(n) < 0.5 for a in RUN.ALL_ARMS}
    for i in range(n):
        common = {"model": "ou_csd", "patient_id": i, "seed": 3_000_000 + i, "tipped": False, "N_i": 40 + i % 21,
                  "never_escalated": i % 5 == 0, "block": "P1"}
        for a in RUN.ALL_ARMS:
            for nl in ("ou_ss", "ar1", "shuffle"):
                base = dict(common, arm=a, null=nl, n_obs=40 + i % 21, tau_var=0.1, p_var=0.01, tau_ac=0.1, p_ac=0.01,
                            det_var=bool(det[a][i]), det_ac=bool(det2[a][i]), evaluable_var=True, evaluable_ac=True)
                rows.append(dict(base, analysis="index"))
                rows.append(dict(base, analysis="elapsed"))
        if i < 200:
            for k in range(1, 11):
                rows.append(dict(common, arm=f"informative_reduced_N|draw{k}", null="ou_ss", analysis="index",
                                 n_obs=1, tau_var=0.1, p_var=0.01, tau_ac=0.1, p_ac=0.01,
                                 det_var=bool(g.random() < 0.3), det_ac=False, evaluable_var=True, evaluable_ac=True))
    return rows, det, det2


def test_summarizer_primary_estimand_and_checks(tmp_path):
    rows, det, det2 = _synthetic_p1()
    RUN.write_rows(rows, str(tmp_path / "t_P1_rows.csv"))
    summ, path = P2.summarize(str(tmp_path), B=500, label="t")
    assert os.path.exists(path) and summ["blocks_completed"] == ["P1"]
    pr = summ["PRIMARY_RESULT"]
    manual = det["informative_reduced_N"].mean() - det["regular_N"].mean()
    assert np.isclose(pr["variance"]["diff"], manual)
    lo, hi = pr["variance"]["ci95_bootstrap"]
    assert lo <= manual <= hi
    assert pr["variance"]["interpretation"] == ST.interpret(lo, hi)
    assert np.isclose(pr["autocorrelation"]["diff"], det2["informative_reduced_N"].mean() - det2["regular_N"].mean())
    sec = summ["blocks"]["P1"]["secondary"]
    assert sec["elapsed_verification"]["mismatches_with_index"] == 0
    assert summ["blocks"]["P1"]["pre_escalation"]["n_never_escalated"] == 40
    assert summ["blocks"]["P1"]["pre_escalation"]["never_escalated_only"]["n"] == 40
    assert summ["bootstrap_streams_used_not_planned"] == []
    assert len(sec["variability_substudy"]["var"]["extra_draw_rates"]) == 10
    assert summ["blocks"]["P1"]["count_report"]["n_by_category"]["<40"] == 0


def test_elapsed_mismatch_is_flagged(tmp_path):
    rows, _, _ = _synthetic_p1()
    for r in rows:
        if r["analysis"] == "elapsed" and r["patient_id"] == 3 and r["arm"] == "regular_N" and r["null"] == "ou_ss":
            r["p_var"] = 0.5
            break
    RUN.write_rows(rows, str(tmp_path / "t_P1_rows.csv"))
    summ, _ = P2.summarize(str(tmp_path), B=200, label="t")
    assert summ["blocks"]["P1"]["secondary"]["elapsed_verification"]["mismatches_with_index"] == 1


def test_bootstrap_streams_same_ids_for_both_outcomes():
    a = np.array([1, 0, 1, 1, 0, 0, 1, 0] * 10, float); b = np.array([0, 0, 1, 0, 0, 1, 0, 0] * 10, float)
    from ewsv322.rng import bootstrap_stream
    r1 = ST.paired_bootstrap(a, b, bootstrap_stream(3_000_000, "ou_csd", "regular_N-vs-informative_reduced_N"), B=300)
    r2 = ST.paired_bootstrap(a, b, bootstrap_stream(3_000_000, "ou_csd", "regular_N-vs-informative_reduced_N"), B=300)
    assert r1 == r2


def test_summarizer_only_counts_recorded_blocks(tmp_path):
    rows, _, _ = _synthetic_p1(n=200)
    for r in rows:
        r["config_hash"] = S.CONFIG_HASH
    rp = tmp_path / "phase2_P1_rows.csv"
    RUN.write_rows(rows, str(rp))
    done, rej = P2.load_completed(str(tmp_path), "phase2")
    assert "P1" not in done and "record" in rej["P1"]                       # no completion record
    (tmp_path / "phase2_P1_record.json").write_text(json.dumps({"rows_sha256": "0" * 64}))
    done, rej = P2.load_completed(str(tmp_path), "phase2")
    assert "P1" not in done and "SHA-256" in rej["P1"]                      # rows changed after recording
    (tmp_path / "phase2_P1_record.json").write_text(json.dumps({"rows_sha256": RUN._sha(str(rp))}))
    done, rej = P2.load_completed(str(tmp_path), "phase2")
    assert "P1" in done and not rej
