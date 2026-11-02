"""Newcombe (1998) published examples, exact TOST, paired bootstrap, Phase 1a/1b decision rules and gates."""
import json
import os

import numpy as np
import pytest

from ewsv322 import stats as ST
from ewsv322.rng import bootstrap_stream

# Newcombe RG. Stat Med 1998;17:2635-2650, Table III, method 10 (cells e, f, g, h; delta = (f-g)/n)
NEWCOMBE_TABLE_III = [
    ((36, 12, 2, 0), (0.0569, 0.3404)), ((20, 12, 2, 16), (0.0562, 0.3292)), ((18, 12, 2, 18), (0.0562, 0.3290)),
    ((36, 14, 0, 0), (0.1528, 0.4167)), ((35, 14, 0, 1), (0.1461, 0.4175)), ((18, 14, 0, 18), (0.1441, 0.3963)),
    ((2, 97, 1, 0), (0.8721, 0.9854)), ((1, 97, 1, 1), (0.8736, 0.9850)), ((0, 29, 1, 0), (0.6666, 0.9882)),
    ((2, 98, 0, 0), (0.9178, 0.9945)), ((1, 98, 0, 1), (0.9171, 0.9916)), ((0, 30, 0, 0), (0.8395, 1.0)),
    ((54, 0, 0, 0), (-0.0664, 0.0664)), ((53, 0, 0, 1), (-0.0729, 0.0729)), ((30, 0, 0, 24), (-0.0358, 0.0358)),
    ((29, 0, 0, 25), (-0.0354, 0.0354)), ((28, 0, 0, 26), (-0.0352, 0.0352)), ((27, 0, 0, 27), (-0.0351, 0.0351)),
]


@pytest.mark.parametrize("cells,expected", NEWCOMBE_TABLE_III)
def test_newcombe_matched_pair_published_examples(cells, expected):
    d, L, U = ST.newcombe_paired(*cells)
    assert abs(L - expected[0]) <= 2e-4 and abs(U - expected[1]) <= 2e-4


def test_tost_equals_clopper_pearson_90_inclusion():
    n = 2000
    for x in range(0, 301):
        r = ST.tost(x, n)
        lo, hi = r["cp90"]
        assert r["pass"] == (lo > 0.025 and hi < 0.075), x
    assert ST.tost(100, 2000)["pass"] and not ST.tost(40, 2000)["pass"] and not ST.tost(160, 2000)["pass"]


def test_paired_bootstrap_reproducible_cohort_stream():
    g = np.random.default_rng(0)
    a = (g.random(500) < 0.6).astype(int); b = (g.random(500) < 0.4).astype(int)
    r1 = ST.paired_bootstrap(a, b, bootstrap_stream(3_000_000, "ou_csd", "x"), B=2000)
    r2 = ST.paired_bootstrap(a, b, bootstrap_stream(3_000_000, "ou_csd", "x"), B=2000)
    assert r1 == r2 and r1[1] < r1[0] < r1[2] and np.isclose(r1[0], a.mean() - b.mean())


def _cells(pass_map):
    out = {}
    for ctrl in ("stable", "noise_only"):
        for arm in ("regular_N", "informative_reduced_N"):
            for ind in ("var", "ac"):
                out[(ctrl, arm, ind)] = {"pass": pass_map}
    return out


def test_phase1a_selection_rules():
    allfail = {n: _cells(False) for n in ("ou_ss", "ar1", "shuffle")}
    assert ST.select_null(allfail) is None                          # no candidate -> stop
    c = {"ou_ss": _cells(False), "ar1": _cells(True), "shuffle": _cells(True)}
    assert ST.select_null(c) == "ar1"                               # first passing in prespecified order
    one_cell_fails = _cells(True); one_cell_fails[("noise_only", "regular_N", "ac")] = {"pass": False}
    assert ST.select_null({"ou_ss": one_cell_fails, "ar1": _cells(False), "shuffle": _cells(False)}) is None


def test_phase1b_decision_rules():
    assert not ST.phase1b_decision({"ar1": _cells(True)}, None)["calibrated"]
    assert ST.phase1b_decision({"ar1": _cells(True)}, "ar1")["calibrated"]
    bad = _cells(True); bad[("stable", "informative_reduced_N", "var")] = {"pass": False}
    d = ST.phase1b_decision({"ar1": bad, "shuffle": _cells(True)}, "ar1")
    assert not d["calibrated"] and "uncalibrated" in d["reason"]       # no switching to 'shuffle'


def test_manifest_records_required_fields():
    from ewsv322 import run
    planned = run.planned_streams([run.calibration_design("primary")], ["noise_only"])
    m = run.manifest("unit", 9_000_000, {}, planned, ["latent", "not_planned"], [], "t0")
    for k in ("config_sha256", "python", "numpy", "scipy", "platform", "code_sha256",
              "stream_keys_planned_and_checked_unique_before_run"):
        assert k in m
    assert m["config_sha256"] == run.S.CONFIG_HASH
    assert m["streams_used_not_in_plan"] == ["not_planned"] and "ERROR" in m


def test_gates(tmp_path, monkeypatch):
    from ewsv322 import run
    monkeypatch.setattr(run, "OUT", str(tmp_path))
    with pytest.raises(SystemExit):
        run.cmd_phase1b(1)                                          # no Phase 1a freeze record
    rec = tmp_path / "phase1a_freeze_record.json"
    rec.write_text(json.dumps({"config_sha256": run.S.CONFIG_HASH, "selected_null": None}))
    with pytest.raises(SystemExit, match="selected no null"):
        run.cmd_phase1b(1)                                          # Phase 1a failure stops the workflow
    with pytest.raises(SystemExit, match="already exists"):
        run.cmd_phase1a(1)                                          # Phase 1a cannot be re-run
    p = tmp_path / "x.json"
    run.write_immutable({"a": 1}, str(p))
    with pytest.raises(RuntimeError):
        run.write_immutable({"a": 2}, str(p))                       # freeze records are immutable
    with pytest.raises(SystemExit, match="Phase 2 gate"):
        monkeypatch.setattr("sys.argv", ["run", "phase2", "--blocks", "P1"])
        run.main()                                                  # no Phase 1b record -> Phase 2 refused
