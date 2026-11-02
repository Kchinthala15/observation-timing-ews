"""Protocol Addendum A1 v1.2 (frozen) - runner.

Stage 0 (no production results; plumbing and timing use the plumbing seed block 9,000,000):
  python -m pytest tests -v 2>&1 | Tee-Object -FilePath outputs_addendum_A1\\stage0\\unit_tests.txt
  python -m ewsv322.a1_run plumbing          every block, 2 patients, 20 surrogates, outputs discarded
  python -m ewsv322.a1_run identity          original-seed blocks reproduce completed Phase 2 rows exactly;
                                             P5/P5FA observed statistics reproduce E1/E3 (needs both full E runs)
  python -m ewsv322.a1_run timing            every block, 5 patients, 200 surrogates -> projected runtime
  python -m ewsv322.a1_run stage0-report     one file summarising tests and checks
After the Stage 0 report is approved:
  python -m ewsv322.a1_run approve-stage0    binds the approval to the current code hashes
  python -m ewsv322.a1_run run --block G1a   blocks in the frozen run_order (enforced)
  python -m ewsv322.a1_run fold-select       Stage 5: FOLDP pilot + frozen selection rule (then: run --block FOLD2)
  python -m ewsv322.a1_run verify --block XR5  full-cohort identity check after each original-seed block

All output goes to outputs_addendum_A1/. outputs/ (Phase 2) and outputs_explore/ are read only.
"""
import argparse
import datetime as dt
import json
import os
import time
from fractions import Fraction
from multiprocessing import Pool

import numpy as np

from . import spec as S
from . import rng as R
from . import run as RUN
from . import phase2 as P2
from . import patient
from . import a1_spec as A
from . import a1_patient

OUT_A1 = os.path.normpath(os.path.join(RUN.PKG, "..", "outputs_addendum_A1"))
OUT_EXPLORE = os.path.normpath(os.path.join(RUN.PKG, "..", "outputs_explore"))
STAGE0 = os.path.join(OUT_A1, "stage0")
FOLD_SELECTION = os.path.join(OUT_A1, "a1_FOLDP_selection_record.json")
APPROVAL = os.path.join(STAGE0, "stage0_APPROVED.json")
DEVIATIONS = os.path.join(OUT_A1, "deviations.json")
PLUMBING_BASE = int(A.A1["seed_blocks"]["plumbing"])
COMPARE = ("tau_var", "p_var", "det_var", "tau_ac", "p_ac", "det_ac", "n_obs")

# frozen run_order (A1 JSON): a block may run only when every block of every earlier stage is complete
STAGES = [["G1a"], ["F1"],
          ["G1b", "G1c", "G1d", "XN1", "XN2", "XS1"] + [f"XR{i}" for i in range(1, 11)] + ["XS2"],
          ["P5", "P5FA"], ["FOLDP", "FOLD2"], ["E2", "E2FA"]]


def code_hashes():
    return {f: RUN._sha(os.path.join(RUN.PKG, f)) for f in sorted(os.listdir(RUN.PKG)) if f.endswith(".py")}


class Log:
    """Prints and appends to a log file (execution log for every Stage 0 check and production block)."""
    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def __call__(self, *msg):
        line = " ".join(str(m) for m in msg)
        print(line, flush=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(f"{dt.datetime.now().isoformat(timespec='seconds')}  {line}\n")


# ------------------------------------------------------------------ block resolution
def block_list():
    out = []
    for b in A.A1["blocks"]:
        if b["id"] == "FOLDP":
            out += [f"FOLDP|{A.fold_model_name(sf, re)}" for sf, re in A.FOLD_GRID]
        else:
            out.append(b["id"])
    return out


def resolve(entry, fold_model=None):
    """-> dict(model, design, ids, base, role, base_block)."""
    if entry.startswith("FOLDP|"):
        b = A.BLOCKS["FOLDP"]; model = entry.split("|", 1)[1]
    else:
        b = A.BLOCKS[entry]; model = b.get("model")
    if "frozen_block" in b:                                   # E2 exactly as frozen
        fb = P2.BLOCKS[b["frozen_block"]]
        lo, hi = fb["patient_ids"]
        return {"model": fb["model"], "design": P2.block_design(fb), "ids": list(range(lo, hi + 1)),
                "base": P2.BASE, "role": b["role"], "base_block": b["frozen_block"]}
    if "base_block" in b:
        fb = P2.BLOCKS[b["base_block"]]
        model, schedule, m, trigger = fb["model"], fb["schedule"], fb["m"], fb["trigger"]
    else:
        schedule, m, trigger = b["schedule"], b["m"], b["trigger"]
    if entry == "FOLD2":
        if fold_model is None:
            sel = read_fold_selection()
            if not sel.get("selected_model"):
                raise SystemExit("FOLDP selected no eligible setting: FOLD2 is not run (frozen rule).")
            fold_model = sel["selected_model"]
        model = fold_model
    design = {"schedule": schedule, "m": m, "trigger": trigger, "arms": list(b["arms"]),
              "analyses": list(b["analyses"]), "nulls": list(b["nulls"]), "n_variability_draws": 0}
    lo, hi = b["patient_ids"]
    return {"model": model, "design": design, "ids": list(range(lo, hi + 1)), "base": int(b["seed_base"]),
            "role": b["role"], "base_block": b.get("base_block")}


def read_fold_selection():
    if not os.path.exists(FOLD_SELECTION):
        raise SystemExit("FOLD2 needs the FOLDP selection record: run 'fold-select' first.")
    with open(FOLD_SELECTION, encoding="utf-8") as f:
        return json.load(f)


def planned(model, design):
    names = set(RUN.planned_streams([design], [model], include_bootstrap=False))
    names |= set(A.MODEL_STREAM_NEW)
    return R.check_unique(names)


def prepare_design(r, workers, used, base=None):
    """Adds the independent-trigger source for indep blocks (frozen rule: same model and seed block,
    primary trigger, over the frozen S1 cohort ids 0-999). `base` overrides the seed block (plumbing)."""
    d = dict(r["design"])
    if d["trigger"] == "indep":
        lo, hi = P2.BLOCKS["S1"]["patient_ids"]
        d["indep_outcomes"] = RUN.indep_source(r["model"], r["base"] if base is None else base,
                                               list(range(lo, hi + 1)), d["schedule"], d["m"], workers, used)
    return d


# ------------------------------------------------------------------ workers
def _work(job):
    model, seed, pid, design, n_surr, label = job
    R.registry_clear()
    t0 = time.perf_counter()
    rows, _ = a1_patient.run(model, seed, design, n_surr)
    sec = time.perf_counter() - t0
    for x in rows:
        x["patient_id"] = pid; x["block"] = label
    return rows, list(R.registry_snapshot().keys()), sec, label


def _frozen_work(job):
    model, seed, pid, design, n_surr, label = job
    R.registry_clear()
    t0 = time.perf_counter()
    patient.run(model, seed, design, n_surr)
    return [], list(R.registry_snapshot().keys()), time.perf_counter() - t0, label


def _dispatch(job):
    return _frozen_work(job[1:]) if job[0] == "frozen" else _work(job[1:])


def _pool_run(jobs, workers, progress=None, ordered=False, log=print):
    """jobs: tuples for _dispatch ('a1'|'frozen', model, seed, pid, design, n_surr, label).
    Returns rows, streams used per label, seconds per label."""
    rows, used, secs = [], {}, {}
    with Pool(workers) as pool:
        it = pool.imap(_dispatch, jobs, chunksize=2) if ordered else pool.imap_unordered(_dispatch, jobs, chunksize=1)
        for k, (r, names, sec, label) in enumerate(it, 1):
            rows.extend(r); used.setdefault(label, set()).update(names); secs.setdefault(label, []).append(sec)
            if progress and (k % progress == 0 or k == len(jobs)):
                log(f"  {k}/{len(jobs)} patients")
    return rows, used, secs


def _same(a, b):
    """Exact equality; NaN equals NaN; booleans compared as booleans."""
    if isinstance(a, bool) or isinstance(b, bool):
        return bool(a) == bool(b)
    try:
        fa, fb = float(a), float(b)
    except (TypeError, ValueError):
        return a == b
    return (np.isnan(fa) and np.isnan(fb)) or fa == fb


# ------------------------------------------------------------------ Stage 0: plumbing
def structural_problems(rows, r, entry, ids, n_surr):
    probs = []
    d = r["design"]
    by_pid = {}
    for x in rows:
        if x.get("block") == entry:
            by_pid.setdefault(int(x["patient_id"]), []).append(x)
    if set(by_pid) != set(ids):
        probs.append(f"{entry}: rows for patients {sorted(by_pid)} but expected {sorted(ids)}")
    arms_expected = [a for a in d["arms"] if not (a == "thinned" and d["schedule"] == "smooth")]
    for pid, rs in by_pid.items():
        if any(x.get("tipped") for x in rs):
            continue
        n_exp = len(arms_expected) * len(d["analyses"]) * len(d["nulls"])
        if len(rs) != n_exp:
            probs.append(f"{entry} patient {pid}: {len(rs)} rows, expected {n_exp}")
        if {x["arm"] for x in rs} != set(arms_expected):
            probs.append(f"{entry} patient {pid}: arms {sorted({x['arm'] for x in rs})}, expected {sorted(arms_expected)}")
        for x in rs:
            if x["arm"] in ("regular_N", "informative_reduced_N", "poisson_matched") and x["n_obs"] != x["N_i"]:
                probs.append(f"{entry} patient {pid} {x['arm']}: n_obs {x['n_obs']} != N_i {x['N_i']}")
            if x["n_baseline_in_arm"] != S.N_BASE:
                probs.append(f"{entry} patient {pid} {x['arm']}: {x['n_baseline_in_arm']} baseline observations")
            for ind in ("var", "ac"):
                p = x[f"p_{ind}"]
                if x[f"evaluable_{ind}"] and not (1.0 / (n_surr + 1) - 1e-12 <= p <= 1):
                    probs.append(f"{entry} patient {pid} {x['arm']}/{x['analysis']}/{x['null']}: p_{ind}={p}")
    return probs


def cmd_plumbing(workers, n_patients=2, n_surr=20):
    P2.check_gate(RUN.OUT)
    log = Log(os.path.join(STAGE0, "plumbing_log.txt"))
    log(f"plumbing: every block, {n_patients} patients, {n_surr} surrogates, seed block {PLUMBING_BASE}")
    jobs, plans, resolved, src_used = [], {}, {}, {}
    fold_probe = A.fold_model_name(*A.FOLD_GRID[0])
    ids = list(range(n_patients))
    for entry in block_list():
        r = resolve(entry, fold_model=fold_probe if entry == "FOLD2" else None)
        resolved[entry] = r
        plans[entry] = set(planned(r["model"], r["design"]))
        u = set()
        d = prepare_design(r, workers, u, base=PLUMBING_BASE)
        src_used[entry] = u
        jobs += [("a1", r["model"], PLUMBING_BASE + i, i, d, n_surr, entry) for i in ids]
    t0 = dt.datetime.now().isoformat(timespec="seconds")
    rows, used, _ = _pool_run(jobs, workers, progress=10, log=log)
    problems = []
    for entry, r in resolved.items():
        problems += structural_problems(rows, r, entry, ids, n_surr)
        extra = sorted((used.get(entry, set()) | src_used[entry]) - plans[entry])
        if extra:
            problems.append(f"{entry}: streams used but not planned: {extra}")
    path = os.path.join(STAGE0, "plumbing_rows_DISCARD.csv")
    RUN.write_rows(rows, path)
    check = {"check": "plumbing", "started": t0, "finished": dt.datetime.now().isoformat(timespec="seconds"),
             "seed_block": PLUMBING_BASE, "blocks": list(resolved), "patients_per_block": n_patients,
             "surrogates": n_surr, "rows": len(rows), "problems": problems, "passed": not problems,
             "a1_sha256": A.A1_SHA256, "config_sha256": S.CONFIG_HASH, "code_sha256": code_hashes(),
             "note": "Plumbing seeds (9,000,000 + id) are outside every production seed block. FOLD2 plumbing uses "
                     "the first fold_v2 grid point only to exercise the code path."}
    with open(os.path.join(STAGE0, "plumbing_check.json"), "w", encoding="utf-8") as f:
        json.dump(check, f, indent=2, default=str)
    log("PLUMBING PASSED" if check["passed"] else "PLUMBING FOUND PROBLEMS:\n" + "\n".join(problems[:40]))


# ------------------------------------------------------------------ identity
def _frozen_rows(block_id):
    rec_p = os.path.join(RUN.OUT, f"phase2_{block_id}_record.json")
    rows_p = os.path.join(RUN.OUT, f"phase2_{block_id}_rows.csv")
    with open(rec_p, encoding="utf-8") as f:
        rec = json.load(f)
    if RUN._sha(rows_p) != rec["rows_sha256"] or rec.get("config_sha256") != S.CONFIG_HASH:
        raise SystemExit(f"{rows_p} does not match its completion record or configuration.")
    return P2.read_rows(rows_p)


def _explore_rows(block):
    """E1: explore_E1__all; E3: explore_E3__all or explore_E3__noct. Verified against their records."""
    tags = ["E1__all"] if block == "E1" else ["E3__all", "E3__noct"]
    for tag in tags:
        rec_p = os.path.join(OUT_EXPLORE, f"explore_{tag}_record.json")
        rows_p = os.path.join(OUT_EXPLORE, f"explore_{tag}_rows.csv")
        if os.path.exists(rec_p) and os.path.exists(rows_p):
            with open(rec_p, encoding="utf-8") as f:
                rec = json.load(f)
            if RUN._sha(rows_p) != rec["rows_sha256"] or rec.get("config_sha256") != S.CONFIG_HASH:
                raise SystemExit(f"{rows_p} does not match its record or configuration.")
            return tag, P2.read_rows(rows_p)
    return None, None


def compare_to_frozen(rows, frozen, keys_of, ids, arms, tests):
    """Every completed row for these patients, arms and tests (index analysis) must exist in the A1 rows
    and match exactly."""
    ids, arms, tests = set(ids), set(arms), set(tests)
    ref = {}
    for x in frozen:
        if x.get("arm") in arms and x.get("analysis") == "index" and x.get("null") in tests \
                and int(x["patient_id"]) in ids:
            ref[(int(x["patient_id"]), x["arm"], x["analysis"], x["null"])] = x
    n_cmp, bad, seen = 0, [], set()
    for x in rows:
        if not x.get("arm"):
            continue
        k = (int(x["patient_id"]), x["arm"], x["analysis"], x["null"])
        if k not in ref:
            continue
        seen.add(k); n_cmp += 1
        diff = [f for f in keys_of if not _same(x[f], ref[k][f])]
        if diff:
            bad.append(f"{k}: differs in {diff}")
    bad += [f"{k}: completed row missing from the A1 output" for k in sorted(set(ref) - seen)]
    return n_cmp, bad


def compare_obs_to_explore(rows, ex, ids, analyses):
    ids, analyses = set(ids), set(analyses)
    ref = {(int(x["patient_id"]), x["arm"], x["analysis"]): x for x in ex
           if x.get("arm") and int(x["patient_id"]) in ids and x.get("analysis") in analyses}
    n_cmp, bad, seen = 0, [], set()
    for x in rows:
        if not x.get("arm"):
            continue
        k = (int(x["patient_id"]), x["arm"], x["analysis"])
        if k not in ref or k in seen:
            continue
        seen.add(k); n_cmp += 1
        diff = [f for f in ("tau_var", "tau_ac", "n_obs") if not _same(x[f], ref[k][f])]
        if diff:
            bad.append(f"{k}: observed statistic differs in {diff}")
    bad += [f"{k}: E-run row missing from the A1 output" for k in sorted(set(ref) - seen)]
    return n_cmp, bad


def cmd_identity(workers, n_patients=10):
    """Stage 0 identity on the first n_patients of every original-seed block, running the FULL production
    design (all arms and tests); rows that exist in the completed outputs must match exactly. P5/P5FA:
    the observed statistic of every representation must match E1/E3. A skipped comparison is a failure."""
    P2.check_gate(RUN.OUT)
    log = Log(os.path.join(STAGE0, "identity_log.txt"))
    report, problems = {}, []
    for entry in block_list():
        if entry.startswith("FOLDP") or entry == "FOLD2" or "frozen_block" in A.BLOCKS.get(entry, {}):
            continue                                   # E2 was never run in Phase 2: nothing to reproduce
        r = resolve(entry)
        if r["base"] != P2.BASE:
            continue
        ids = r["ids"][:n_patients]
        if entry in ("P5", "P5FA"):
            src_block = "E1" if entry == "P5" else "E3"
            tag, ex = _explore_rows(src_block)
            if ex is None:
                problems.append(f"{entry}: {src_block} full-run outputs not found in outputs_explore - identity not shown")
                report[entry] = {"against": src_block, "status": "NOT RUN (counts as failure)"}
                continue
            d = dict(r["design"], nulls=["shuffle_rep"])          # observed statistic does not depend on the test
            rows, _, _ = _pool_run([("a1", r["model"], r["base"] + i, i, d, 1, entry) for i in ids], workers, log=log)
            n_cmp, bad = compare_obs_to_explore(rows, ex, ids, r["design"]["analyses"])
            report[entry] = {"against": f"explore_{tag}", "patients": ids, "rows_compared": n_cmp, "rows_differing": len(bad)}
        elif r["base_block"]:
            frozen = _frozen_rows(r["base_block"])
            d = prepare_design(r, workers, set())
            rows, _, _ = _pool_run([("a1", r["model"], r["base"] + i, i, d, S.N_SURR, entry) for i in ids], workers, log=log)
            n_cmp, bad = compare_to_frozen(rows, frozen, COMPARE, ids, r["design"]["arms"], r["design"]["nulls"])
            report[entry] = {"against": r["base_block"], "patients": ids, "design": "full production design",
                             "rows_compared": n_cmp, "rows_differing": len(bad)}
        else:
            continue
        if n_cmp == 0:
            bad.append("nothing compared")
        problems += [f"{entry}: {b}" for b in bad]
        log(f"{entry}: compared {n_cmp} rows, {len(bad)} differing")
    check = {"check": "identity", "written": dt.datetime.now().isoformat(timespec="seconds"),
             "report": report, "problems": problems, "passed": not problems,
             "a1_sha256": A.A1_SHA256, "config_sha256": S.CONFIG_HASH, "code_sha256": code_hashes()}
    with open(os.path.join(STAGE0, "identity_check.json"), "w", encoding="utf-8") as f:
        json.dump(check, f, indent=2, default=str)
    log("IDENTITY PASSED" if check["passed"] else "IDENTITY PROBLEMS:\n" + "\n".join(problems[:40]))


# ------------------------------------------------------------------ timing
def p1_actual_minutes():
    with open(os.path.join(RUN.OUT, "phase2_P1_record.json"), encoding="utf-8") as f:
        return float(json.load(f)["minutes"])


def cmd_timing(workers, n_patients=5):
    """Every block on n_patients (plumbing seeds, production 200 surrogates) and 2 x workers P1-design
    benchmark patients through the frozen code, all interleaved in ONE pool so every patient is timed
    under the same contention. Efficiency = ideal P1 minutes / measured P1 minutes (the P1 run used the
    default worker count, cpu_count - 1); projections are invalid if a different worker count is used."""
    P2.check_gate(RUN.OUT)
    log = Log(os.path.join(STAGE0, "timing_log.txt"))
    default_workers = max(1, (os.cpu_count() or 2) - 1)
    jobs, info = [], {}
    fold_probe = A.fold_model_name(*A.FOLD_GRID[0])
    for entry in block_list():
        r = resolve(entry, fold_model=fold_probe if entry == "FOLD2" else None)
        d = prepare_design(r, workers, set(), base=PLUMBING_BASE)
        info[entry] = {"n_patients_production": len(r["ids"])}
        jobs += [("a1", r["model"], PLUMBING_BASE + i, i, d, S.N_SURR, entry) for i in range(n_patients)]
    p1d = dict(P2.block_design(P2.BLOCKS["P1"]), n_variability_draws=0)
    bench = [("frozen", "ou_csd", PLUMBING_BASE + 100 + i, 100 + i, p1d, S.N_SURR, "BENCH_P1") for i in range(2 * workers)]
    allj = []
    for i in range(max(len(jobs), len(bench))):                 # interleave
        if i < len(bench): allj.append(bench[i])
        if i < len(jobs): allj.append(jobs[i])
    t0 = time.time()
    _, _, secs = _pool_run(allj, workers, progress=20, log=log)
    wall = time.time() - t0
    s_p1 = float(np.mean(secs["BENCH_P1"]))
    p1_min = p1_actual_minutes()
    ideal_p1_min = s_p1 * 1000 / workers / 60
    eff = ideal_p1_min / p1_min
    total = 0.0
    for entry, v in info.items():
        s = float(np.mean(secs[entry]))
        proj_h = s * v["n_patients_production"] / workers / eff / 3600
        v.update({"seconds_per_patient_measured": round(s, 2), "projected_hours": round(proj_h, 2)})
        total += proj_h
    rep = {"check": "timing", "written": dt.datetime.now().isoformat(timespec="seconds"), "code_sha256": code_hashes(),
           "workers": workers, "default_workers_on_this_machine": default_workers,
           "valid": workers == default_workers,
           "patients_per_block": n_patients, "surrogates": S.N_SURR, "seed_block": PLUMBING_BASE,
           "benchmark": {"P1_design_patients": len(bench), "P1_seconds_per_patient": round(s_p1, 2),
                         "P1_ideal_minutes": round(ideal_p1_min, 1), "P1_actual_minutes": p1_min,
                         "scaling_efficiency": round(eff, 3),
                         "note": "benchmark omits P1's variability draws (ids 0-199 only in P1), so it slightly under-costs "
                                 "P1; efficiency is then slightly low and projections err long."},
           "blocks": info, "projected_total_hours": round(total, 1), "timing_wall_minutes": round(wall / 60, 1),
           "method": "projected hours = measured seconds per patient (under full load) x production patients / workers / efficiency. "
                     "FOLD2 is projected with the first grid point; FOLDP entries are per grid point (6 x 300 patients)."}
    if not rep["valid"]:
        rep["WARNING"] = f"--workers {workers} differs from the default {default_workers} used for P1; rerun timing without --workers."
    with open(os.path.join(STAGE0, "timing_report.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2, default=str)
    for k, v in info.items():
        log(f"{k:38s} {v['projected_hours']:7.2f} h")
    log(f"Projected total: {rep['projected_total_hours']} hours with {workers} workers" + ("" if rep["valid"] else "  (INVALID: see WARNING)"))


# ------------------------------------------------------------------ Stage 0 report and approval
def cmd_stage0_report():
    out = {"written": dt.datetime.now().isoformat(timespec="seconds"), "a1_sha256": A.A1_SHA256,
           "code_sha256": code_hashes()}
    ok = True
    ut = os.path.join(STAGE0, "unit_tests.txt")
    if os.path.exists(ut):
        with open(ut, encoding="utf-8", errors="replace") as f:
            txt = f.read().replace("\x00", "")
        import re
        tail = [ln for ln in txt.splitlines() if re.search(r"=+ .*\d+ (passed|failed|error)", ln)][-1:]
        passed = bool(tail) and re.search(r"\d+ passed", tail[0]) is not None \
            and " failed" not in tail[0] and " error" not in tail[0]
        newest_py = max(os.path.getmtime(os.path.join(RUN.PKG, f)) for f in os.listdir(RUN.PKG) if f.endswith(".py"))
        after_code = os.path.getmtime(ut) >= newest_py
        out["unit_tests"] = {"summary_line": tail[0] if tail else "not found", "passed": passed,
                             "run_after_last_code_change": after_code, "sha256": RUN._sha(ut)}
        ok &= passed and after_code
    else:
        out["unit_tests"] = "MISSING"; ok = False
    for name in ("plumbing_check", "identity_check", "timing_report"):
        p = os.path.join(STAGE0, f"{name}.json")
        if not os.path.exists(p):
            out[name] = "MISSING"; ok = False; continue
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        code_match = d.get("code_sha256") == out["code_sha256"]
        passed = bool(d.get("passed", d.get("valid", True))) and code_match
        out[name] = {"sha256": RUN._sha(p), "passed": passed, "code_matches_current": code_match,
                     "problems": d.get("problems", [])[:20]}
        if name == "timing_report":
            out[name]["projected_total_hours"] = d["projected_total_hours"]
            out[name]["by_block_hours"] = {k: v["projected_hours"] for k, v in d["blocks"].items()}
        ok &= passed
    out["stage0_checks_passed"] = ok
    out["people_decide"] = ("E2/E2FA run only if the projected runtime is judged feasible; a reduction of E2FA to 500 "
                            "patients needs a deviations.json entry before the run and is strictly descriptive.")
    with open(os.path.join(STAGE0, "stage0_report.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print(json.dumps(out, indent=2, default=str))


def read_deviations():
    if not os.path.exists(DEVIATIONS):
        return []
    with open(DEVIATIONS, encoding="utf-8") as f:
        return json.load(f)


def add_deviation(entry):
    """Append-only: existing entries are never changed."""
    devs = read_deviations()
    entry = dict(entry, number=len(devs) + 1, recorded=dt.datetime.now().isoformat(timespec="seconds"))
    devs.append(entry)
    tmp = DEVIATIONS + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(devs, f, indent=2)
    os.replace(tmp, DEVIATIONS)
    return entry


def cmd_approve():
    p = os.path.join(STAGE0, "stage0_report.json")
    if not os.path.exists(p):
        raise SystemExit("Run stage0-report first.")
    with open(p, encoding="utf-8") as f:
        rep = json.load(f)
    if not rep.get("stage0_checks_passed"):
        raise SystemExit("Stage 0 checks did not all pass; production runs are not allowed.")
    if rep["code_sha256"] != code_hashes():
        raise SystemExit("Code has changed since the Stage 0 report; rerun Stage 0.")
    if not any(d.get("block") == "E2FA" for d in read_deviations()):
        add_deviation({"block": "E2FA", "deviation": A.BLOCKS["E2FA"]["deviation"],
                       "source": "Protocol Addendum A1 v1.2, Section 6.7 (entry 1 by protocol)"})
    if os.path.exists(APPROVAL):
        print("Stage 0 approval already recorded:", APPROVAL); return
    RUN.write_immutable({"approved": dt.datetime.now().isoformat(timespec="seconds"),
                         "stage0_report_sha256": RUN._sha(p), "a1_sha256": A.A1_SHA256, "code_sha256": code_hashes(),
                         "note": "Recorded after the Stage 0 report and projected runtime were approved."},
                        APPROVAL)
    print("Stage 0 approval recorded:", APPROVAL)


# ------------------------------------------------------------------ production
def _require_approval():
    if not os.path.exists(APPROVAL):
        raise SystemExit("Production runs need the Stage 0 approval record (approve-stage0).")
    with open(APPROVAL, encoding="utf-8") as f:
        ap = json.load(f)
    if ap["code_sha256"] != code_hashes():
        raise SystemExit("Code differs from the code approved at Stage 0; production refused.")


def _tag(entry):
    return entry.replace("|", "_").replace("=", "")


def _done(entry):
    return os.path.exists(os.path.join(OUT_A1, f"a1_{_tag(entry)}_record.json"))


def _require_order(entry):
    base = entry.split("|")[0]
    stage = next((i for i, s in enumerate(STAGES) if base in s), None)
    if stage is None:
        raise SystemExit(f"Unknown block {entry}. Blocks: {', '.join(block_list())}")
    missing = []
    for s in STAGES[:stage]:
        for b in s:
            if b == "FOLDP":
                missing += [f"FOLDP|{A.fold_model_name(sf, re)}" for sf, re in A.FOLD_GRID
                            if not _done(f"FOLDP|{A.fold_model_name(sf, re)}")]
            elif b == "FOLD2":
                if not _done("FOLD2") and not (os.path.exists(FOLD_SELECTION) and not read_fold_selection().get("selected_model")):
                    missing.append("FOLD2")
            elif not _done(b):
                missing.append(b)
    if base == "FOLD2" and not os.path.exists(FOLD_SELECTION):
        missing.append("FOLDP selection (fold-select)")
    if base == "E2FA" and not any(d.get("block") == "E2FA" for d in read_deviations()):
        missing.append("deviations.json entry for E2FA")
    if missing:
        raise SystemExit(f"{entry}: earlier stages not complete (frozen run_order): {missing}")


def run_entry(entry, workers):
    _require_approval()
    _require_order(entry)
    P2.check_gate(RUN.OUT)
    tag = _tag(entry)
    rec_path = os.path.join(OUT_A1, f"a1_{tag}_record.json")
    if os.path.exists(rec_path):
        print(f"{entry}: already completed (record exists) - skipped."); return rec_path
    log = Log(os.path.join(OUT_A1, f"a1_{tag}_log.txt"))
    r = resolve(entry)
    rows_path = os.path.join(OUT_A1, f"a1_{tag}_rows.csv")
    tmp = rows_path + ".tmp"
    open(tmp, "w").close(); os.remove(tmp)                       # pre-flight path check
    plan = planned(r["model"], r["design"])
    used = set()
    d = prepare_design(r, workers, used)
    t0 = dt.datetime.now().isoformat(timespec="seconds"); t1 = time.time()
    log(f"{entry}: model {r['model']}, {len(r['ids'])} patients, seed block {r['base']}, workers {workers}")
    jobs = [("a1", r["model"], r["base"] + i, i, d, S.N_SURR, entry) for i in r["ids"]]
    rows, u, _ = _pool_run(jobs, workers, progress=50, ordered=True, log=log)
    used |= set().union(*u.values()) if u else set()
    for x in rows:
        x["phase"] = "addendum_A1"; x["seed_base"] = r["base"]
    RUN.write_rows(rows, tmp)
    RUN._abort_if_unplanned(plan, used)
    os.replace(tmp, rows_path)
    record = {"phase": "addendum_A1", "block": entry, "role": r["role"], "model": r["model"],
              "design": {k: v for k, v in d.items() if k != "indep_outcomes"},
              "seed_base": r["base"], "patient_ids": [r["ids"][0], r["ids"][-1]], "n_patients": len(r["ids"]),
              "rows": len(rows), "tipped": len({x["patient_id"] for x in rows if x.get("tipped")}),
              "minutes": round((time.time() - t1) / 60, 1), "rows_sha256": RUN._sha(rows_path),
              "a1_sha256": A.A1_SHA256, "config_sha256": S.CONFIG_HASH, "parent_config_sha256": S.PARENT_SHA256,
              "written": dt.datetime.now().isoformat(timespec="seconds")}
    man = RUN.manifest(f"a1_{tag}", r["base"], {"a1_block": A.BLOCKS.get(entry.split("|")[0]), "record": record,
                                                "a1_sha256": A.A1_SHA256}, plan, used, [rows_path], t0)
    with open(os.path.join(OUT_A1, f"a1_{tag}_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(man, f, indent=2, default=str)
    RUN.write_immutable(record, rec_path)
    log(f"{entry}: done in {record['minutes']} min, {len(rows)} rows.")
    return rec_path


def _read_a1(entry):
    tag = _tag(entry)
    with open(os.path.join(OUT_A1, f"a1_{tag}_record.json"), encoding="utf-8") as f:
        rec = json.load(f)
    rows_p = os.path.join(OUT_A1, f"a1_{tag}_rows.csv")
    if RUN._sha(rows_p) != rec["rows_sha256"]:
        raise SystemExit(f"{rows_p} does not match its record.")
    return rec, P2.read_rows(rows_p)


def cmd_verify(entry):
    """Full-cohort identity (A1 Section 9): every row of a completed original-seed block that also exists
    in the completed Phase 2 outputs must match exactly; P5/P5FA observed statistics must match E1/E3."""
    log = Log(os.path.join(OUT_A1, f"a1_{_tag(entry)}_identity_log.txt"))
    rec, rows = _read_a1(entry)
    if rec.get("a1_sha256") != A.A1_SHA256_FROZEN_ORIGINAL or rec.get("config_sha256") != S.CONFIG_HASH:
        raise SystemExit(f"{entry}: record was written under a different A1 or Phase 2 configuration.")
    r = resolve(entry) if entry != "FOLD2" else None
    if entry in ("P5", "P5FA"):
        tag, ex = _explore_rows("E1" if entry == "P5" else "E3")
        if ex is None:
            raise SystemExit("E1/E3 full-run outputs not found.")
        n_cmp, bad = compare_obs_to_explore(rows, ex, r["ids"], r["design"]["analyses"])
        against = f"explore_{tag}"
    elif r and r["base"] == P2.BASE and r["base_block"] and r["base_block"] != "E2":
        n_cmp, bad = compare_to_frozen(rows, _frozen_rows(r["base_block"]), COMPARE, r["ids"],
                                       r["design"]["arms"], r["design"]["nulls"])
        against = r["base_block"]
    else:
        raise SystemExit(f"{entry} is not an original-seed block; nothing to verify.")
    out = {"block": entry, "against": against, "rows_compared": n_cmp, "rows_differing": len(bad),
           "differences": bad[:50], "passed": n_cmp > 0 and not bad, "rows_sha256": rec["rows_sha256"],
           "written": dt.datetime.now().isoformat(timespec="seconds")}
    RUN.write_immutable(out, os.path.join(OUT_A1, f"a1_{_tag(entry)}_identity_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"))
    log(f"{entry} vs {against}: {n_cmp} rows compared, {len(bad)} differing -> " + ("PASSED" if out["passed"] else "FAILED: stop and report"))


# ------------------------------------------------------------------ fold pilot
def select_fold(results):
    """Frozen A1 rule with exact integer arithmetic. Each result: model, sigma_f, r_end, n_enrolled,
    tipped, n_nontipped, x (regular_N variance detections among non-tipped).
    Eligible: tipping <= 5%. Window: detection in [35%, 65%]. Choose closest to 50%; ties -> smaller
    sigma_f, then larger r_end. None if no point qualifies."""
    elig = []
    for r in results:
        n, x = r["n_nontipped"], r["x"]
        if 20 * r["tipped"] <= r["n_enrolled"] and n > 0 and 20 * x >= 7 * n and 20 * x <= 13 * n:
            elig.append(r)
    elig.sort(key=lambda r: (Fraction(abs(2 * r["x"] - r["n_nontipped"]), 2 * r["n_nontipped"]), r["sigma_f"], -r["r_end"]))
    return elig[0]["model"] if elig else None


def cmd_fold_select(workers):
    _require_approval()
    if os.path.exists(FOLD_SELECTION):
        print("FOLDP selection already recorded."); return
    results = []
    for sf, re in A.FOLD_GRID:
        model = A.fold_model_name(sf, re)
        entry = f"FOLDP|{model}"
        run_entry(entry, workers)
        rec, rows = _read_a1(entry)
        tipped = len({x["patient_id"] for x in rows if x.get("tipped")})
        reg = [x for x in rows if x.get("arm") == "regular_N" and x.get("null") == "ou_ss" and x.get("analysis") == "index"]
        x = sum(bool(z["det_var"]) for z in reg)
        results.append({"model": model, "sigma_f": sf, "r_end": re, "n_enrolled": rec["n_patients"], "tipped": tipped,
                        "n_nontipped": len(reg), "x": x,
                        "tipping": tipped / rec["n_patients"], "regular_N_variance_detection": x / len(reg) if reg else None})
    sel = select_fold(results)
    RUN.write_immutable({"written": dt.datetime.now().isoformat(timespec="seconds"), "grid_results": results,
                         "rule": A.A1["new_models"]["fold_v2"]["selection_rule"], "selected_model": sel,
                         "a1_sha256": A.A1_SHA256}, FOLD_SELECTION)
    print("Selected:", sel if sel else "none eligible - FOLD2 is not run; fold reported as uninformative")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["plumbing", "identity", "timing", "stage0-report", "approve-stage0",
                                        "run", "fold-select", "verify", "list"])
    ap.add_argument("--block")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    os.makedirs(STAGE0, exist_ok=True)
    if a.command == "plumbing":
        cmd_plumbing(a.workers)
    elif a.command == "identity":
        cmd_identity(a.workers)
    elif a.command == "timing":
        cmd_timing(a.workers)
    elif a.command == "stage0-report":
        cmd_stage0_report()
    elif a.command == "approve-stage0":
        cmd_approve()
    elif a.command == "fold-select":
        cmd_fold_select(a.workers)
    elif a.command == "verify":
        if not a.block:
            raise SystemExit("--block required")
        cmd_verify(a.block)
    elif a.command == "list":
        print("\n".join(block_list()))
    else:
        if not a.block or a.block.startswith("FOLDP"):
            raise SystemExit("--block <ID> required (FOLDP runs only through fold-select)")
        run_entry(a.block, a.workers)


if __name__ == "__main__":
    main()
