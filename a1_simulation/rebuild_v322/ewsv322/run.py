"""Command-line runner with phase gates.

    python -m ewsv322.run plumbing  [--patients 2] [--surrogates 20] [--workers N]
    python -m ewsv322.run phase1a   [--workers N]
    python -m ewsv322.run phase1b   [--workers N]
    python -m ewsv322.run plumbing2 [--patients 2] [--surrogates 20]   (Phase 2 pipeline check, discarded)
    python -m ewsv322.run phase2    [--blocks required|optional|<block id>]
    python -m ewsv322.run summarize

Gates (nulls.phase1a_rule / phase1b_rule):
  * phase1a refuses to run if its freeze record already exists (records are immutable).
  * phase1b refuses to run without the Phase 1a freeze record, and stops if no null was selected.
Every run writes a manifest: config hash, software versions, platform, command, seed block, stream
names and SHA-256 keys (checked unique BEFORE the run), code file hashes, output file hashes.
"""
import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import platform
import stat
import sys
import time
from multiprocessing import Pool

import numpy as np
import scipy

from . import spec as S
from . import rng as R
from . import patient, stats
from .analysis import ANALYSES_BASE

PKG = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(PKG, "..", "outputs")
CAL_ARMS = ("regular_N", "informative_reduced_N")
ALL_ARMS = ("regular_N", "informative_reduced_N", "informative_all", "regular_matched", "thinned",
            "informative_pre", "regular_pre_trunc", "regular_pre_matched")


# ------------------------------------------------------------------ stream planning
def planned_streams(designs, models, include_bootstrap=True):
    names = {"latent", "err_baseline", "err_reg_N", "err_reg_matched", "err_reg_pre_matched",
             "subsample", "indep_trigger", "surrogates"}
    names |= {f"subsample|draw{d}" for d in range(1, S.N_SUBSAMPLE_DRAWS + 1)}
    for d in designs:
        for trig in {d["trigger"], "primary"}:
            tg = R.tag(d["schedule"], d.get("m"), trig)
            names |= {f"schedule|{tg}", f"err_inf|{tg}", f"thin|{tg}"}
        arm_names = set(d["arms"]) | {f"informative_reduced_N|draw{k}" for k in range(1, d.get("n_variability_draws", 0) + 1)}
        for a in arm_names:
            for an in d["analyses"]:
                for nl in d["nulls"]:
                    names.add(f"{a}|{'index' if an == 'elapsed' else an}|{nl}")
    if include_bootstrap:            # Phase 2 block manifests list patient streams only; bootstrap streams go in the summary
        for mdl in models:
            names.add(f"bootstrap|{mdl}|regular_N-vs-informative_reduced_N")
    return R.check_unique(names)


# ------------------------------------------------------------------ worker
def _work(job):
    model, seed, pid, design, n_surr = job
    R.registry_clear()
    rows, _ = patient.run(model, seed, design, n_surr)
    for r in rows:
        r["patient_id"] = pid
    return rows, list(R.registry_snapshot().keys())


def _primary_job(job):
    model, seed, sched, m = job
    R.registry_clear()
    out = patient.primary_outcome(model, seed, sched, m)
    return out, list(R.registry_snapshot().keys())


def run_cohort(model, base, ids, design, workers, n_surr, used):
    jobs = [(model, base + i, i, design, n_surr) for i in ids]
    rows = []
    with Pool(workers) as pool:
        for k, (r, names) in enumerate(pool.imap(_work, jobs, chunksize=2), 1):
            rows.extend(r); used.update(names)
            if k % 50 == 0 or k == len(jobs):
                print(f"  {model}: {k}/{len(jobs)} patients", flush=True)
    return rows


def indep_source(model, base, ids, sched, m, workers, used):
    """trigger.independent_implementation: empirical primary-trigger outcomes of the complete cohort.
    IMPL-NOTE (see IMPLEMENTATION_NOTES.md): tipped fold patients (NaN) are excluded from the source distribution."""
    with Pool(workers) as pool:
        res = pool.map(_primary_job, [(model, base + i, sched, m) for i in ids], chunksize=8)
    for _, names in res:
        used.update(names)
    out = np.array([o for o, _ in res], float)
    return out[~np.isnan(out)]


# ------------------------------------------------------------------ io
def _sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def write_rows(rows, path):
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_immutable(obj, path):
    if os.path.exists(path):
        raise RuntimeError(f"{path} already exists; freeze records are immutable")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
    os.chmod(path, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)


def manifest(kind, base, extra, planned, used, outputs, t0):
    code = {f: _sha(os.path.join(PKG, f)) for f in sorted(os.listdir(PKG)) if f.endswith(".py")}
    unplanned = sorted(set(used) - set(planned))
    m = {"run": kind, "started": t0, "finished": dt.datetime.now().isoformat(timespec="seconds"),
         "command": " ".join(sys.argv), "config_file": os.path.basename(S.CONFIG_PATH),
         "config_sha256": S.CONFIG_HASH, "python": sys.version, "numpy": np.__version__,
         "scipy": scipy.__version__, "platform": platform.platform(), "seed_block_base": base,
         "stream_keys_planned_and_checked_unique_before_run": planned,
         "streams_used_not_in_plan": unplanned, "code_sha256": code,
         "outputs_sha256": {os.path.basename(p): _sha(p) for p in outputs},
         "stream_identities": {
             "method": "Generator(PCG64(SeedSequence(entropy=E, spawn_key=S)))",
             "patient_streams": "E = seed_block_base + patient_id; S = (K(name),) for every planned name except "
                                "'surrogates', bootstrap names and '<arm>|<analysis>|<null>' labels",
             "surrogate_replicates": f"E = seed_block_base + patient_id; S = (K('surrogates'), K('<arm>|<analysis>|<null>'), k), "
                                     f"k = 0..B-1 (B = {S.N_SURR} in production)",
             "bootstrap": "E = seed_block_base; S = (K('bootstrap|<model>|<comparison>'),)",
             "K": "first 16 hex digits of SHA-256(UTF-8 name) as an integer (values listed in "
                  "stream_keys_planned_and_checked_unique_before_run)"}}
    m.update(extra)
    if unplanned:
        m["ERROR"] = "streams used that were not planned/checked"
    return m


# ------------------------------------------------------------------ calibration summary
def calibration_table(rows, controls):
    """False-alarm rate = detections / all non-tipped patients (non-evaluable = non-detection;
    IMPL-NOTE (see IMPLEMENTATION_NOTES.md): calibration uses the primary 'index' analysis)."""
    calib, evals = {}, {}
    for null in S.NULLS:
        calib[null] = {}
        for ctrl in controls:
            for arm in CAL_ARMS:
                sel = [r for r in rows if r.get("model") == ctrl and r.get("arm") == arm
                       and r.get("null") == null and r.get("analysis") == "index"]
                n = len(sel)
                for ind in ("var", "ac"):
                    x = sum(bool(r[f"det_{ind}"]) for r in sel)
                    calib[null][(ctrl, arm, ind)] = stats.tost(x, n)
                    ne = sum(bool(r[f"evaluable_{ind}"]) for r in sel)
                    evals[f"{null}|{ctrl}|{arm}|{ind}"] = {"evaluable": ne, "n": n,
                                                          "cp95": stats.clopper_pearson(ne, n)}
    lb = {}
    for arm in CAL_ARMS:
        ps = [r["ljung_box_p"] for r in rows if r.get("model") == "stable" and r.get("arm") == arm
              and r.get("null") == "ou_ss" and r.get("analysis") == "index"
              and np.isfinite(r.get("ljung_box_p", np.nan))]
        lb[arm] = {"n": len(ps), "share_p_below_0.05": float(np.mean(np.array(ps) < 0.05)) if ps else None}
    return calib, evals, lb


def _jsonable_calib(calib):
    return {null: {"|".join(k): v for k, v in cells.items()} for null, cells in calib.items()}


def calibration_design(trigger):
    return {"schedule": "step", "m": S.M_PRIMARY, "trigger": trigger, "arms": list(CAL_ARMS),
            "analyses": ["index"], "nulls": list(S.NULLS), "n_variability_draws": 0}


def run_calibration(phase, workers):
    base = S.PHASE_BASE[phase]
    ids = range(S.N_CALIB)
    t0 = dt.datetime.now().isoformat(timespec="seconds")
    d_stable, d_noise = calibration_design("indep"), calibration_design("primary")
    planned = planned_streams([d_stable, d_noise], ["stable", "noise_only"])
    used = set()
    print(f"{phase}: building independent-trigger source distribution (stable cohort, primary trigger)")
    src = indep_source("stable", base, ids, "step", S.M_PRIMARY, workers, used)
    d_stable["indep_outcomes"] = src
    rows = run_cohort("stable", base, ids, d_stable, workers, S.N_SURR, used)
    rows += run_cohort("noise_only", base, ids, d_noise, workers, S.N_SURR, used)
    for r in rows:
        r["phase"] = phase
    os.makedirs(OUT, exist_ok=True)
    rows_path = os.path.join(OUT, f"{phase}_rows.csv")
    write_rows(rows, rows_path)
    calib, evals, lb = calibration_table(rows, ("stable", "noise_only"))
    src_summary = {"n": int(len(src)), "never_escalated_share": float(np.mean(~np.isfinite(src))),
                   "escalation_time_quantiles": [float(v) for v in np.percentile(src[np.isfinite(src)], [5, 25, 50, 75, 95])]
                   if np.isfinite(src).any() else None}
    realised = np.array([r["t_esc"] for r in rows if r.get("model") == "stable" and r.get("arm") == "regular_N"
                         and r.get("null") == "ou_ss"], float)
    fin = realised[np.isfinite(realised)]
    realised_summary = {"n": int(len(realised)), "never_escalated_share": float(np.mean(~np.isfinite(realised))),
                        "escalation_time_quantiles": [float(v) for v in np.percentile(fin, [5, 25, 50, 75, 95])]
                        if len(fin) else None}
    return base, t0, rows_path, calib, evals, lb, planned, used, src_summary, realised_summary


def _abort_if_unplanned(planned, used):
    extra = sorted(set(used) - set(planned))
    if extra:
        raise SystemExit("Streams used that were not planned and checked before the run: " + ", ".join(extra)
                         + " — no record written.")


def cmd_phase1a(workers):
    freeze_path = os.path.join(OUT, "phase1a_freeze_record.json")
    if os.path.exists(freeze_path):
        raise SystemExit("Phase 1a freeze record already exists — Phase 1a cannot be re-run.")
    base, t0, rows_path, calib, evals, lb, planned, used, src, real = run_calibration("phase1a", workers)
    selected = stats.select_null(calib)
    record = {"phase": "phase1a", "written": dt.datetime.now().isoformat(timespec="seconds"),
              "config_sha256": S.CONFIG_HASH, "selected_null": selected,
              "decision": ("selected: proceed to Phase 1b" if selected else
                           "NO candidate passed: stop the calibrated inferential workflow; no calibrated Phase 1b/2"),
              "tost_all_candidates": _jsonable_calib(calib), "evaluability": evals,
              "ljung_box_stable_control": lb, "indep_source_distribution": src,
              "indep_realised_distribution": real, "rows_sha256": _sha(rows_path)}
    os.makedirs(OUT, exist_ok=True)
    _abort_if_unplanned(planned, used)
    write_immutable(record, freeze_path)
    write_immutable(record, os.path.join(OUT, f"phase1a_freeze_record_{dt.date.today().isoformat()}.json"))
    man = manifest("phase1a", base, {"selected_null": selected}, planned, used, [rows_path, freeze_path], t0)
    with open(os.path.join(OUT, "phase1a_manifest.json"), "w") as f:
        json.dump(man, f, indent=2, default=str)
    print(json.dumps({"selected_null": selected}, indent=2))


def cmd_phase1b(workers):
    freeze_path = os.path.join(OUT, "phase1a_freeze_record.json")
    rec_path = os.path.join(OUT, "phase1b_record.json")
    if not os.path.exists(freeze_path):
        raise SystemExit("Phase 1b requires the Phase 1a freeze record.")
    if os.path.exists(rec_path):
        raise SystemExit("Phase 1b record already exists — Phase 1b cannot be re-run.")
    with open(freeze_path, encoding="utf-8") as f:
        fr = json.load(f)
    if fr["config_sha256"] != S.CONFIG_HASH:
        raise SystemExit("Config changed since Phase 1a freeze — refusing to run.")
    if fr["selected_null"] is None:
        raise SystemExit("Phase 1a selected no null: the calibrated workflow stops here (nulls.phase1a_rule.fail).")
    base, t0, rows_path, calib, evals, lb, planned, used, src, real = run_calibration("phase1b", workers)
    decision = stats.phase1b_decision(calib, fr["selected_null"])
    record = {"phase": "phase1b", "written": dt.datetime.now().isoformat(timespec="seconds"),
              "config_sha256": S.CONFIG_HASH, "selected_null_from_phase1a": fr["selected_null"],
              "decision": decision, "tost_all_candidates": _jsonable_calib(calib), "evaluability": evals,
              "ljung_box_stable_control": lb, "indep_source_distribution": src,
              "indep_realised_distribution": real, "rows_sha256": _sha(rows_path)}
    _abort_if_unplanned(planned, used)
    write_immutable(record, rec_path)
    write_immutable(record, os.path.join(OUT, f"phase1b_record_{dt.date.today().isoformat()}.json"))
    man = manifest("phase1b", base, {"decision": decision}, planned, used, [rows_path, rec_path], t0)
    with open(os.path.join(OUT, "phase1b_manifest.json"), "w") as f:
        json.dump(man, f, indent=2, default=str)
    print(json.dumps(decision, indent=2))


def cmd_plumbing(n_patients, n_surr, workers):
    """Pipeline check on seed block 9,000,000 (outside every phase). Results are NOT summarised and the
    row file is labelled DISCARD. Checks structure only; reports run time per patient."""
    base = S.PLUMBING_BASE
    t0 = dt.datetime.now().isoformat(timespec="seconds")
    ids = range(n_patients)
    designs = []
    for model in ("ou_csd", "ou_combined", "fold", "stable", "level_drop_only", "noise_only"):
        designs.append((model, {"schedule": "step", "m": S.M_PRIMARY, "trigger": "primary"}))
        if model != "noise_only":
            designs.append((model, {"schedule": "ramp", "m": S.M_PRIMARY, "trigger": "primary"}))
            designs.append((model, {"schedule": "smooth", "m": None, "trigger": "none"}))
    designs.append(("stable", {"schedule": "step", "m": S.M_PRIMARY, "trigger": "indep"}))
    for _, d in designs:
        d.update(arms=list(ALL_ARMS), analyses=list(ANALYSES_BASE), nulls=list(S.NULLS), n_variability_draws=2)
    planned = planned_streams([d for _, d in designs], sorted({m for m, _ in designs}))
    used, rows, timing, problems = set(), [], {}, []
    for model, d in designs:
        if d["trigger"] == "indep":
            d["indep_outcomes"] = indep_source(model, base, ids, "step", S.M_PRIMARY, workers, used)
        t1 = time.time()
        r = run_cohort(model, base, ids, d, workers, n_surr, used)
        timing[f"{model}|{R.tag(d['schedule'], d['m'], d['trigger'])}"] = round((time.time() - t1) / n_patients, 1)
        rows += r
    # structural checks
    for r in rows:
        if r.get("tipped"):
            continue
        if not (8 <= r["N_i"] <= 60):
            problems.append(f"N_i out of range: {r['seed']} {r['N_i']}")
        if r["arm"] in ("regular_N", "informative_reduced_N") and r["n_obs"] != r["N_i"]:
            problems.append(f"matched count broken: {r['seed']} {r['arm']} n_obs={r['n_obs']} N_i={r['N_i']}")
        if r["n_baseline_in_arm"] != 8 and r["arm"] != "regular_pre_trunc":
            problems.append(f"baseline count != 8: {r['seed']} {r['arm']}")
        for ind in ("var", "ac"):
            p = r[f"p_{ind}"]
            if np.isfinite(r[f"tau_{ind}"]) and not (1 / (n_surr + 1) - 1e-12 <= p <= 1 + 1e-12):
                problems.append(f"p out of range: {r['seed']} {r['arm']} {r['analysis']} {ind} {p}")
    os.makedirs(OUT, exist_ok=True)
    rows_path = os.path.join(OUT, "plumbing_rows_DISCARD_not_results.csv")
    write_rows(rows, rows_path)
    check = {"rows": len(rows), "tipped_rows": sum(bool(r.get("tipped")) for r in rows),
             "structural_problems": problems, "seconds_per_patient_by_design": timing,
             "surrogates_used_in_plumbing_only": n_surr}
    man = manifest("plumbing", base, {"structural_check": check}, planned, used, [rows_path], t0)
    with open(os.path.join(OUT, "plumbing_manifest.json"), "w") as f:
        json.dump(man, f, indent=2, default=str)
    print(json.dumps(check, indent=2))
    print("Plumbing PASSED" if not problems and not man.get("ERROR") else "Plumbing FOUND PROBLEMS — see manifest")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["plumbing", "phase1a", "phase1b", "plumbing2", "phase2", "summarize"])
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--patients", type=int, default=2, help="plumbing / plumbing2 only")
    ap.add_argument("--surrogates", type=int, default=20, help="plumbing / plumbing2 only; production always uses the config value (200)")
    ap.add_argument("--blocks", default="required",
                    help="phase2 only: 'required' (P1,S1-S3,R1-R10,N1-N2), 'optional' (E1-E3), or one block id such as P1")
    a = ap.parse_args()
    if a.command == "plumbing":
        cmd_plumbing(a.patients, a.surrogates, a.workers)
    elif a.command == "phase1a":
        cmd_phase1a(a.workers)
    elif a.command == "phase1b":
        cmd_phase1b(a.workers)
    else:
        from . import phase2          # imported here to avoid a circular import
        if a.command == "plumbing2":
            phase2.cmd_plumbing2(a.patients, a.surrogates, a.workers)
        elif a.command == "phase2":
            phase2.cmd_phase2(a.blocks, a.workers)
        else:
            phase2.check_gate()
            summ, path = phase2.summarize()
            print("Summary written to", path)
            print(json.dumps(summ.get("PRIMARY_RESULT", "P1 not completed yet"), indent=2, default=str))


if __name__ == "__main__":
    main()
