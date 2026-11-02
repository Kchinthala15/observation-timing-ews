"""Phase 2 — run blocks from the frozen phase2_matrix and summarise them.

Governing: config/Simulation_Rebuild_Config_v3.2.2_Phase2.json (key phase2_matrix), approved in
the Phase 2 run-matrix proposal (revision 3).

Gate (phase2_matrix.gate): outputs/phase1b_record.json must exist with decision.calibrated = true,
selected_null_from_phase1a = 'ou_ss', and config_sha256 equal to the parent hash recorded in
config_lineage. Each block writes an immutable record; a completed block cannot be re-run.
"""
import csv
import datetime as dt
import json
import math
import os
import time

import numpy as np

from . import spec as S
from . import rng as R
from . import stats
from . import run as RUN

MATRIX = S.CFG["phase2_matrix"]
BLOCKS = {b["id"]: b for b in MATRIX["blocks"]}
REQUIRED = [b["id"] for b in MATRIX["blocks"] if b["role"] != "exploratory"]
OPTIONAL = [b["id"] for b in MATRIX["blocks"] if b["role"] == "exploratory"]
BASE = S.PHASE_BASE["phase2"]
PRIMARY_STREAM_COMPARISON = "regular_N-vs-informative_reduced_N"   # bootstrap|ou_csd|regular_N-vs-informative_reduced_N


# ------------------------------------------------------------------ gate
def check_gate(out_dir=None):
    out_dir = out_dir or RUN.OUT
    p = os.path.join(out_dir, "phase1b_record.json")
    if not os.path.exists(p):
        raise SystemExit("Phase 2 gate: outputs/phase1b_record.json not found.")
    with open(p, encoding="utf-8") as f:
        rec = json.load(f)
    if not rec.get("decision", {}).get("calibrated"):
        raise SystemExit("Phase 2 gate: Phase 1b did not pass (calibrated is not true). Calibrated Phase 2 not permitted.")
    if rec.get("selected_null_from_phase1a") != "ou_ss":
        raise SystemExit("Phase 2 gate: the selected null is not ou_ss.")
    if rec.get("config_sha256") != S.CFG["config_lineage"]["parent_config_sha256"]:
        raise SystemExit("Phase 2 gate: Phase 1b ran on a configuration other than the recorded parent.")
    return rec


# ------------------------------------------------------------------ design
def block_design(b):
    d = {"schedule": b["schedule"], "m": b["m"], "trigger": b["trigger"], "arms": list(b["arms"]),
         "analyses": list(b["analyses"]), "nulls": list(b["nulls"]), "n_variability_draws": 0}
    v = b.get("variability_draws")
    if v:
        d["variability_analyses"] = [v["analysis"]]
        d["variability_nulls"] = [v["null"]]
    return d


def block_ids(b, n_limit=None):
    lo, hi = b["patient_ids"]
    ids = list(range(lo, hi + 1))
    return ids[:n_limit] if n_limit else ids


def _jobs(b, design, ids, base, n_surr):
    v = b.get("variability_draws")
    jobs = []
    for i in ids:
        d = design
        if v and v["patient_ids"][0] <= i <= v["patient_ids"][1]:
            d = dict(design, n_variability_draws=int(v["draws"]))
        jobs.append((b["model"], base + i, i, d, n_surr))
    return jobs


def _planned(b, design):
    d = dict(design)
    if b.get("variability_draws"):
        d["n_variability_draws"] = int(b["variability_draws"]["draws"])
    return RUN.planned_streams([d], [b["model"]], include_bootstrap=False)


def run_block_rows(b, base, ids, n_surr, workers, used):
    from multiprocessing import Pool
    design = block_design(b)
    if b["trigger"] == "indep":
        # trigger.independent_implementation: source = same model and seed block, primary trigger
        design["indep_outcomes"] = RUN.indep_source(b["model"], base, ids, b["schedule"], b["m"], workers, used)
    rows = []
    jobs = _jobs(b, design, ids, base, n_surr)
    with Pool(workers) as pool:
        for k, (r, names) in enumerate(pool.imap(RUN._work, jobs, chunksize=2), 1):
            for x in r:
                x["block"] = b["id"]
            rows.extend(r); used.update(names)
            if k % 50 == 0 or k == len(jobs):
                print(f"  {b['id']} ({b['model']}): {k}/{len(jobs)} patients", flush=True)
    return rows, design


def cmd_block(block_id, workers, out_dir=None):
    out_dir = out_dir or RUN.OUT
    check_gate(out_dir)
    b = BLOCKS[block_id]
    rec_path = os.path.join(out_dir, f"phase2_{block_id}_record.json")
    if os.path.exists(rec_path):
        print(f"{block_id}: already completed (record exists) - skipped.")
        return
    t0 = dt.datetime.now().isoformat(timespec="seconds")
    t1 = time.time()
    used = set()
    design = block_design(b)
    planned = _planned(b, design)
    rows, _ = run_block_rows(b, BASE, block_ids(b), S.N_SURR, workers, used)
    for r in rows:
        r["phase"] = "phase2"
    rows_path = os.path.join(out_dir, f"phase2_{block_id}_rows.csv")
    tmp_path = rows_path + ".tmp"
    RUN.write_rows(rows, tmp_path)
    RUN._abort_if_unplanned(planned, used)          # aborts before anything is final; the .tmp file is not summarised
    os.replace(tmp_path, rows_path)
    record = {"phase": "phase2", "block": block_id, "role": b["role"],
              "written": dt.datetime.now().isoformat(timespec="seconds"),
              "config_sha256": S.CONFIG_HASH, "parent_config_sha256": S.PARENT_SHA256,
              "n_patients": len(block_ids(b)), "rows": len(rows),
              "tipped": sum(bool(r.get("tipped")) for r in rows),
              "minutes": round((time.time() - t1) / 60, 1), "rows_sha256": RUN._sha(rows_path)}
    man = RUN.manifest(f"phase2_{block_id}", BASE, {"block": b, "parent_config_sha256": S.PARENT_SHA256,
                                                     "record": record}, planned, used, [rows_path], t0)
    with open(os.path.join(out_dir, f"phase2_{block_id}_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(man, f, indent=2, default=str)
    RUN.write_immutable(record, rec_path)          # last step: the block counts as complete only from here
    print(f"{block_id}: done in {record['minutes']} min, {len(rows)} rows.")


def cmd_phase2(which, workers):
    ids = REQUIRED if which == "required" else OPTIONAL if which == "optional" else [which]
    for bid in ids:
        if bid not in BLOCKS:
            raise SystemExit(f"Unknown block {bid}. Blocks: {', '.join(BLOCKS)}")
        cmd_block(bid, workers)


# ------------------------------------------------------------------ reading rows
_NUM = {"tau_var", "p_var", "tau_ac", "p_ac", "t_esc", "t_tip", "m", "N_i", "n_obs", "patient_id", "seed",
        "n_valid_windows_var", "n_valid_windows_ac", "n_post", "n_inf_total"}
_BOOL = {"det_var", "det_ac", "evaluable_var", "evaluable_ac", "tipped", "never_escalated", "escalated"}


def _parse(r):
    out = dict(r)
    for k in _NUM & r.keys():
        v = r[k]
        out[k] = float(v) if v not in ("", "None") else math.nan
    for k in _BOOL & r.keys():
        out[k] = r[k] == "True"
    return out


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return [_parse(r) for r in csv.DictReader(f)]


# ------------------------------------------------------------------ summaries
def _cp(x, n):
    return [round(v, 4) for v in stats.clopper_pearson(int(x), int(n))] if n else None


def outcome_vector(rows, arm, analysis, null, ind, ids):
    """0/1 detection per patient id, in the order of `ids` (non-evaluable = 0)."""
    d = {int(r["patient_id"]): int(bool(r[f"det_{ind}"])) for r in rows
         if r.get("arm") == arm and r.get("analysis") == analysis and r.get("null") == null}
    return np.array([d[i] for i in ids], float)


def analysed_ids(rows):
    """Patient ids that were analysed (fold: non-tipped only, the labelled conditional subset)."""
    return sorted({int(r["patient_id"]) for r in rows if not r.get("tipped")})


def comparison_name(bid, an, nl):
    """Bootstrap stream comparison label: the primary comparison keeps the exact name from the matrix;
    every other block/analysis/null comparison gets its own named stream."""
    if bid == "P1" and an == "index" and nl == "ou_ss":
        return PRIMARY_STREAM_COMPARISON
    return f"{bid}:{an}|{nl}:{PRIMARY_STREAM_COMPARISON}"


def paired(rows, block, model, armA, armB, analysis, null, ids, B, stream_comparison, primary=False, base=BASE):
    """d = detection(armA) - detection(armB); both outcomes use the same resampled patient ids
    (a fresh generator from the same named cohort stream for each outcome)."""
    out = {"block": block, "armA": armA, "armB": armB, "analysis": analysis, "null": null, "n": len(ids),
           "null_role": "calibrated" if null == "ou_ss" else "sensitivity (side by side)",
           "bootstrap_stream": f"bootstrap|{model}|{stream_comparison}"}
    for ind in ("var", "ac"):
        a = outcome_vector(rows, armA, analysis, null, ind, ids)
        b = outcome_vector(rows, armB, analysis, null, ind, ids)
        est, lo, hi = stats.paired_bootstrap(a, b, R.bootstrap_stream(base, model, stream_comparison), B=B)
        e = int(np.sum((a == 1) & (b == 1))); f = int(np.sum((a == 1) & (b == 0)))
        g = int(np.sum((a == 0) & (b == 1))); h = int(np.sum((a == 0) & (b == 0)))
        _, nl, nu = stats.newcombe_paired(e, f, g, h)
        res = {"rate_A": float(a.mean()), "rate_B": float(b.mean()), "diff": est, "ci95_bootstrap": [lo, hi],
               "newcombe_crosscheck": [nl, nu], "cells_e_f_g_h": [e, f, g, h]}
        if primary:
            res["interpretation"] = stats.interpret(lo, hi)
        out[ind] = res
    return out


def rates(rows):
    """Detection and evaluability rates with Clopper-Pearson 95% CIs per arm x analysis x null.
    Elapsed rows (verification only) and the extra variability draws are excluded here."""
    out = {}
    keys = sorted({(r["arm"], r["analysis"], r["null"]) for r in rows if not r.get("tipped")
                   and r.get("analysis") != "elapsed" and not str(r.get("arm")).startswith("informative_reduced_N|draw")})
    for arm, an, nl in keys:
        sel = [r for r in rows if r.get("arm") == arm and r.get("analysis") == an and r.get("null") == nl]
        n = len(sel)
        cell = {"n": n}
        for ind in ("var", "ac"):
            x = sum(bool(r[f"det_{ind}"]) for r in sel); ev = sum(bool(r[f"evaluable_{ind}"]) for r in sel)
            cell[ind] = {"detection": x / n if n else None, "detection_cp95": _cp(x, n),
                         "evaluable": ev / n if n else None, "evaluable_cp95": _cp(ev, n)}
        out[f"{arm}|{an}|{nl}"] = cell
    return out


def count_report(rows, ids, analysis="index"):
    """N_i distribution (N_i is the same for every row of a patient)."""
    reg = {int(r["patient_id"]): r for r in rows if r.get("arm") == "regular_N" and r.get("analysis") == analysis
           and r.get("null") == "ou_ss"}
    N = np.array([reg[i]["N_i"] for i in ids if i in reg], float)
    if not len(N):
        return None
    cats = {"<40": N < 40, "40-49": (N >= 40) & (N < 50), "50-59": (N >= 50) & (N < 60), "60": N == 60}
    return {"mean": float(N.mean()), "sd": float(N.std(ddof=1)) if len(N) > 1 else None,
            "quantiles_5_25_50_75_95": [float(v) for v in np.percentile(N, [5, 25, 50, 75, 95])],
            "share_at_60": float(np.mean(N == 60)),
            "n_by_category": {k: int(v.sum()) for k, v in cats.items()}}


def by_count_category(rows, ids):
    reg = {int(r["patient_id"]): r["N_i"] for r in rows if r.get("arm") == "regular_N" and r.get("analysis") == "index"
           and r.get("null") == "ou_ss"}
    out = {}
    for name, f in (("<40", lambda n: n < 40), ("40-49", lambda n: 40 <= n < 50),
                    ("50-59", lambda n: 50 <= n < 60), ("60", lambda n: n == 60)):
        sub = [i for i in ids if i in reg and f(reg[i])]
        cell = {"n": len(sub)}
        for ind in ("var", "ac"):
            if sub:
                a = outcome_vector(rows, "informative_reduced_N", "index", "ou_ss", ind, sub)
                b = outcome_vector(rows, "regular_N", "index", "ou_ss", ind, sub)
                cell[ind] = {"rate_informative": float(a.mean()), "rate_regular": float(b.mean()),
                             "diff": float((a - b).mean())}
        out[name] = cell
    return out


def evaluability_paired(rows, ids, model, block, B, base=BASE):
    out = {}
    for ind in ("var", "ac"):
        ev = {}
        for arm in ("informative_reduced_N", "regular_N"):
            d = {int(r["patient_id"]): int(bool(r[f"evaluable_{ind}"])) for r in rows
                 if r.get("arm") == arm and r.get("analysis") == "index" and r.get("null") == "ou_ss"}
            ev[arm] = np.array([d[i] for i in ids], float)
        est, lo, hi = stats.paired_bootstrap(ev["informative_reduced_N"], ev["regular_N"],
                                             R.bootstrap_stream(base, model, f"{block}:evaluability"), B=B)
        both = [i for k, i in enumerate(ids) if ev["informative_reduced_N"][k] == 1 and ev["regular_N"][k] == 1]
        cond = None
        if both:
            a = outcome_vector(rows, "informative_reduced_N", "index", "ou_ss", ind, both)
            b = outcome_vector(rows, "regular_N", "index", "ou_ss", ind, both)
            cond = {"n_both_evaluable": len(both), "rate_informative": float(a.mean()), "rate_regular": float(b.mean()),
                    "diff": float((a - b).mean()), "label": "secondary, conditional on both arms evaluable"}
        out[ind] = {"evaluability_diff_informative_minus_regular": est, "ci95": [lo, hi], "conditional_detection": cond}
    return out


def elapsed_check(rows):
    idx = {(r["patient_id"], r["arm"], r["null"]): r for r in rows if r.get("analysis") == "index"}
    mism, n = 0, 0
    for r in rows:
        if r.get("analysis") != "elapsed":
            continue
        n += 1
        q = idx.get((r["patient_id"], r["arm"], r["null"]))
        same = q is not None and all((q[k] == r[k]) or (math.isnan(q[k]) and math.isnan(r[k]))
                                     for k in ("tau_var", "p_var", "tau_ac", "p_ac"))
        mism += not same
    return {"elapsed_rows": n, "mismatches_with_index": mism,
            "rule": "verification only; any mismatch is an implementation error, not a finding"}


def variability(rows, b):
    v = b.get("variability_draws")
    if not v:
        return None
    present = {int(r["patient_id"]) for r in rows if r.get("arm") == "informative_reduced_N|draw1"
               and r.get("analysis") == v["analysis"] and r.get("null") == v["null"]}
    ids = [i for i in range(v["patient_ids"][0], v["patient_ids"][1] + 1) if i in present]
    if not ids:
        return None
    out = {"n": len(ids)}
    for ind in ("var", "ac"):
        per_draw = []
        for k in range(1, int(v["draws"]) + 1):
            a = outcome_vector(rows, f"informative_reduced_N|draw{k}", v["analysis"], v["null"], ind, ids)
            per_draw.append(float(a.mean()))
        prim = outcome_vector(rows, "informative_reduced_N", v["analysis"], v["null"], ind, ids)
        out[ind] = {"primary_draw_rate": float(prim.mean()), "extra_draw_rates": per_draw,
                    "mean_extra": float(np.mean(per_draw)),
                    "sd_extra": float(np.std(per_draw, ddof=1)) if len(per_draw) > 1 else None}
    return out


def _pre_escalation(rows, bid, model, ids, B, base):
    never_flag = {int(r["patient_id"]): bool(r.get("never_escalated")) for r in rows
                  if r.get("arm") == "regular_N" and r.get("analysis") == "index" and r.get("null") == "ou_ss"}
    esc = [i for i in ids if not never_flag.get(i, False)]
    never = [i for i in ids if never_flag.get(i, False)]
    out = {"all": paired(rows, bid, model, "informative_pre", "regular_pre_matched", "index", "ou_ss", ids, B,
                         f"{bid}:informative_pre-vs-regular_pre_matched", base=base),
           "n_escalated": len(esc), "n_never_escalated": len(never)}
    if esc:
        out["escalated_only"] = paired(rows, bid, model, "informative_pre", "regular_pre_matched", "index", "ou_ss",
                                       esc, B, f"{bid}:pre-escalated-only", base=base)
    if never:
        out["never_escalated_only"] = paired(rows, bid, model, "informative_pre", "regular_pre_matched", "index", "ou_ss",
                                             never, B, f"{bid}:pre-never-escalated", base=base)
        out["never_escalated_only"]["label"] = "never escalated: full series used, flagged and reported separately"
    return out


def planned_summary_streams(blocks_present):
    """Every bootstrap stream name the summariser may use, built and checked for unique keys before any draw."""
    names = set()
    for bid in blocks_present:
        b = BLOCKS[bid]
        for an in b["analyses"]:
            if an == "elapsed":
                continue
            for nl in b["nulls"]:
                names.add(f"bootstrap|{b['model']}|{comparison_name(bid, an, nl)}")
        names.add(f"bootstrap|{b['model']}|{bid}:evaluability")
        for c in ("informative_pre-vs-regular_pre_matched", "pre-escalated-only", "pre-never-escalated"):
            names.add(f"bootstrap|{b['model']}|{bid}:{c}")
        for c in ("informative_all-vs-regular_matched", "thinned-vs-informative_reduced_N", "thinned-vs-regular_N"):
            names.add(f"bootstrap|{b['model']}|{bid}:{c}")
    names.add("bootstrap|ou_csd|P1-vs-S1:trigger-selection")
    return R.check_unique(names)


def load_completed(out_dir, label):
    """Production ('phase2'): a block counts only if its immutable record exists, the rows file matches the
    record's SHA-256, and every row carries the current config hash. Plumbing labels: rows file only."""
    done, rejected = {}, {}
    for bid in BLOCKS:
        rp = os.path.join(out_dir, f"{label}_{bid}_rows.csv")
        if not os.path.exists(rp):
            continue
        if label == "phase2":
            rec = os.path.join(out_dir, f"phase2_{bid}_record.json")
            if not os.path.exists(rec):
                rejected[bid] = "rows file without a completion record"
                continue
            with open(rec, encoding="utf-8") as f:
                r = json.load(f)
            if RUN._sha(rp) != r.get("rows_sha256"):
                rejected[bid] = "rows file does not match the SHA-256 in its record"
                continue
        rows = read_rows(rp)
        if label == "phase2" and any(x.get("config_hash") != S.CONFIG_HASH for x in rows):
            rejected[bid] = "rows produced under a different configuration"
            continue
        done[bid] = rows
    return done, rejected


def summarize(out_dir=None, B=None, base=BASE, label="phase2"):
    out_dir = out_dir or RUN.OUT
    B = B or S.N_BOOT
    done, rejected = load_completed(out_dir, label)
    planned = planned_summary_streams(done)
    R.registry_clear()
    summ = {"config_sha256": S.CONFIG_HASH, "parent_config_sha256": S.PARENT_SHA256,
            "written": dt.datetime.now().isoformat(timespec="seconds"), "bootstrap_resamples": B,
            "blocks_completed": sorted(done), "blocks_rejected": rejected,
            "blocks_missing": [b for b in BLOCKS if b not in done and b not in rejected],
            "bootstrap_streams_planned_and_checked": planned, "blocks": {}}
    for bid, rows in done.items():
        b = BLOCKS[bid]
        ids = analysed_ids(rows)
        tipped = sorted({int(r["patient_id"]) for r in rows if r.get("tipped")})
        s = {"role": b["role"], "model": b["model"], "label": b["note"],
             "n_patients": len(set(ids) | set(tipped)), "n_analysed": len(ids), "n_tipped": len(tipped),
             "rates": rates(rows)}
        if b["model"] == "fold":
            s["conditional_label"] = "fold: results are conditional on non-tipping (tipped patients counted, not analysed)"
        if b["role"] == "exploratory":
            s["exploratory_label"] = "exploratory: uncalibrated, descriptive only"
        if b["role"] == "negative_control":
            s["false_alarm_label"] = "false-alarm rates (no slowing in this model)"
        if b["role"] in ("primary", "secondary", "sensitivity", "exploratory") and ids:
            comps = []
            for an in b["analyses"]:
                if an == "elapsed":
                    continue
                for nl in b["nulls"]:
                    comps.append(paired(rows, bid, b["model"], "informative_reduced_N", "regular_N", an, nl, ids, B,
                                        comparison_name(bid, an, nl),
                                        primary=(bid == "P1" and an == "index" and nl == "ou_ss"), base=base))
            s["paired_informative_reduced_minus_regular"] = comps
        if ids:
            s["count_report"] = count_report(rows, ids, "index" if "index" in b["analyses"] else b["analyses"][0])
            if "index" in b["analyses"] and "ou_ss" in b["nulls"]:
                s["by_count_category"] = by_count_category(rows, ids)
                s["evaluability"] = evaluability_paired(rows, ids, b["model"], bid, B, base=base)
        if bid in ("P1", "S1") and ids:
            s["pre_escalation"] = _pre_escalation(rows, bid, b["model"], ids, B, base)
        if bid == "P1" and ids:
            sec = {}
            sec["informative_all_vs_regular_matched"] = paired(rows, bid, b["model"], "informative_all", "regular_matched",
                                                               "index", "ou_ss", ids, B, "P1:informative_all-vs-regular_matched", base=base)
            sec["thinned_vs_informative_reduced"] = paired(rows, bid, b["model"], "thinned", "informative_reduced_N", "index",
                                                           "ou_ss", ids, B, "P1:thinned-vs-informative_reduced_N", base=base)
            sec["thinned_vs_regular"] = paired(rows, bid, b["model"], "thinned", "regular_N", "index", "ou_ss", ids, B,
                                               "P1:thinned-vs-regular_N", base=base)
            sec["elapsed_verification"] = elapsed_check(rows)
            sec["variability_substudy"] = variability(rows, b)
            s["secondary"] = sec
        summ["blocks"][bid] = s
    if "P1" in done and "S1" in done:
        p1, s1 = done["P1"], done["S1"]
        ids = sorted(set(analysed_ids(p1)) & set(analysed_ids(s1)))
        did = {"n": len(ids), "bootstrap_stream": "bootstrap|ou_csd|P1-vs-S1:trigger-selection",
               "definition": "per patient: [det(inf_reduced)-det(regular)] in P1 minus the same in S1; same patient ids, same latent paths"}
        if ids:
            for ind in ("var", "ac"):
                dP = outcome_vector(p1, "informative_reduced_N", "index", "ou_ss", ind, ids) - outcome_vector(p1, "regular_N", "index", "ou_ss", ind, ids)
                dS = outcome_vector(s1, "informative_reduced_N", "index", "ou_ss", ind, ids) - outcome_vector(s1, "regular_N", "index", "ou_ss", ind, ids)
                est, lo, hi = stats.paired_bootstrap(dP, dS, R.bootstrap_stream(base, "ou_csd", "P1-vs-S1:trigger-selection"), B=B)
                did[ind] = {"P1_diff": float(dP.mean()), "S1_diff": float(dS.mean()), "difference_in_differences": est, "ci95": [lo, hi]}
        summ["trigger_selection_P1_vs_S1"] = did
    comps = summ["blocks"].get("P1", {}).get("paired_informative_reduced_minus_regular", [])
    prim = [c for c in comps if c["analysis"] == "index" and c["null"] == "ou_ss"]
    if prim:
        summ["PRIMARY_RESULT"] = {"comparison": "P1, index, ou_ss: detection(informative_reduced_N) - detection(regular_N)",
                                  "variance": prim[0]["var"], "autocorrelation": prim[0]["ac"]}
    used = {n for n in R.registry_snapshot() if n.startswith("bootstrap|")}
    summ["bootstrap_streams_used_not_planned"] = sorted(used - set(planned))
    path = os.path.join(out_dir, f"{label}_summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summ, f, indent=2, default=str)
    return summ, path


# ------------------------------------------------------------------ plumbing
def cmd_plumbing2(n_patients, n_surr, workers, B=200):
    """Every block on n_patients patients, seed block 9,000,000, few surrogates; then the summariser on the
    result with B bootstrap resamples. Results are discarded (files labelled plumbing2_*); checks structure only."""
    out_dir = RUN.OUT
    os.makedirs(out_dir, exist_ok=True)
    timing, problems, used, planned_all = {}, [], set(), {}
    for bid, b in BLOCKS.items():
        design = block_design(b)
        planned_all.update(_planned(b, design))
        ids = block_ids(b, n_patients)
        if b.get("variability_draws"):
            ids = sorted(set(ids) | {b["variability_draws"]["patient_ids"][0]})
        t1 = time.time()
        rows, _ = run_block_rows(b, S.PLUMBING_BASE, ids, n_surr, workers, used)
        timing[bid] = round((time.time() - t1) / max(1, len(ids)), 1)
        RUN.write_rows(rows, os.path.join(out_dir, f"plumbing2_{bid}_rows.csv"))
        for r in rows:
            if r.get("tipped"):
                continue
            if not (8 <= r["N_i"] <= 60):
                problems.append(f"{bid}: N_i out of range {r['N_i']}")
            if r["arm"] in ("regular_N", "informative_reduced_N") and r["n_obs"] != r["N_i"]:
                problems.append(f"{bid}: matched count broken")
            if r["arm"].startswith("informative_reduced_N|draw") and (r["analysis"] != "index" or r["null"] != "ou_ss"):
                problems.append(f"{bid}: variability arm ran a non-listed analysis/null")
    summ, spath = summarize(out_dir, B=B, base=S.PLUMBING_BASE, label="plumbing2")
    if summ["blocks_missing"]:
        problems.append("summariser missing blocks: " + ", ".join(summ["blocks_missing"]))
    if summ["bootstrap_streams_used_not_planned"]:
        problems.append("bootstrap streams used but not planned: " + ", ".join(summ["bootstrap_streams_used_not_planned"]))
    ev = summ["blocks"].get("P1", {}).get("secondary", {}).get("elapsed_verification", {})
    if ev.get("mismatches_with_index"):
        problems.append("elapsed does not equal index")
    unplanned = sorted(set(used) - set(planned_all))
    if unplanned:
        problems.append("unplanned streams: " + ", ".join(unplanned))
    check = {"blocks": len(BLOCKS), "seconds_per_patient_by_block": timing, "structural_problems": problems,
             "surrogates_used_in_plumbing_only": n_surr, "bootstrap_resamples_in_plumbing_only": B,
             "config_sha256": S.CONFIG_HASH, "parent_config_sha256": S.PARENT_SHA256}
    with open(os.path.join(out_dir, "plumbing2_check.json"), "w", encoding="utf-8") as f:
        json.dump(check, f, indent=2, default=str)
    print(json.dumps(check, indent=2))
    print("Phase 2 plumbing PASSED" if not problems else "Phase 2 plumbing FOUND PROBLEMS - see outputs/plumbing2_check.json")
