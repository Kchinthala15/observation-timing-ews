"""Exploratory blocks E1 / E3 of the frozen Phase 2 matrix (alternative representations), run without
touching the completed Phase 2 outputs.

What this adds (and nothing else):
  * runs a pre-specified exploratory block (E1 or E3), optionally restricted to a SUBSET of the
    analyses that block already lists in the frozen config (no new representations, no new settings);
  * writes everything to a separate folder (default: outputs_explore/), never into outputs/;
  * logs, per patient and arm, how many bins were empty before filling (missingness);
  * summarises E1 against the completed P1 index results, and E3 against the completed N2 results.

Same seeds, latent paths, schedules, arms, surrogate streams and detection rules as the frozen package:
a patient's result for a given analysis is identical whether the block is run whole or as a subset,
because every surrogate stream is keyed by (patient, arm, analysis, null, replicate).

Usage (from the rebuild_v322 folder):
  python -m ewsv322.explore run --block E1 --analyses "binned|60|linear_interp"
  python -m ewsv322.explore run --block E3 --analyses "binned|60|linear_interp"
  python -m ewsv322.explore run --block E1 --analyses all
  python -m ewsv322.explore run --block E3 --analyses all-except-ct
  python -m ewsv322.explore summarize
"""
import argparse
import datetime as dt
import hashlib
import math
import json
import os
import time

import numpy as np

from . import spec as S
from . import rng as R
from . import stats
from . import run as RUN
from . import patient
from . import phase2 as P2

EXPLORE_OUT = os.path.normpath(os.path.join(RUN.PKG, "..", "outputs_explore"))
ALLOWED_BLOCKS = ("E1", "E3")
# Completed Phase 2 comparators (read only)
COMPARATOR = {"E1": ("P1", "index"), "E3": ("N2", "index")}


# ------------------------------------------------------------------ run
def resolve_analyses(block_id, spec_arg):
    listed = list(P2.BLOCKS[block_id]["analyses"])
    if spec_arg == "all":
        return listed
    if spec_arg == "all-except-ct":
        return [a for a in listed if not a.startswith("ct|")]
    wanted = [a.strip() for a in spec_arg.split(",") if a.strip()]
    bad = [a for a in wanted if a not in listed]
    if bad:
        raise SystemExit(f"{block_id} does not list these analyses in the frozen config: {bad}. "
                         f"Listed: {listed}")
    return [a for a in listed if a in wanted]          # keep the frozen order


def tag_for(block_id, analyses):
    """Short file tag (Windows 260-character path limit). The full analysis list is kept in the record."""
    listed = list(P2.BLOCKS[block_id]["analyses"])
    if analyses == listed:
        return f"{block_id}__all"
    if analyses == [a for a in listed if not a.startswith("ct|")]:
        return f"{block_id}__noct"
    if len(analyses) == 1:
        return f"{block_id}__" + analyses[0].replace("|", "-")
    return f"{block_id}__sub{len(analyses)}_" + hashlib.sha256("\n".join(analyses).encode()).hexdigest()[:10]


def _bin_occupancy(t, nb):
    idx = np.clip(np.floor(np.asarray(t, float) / (S.T / nb)).astype(int), 0, nb - 1)
    cnt = np.bincount(idx, minlength=nb)
    return int(np.sum(cnt == 0))


def _work(job):
    """Same as run._work, plus a per-arm missingness log for the binned and interp analyses."""
    model, seed, pid, design, n_surr = job
    R.registry_clear()
    rows, dbg = patient.run(model, seed, design, n_surr)
    arms_ts = dbg["arms"] if dbg else {}
    for r in rows:
        r["patient_id"] = pid
        an = str(r.get("analysis", ""))
        ser = arms_ts.get(r.get("arm"))
        if ser is None:
            continue
        t = np.asarray(ser["t"], float)
        if an.startswith("binned|"):
            nb = int(an.split("|")[1])
            r["bins_total"] = nb
            r["bins_empty_before_fill"] = _bin_occupancy(t, nb)
        elif an.startswith("interp|") and len(t):
            grid = np.linspace(0.0, S.T, int(an.split("|")[1]))
            r["grid_points_extrapolated"] = int(np.sum((grid < t.min()) | (grid > t.max())))
    return rows, list(R.registry_snapshot().keys())


def cmd_run(block_id, analyses_arg, workers, out_dir):
    from multiprocessing import Pool
    if block_id not in ALLOWED_BLOCKS:
        raise SystemExit(f"This tool runs only the exploratory blocks {ALLOWED_BLOCKS}.")
    P2.check_gate(RUN.OUT)                     # Phase 1b record in the original outputs folder
    b = P2.BLOCKS[block_id]
    analyses = resolve_analyses(block_id, analyses_arg)
    os.makedirs(out_dir, exist_ok=True)
    tag = tag_for(block_id, analyses)
    rec_path = os.path.join(out_dir, f"explore_{tag}_record.json")
    if os.path.exists(rec_path):
        print(f"{tag}: already completed (record exists) - skipped.")
        return
    rows_path = os.path.join(out_dir, f"explore_{tag}_rows.csv")
    tmp = rows_path + ".tmp"
    open(tmp, "w").close()                     # pre-flight: a bad or over-long path fails now, not after the run
    os.remove(tmp)
    b_run = dict(b, analyses=analyses)
    design = P2.block_design(b_run)
    planned = P2._planned(b_run, design)
    ids = P2.block_ids(b)
    t0 = dt.datetime.now().isoformat(timespec="seconds"); t1 = time.time()
    jobs = P2._jobs(b_run, design, ids, P2.BASE, S.N_SURR)
    rows, used = [], set()
    with Pool(workers) as pool:
        for k, (r, names) in enumerate(pool.imap(_work, jobs, chunksize=2), 1):
            for x in r:
                x["block"] = block_id; x["phase"] = "explore"
            rows.extend(r); used.update(names)
            if k % 50 == 0 or k == len(jobs):
                print(f"  {tag}: {k}/{len(jobs)} patients", flush=True)
    RUN.write_rows(rows, tmp)
    RUN._abort_if_unplanned(planned, used)
    os.replace(tmp, rows_path)
    record = {"phase": "explore", "block": block_id, "role": b["role"], "tag": tag,
              "analyses_run": analyses, "analyses_listed_in_frozen_block": list(b["analyses"]),
              "subset_of_frozen_block": analyses != list(b["analyses"]),
              "written": dt.datetime.now().isoformat(timespec="seconds"),
              "config_sha256": S.CONFIG_HASH, "parent_config_sha256": S.PARENT_SHA256,
              "n_patients": len(ids), "rows": len(rows),
              "minutes": round((time.time() - t1) / 60, 1), "rows_sha256": RUN._sha(rows_path),
              "status_note": "Exploratory, uncalibrated representation (phase2_matrix.exploratory): descriptive only; "
                             "never used to support or replace the primary result."}
    man = RUN.manifest(f"explore_{tag}", P2.BASE, {"block": b, "analyses_run": analyses, "record": record},
                       planned, used, [rows_path], t0)
    with open(os.path.join(out_dir, f"explore_{tag}_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(man, f, indent=2, default=str)
    RUN.write_immutable(record, rec_path)
    print(f"{tag}: done in {record['minutes']} min, {len(rows)} rows. Output folder: {os.path.abspath(out_dir)}")


# ------------------------------------------------------------------ summarise
def _load_completed_block(block_id):
    rec_p = os.path.join(RUN.OUT, f"phase2_{block_id}_record.json")
    rows_p = os.path.join(RUN.OUT, f"phase2_{block_id}_rows.csv")
    with open(rec_p, encoding="utf-8") as f:
        rec = json.load(f)
    if RUN._sha(rows_p) != rec["rows_sha256"]:
        raise SystemExit(f"{rows_p} does not match its completion record; refusing to compare.")
    if rec.get("config_sha256") != S.CONFIG_HASH:
        raise SystemExit(f"{rec_p} was written under a different configuration; refusing to compare.")
    rows = P2.read_rows(rows_p)
    if any(r.get("config_hash") != S.CONFIG_HASH for r in rows):
        raise SystemExit(f"{rows_p} contains rows from a different configuration; refusing to compare.")
    return rows


def _load_explore(out_dir):
    rows, recs = [], []
    for fn in sorted(os.listdir(out_dir)):
        if fn.startswith("explore_") and fn.endswith("_record.json"):
            with open(os.path.join(out_dir, fn), encoding="utf-8") as f:
                rec = json.load(f)
            rp = os.path.join(out_dir, f"explore_{rec['tag']}_rows.csv")
            if rec.get("config_sha256") != S.CONFIG_HASH or RUN._sha(rp) != rec["rows_sha256"]:
                raise SystemExit(f"{rp}: hash does not match its record.")
            recs.append(rec)
            rows.extend(P2.read_rows(rp))
    # Overlapping runs (e.g. one analysis run alone, then the whole block) give identical rows for the same
    # patient/arm/analysis/null. Keep one copy; stop if two copies ever disagree.
    keep, seen = [], {}
    fields = ("tau_var", "p_var", "tau_ac", "p_ac", "det_var", "det_ac")
    def same(a, b):
        return a == b or (isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b))
    for r in rows:
        if r.get("arm") in (None, ""):
            keep.append(r); continue
        k = (r.get("block"), int(r["patient_id"]), r.get("arm"), r.get("analysis"), r.get("null"))
        if k in seen:
            if not all(same(seen[k].get(f), r.get(f)) for f in fields):
                raise SystemExit(f"Two runs disagree for {k}; the outputs are not reproducible - stop and report.")
            continue
        seen[k] = r; keep.append(r)
    return keep, recs


def _vec(rows, arm, analysis, ind, ids):
    d = {int(r["patient_id"]): int(bool(r[f"det_{ind}"])) for r in rows
         if r.get("arm") == arm and r.get("analysis") == analysis and r.get("null") == "ou_ss"}
    missing = [i for i in ids if i not in d]
    if missing:
        raise SystemExit(f"missing patients for {arm}/{analysis}: {missing[:5]}...")
    return np.array([d[i] for i in ids], float)


def _paired(a, b, model, name, B):
    est, lo, hi = stats.paired_bootstrap(a, b, R.bootstrap_stream(P2.BASE, model, name), B=B)
    e = int(np.sum((a == 1) & (b == 1))); f = int(np.sum((a == 1) & (b == 0)))
    g = int(np.sum((a == 0) & (b == 1))); h = int(np.sum((a == 0) & (b == 0)))
    _, nl, nu = stats.newcombe_paired(e, f, g, h)
    return {"rate_A": float(a.mean()), "rate_B": float(b.mean()), "diff": est, "ci95_bootstrap": [lo, hi],
            "newcombe_crosscheck": [nl, nu], "cells_e_f_g_h": [e, f, g, h], "bootstrap_stream": f"bootstrap|{model}|{name}"}


def _did(a1, b1, a2, b2, model, name, B):
    """Patient bootstrap of (mean(a1-b1) - mean(a2-b2)); same resampled ids for both terms."""
    d = (a1 - b1) - (a2 - b2)
    est, lo, hi = stats.paired_bootstrap(d, np.zeros_like(d), R.bootstrap_stream(P2.BASE, model, name), B=B)
    return {"diff_in_diff": est, "ci95_bootstrap": [lo, hi], "bootstrap_stream": f"bootstrap|{model}|{name}"}


def _cell(rows, arm, analysis, ids):
    idset = set(ids)
    sel = [r for r in rows if r.get("arm") == arm and r.get("analysis") == analysis and r.get("null") == "ou_ss"
           and int(r["patient_id"]) in idset]
    n = len(sel); out = {"n": n}
    for ind in ("var", "ac"):
        x = sum(bool(r[f"det_{ind}"]) for r in sel); ev = sum(bool(r[f"evaluable_{ind}"]) for r in sel)
        vw = [float(r[f"n_valid_windows_{ind}"]) for r in sel if r.get(f"n_valid_windows_{ind}") not in (None, "")]
        out[ind] = {"rate": x / n if n else None, "x": x,
                    "cp95": [round(v, 4) for v in stats.clopper_pearson(x, n)] if n else None,
                    "cp90": [round(v, 4) for v in stats.clopper_pearson(x, n, 0.90)] if n else None,
                    "evaluable": ev / n if n else None, "not_evaluable_n": n - ev,
                    "valid_windows_median": float(np.median(vw)) if vw else None}
    be = [float(r["bins_empty_before_fill"]) / float(r["bins_total"]) for r in sel
          if r.get("bins_empty_before_fill") not in (None, "")]
    if be:
        out["empty_bin_fraction_before_fill"] = {"mean": float(np.mean(be)), "median": float(np.median(be)),
                                                 "p90": float(np.percentile(be, 90)), "max": float(np.max(be))}
    ge = [float(r["grid_points_extrapolated"]) for r in sel if r.get("grid_points_extrapolated") not in (None, "")]
    if ge:
        out["interp_grid_points_extrapolated_median"] = float(np.median(ge))
    return out


def cmd_summarize(out_dir, B=S.N_BOOT):
    rows, recs = _load_explore(out_dir)
    if not recs:
        raise SystemExit(f"No completed exploratory runs in {out_dir}.")
    summ = {"written": dt.datetime.now().isoformat(timespec="seconds"), "config_sha256": S.CONFIG_HASH,
            "status": "EXPLORATORY, UNCALIBRATED, DESCRIPTIVE ONLY (phase2_matrix.exploratory). Not a replacement "
                      "for the primary P1 result.",
            "runs": [{k: r[k] for k in ("tag", "analyses_run", "subset_of_frozen_block", "n_patients", "rows",
                                        "minutes", "rows_sha256")} for r in recs],
            "E1_detection": {}, "E3_false_alarms": {}}
    flat = []
    for block_id in ("E1", "E3"):
        b_rows = [r for r in rows if r.get("block") == block_id]
        if not b_rows:
            continue
        comp_block, comp_an = COMPARATOR[block_id]
        comp = _load_completed_block(comp_block)
        model = P2.BLOCKS[block_id]["model"]
        tipped = {int(r["patient_id"]) for r in b_rows if r.get("tipped")}
        ids = [i for i in P2.block_ids(P2.BLOCKS[block_id]) if i not in tipped]
        analyses = [a for a in P2.BLOCKS[block_id]["analyses"] if any(r.get("analysis") == a for r in b_rows)]
        for an in analyses:
            res = {"regular_N": _cell(b_rows, "regular_N", an, ids),
                   "informative_reduced_N": _cell(b_rows, "informative_reduced_N", an, ids),
                   f"{comp_block}_index_regular_N": _cell(comp, "regular_N", comp_an, ids),
                   f"{comp_block}_index_informative_reduced_N": _cell(comp, "informative_reduced_N", comp_an, ids)}
            contrasts, dids = {}, {}
            for ind in ("var", "ac"):
                inf_a = _vec(b_rows, "informative_reduced_N", an, ind, ids)
                reg_a = _vec(b_rows, "regular_N", an, ind, ids)
                inf_i = _vec(comp, "informative_reduced_N", comp_an, ind, ids)
                reg_i = _vec(comp, "regular_N", comp_an, ind, ids)
                pre = f"{block_id}:{an}:{ind}"
                contrasts[ind] = {
                    "informative_minus_regular_same_representation":
                        _paired(inf_a, reg_a, model, P2.comparison_name(block_id, an, "ou_ss"), B),
                    "informative_representation_minus_informative_index":
                        _paired(inf_a, inf_i, model, f"{pre}:inf-vs-inf_index", B),
                    "regular_representation_minus_regular_index":
                        _paired(reg_a, reg_i, model, f"{pre}:reg-vs-reg_index", B),
                    "informative_representation_minus_regular_index":
                        _paired(inf_a, reg_i, model, f"{pre}:inf-vs-reg_index", B)}
                dids[ind] = _did(inf_a, reg_a, inf_i, reg_i, model, f"{pre}:did-vs-index", B)
            res["paired_contrasts"] = contrasts
            res["change_in_informative_minus_regular_vs_index"] = dids
            if block_id == "E3":
                for arm in ("regular_N", "informative_reduced_N"):
                    for ind in ("var", "ac"):
                        c = res[arm][ind]
                        res[arm][ind]["tost_2.5_7.5_descriptive"] = stats.tost(c["x"], res[arm]["n"])["pass"]
                summ["E3_false_alarms"][an] = res
            else:
                summ["E1_detection"][an] = res
            for ind in ("var", "ac"):
                ct = contrasts[ind]
                flat.append({"block": block_id, "analysis": an, "indicator": ind,
                             "regular_rate": res["regular_N"][ind]["rate"],
                             "informative_rate": res["informative_reduced_N"][ind]["rate"],
                             "inf_minus_reg": ct["informative_minus_regular_same_representation"]["diff"],
                             "inf_minus_reg_ci": ct["informative_minus_regular_same_representation"]["ci95_bootstrap"],
                             "inf_minus_inf_index": ct["informative_representation_minus_informative_index"]["diff"],
                             "inf_minus_inf_index_ci": ct["informative_representation_minus_informative_index"]["ci95_bootstrap"],
                             "did_vs_index": dids[ind]["diff_in_diff"], "did_vs_index_ci": dids[ind]["ci95_bootstrap"],
                             "regular_evaluable": res["regular_N"][ind]["evaluable"],
                             "informative_evaluable": res["informative_reduced_N"][ind]["evaluable"]})
    import platform, scipy
    summ["software"] = {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                        "platform": platform.platform()}
    summ["code_sha256"] = {f: RUN._sha(os.path.join(RUN.PKG, f)) for f in sorted(os.listdir(RUN.PKG)) if f.endswith(".py")}
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    jp = os.path.join(out_dir, f"explore_summary_{stamp}.json")
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(summ, f, indent=2, default=str)
    cp = os.path.join(out_dir, f"explore_summary_table_{stamp}.csv")
    RUN.write_rows(flat, cp)
    print("Summary written to", jp)
    print("Table written to", cp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["run", "summarize"])
    ap.add_argument("--block", choices=list(ALLOWED_BLOCKS))
    ap.add_argument("--analyses", default="binned|60|linear_interp",
                    help="comma list of analyses already listed for the block, or 'all', or 'all-except-ct'")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out", default=EXPLORE_OUT)
    a = ap.parse_args()
    if a.command == "run":
        if not a.block:
            raise SystemExit("--block E1 or --block E3 is required")
        cmd_run(a.block, a.analyses, a.workers, a.out)
    else:
        cmd_summarize(a.out)


if __name__ == "__main__":
    main()
