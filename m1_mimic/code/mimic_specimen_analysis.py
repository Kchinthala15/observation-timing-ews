#!/usr/bin/env python3
"""
mimic_specimen_analysis.py  (MIMIC-IV stage, pre-specified plan M1 v1.2; script v1.2.2)

Question: before a genuine ward-to-ICU escalation, does the number of laboratory
DRAWS (distinct specimens, MIMIC-IV specimen_id) per hour rise, compared with
matched ward patients who were not escalated?

Runs locally on the full MIMIC-IV. Writes AGGREGATES ONLY (no patient rows, no
identifiers, no dates). Every patient count below 10 is suppressed in the output
unless --demo is given (the open 100-patient demo, which has no such restriction).

Usage (Windows PowerShell):
    python mimic_specimen_analysis.py --hosp "C:\\path\\to\\hosp" --out M1_results
Demo test:
    python mimic_specimen_analysis.py --hosp "...\\mimic-iv-clinical-database-demo-2.2\\hosp" --out demo_out --demo
Self-test of the estimator (no data needed):
    python mimic_specimen_analysis.py --selftest

Design (fixed before running on the full data):
  Cases     first ICU entry in an admission whose preceding unit is a ward, with the
            48 h before ICU entry spent entirely on ward units (no ED, procedural
            area, ICU or unknown unit in that 48 h). One admission per patient
            (the earliest qualifying one).
  Controls  risk-set matching on clock hour with an elapsed-time caliper. Cases are
            taken in seeded random order; for each, candidate ward admissions with
            no ICU entry (patients never cases) are examined in seeded random order.
            The pseudo-event is the time with the case's exact clock hour and minute
            nearest to (candidate admission + case hours-since-admission), so the
            elapsed-time difference is at most +/-12 h by construction. Accepted
            only if the 48 h before it lie within one continuous ward run. Up to 2
            controls per case, without replacement (each control patient used once).
            Selection uses transfer records only, never laboratory data. Cases with
            no eligible control are counted and reported.
            v1.2.1 fix: a ward run is ended by every non-ward segment (ward_runs), so
            the control rule is identical to the case rule (all_ward); estimand,
            windows, seed, bootstrap count and outcomes are unchanged.
            v1.2.2: the control search is exhaustive (every candidate admission is
            checked for every case; the 5,000-candidate cap is removed), and unmatched
            cases are split into "no eligible control" and "eligible controls already
            assigned". Undefined bootstrap ratio replicates are dropped explicitly and
            the valid replicate counts are reported.
  Windows   baseline = 48 to 24 h before the event; pre = the 24 h before it.
  Outcomes  primary: draws per hour (distinct specimen_id).
            secondary: distinct collection times per hour; tests per hour; tests per
            draw; the old 60-second grouping and its agreement with specimen_id.
  Estimand  equal-set-weighted difference-in-differences. For each matched set i:
            contrast_i = (case pre - case baseline) - mean over the set's controls
            of (control pre - control baseline). Primary estimate = mean of
            contrast_i over matched sets (each set weighted equally, whether it has
            1 or 2 controls). Control means and ratios are likewise means of
            set-level control means. 95% interval: bootstrap of matched sets
            (2,000 resamples, seed 20261101), recomputing the same estimand.
  Sensitivity  12 h windows (12-24 h vs 0-12 h before the event).
  Also      48-hour profile of draws, draws by clock hour, ED-origin escalations
            descriptively, full matching diagnostics.
"""
import argparse, json, os, sys, hashlib, datetime as dt
import numpy as np
import pandas as pd

PLAN = "M1 v1.2.2"
SEED = 20261101
NBOOT = 2000
SUPPRESS = 10


def classify(unit):
    if not isinstance(unit, str) or not unit.strip():
        return "unknown"
    u = unit.lower()
    if "icu" in u or "intensive care" in u or "cardiac vascular" in u or "coronary care" in u:
        return "icu"
    if "emergency" in u:
        return "emergency"
    if any(k in u for k in ("pacu", "post-anesthesia", "operating", "procedure", "cath lab", "endoscopy", "radiology")):
        return "procedural"
    if any(k in u for k in ("medicine", "surgery", "surgical", "med/surg", "medical", "oncology", "obstetrics", "psychiatry",
                            "neurology", "hematology", "transplant", "vascular", "cardiology", "observation", "stepdown",
                            "step down", "intermediate", "trauma", "gyn")):
        return "ward"
    return "other"


def build_segments(tr):
    tr = tr.dropna(subset=["hadm_id", "intime"]).copy()
    tr["intime"] = pd.to_datetime(tr["intime"]); tr["outtime"] = pd.to_datetime(tr["outtime"])
    tr = tr[tr["eventtype"].fillna("") != "discharge"]
    tr["cls"] = tr["careunit"].map(classify)
    tr = tr.sort_values(["hadm_id", "intime"])
    return tr


def all_ward(seg, start, end):
    """True if [start, end) is fully covered by ward segments."""
    s = seg[(seg["outtime"] > start) & (seg["intime"] < end)]
    if s.empty or s["intime"].min() > start or s["outtime"].max() < end:
        return False
    if (s["cls"] != "ward").any():
        return False
    s = s.sort_values("intime"); cur = start
    for a, b in zip(s["intime"], s["outtime"]):
        if a > cur + pd.Timedelta(minutes=1):
            return False
        cur = max(cur, b)
    return cur >= end


def find_cases(tr):
    cases, ed_cases = [], []
    for i, (hadm, seg) in enumerate(tr.groupby("hadm_id", sort=False)):
        if i and i % 100000 == 0:
            print(f"    cases: {i:,} admissions checked", flush=True)
        icu = seg[seg["cls"] == "icu"]
        if icu.empty:
            continue
        t_icu = icu["intime"].iloc[0]
        before = seg[seg["intime"] < t_icu]
        if before.empty:
            continue
        prev_cls = before["cls"].iloc[-1]
        admit = seg["intime"].iloc[0]
        rec = dict(hadm_id=hadm, subject_id=seg["subject_id"].iloc[0], event=t_icu,
                   hours_since_admit=(t_icu - admit).total_seconds() / 3600)
        if prev_cls == "ward" and all_ward(seg, t_icu - pd.Timedelta(hours=48), t_icu):
            cases.append(rec)
        elif prev_cls == "emergency" and (t_icu - admit) >= pd.Timedelta(hours=48):
            ed_cases.append(rec)
    c = pd.DataFrame(cases)
    if not c.empty:
        c = c.sort_values("event").groupby("subject_id", as_index=False).first()
    e = pd.DataFrame(ed_cases)
    if not e.empty:
        e = e.sort_values("event").groupby("subject_id", as_index=False).first()
    return c, e


K_CONTROLS = 2          # up to 2 controls per case
TOL_H = 12.0            # elapsed-time tolerance; the smallest that always allows an exact clock-hour match


def ward_runs(seg):
    """Maximal periods of continuous ward occupancy in one admission: list of [start, end].
    v1.2.1: ward segments are merged (gaps of at most 1 minute, as in all_ward), and then
    EVERY non-ward segment (ED, ICU, procedural, other, unknown; including zero-length or
    overlapping records) terminates the run it touches: its interval is cut out, so a run
    can never span a non-ward segment. A 48 h window lies inside one of these runs exactly
    when all_ward() accepts it, i.e. the same rule that defines cases."""
    seg = seg[seg["outtime"].notna()].sort_values("intime")
    runs = []
    for a, b in zip(seg.loc[seg["cls"] == "ward", "intime"], seg.loc[seg["cls"] == "ward", "outtime"]):
        if runs and a <= runs[-1][1] + pd.Timedelta(minutes=1):
            runs[-1][1] = max(runs[-1][1], b)
        else:
            runs.append([a, b])
    for a, b in zip(seg.loc[seg["cls"] != "ward", "intime"], seg.loc[seg["cls"] != "ward", "outtime"]):
        cut = []
        for lo, hi in runs:
            if a < hi and b > lo:              # non-ward segment overlaps the run: split it
                if a > lo:
                    cut.append([lo, a])
                if b < hi:
                    cut.append([b, hi])
            elif a == b and lo < a < hi:       # zero-length non-ward record inside the run
                cut += [[lo, a], [a, hi]]
            else:
                cut.append([lo, hi])
        runs = cut
    return runs


DAY_NS = 86_400 * 10**9
TOL_NS = int(TOL_H * 3600 * 10**9)


def pseudo_event_ns(admit_ns, h, clock_s):
    """Vectorised pseudo-event times (int64 ns) for candidate admissions: the time with the
    case's exact clock time nearest to (admission + h hours), within +/-12 h. Same rule as
    pseudo_event_scalar(), the original v1.2 per-candidate code; the only difference is that
    the original kept a sub-microsecond floating-point residue, which is dropped here."""
    target = admit_ns + pd.Timedelta(hours=h).value
    t = target - (target % DAY_NS) + clock_s * 10**9
    t = np.where(t - target > TOL_NS, t - DAY_NS, t)
    t = np.where(target - t > TOL_NS, t + DAY_NS, t)
    return t


def pseudo_event_scalar(admit, h, ev):
    target = admit + pd.Timedelta(hours=h)
    t = target.replace(hour=ev.hour, minute=ev.minute, second=ev.second, microsecond=0)
    if t - target > pd.Timedelta(hours=TOL_H):
        t -= pd.Timedelta(days=1)
    elif target - t > pd.Timedelta(hours=TOL_H):
        t += pd.Timedelta(days=1)
    return t


def build_candidates(tr, cases):
    """Candidate control admissions: no ICU entry, patient never a case, at least one
    continuous ward run of >= 48 h. Flattened arrays so every case can be checked against
    the whole pool at once."""
    case_subj = set(cases["subject_id"]) if not cases.empty else set()
    icu_hadm = set(tr.loc[tr["cls"] == "icu", "hadm_id"])
    hadm, subj, admit, rc, rlo, rhi = [], [], [], [], [], []
    for i, (h, seg) in enumerate(tr.groupby("hadm_id", sort=False)):
        if i and i % 100000 == 0:
            print(f"    controls: {i:,} admissions indexed", flush=True)
        sj = seg["subject_id"].iloc[0]
        if h in icu_hadm or sj in case_subj:
            continue
        runs = [(a + pd.Timedelta(hours=48), b) for a, b in ward_runs(seg) if b - a >= pd.Timedelta(hours=48)]
        if not runs:
            continue
        k = len(hadm)
        hadm.append(h); subj.append(sj); admit.append(seg["intime"].iloc[0])
        for lo, hi in runs:
            rc.append(k); rlo.append(lo.value); rhi.append(hi.value)
    return {"hadm": np.array(hadm, dtype=object), "subj": np.array(subj, dtype=object),
            "admit": np.array([pd.Timestamp(x).value for x in admit], dtype=np.int64),
            "run_cand": np.array(rc, dtype=np.int64), "run_lo": np.array(rlo, dtype=np.int64), "run_hi": np.array(rhi, dtype=np.int64)}


def eligible_for_case(C, h, ev):
    """Indices of ALL candidate admissions eligible for this case (exhaustive), with their
    pseudo-event times. Eligible = pseudo-event lies inside a 48 h-complete ward run."""
    clock_s = ev.hour * 3600 + ev.minute * 60 + ev.second
    t = pseudo_event_ns(C["admit"], h, clock_s)
    tr_ = t[C["run_cand"]]
    ok = (C["run_lo"] <= tr_) & (tr_ <= C["run_hi"])
    idx = np.unique(C["run_cand"][ok])
    return idx, t


def find_controls(tr, cases, rng):
    """Risk-set matching with an EXHAUSTIVE search (v1.2.2; replaces the 5,000-candidate cap).
    Cases are processed in seeded random order. For each case, every candidate admission is
    checked; among those eligible whose patient has not yet been used, up to 2 are taken in
    seeded random order (distinct patients). Each control patient is used once. Matching
    criteria are unchanged: exact clock hour and minute, elapsed caliper +/-12 h by
    construction, 48 h continuous ward before the pseudo-event, transfers only."""
    C = build_candidates(tr, cases)
    print(f"    candidate control admissions with >=48 h continuous ward time: {len(C['hadm']):,}", flush=True)
    subj_codes, subj_index = pd.factorize(pd.Series(C["subj"]))
    used = np.zeros(len(subj_index), dtype=bool)
    out, per_case = [], []
    order = rng.permutation(len(cases))
    for n, ci in enumerate(order):
        if n and n % 2000 == 0:
            print(f"    matched {n:,} of {len(cases):,} cases", flush=True)
        case = cases.iloc[ci]
        h, ev = float(case["hours_since_admit"]), case["event"]
        idx, t = eligible_for_case(C, h, ev)
        avail = idx[~used[subj_codes[idx]]]
        n_avail_patients = int(len(np.unique(subj_codes[avail])))
        got = 0
        for k in rng.permutation(avail):
            if used[subj_codes[k]]:
                continue
            used[subj_codes[k]] = True
            te = pd.Timestamp(int(t[k])); admit = pd.Timestamp(int(C["admit"][k]))
            out.append(dict(hadm_id=C["hadm"][k], subject_id=C["subj"][k], event=te, case_hadm_id=case["hadm_id"],
                            hours_since_admit=(te - admit).total_seconds() / 3600, case_hours_since_admit=h,
                            case_clock=(ev.hour, ev.minute)))
            got += 1
            if got == K_CONTROLS:
                break
        per_case.append(dict(n_eligible_admissions=int(len(idx)), n_eligible_patients=int(len(np.unique(subj_codes[idx]))),
                             n_available_patients=n_avail_patients, n_selected=got))
    pc = pd.DataFrame(per_case)
    q = lambda s: [float(x) for x in s.quantile([0, .01, .05, .10, .25, .50]).to_numpy()] if len(s) else []
    mdiag = {"search": "exhaustive over all candidate admissions (no cap)",
             "candidate_control_admissions": int(len(C["hadm"])),
             "eligible_patients_per_case_quantiles_0_1_5_10_25_50": q(pc["n_eligible_patients"]) if len(pc) else [],
             "available_patients_at_turn_quantiles_0_1_5_10_25_50": q(pc["n_available_patients"]) if len(pc) else [],
             "cases_no_eligible_control": int((pc["n_eligible_patients"] == 0).sum()) if len(pc) else 0,
             "cases_eligible_but_all_already_assigned": int(((pc["n_eligible_patients"] > 0) & (pc["n_available_patients"] == 0)).sum()) if len(pc) else 0,
             "cases_one_control_only_one_eligible": int(((pc["n_selected"] == 1) & (pc["n_eligible_patients"] == 1)).sum()) if len(pc) else 0,
             "cases_one_control_others_already_assigned": int(((pc["n_selected"] == 1) & (pc["n_eligible_patients"] > 1)).sum()) if len(pc) else 0,
             "coverage_share_cases_matched": float((pc["n_selected"] > 0).mean()) if len(pc) else None,
             "coverage_share_cases_two_controls": float((pc["n_selected"] == 2).mean()) if len(pc) else None}
    return pd.DataFrame(out), mdiag


def matching_diagnostics(tr, cases, controls):
    segs = {h: g for h, g in tr.groupby("hadm_id", sort=False)}
    d = controls["hours_since_admit"] - controls["case_hours_since_admit"]
    clock_ok = [(e.hour, e.minute) == c for e, c in zip(controls["event"], controls["case_clock"])]
    ward_ok_ctrl = [all_ward(segs[h], e - pd.Timedelta(hours=48), e) for h, e in zip(controls["hadm_id"], controls["event"])]
    ward_ok_case = [all_ward(segs[h], e - pd.Timedelta(hours=48), e) for h, e in zip(cases["hadm_id"], cases["event"])]
    per_case = controls.groupby("case_hadm_id").size()
    return {"elapsed_diff_hours_control_minus_case": {"mean": float(d.mean()), "median": float(d.median()),
                                                      "max_abs": float(d.abs().max()), "min": float(d.min()), "max": float(d.max()),
                                                      "abs_quantiles_10_25_50_75_90": [float(x) for x in d.abs().quantile([.1, .25, .5, .75, .9])],
                                                      "share_within_12h": float((d.abs() <= TOL_H + 1e-9).mean())},
            "case_hours_since_admit_median": float(cases["hours_since_admit"].median()),
            "control_hours_since_admit_median": float(controls["hours_since_admit"].median()),
            "clock_hour_and_minute_exact_match_share": float(np.mean(clock_ok)),
            "controls_48h_continuous_ward_share": float(np.mean(ward_ok_ctrl)),
            "cases_48h_continuous_ward_share": float(np.mean(ward_ok_case)),
            "one_admission_per_control_patient": bool(controls["subject_id"].is_unique),
            "one_admission_per_case_patient": bool(cases["subject_id"].is_unique),
            "no_patient_both_case_and_control": bool(not (set(cases["subject_id"]) & set(controls["subject_id"]))),
            "cases_with_2_controls": int((per_case == 2).sum()), "cases_with_1_control": int((per_case == 1).sum()),
            "cases_with_no_control": int(len(cases) - len(per_case))}


def load_labs(path, hadm_ids):
    keep = []
    for ch in pd.read_csv(path, usecols=["hadm_id", "specimen_id", "charttime"], chunksize=5_000_000):
        ch = ch[ch["hadm_id"].isin(hadm_ids)]
        if len(ch):
            keep.append(ch)
    labs = pd.concat(keep) if keep else pd.DataFrame(columns=["hadm_id", "specimen_id", "charttime"])
    labs["charttime"] = pd.to_datetime(labs["charttime"])
    return labs


def per_stay(stays, labs, w_hours):
    """Counts per stay in baseline [-2w,-w) and pre [-w,0) windows, plus 48 hourly bins."""
    lab = labs.merge(stays[["hadm_id", "event"]], on="hadm_id")
    lab["h"] = (lab["event"] - lab["charttime"]).dt.total_seconds() / 3600
    lab = lab[(lab["h"] > 0) & (lab["h"] <= 2 * w_hours)]
    lab["win"] = np.where(lab["h"] <= w_hours, "pre", "base")
    keep = ["hadm_id"] + (["case_hadm_id"] if "case_hadm_id" in stays.columns else [])
    res = stays[keep].copy()
    for win in ("base", "pre"):
        sub = lab[lab["win"] == win]
        tests = sub.groupby("hadm_id").size()
        draws = sub.groupby("hadm_id")["specimen_id"].nunique()
        times = sub.groupby("hadm_id")["charttime"].nunique()
        # old definition: timestamps more than 60 s apart start a new draw
        old = sub.sort_values(["hadm_id", "charttime"]).groupby("hadm_id")["charttime"].apply(
            lambda s: int((s.diff().dt.total_seconds().fillna(1e9) > 60).sum()))
        res[f"tests_{win}"] = res["hadm_id"].map(tests).fillna(0) / w_hours
        res[f"draws_{win}"] = res["hadm_id"].map(draws).fillna(0) / w_hours
        res[f"old_{win}"] = res["hadm_id"].map(old).fillna(0) / w_hours
        res[f"times_{win}"] = res["hadm_id"].map(times).fillna(0) / w_hours
    return res, lab


def set_level(cs, ct, k):
    """Per matched set: case base/pre and the mean control base/pre (equal weight per set)."""
    pos = {h: i for i, h in enumerate(cs["hadm_id"])}
    grp = ct["case_hadm_id"].map(pos).to_numpy()
    cnt = np.bincount(grp, minlength=len(cs)).astype(float)
    kb = np.bincount(grp, weights=ct[f"{k}_base"].to_numpy(float), minlength=len(cs)) / cnt
    kp = np.bincount(grp, weights=ct[f"{k}_pre"].to_numpy(float), minlength=len(cs)) / cnt
    return cs[f"{k}_base"].to_numpy(float), cs[f"{k}_pre"].to_numpy(float), kb, kp


def estimand(cb, cp, kb, kp, idx=None):
    """Equal-set-weighted estimates for sets idx (rows of a bootstrap index matrix, or all sets)."""
    if idx is None:
        idx = np.arange(len(cb))[None, :]
    c = (cp[idx] - cb[idx]) - (kp[idx] - kb[idx])
    mcb, mcp, mkb, mkp = cb[idx].mean(1), cp[idx].mean(1), kb[idx].mean(1), kp[idx].mean(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return {"did": c.mean(1), "case_ratio": mcp / mcb, "control_ratio": mkp / mkb, "ror": (mcp / mcb) / (mkp / mkb),
                "mcb": mcb, "mcp": mcp, "mkb": mkb, "mkp": mkp}


def summarise(cs, ct, rng, label):
    """Matched sets only (cases with >=1 control). Every estimate is equal-set-weighted."""
    cs = cs[cs["hadm_id"].isin(set(ct["case_hadm_id"]))].reset_index(drop=True) if len(ct) else cs.iloc[0:0]
    out = {"window": label, "n_matched_sets": int(len(cs)), "n_controls": int(len(ct))}
    if len(cs) == 0:
        return out
    ib = rng.integers(len(cs), size=(NBOOT, len(cs))) if len(cs) > 1 else None
    def pct(x):
        x = np.asarray(x, float); x = x[np.isfinite(x)]     # undefined replicates (0 denominators) dropped explicitly
        return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))] if len(x) else None
    nvalid = lambda x: int(np.isfinite(np.asarray(x, float)).sum())
    for k in ("draws", "times", "tests", "old"):
        cb, cp, kb, kp = set_level(cs, ct, k)
        e = {key: float(v[0]) for key, v in estimand(cb, cp, kb, kp).items()}
        r = {"case_base_per_h": e["mcb"], "case_pre_per_h": e["mcp"], "control_base_per_h": e["mkb"], "control_pre_per_h": e["mkp"],
             "case_ratio": e["case_ratio"], "control_ratio": e["control_ratio"], "did_per_h": e["did"], "ratio_of_ratios": e["ror"],
             "share_sets_case_increase": float((cp > cb).mean()), "share_sets_control_mean_increase": float((kp > kb).mean()),
             "share_sets_contrast_positive": float((((cp - cb) - (kp - kb)) > 0).mean())}
        if ib is not None:
            bt = estimand(cb, cp, kb, kp, ib)
            r["did_ci95"] = pct(bt["did"]); r["case_ratio_ci95"] = pct(bt["case_ratio"]); r["ratio_of_ratios_ci95"] = pct(bt["ror"])
            r["valid_bootstrap_replicates"] = {"did": nvalid(bt["did"]), "case_ratio": nvalid(bt["case_ratio"]),
                                               "ratio_of_ratios": nvalid(bt["ror"]), "of": NBOOT}
        out[k] = r
    tpd = lambda df, w: (df[f"tests_{w}"].sum() / df[f"draws_{w}"].sum()) if len(df) and df[f"draws_{w}"].sum() else None
    out["tests_per_draw"] = {"case_base": tpd(cs, "base"), "case_pre": tpd(cs, "pre"), "control_base": tpd(ct, "base"), "control_pre": tpd(ct, "pre")}
    allw = pd.concat([cs, ct])
    d = np.concatenate([allw["draws_base"], allw["draws_pre"]]); o = np.concatenate([allw["old_base"], allw["old_pre"]])
    out["old_vs_specimen"] = {"mean_draws_specimen_per_h": float(d.mean()), "mean_draws_60s_per_h": float(o.mean()),
                              "share_windows_equal": float(np.mean(np.isclose(d, o))),
                              "correlation": float(np.corrcoef(d, o)[0, 1]) if d.std() > 0 and o.std() > 0 else None}
    return out


def selftest():
    """Hand-checkable test: set A has 2 controls, set B has 1 control."""
    cols = lambda b, p: {f"{k}_{w}": (b if w == "base" else p) for k in ("draws", "times", "tests", "old") for w in ("base", "pre")}
    cs = pd.DataFrame([dict(hadm_id="A", **cols(1.0, 3.0)), dict(hadm_id="B", **cols(2.0, 2.0))])
    ct = pd.DataFrame([dict(hadm_id="a1", case_hadm_id="A", **cols(1.0, 1.0)), dict(hadm_id="a2", case_hadm_id="A", **cols(1.0, 2.0)),
                       dict(hadm_id="b1", case_hadm_id="B", **cols(4.0, 0.0))])
    # Set A: case change +2; control changes 0 and +1 -> mean +0.5; contrast +1.5
    # Set B: case change 0; control change -4; contrast +4
    # Equal-set-weighted DiD = (1.5 + 4) / 2 = 2.75. (Pooling the 3 controls would give
    # case change mean 1 minus control change mean (0+1-4)/3 = -1 -> 2.0, which is NOT the estimand.)
    cb, cp, kb, kp = set_level(cs, ct, "draws")
    e = estimand(cb, cp, kb, kp)
    checks = {"set_A_control_base": kb[0] == 1.0, "set_A_control_pre": kp[0] == 1.5, "set_B_control_base": kb[1] == 4.0,
              "set_B_control_pre": kp[1] == 0.0, "did_equals_2.75": abs(e["did"][0] - 2.75) < 1e-12,
              "control_base_mean_equals_2.5": abs(e["mkb"][0] - 2.5) < 1e-12, "control_pre_mean_equals_0.75": abs(e["mkp"][0] - 0.75) < 1e-12,
              "not_pooled_2.0": abs(e["did"][0] - 2.0) > 1e-6}
    # bootstrap recomputation: a resample of [A, A] must give contrast 1.5; [B, B] must give 4.0
    bt = estimand(cb, cp, kb, kp, np.array([[0, 0], [1, 1], [0, 1]]))
    checks["bootstrap_AA_1.5"] = abs(bt["did"][0] - 1.5) < 1e-12
    checks["bootstrap_BB_4.0"] = abs(bt["did"][1] - 4.0) < 1e-12
    checks["bootstrap_AB_2.75"] = abs(bt["did"][2] - 2.75) < 1e-12
    # summarise end-to-end gives the same point estimate
    out = summarise(cs, ct, np.random.default_rng(1), "test")
    checks["summarise_did_2.75"] = abs(out["draws"]["did_per_h"] - 2.75) < 1e-12
    return {k: bool(v) for k, v in checks.items()}


def selftest_ward_runs():
    """ward -> non-ward -> ward must never form one run, and runs must agree with all_ward()."""
    T = lambda h: pd.Timestamp("2150-01-01") + pd.Timedelta(hours=h)
    mk = lambda rows: pd.DataFrame([dict(intime=T(a), outtime=T(b), cls=c) for a, b, c in rows])
    ck = {}
    # 1. ward 0-40, procedural 40-41, ward 41-100 (back to back, no gap)
    s1 = mk([(0, 40, "ward"), (40, 41, "procedural"), (41, 100, "ward")])
    r1 = ward_runs(s1)
    ck["wnw_two_runs"] = len(r1) == 2 and r1[0] == [T(0), T(40)] and r1[1] == [T(41), T(100)]
    ck["wnw_no_48h_window_across"] = not all_ward(s1, T(30), T(78)) and not any(lo <= T(30) and T(78) <= hi for lo, hi in r1)
    # 2. same, but the non-ward record lasts 30 s (shorter than the 1-minute merge tolerance)
    s2 = mk([(0, 40, "ward"), (40, 40 + 30 / 3600, "emergency"), (40 + 30 / 3600, 100, "ward")])
    ck["short_nonward_still_splits"] = len(ward_runs(s2)) == 2
    # 3. a non-ward record overlapping inside one long ward record (ward 0-100, unknown 50-51)
    s3 = mk([(0, 100, "ward"), (50, 51, "unknown")])
    ck["overlapping_nonward_splits"] = ward_runs(s3) == [[T(0), T(50)], [T(51), T(100)]]
    # 4. zero-length non-ward record inside a ward record
    s4 = mk([(0, 100, "ward"), (60, 60, "icu")])
    ck["zero_length_nonward_splits"] = ward_runs(s4) == [[T(0), T(60)], [T(60), T(100)]]
    # 5. ward -> ward with a 30 s gap is still one run (unchanged tolerance)
    ck["ward_ward_small_gap_one_run"] = len(ward_runs(mk([(0, 40, "ward"), (40 + 30 / 3600, 100, "ward")]))) == 1
    # 6. agreement with all_ward on random synthetic admissions and random event times
    g = np.random.default_rng(7); agree = True
    for _ in range(300):
        rows, t = [], 0.0
        for _ in range(g.integers(1, 7)):
            t += float(g.choice([0, 0, 1 / 120, 0.5, 3]))
            d = float(g.uniform(0, 80)) if g.random() > .1 else 0.0
            rows.append((t, t + d, str(g.choice(["ward", "ward", "ward", "procedural", "unknown", "emergency"]))))
            t += d * float(g.choice([1, 1, 0.5]))
        sg = mk(rows); rr = ward_runs(sg)
        for _ in range(40):
            e = T(float(g.uniform(0, t + 5)))
            agree &= any(lo <= e - pd.Timedelta(hours=48) and e <= hi for lo, hi in rr) == all_ward(sg, e - pd.Timedelta(hours=48), e)
    ck["runs_agree_with_all_ward_12000_random_checks"] = bool(agree)
    return {k: bool(v) for k, v in ck.items()}


def selftest_matching():
    """Exhaustive search: eligibility equals a brute-force check with the case rule (all_ward),
    pseudo-event vectorisation equals the original code, and unmatched categories are right."""
    ck = {}
    g = np.random.default_rng(11)
    # 1. vectorised pseudo-event == original scalar code
    base = pd.Timestamp("2150-01-01").value
    adm = base + g.integers(0, 400 * DAY_NS // 10**9, 3000) * 10**9
    same = True
    for _ in range(40):
        h = float(g.uniform(0, 400)); ev = pd.Timestamp(base + int(g.integers(0, DAY_NS // 10**9)) * 10**9)
        vec = pseudo_event_ns(adm, h, ev.hour * 3600 + ev.minute * 60 + ev.second)
        sc = np.array([pseudo_event_scalar(pd.Timestamp(int(a)), h, ev).value for a in adm[:300]])
        same &= bool(np.abs(vec[:300] - sc).max() < 1000)   # original keeps sub-microsecond float residue; v1.2.2 gives the exact clock time
    ck["pseudo_event_vectorised_equals_original_within_1_microsecond"] = same
    # 2. random synthetic transfers: exhaustive eligible set == brute force with all_ward()
    T = lambda x: pd.Timestamp("2150-01-01") + pd.Timedelta(hours=x)
    rows = []
    for a in range(400):
        t = float(g.uniform(0, 500))
        for _ in range(g.integers(1, 5)):
            d = float(g.uniform(0, 150)); c = str(g.choice(["ward", "ward", "ward", "procedural", "emergency", "unknown", "icu"], p=[.3, .3, .2, .06, .06, .04, .04]))
            rows.append(dict(hadm_id=a, subject_id=a // 2, intime=T(t), outtime=T(t + d), cls=c))
            t += d + float(g.choice([0, 0, 1 / 120, 2]))
    tr = pd.DataFrame(rows).sort_values(["hadm_id", "intime"]).reset_index(drop=True)
    cases = pd.DataFrame([dict(hadm_id=10_000 + i, subject_id=10_000 + i, event=T(float(g.uniform(0, 500))),
                               hours_since_admit=float(g.uniform(48, 300))) for i in range(60)])
    C = build_candidates(tr, cases)
    icu = set(tr.loc[tr["cls"] == "icu", "hadm_id"]); segs = {h: s_ for h, s_ in tr.groupby("hadm_id")}
    agree = True
    for _, cs in cases.iterrows():
        idx, t = eligible_for_case(C, cs["hours_since_admit"], cs["event"])
        mine = set(C["hadm"][idx])
        brute = set()
        for h, s_ in segs.items():
            if h in icu:
                continue
            e = pseudo_event_scalar(s_["intime"].iloc[0], cs["hours_since_admit"], cs["event"])
            if all_ward(s_, e - pd.Timedelta(hours=48), e):
                brute.add(h)
        agree &= mine == brute
    ck["exhaustive_eligible_equals_brute_force_all_ward"] = bool(agree)
    # 3. categories: cases 101, 102, 104 (same clock and elapsed time) compete for the same 3
    #    eligible patients (1, 3, 4); case 103 (20 h since admission) has none eligible.
    #    Whatever the random order: one case gets 2, one gets 1 (others assigned), one gets 0
    #    (eligible but all assigned), and 103 has no eligible control.
    rows = [dict(hadm_id=1, subject_id=1, intime=T(0), outtime=T(200), cls="ward"),      # eligible for X, Y (event 08:00)
            dict(hadm_id=2, subject_id=2, intime=T(0), outtime=T(50), cls="ward"),       # too short for anyone
            dict(hadm_id=3, subject_id=3, intime=T(1000), outtime=T(1200), cls="ward"),  # eligible for W only
            dict(hadm_id=4, subject_id=4, intime=T(1000), outtime=T(1200), cls="ward")]  # eligible for W only
    tr3 = pd.DataFrame(rows)
    ev8 = pd.Timestamp("2150-03-01 08:00:00")
    cases3 = pd.DataFrame([dict(hadm_id=101, subject_id=101, event=ev8, hours_since_admit=100.0),
                           dict(hadm_id=102, subject_id=102, event=ev8, hours_since_admit=100.0),
                           dict(hadm_id=103, subject_id=103, event=ev8, hours_since_admit=20.0),   # < 48 h: nobody eligible
                           dict(hadm_id=104, subject_id=104, event=ev8, hours_since_admit=100.0)])
    ctl, md = find_controls(tr3, cases3, np.random.default_rng(3))
    ck["no_eligible_counted"] = md["cases_no_eligible_control"] == 1
    ck["every_case_with_eligible_resolved"] = (md["cases_eligible_but_all_already_assigned"] + md["cases_one_control_only_one_eligible"]
                                              + md["cases_one_control_others_already_assigned"] + int(md["coverage_share_cases_two_controls"] * 4)
                                              + md["cases_no_eligible_control"]) == 4
    ck["each_control_patient_used_once"] = bool(ctl["subject_id"].is_unique) and len(ctl) == 3
    ck["eligible_but_all_assigned_equals_1"] = md["cases_eligible_but_all_already_assigned"] == 1
    ck["one_control_others_assigned_equals_1"] = md["cases_one_control_others_already_assigned"] == 1
    ck["two_controls_share_equals_0.25"] = abs(md["coverage_share_cases_two_controls"] - 0.25) < 1e-12
    # 4. exact clock and caliper on all selected controls
    clock = all((e.hour, e.minute) == c for e, c in zip(ctl["event"], ctl["case_clock"]))
    ck["selected_controls_exact_clock_and_caliper"] = bool(clock and (ctl["hours_since_admit"] - ctl["case_hours_since_admit"]).abs().max() <= TOL_H + 1e-9)
    return {k: bool(v) for k, v in ck.items()}


def suppress(obj, demo):
    if demo:
        return obj
    if isinstance(obj, dict):
        small = lambda k, v: isinstance(v, int) and not isinstance(v, bool) and 0 < v < SUPPRESS and (k.startswith("n_") or k.startswith("cases_") or k == "n")
        return {k: ("<10 suppressed" if small(k, v) else suppress(v, demo)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [suppress(v, demo) for v in obj]
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hosp", help="folder containing transfers.csv.gz and labevents.csv.gz")
    ap.add_argument("--selftest", action="store_true", help="run the estimator self-test and exit")
    ap.add_argument("--out", default="mimic_M1_results")
    ap.add_argument("--demo", action="store_true", help="open demo data: no small-cell suppression")
    a = ap.parse_args()
    if a.selftest:
        res = selftest(); res.update(selftest_ward_runs()); res.update(selftest_matching()); print(json.dumps(res, indent=1)); print("ALL PASSED" if all(res.values()) else "FAILED"); return
    if not a.hosp:
        sys.exit("--hosp is required")
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(SEED)
    t0 = dt.datetime.now()
    print(f"Plan {PLAN}. Reading transfers ...")
    tr = build_segments(pd.read_csv(os.path.join(a.hosp, "transfers.csv.gz")))
    cases, ed = find_cases(tr)
    print(f"  ward-origin escalations (one per patient): {len(cases):,}; ED-origin (descriptive): {len(ed):,}")
    controls, mdiag = find_controls(tr, cases, rng)
    print(f"  matched non-escalated ward controls: {len(controls):,}")
    diag = matching_diagnostics(tr, cases, controls) if len(controls) else {}
    diag.update(mdiag)
    print(f"  matching: clock exact {diag.get('clock_hour_and_minute_exact_match_share')}, max elapsed gap "
          f"{diag.get('elapsed_diff_hours_control_minus_case', {}).get('max_abs')} h, unmatched cases {suppress({'cases_with_x': diag.get('cases_with_no_control')}, a.demo)['cases_with_x']}")
    sm = suppress({k: v for k, v in diag.items() if k.startswith("cases_")}, a.demo)
    print(f"  exhaustive search: no eligible control {sm.get('cases_no_eligible_control')}; eligible but all already assigned "
          f"{sm.get('cases_eligible_but_all_already_assigned')}; one control (only one eligible) {sm.get('cases_one_control_only_one_eligible')}; "
          f"one control (others already assigned) {sm.get('cases_one_control_others_already_assigned')}")
    if diag and (diag["controls_48h_continuous_ward_share"] < 1 or diag["cases_48h_continuous_ward_share"] < 1):
        print("  WARNING: independent 48 h ward check is below 100%; do not use these results.")
    else:
        print("  independent 48 h ward check: cases 100%, controls 100%")
    ids = set(cases["hadm_id"]) | set(controls["hadm_id"]) | (set(ed["hadm_id"]) if not ed.empty else set())
    print("Reading labevents in chunks (this is the slow step on the full data) ...")
    labs = load_labs(os.path.join(a.hosp, "labevents.csv.gz"), ids)
    print(f"  lab rows kept: {len(labs):,}")
    res = {"plan": PLAN, "seed": SEED, "bootstrap_resamples": NBOOT, "demo": a.demo,
           "software": {"python": sys.version.split()[0], "pandas": pd.__version__, "numpy": np.__version__},
           "cohort": {"n_ward_escalations": int(len(cases)), "n_controls": int(len(controls)), "n_ed_escalations": int(len(ed)),
                      "case_hours_since_admit_median": float(cases["hours_since_admit"].median()) if len(cases) else None,
                      "control_hours_since_admit_median": float(controls["hours_since_admit"].median()) if len(controls) else None},
           "matching": dict(diag, method="risk-set matching on exact clock hour and minute, elapsed-time caliper +/-12 h, "
                            "up to 2 controls per case, without replacement (each control patient used once); cases processed "
                            "in seeded random order, candidates examined in seeded random order, first eligible accepted; "
                            "selection uses transfers only, never laboratory data; v1.2.2: exhaustive eligible-control search, no candidate cap")}
    for w, lab_name in ((24, "primary_24h"), (12, "sensitivity_12h")):
        cs, lc = per_stay(cases, labs, w); ct, lk = per_stay(controls, labs, w) if len(controls) else (pd.DataFrame(), None)
        res[lab_name] = summarise(cs, ct, rng, f"{w} h windows")
        if w == 24:
            prof = {}
            for name, lab in (("cases", lc), ("controls", lk)):
                n = len(cases) if name == "cases" else len(controls)
                if lab is None or n == 0:
                    continue
                hb = np.floor(lab["h"]).clip(0, 47).astype(int)
                per = lab.assign(hb=hb).groupby("hb")["specimen_id"].nunique().reindex(range(48), fill_value=0) / n
                prof[name + "_mean_draws_by_hour_before_event"] = [round(float(x), 4) for x in per.to_numpy()]
                clk = lab.groupby(lab["charttime"].dt.hour)["specimen_id"].nunique().reindex(range(24), fill_value=0)
                prof[name + "_share_of_draws_by_clock_hour"] = [round(float(x), 4) for x in (clk / clk.sum()).to_numpy()] if clk.sum() else None
            res["profile"] = prof
    if not ed.empty:
        ce, _ = per_stay(ed, labs, 24)
        res["ed_origin_descriptive"] = {"n": int(len(ce)), "draws_base_per_h": float(ce["draws_base"].mean()),
                                        "draws_pre_per_h": float(ce["draws_pre"].mean()), "tests_base_per_h": float(ce["tests_base"].mean()),
                                        "tests_pre_per_h": float(ce["tests_pre"].mean())}
    res["runtime_minutes"] = round((dt.datetime.now() - t0).total_seconds() / 60, 1)
    res["script_sha256"] = hashlib.sha256(open(__file__, "rb").read()).hexdigest()
    res = suppress(res, a.demo)
    p = os.path.join(a.out, "mimic_M1_summary.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, default=str)
    pr = res["primary_24h"]["draws"]
    print("\nPRIMARY (draws per hour, distinct specimen_id, 24 h windows)")
    print(f"  cases:    {pr['case_base_per_h']:.3f} -> {pr['case_pre_per_h']:.3f}")
    print(f"  controls: {pr['control_base_per_h']:.3f} -> {pr['control_pre_per_h']:.3f}")
    print(f"  difference-in-differences: {pr['did_per_h']:+.3f} per hour, 95% CI {pr['did_ci95']}")
    print(f"\nAggregates only written to {p}. Send that one file.")


if __name__ == "__main__":
    main()
