"""Protocol Addendum A1 v1.2 (frozen) - summariser.

Implements the frozen A1 analysis rules (A1 JSON: inference, calibration_status, comparative_claim_rules,
detection_change_classification, gate, hypotheses; PDF Sections 5, 5b, 6, 7, 9):

  * calibration label for every test x arm x analysis x indicator in every null-process block
    (CALIBRATED / LOW / HIGH_NOT_EXCLUDED / HIGH from the 90% Clopper-Pearson interval)
  * gate G1 (G1a, ou_ss, regular_N / informative_reduced_N / poisson_matched, variance and AC)
  * confirmatory H1, H2 in F1 (ar1, variance): two-sided 97.5% paired bootstrap CIs (10,000 patient resamples),
    Newcombe cross-check at the same level; claim wording governed by the calibration of ou_ss AND the
    comparator in G1a and XN2 (arms used), with the pre-specified size-adjusted analysis (alpha* from G1a;
    XN2 as sensitivity)
  * secondary families (95% unadjusted CIs + Holm-adjusted exact McNemar p): S-Q1 (total, irregularity,
    escalation), S-Q3 (gain of ou_ss over ar1 / shuffle; the confirmatory F1 ar1-variance gain excluded),
    S-Q2 (P5 per Section 6.5; joint Holm with E2 also reported), S-FA (G1b-d, XN1, XN2, P5FA, E2FA;
    G1a is the gate and has no multiplicity)
  * four-way classification of every recovery / gain; scope, restriction and replication statements
  * mean Kendall tau per arm and paired tau differences (or tau difference-in-differences) for every contrast
  * frozen +/-5 pp interpretation categories, with +/-3 and +/-10 sensitivity, and the U < 0 flag

Refuses to run if an identity check failed (implementation error, A1 Section 9); skips a block whose
identity check has not been run. Usage (from the rebuild_v322 folder):  python -m a1_summary.summarize
Kept OUTSIDE ewsv322 so adding it does not change the code fingerprints the Stage 0 approval is bound to.
"""
import datetime as dt
import json
import math
import os

import numpy as np
from scipy.stats import binom, norm

from ewsv322 import spec as S
from ewsv322 import rng as R
from ewsv322 import stats
from ewsv322 import run as RUN
from ewsv322 import phase2 as P2
from ewsv322 import a1_spec as A
from ewsv322 import a1_run as AR

B_BOOT = S.N_BOOT                                   # 10,000 (frozen)
MARGINS = [3, 5, 10]
NULL_BLOCKS = ["G1a", "G1b", "G1c", "G1d", "XN1", "XN2"]
Q1_BLOCKS = ["F1", "XS1"] + [f"XR{i}" for i in range(1, 11)] + ["XS2", "FOLD2"]
TESTS = ["ou_ss", "ar1", "shuffle"]
INDS = ("var", "ac")
STREAMS_USED = []                                   # every bootstrap stream name, checked unique at the end


# ------------------------------------------------------------------ loading
def load_block(entry):
    """Rows of a completed A1 block, verified against its record. Original-seed blocks also need a passed
    full-cohort identity check; a FAILED identity check stops the summariser."""
    tag = AR._tag(entry)
    rec_p = os.path.join(AR.OUT_A1, f"a1_{tag}_record.json")
    if not os.path.exists(rec_p):
        return None, "not run"
    rec, rows = AR._read_a1(entry)
    if rec.get("a1_sha256") != A.A1_SHA256_FROZEN_ORIGINAL or rec.get("config_sha256") != S.CONFIG_HASH:
        raise SystemExit(f"{entry}: record written under a different configuration.")
    r = AR.resolve(entry)
    pids = {int(x["patient_id"]) for x in rows}
    if pids != set(r["ids"]):
        raise SystemExit(f"{entry}: rows cover {len(pids)} patients, block defines {len(r['ids'])} - stop and report.")
    needs_identity = entry in ("P5", "P5FA") or (r["base"] == P2.BASE and r["base_block"] and r["base_block"] != "E2")
    if needs_identity:
        files = [f for f in os.listdir(AR.OUT_A1) if f.startswith(f"a1_{tag}_identity_") and f.endswith(".json")]
        ok = False
        for f in files:
            with open(os.path.join(AR.OUT_A1, f), encoding="utf-8") as fh:
                d = json.load(fh)
            if d.get("rows_sha256") == rec["rows_sha256"] and not d.get("passed"):
                raise SystemExit(f"{entry}: identity check FAILED ({f}) - implementation error, stop and report (A1 Section 9).")
            ok |= bool(d.get("passed")) and d.get("rows_sha256") == rec["rows_sha256"]
        if not ok:
            return None, "identity check (verify) not run - not summarised"
    return rows, "ok"


def load_explore(tags):
    for tag in tags:
        rec_p = os.path.join(AR.OUT_EXPLORE, f"explore_{tag}_record.json")
        rows_p = os.path.join(AR.OUT_EXPLORE, f"explore_{tag}_rows.csv")
        if os.path.exists(rec_p) and os.path.exists(rows_p):
            with open(rec_p, encoding="utf-8") as f:
                rec = json.load(f)
            if RUN._sha(rows_p) != rec["rows_sha256"] or rec.get("config_sha256") != S.CONFIG_HASH:
                raise SystemExit(f"{rows_p} does not match its record or configuration.")
            return P2.read_rows(rows_p)
    return None


# ------------------------------------------------------------------ vectors
def index_rows(rows):
    """dict (arm, analysis, test) -> {patient_id: row}, tipped patients excluded."""
    out = {}
    for x in rows:
        if not x.get("arm") or x.get("tipped"):
            continue
        out.setdefault((x["arm"], x["analysis"], x["null"]), {})[int(x["patient_id"])] = x
    return out


def common_ids(*cells):
    s = None
    for c in cells:
        s = set(c) if s is None else s & set(c)
    return sorted(s or [])


def det_vec(cell, ids, ind, alpha=None):
    if alpha is None:
        return np.array([1.0 if cell[i][f"det_{ind}"] else 0.0 for i in ids])
    out = []
    for i in ids:
        tau, p = cell[i][f"tau_{ind}"], cell[i][f"p_{ind}"]
        out.append(1.0 if (np.isfinite(tau) and np.isfinite(p) and tau > 0 and p <= alpha + 1e-12) else 0.0)
    return np.array(out)


def tau_vec(cell, ids, ind):
    return np.array([float(cell[i][f"tau_{ind}"]) for i in ids])


# ------------------------------------------------------------------ statistics
def cp(x, n, level):
    return stats.clopper_pearson(int(x), int(n), level)


def calib_label(x, n):
    L, U = cp(x, n, 0.90)
    if L >= 0.025 and U <= 0.075:
        return "CALIBRATED"
    if U <= 0.075:
        return "LOW"
    return "HIGH" if L > 0.075 else "HIGH_NOT_EXCLUDED"


def mcnemar_p(a, b):
    f = int(np.sum((a == 1) & (b == 0))); g = int(np.sum((a == 0) & (b == 1)))
    if f + g == 0:
        return 1.0
    return float(min(1.0, 2 * binom.cdf(min(f, g), f + g, 0.5)))


def boot(d, base, model, name, level=0.95):
    """Patient bootstrap of mean(d) (d a per-patient paired quantity, NaN-free); percentile CI."""
    d = np.asarray(d, float)
    STREAMS_USED.append(f"bootstrap|{model}|{name}")
    if len(d) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = R.bootstrap_stream(base, model, name)
    n = len(d)
    boots = np.empty(B_BOOT)
    done = 0
    while done < B_BOOT:
        b = min(500, B_BOOT - done)
        idx = rng.integers(0, n, size=(b, n))
        boots[done:done + b] = d[idx].mean(axis=1)
        done += b
    a = (1 - level) / 2
    lo, hi = np.percentile(boots, [100 * a, 100 * (1 - a)])
    return float(d.mean()), float(lo), float(hi)


def cikey(level):
    return f"ci{level * 100:g}_bootstrap"


def categories(lo, hi):
    out = {f"pm{m}": stats.interpret(lo, hi, margin=m / 100)["category"] for m in MARGINS}
    out["upper_ci_below_zero_any_loss"] = bool(hi < 0)
    return out


def tau_part(ta, tb, base, model, name, level, tc=None, td=None):
    """Paired tau difference (ta - tb), or difference-in-differences (ta - tb) - (tc - td); NaN taus dropped."""
    d = ta - tb if tc is None else (ta - tb) - (tc - td)
    ok = np.isfinite(d)
    m, lo, hi = boot(d[ok], base, model, name + "|tau", level)
    return {"diff": m, "ci": [lo, hi], "n_tau": int(ok.sum()),
            "mean_A": float(np.nanmean(ta)), "mean_B": float(np.nanmean(tb))}


def contrast(a, b, base, model, name, ta=None, tb=None, level=0.95):
    """Paired detection contrast a - b: bootstrap CI, Newcombe cross-check at the same level, exact McNemar p,
    margin categories, and the paired tau difference when tau vectors are given."""
    est, lo, hi = boot(a - b, base, model, name, level)
    e = int(np.sum((a == 1) & (b == 1))); f = int(np.sum((a == 1) & (b == 0)))
    g = int(np.sum((a == 0) & (b == 1))); h = int(np.sum((a == 0) & (b == 0)))
    z = float(norm.ppf(1 - (1 - level) / 2))
    _, nl, nu = stats.newcombe_paired(e, f, g, h, z=z)
    out = {"n": len(a), "rate_A": float(a.mean()) if len(a) else None, "rate_B": float(b.mean()) if len(b) else None,
           "diff": est, cikey(level): [lo, hi], f"newcombe_{level * 100:g}": [nl, nu],
           "cells_e_f_g_h": [e, f, g, h], "mcnemar_p": mcnemar_p(a, b), "categories": categories(lo, hi),
           "bootstrap_stream": f"bootstrap|{model}|{name}"}
    if ta is not None:
        out["tau"] = tau_part(ta, tb, base, model, name, level)
    return out


def did(a, b, c, d, base, model, name, level=0.95, ta=None, tb=None, tc=None, td=None):
    est, lo, hi = boot((a - b) - (c - d), base, model, name, level)
    out = {"diff_in_diff": est, cikey(level): [lo, hi], "categories": categories(lo, hi)}
    if ta is not None:
        out["tau"] = tau_part(ta, tb, base, model, name, level, tc, td)
    return out


def holm(entries, key="holm_p"):
    if not entries:
        return entries
    ps = [e["mcnemar_p"] for e in entries]
    order = np.argsort(ps, kind="stable")
    m = len(ps); running = 0.0
    for rank, k in enumerate(order):
        running = max(running, min(1.0, (m - rank) * ps[k]))
        entries[k][key] = running
    return entries


def classify(lo, label):
    if not (lo > 0):
        return "NO_RECOVERY"
    if label in ("CALIBRATED", "LOW"):
        return "GENUINE_RECOVERY"
    if label == "HIGH":
        return "IMPROVEMENT_AT_FALSE_ALARM_COST"
    return "IMPROVEMENT_FA_CONTROL_NOT_ESTABLISHED"


def worst_label(labels):
    labels = list(labels)
    if any(l == "HIGH" for l in labels):
        return "HIGH"
    if any(l not in ("CALIBRATED", "LOW") for l in labels):
        return "HIGH_NOT_EXCLUDED" if all(l != "NOT_AVAILABLE" for l in labels) else "NOT_AVAILABLE"
    return "LOW" if any(l == "LOW" for l in labels) else "CALIBRATED"


# ------------------------------------------------------------------ calibration, evaluability
def calibration(blocks, e2fa_n500=False):
    table, fam = {}, []
    for bid, rows in blocks.items():
        r = AR.resolve(bid)
        idx = index_rows(rows)
        for test in r["design"]["nulls"]:
            for an in r["design"]["analyses"]:
                for arm in [a for a in r["design"]["arms"] if (a, an, test) in idx]:
                    cell = idx[(arm, an, test)]
                    n = len(cell)
                    for ind in INDS:
                        x = int(sum(1 for v in cell.values() if v[f"det_{ind}"]))
                        L90, U90 = cp(x, n, 0.90); L95, U95 = cp(x, n, 0.95)
                        lab = calib_label(x, n) if n else "NOT_AVAILABLE"
                        if bid == "E2FA" and (e2fa_n500 or n < 1000):
                            lab = "DESCRIPTIVE_N500"      # A1 6.7: an n = 500 E2FA cannot establish calibration
                        table[f"{bid}|{arm}|{an}|{test}|{ind}"] = {
                            "block": bid, "arm": arm, "analysis": an, "test": test, "indicator": ind, "x": x, "n": n,
                            "rate": x / n if n else None, "cp90": [L90, U90], "cp95": [L95, U95],
                            "tost_pass": stats.tost(x, n)["pass"] if n else False, "label": lab}
                if ("regular_N", an, test) in idx and ("informative_reduced_N", an, test) in idx:
                    c1, c2 = idx[("informative_reduced_N", an, test)], idx[("regular_N", an, test)]
                    ids = common_ids(c1, c2)
                    for ind in INDS:
                        e = contrast(det_vec(c1, ids, ind), det_vec(c2, ids, ind), r["base"], r["model"],
                                     f"A1|{bid}|FA|{an}|{test}|{ind}|inf-vs-reg",
                                     tau_vec(c1, ids, ind), tau_vec(c2, ids, ind))
                        e.update({"block": bid, "analysis": an, "test": test, "indicator": ind})
                        fam.append(e)
    holm([e for e in fam if e["block"] != "G1a" and not (e["block"] == "E2FA" and (e2fa_n500 or e["n"] < 1000))])
    s_fa = [e for e in fam if e["block"] != "G1a"]          # descriptive E2FA (n < 1000) reported, not in Holm
    return table, s_fa, [e for e in fam if e["block"] == "G1a"]


def evaluability(blocks):
    out = {}
    for bid, rows in blocks.items():
        idx = index_rows(rows)
        for (arm, an, test), cell in idx.items():
            n = len(cell)
            for ind in INDS:
                ev = sum(1 for v in cell.values() if v[f"evaluable_{ind}"])
                out[f"{bid}|{arm}|{an}|{test}|{ind}"] = {"evaluable": ev / n if n else None, "n": n,
                                                         "cp95": list(cp(ev, n, 0.95)) if n else None}
    return out


def label_of(calib, block, arm, test, ind, analysis="index"):
    c = calib.get(f"{block}|{arm}|{analysis}|{test}|{ind}")
    return c["label"] if c else "NOT_AVAILABLE"


def gate(calib):
    cells = {f"{arm}|{ind}": label_of(calib, "G1a", arm, "ou_ss", ind)
             for arm in ("regular_N", "informative_reduced_N", "poisson_matched") for ind in INDS}
    return {"cells": cells, "passed": all(v == "CALIBRATED" for v in cells.values()), "rule": A.A1["gate"]["G1"]}


def scope_statements(calib):
    out = {}
    scope = {"G1b": "slow process theta 0.2, 6-fold escalation", "G1c": "slow process theta 0.05, 2-fold escalation",
             "G1d": "slow process theta 0.05, 6-fold escalation"}
    for b, what in scope.items():
        labs = {f"{arm}|{ind}": label_of(calib, b, arm, "ou_ss", ind)
                for arm in ("regular_N", "informative_reduced_N", "poisson_matched") for ind in INDS}
        ok = all(v == "CALIBRATED" for v in labs.values())
        if any(v == "NOT_AVAILABLE" for v in labs.values()):
            out[b] = {"labels": labs, "statement": f"{b} not run: scope for {what} not yet assessable."}
            continue
        out[b] = {"labels": labs, "statement": (f"ou_ss CALIBRATED: the Q3 claim extends to {what}." if ok else
                                               f"ou_ss not CALIBRATED in every cell: the paper states the Q3 claim does not extend to {what}.")}
    labs = {f"{arm}|{t}": label_of(calib, "XN1", arm, t, "ac") for arm in ("regular_N", "informative_reduced_N") for t in TESTS}
    out["XN1_AC"] = {"labels": labs, "statement": "Level-drop control AC false alarms are reported for every test; any "
                                                  "exceedance is a limitation, not a finding (A1 6.4)."}
    return out


# ------------------------------------------------------------------ confirmatory
def alpha_star(rows, arm, test, ind):
    cell = index_rows(rows).get((arm, "index", test))
    if not cell:
        return None
    ids = sorted(cell)
    best = None
    for k in range(1, S.N_SURR + 2):
        a = k / (S.N_SURR + 1)
        if det_vec(cell, ids, ind, alpha=a).mean() <= 0.05:
            best = a
        else:
            break
    return best


def wording(ou_labels, cmp_labels):
    """ou_ss status and comparator status reported independently (A1 comparative_claim_rules)."""
    parts = []
    if "NOT_AVAILABLE" in ou_labels.values() or "NOT_AVAILABLE" in cmp_labels.values():
        parts.append("a reference false-alarm block is missing: no claim at controlled false-alarm rates until it is run")
    if any(v not in ("CALIBRATED", "NOT_AVAILABLE") for v in ou_labels.values()):
        parts.append("ou_ss not CALIBRATED in every reference cell: claim restricted to the regimes where it is CALIBRATED")
    cl = set(cmp_labels.values())
    if "HIGH" in cl:
        parts.append("comparator HIGH: tests of demonstrably unequal size; judged only on the size-adjusted analysis")
    elif "HIGH_NOT_EXCLUDED" in cl:
        parts.append("comparator false-alarm control not established; judged on the size-adjusted analysis")
    elif "LOW" in cl:
        parts.append("comparator LOW: 'detects more than a conservative timestamp-blind test', judged on the size-adjusted analysis")
    all_cal = all(v == "CALIBRATED" for v in list(ou_labels.values()) + list(cmp_labels.values()))
    return {"all_calibrated": all_cal, "statements": parts or ["ou_ss and comparator CALIBRATED in G1a and XN2"]}


def confirmatory(f1_rows, calib, null_rows):
    r = AR.resolve("F1")
    idx = index_rows(f1_rows)
    base, model = r["base"], r["model"]
    out = {}
    for comp in ("ar1", "shuffle"):
        c = {k: idx[k] for k in [("informative_reduced_N", "index", "ou_ss"), ("informative_reduced_N", "index", comp),
                                 ("regular_N", "index", "ou_ss"), ("regular_N", "index", comp)]}
        io_c, ic_c = c[("informative_reduced_N", "index", "ou_ss")], c[("informative_reduced_N", "index", comp)]
        ro_c, rc_c = c[("regular_N", "index", "ou_ss")], c[("regular_N", "index", comp)]
        ids = common_ids(*c.values())
        res = {}
        for ind in INDS:
            io, ic, ro, rc = (det_vec(x, ids, ind) for x in (io_c, ic_c, ro_c, rc_c))
            tio, tic, tro, trc = (tau_vec(x, ids, ind) for x in (io_c, ic_c, ro_c, rc_c))
            level = 0.975 if (comp == "ar1" and ind == "var") else 0.95
            h1 = contrast(io, ic, base, model, f"A1|F1|H1|{comp}|{ind}", tio, tic, level)
            h2 = did(io, ic, ro, rc, base, model, f"A1|F1|H2|{comp}|{ind}", level, tio, tic, tro, trc)
            adj = {}
            for ref in ("G1a", "XN2"):
                if null_rows.get(ref) is None:
                    adj[ref] = {"status": "reference block not available"}; continue
                al = {(arm, t): alpha_star(null_rows[ref], arm, t, ind)
                      for arm in ("informative_reduced_N", "regular_N") for t in ("ou_ss", comp)}
                adj[ref] = {"alpha_star": {f"{k[0]}|{k[1]}": v for k, v in al.items()}}
                ai, ac_ = al[("informative_reduced_N", "ou_ss")], al[("informative_reduced_N", comp)]
                if ai is None or ac_ is None:
                    adj[ref]["status"] = "alpha* undefined for the informative arm"; continue
                io_a, ic_a = det_vec(io_c, ids, ind, ai), det_vec(ic_c, ids, ind, ac_)
                e1, l1, u1 = boot(io_a - ic_a, base, model, f"A1|F1|H1adj{ref}|{comp}|{ind}", level)
                adj[ref]["H1"] = {"diff": e1, cikey(level): [l1, u1], "supported": l1 > 0}
                ar_, arc = al[("regular_N", "ou_ss")], al[("regular_N", comp)]
                if ar_ is None or arc is None:
                    adj[ref]["H2_status"] = "alpha* undefined for the regular arm"; continue
                ro_a, rc_a = det_vec(ro_c, ids, ind, ar_), det_vec(rc_c, ids, ind, arc)
                e2, l2, u2 = boot((io_a - ic_a) - (ro_a - rc_a), base, model, f"A1|F1|H2adj{ref}|{comp}|{ind}", level)
                adj[ref]["H2"] = {"diff_in_diff": e2, cikey(level): [l2, u2], "supported": l2 > 0}
            ou1 = {f"{ref}|informative_reduced_N": label_of(calib, ref, "informative_reduced_N", "ou_ss", ind) for ref in ("G1a", "XN2")}
            cm1 = {f"{ref}|informative_reduced_N": label_of(calib, ref, "informative_reduced_N", comp, ind) for ref in ("G1a", "XN2")}
            ou2 = dict(ou1, **{f"{ref}|regular_N": label_of(calib, ref, "regular_N", "ou_ss", ind) for ref in ("G1a", "XN2")})
            cm2 = dict(cm1, **{f"{ref}|regular_N": label_of(calib, ref, "regular_N", comp, ind) for ref in ("G1a", "XN2")})
            res[ind] = {"H1": h1, "H2": h2, "size_adjusted": adj,
                        "H1_wording": wording(ou1, cm1), "H2_wording": wording(ou2, cm2)}
        out[comp] = res
    v = out["ar1"]["var"]
    lo1 = v["H1"][cikey(0.975)][0]; lo2 = v["H2"][cikey(0.975)][0]
    h1, h2 = lo1 > 0, lo2 > 0
    cal1, cal2 = v["H1_wording"]["all_calibrated"], v["H2_wording"]["all_calibrated"]
    sa = v["size_adjusted"]
    sa_flags = {f"{h}_supported_size_adjusted_{ref}": (sa[ref][h]["supported"] if h in sa.get(ref, {}) else None)
                for ref in ("G1a", "XN2") for h in ("H1", "H2")}
    # basis of interpretation (A1 5b/6.2): raw if every compared test is CALIBRATED, otherwise size-adjusted (G1a; XN2 sensitivity)
    eff_h1 = h1 if cal1 else sa_flags["H1_supported_size_adjusted_G1a"]
    eff_h2 = h2 if cal2 else sa_flags["H2_supported_size_adjusted_G1a"]
    basis1 = "raw" if cal1 else "size_adjusted_G1a"
    basis2 = "raw" if cal2 else "size_adjusted_G1a"
    w1 = "; ".join(v["H1_wording"]["statements"]); w2 = "; ".join(v["H2_wording"]["statements"])
    if eff_h1 is None:
        verdict = f"H1 not assessable on its required basis ({basis1}): reference block missing or alpha* undefined."
    elif not eff_h1:
        verdict = f"H1 not supported ({basis1}): Q3 dropped."
    elif eff_h2:
        if cal1 and cal2:
            verdict = ("H1 and H2 supported, all compared tests CALIBRATED in G1a and XN2: 'timestamp-blind surrogate tests lose "
                       "variance detection under irregular sampling; a timing-aware test recovers it at controlled false-alarm rates'.")
        else:
            verdict = (f"H1 ({basis1}) and H2 ({basis2}) supported; wording qualified: H1 - {w1}; H2 - {w2}.")
    else:
        verdict = (f"H1 supported ({basis1}), H2 not {'assessable' if eff_h2 is None else 'supported'} ({basis2}): "
                   f"'ou_ss is more powerful in this setting' (not irregularity-specific). Wording: {w1}.")
    out["decision"] = dict({"H1_supported": h1, "H1_material": lo1 > 0.05, "H2_supported": h2,
                            "all_tests_calibrated_for_H1": cal1, "all_tests_calibrated_for_H2": cal2,
                            "H1_basis": basis1, "H2_basis": basis2, "H1_supported_on_basis": eff_h1,
                            "H2_supported_on_basis": eff_h2, "verdict": verdict,
                            "size_adjusted_limitation": "the bootstrap does not re-estimate alpha* (A1 Section 5b)"}, **sa_flags)
    return out


# ------------------------------------------------------------------ Q1 / Q3 by block
def q1_q3_block(bid, rows):
    r = AR.resolve(bid)
    idx = index_rows(rows)
    base, model = r["base"], r["model"]
    q1, q3 = [], []
    for test in TESTS:
        cells = {a: idx.get((a, "index", test)) for a in ("regular_N", "informative_reduced_N", "poisson_matched", "thinned")}
        if not cells["regular_N"] or not cells["informative_reduced_N"]:
            continue
        ids = common_ids(*[cells[a] for a in ("regular_N", "informative_reduced_N", "poisson_matched") if cells[a]])
        for ind in INDS:
            reg, inf = det_vec(cells["regular_N"], ids, ind), det_vec(cells["informative_reduced_N"], ids, ind)
            treg, tinf = tau_vec(cells["regular_N"], ids, ind), tau_vec(cells["informative_reduced_N"], ids, ind)
            comps = {"total(inf-reg)": (inf, reg, tinf, treg, "S-Q1")}
            if cells["poisson_matched"]:
                pm, tpm = det_vec(cells["poisson_matched"], ids, ind), tau_vec(cells["poisson_matched"], ids, ind)
                comps["irregularity(pm-reg)"] = (pm, reg, tpm, treg, "S-Q1")
                comps["escalation(inf-pm)"] = (inf, pm, tinf, tpm, "S-Q1")
            if cells["thinned"]:
                tids = common_ids(cells["thinned"], cells["regular_N"])
                comps["thinned-reg"] = (det_vec(cells["thinned"], tids, ind), det_vec(cells["regular_N"], tids, ind),
                                        tau_vec(cells["thinned"], tids, ind), tau_vec(cells["regular_N"], tids, ind), "descriptive")
            for name, (a, b, ta, tb, fam) in comps.items():
                e = contrast(a, b, base, model, f"A1|{bid}|Q1|{name}|{test}|{ind}", ta, tb)
                e.update({"block": bid, "test": test, "indicator": ind, "component": name, "family": fam})
                q1.append(e)
    for comp in ("ar1", "shuffle"):
        keys = [("informative_reduced_N", "index", "ou_ss"), ("informative_reduced_N", "index", comp),
                ("regular_N", "index", "ou_ss"), ("regular_N", "index", comp)]
        c = {k: idx.get(k) for k in keys}
        if any(v is None for v in c.values()):
            continue
        ids = common_ids(*c.values())
        for ind in INDS:
            io, ic, ro, rc = (det_vec(c[k], ids, ind) for k in keys)
            tio, tic, tro, trc = (tau_vec(c[k], ids, ind) for k in keys)
            g = contrast(io, ic, base, model, f"A1|{bid}|Q3|gain|{comp}|{ind}", tio, tic)
            g["specificity_gain_inf_minus_gain_reg"] = did(io, ic, ro, rc, base, model, f"A1|{bid}|Q3|spec|{comp}|{ind}",
                                                           0.95, tio, tic, tro, trc)
            g.update({"block": bid, "comparator": comp, "indicator": ind,
                      "family": "confirmatory(excluded)" if (bid == "F1" and comp == "ar1" and ind == "var") else "S-Q3"})
            q3.append(g)
    return q1, q3


def q1_replication(f1_rows, p1_rows):
    out = {}
    for bid, rows, base in (("F1", f1_rows, AR.resolve("F1")["base"]), ("P1", p1_rows, P2.BASE)):
        idx = index_rows(rows)
        ci, cr = idx[("informative_reduced_N", "index", "ou_ss")], idx[("regular_N", "index", "ou_ss")]
        ids = common_ids(ci, cr)
        out[bid] = {ind: contrast(det_vec(ci, ids, ind), det_vec(cr, ids, ind), base, "ou_csd",
                                  f"A1|REPL|{bid}|{ind}", tau_vec(ci, ids, ind), tau_vec(cr, ids, ind)) for ind in INDS}
    out["rule"] = "frozen Phase 2 interpretation rule (+/-5 pp categories) applied to F1 and set against P1"
    out["same_category"] = {ind: out["F1"][ind]["categories"]["pm5"] == out["P1"][ind]["categories"]["pm5"] for ind in INDS}
    return out


def state_dependence(f1_rows, xs1_rows):
    """Escalation component (inf - pm) F1 vs XS1 (different patients: independent two-sample bootstrap).
    Primary reading: ou_ss, variance (stated); all tests and indicators reported."""
    out = {"primary": "ou_ss|var"}
    for test in TESTS:
        res = {}
        for ind in INDS:
            comps = []
            for bid, rows in (("F1", f1_rows), ("XS1", xs1_rows)):
                idx = index_rows(rows)
                ci, cpm = idx.get(("informative_reduced_N", "index", test)), idx.get(("poisson_matched", "index", test))
                if not ci or not cpm:
                    break
                ids = common_ids(ci, cpm)
                comps.append(det_vec(ci, ids, ind) - det_vec(cpm, ids, ind))
            if len(comps) < 2:
                continue
            d1, d2 = comps
            n1, n2 = f"A1|SD|F1|{test}|{ind}", f"A1|SD|XS1|{test}|{ind}"
            STREAMS_USED.extend([f"bootstrap|ou_csd|{n1}", f"bootstrap|ou_csd|{n2}"])
            r1 = R.bootstrap_stream(AR.resolve("F1")["base"], "ou_csd", n1)
            r2 = R.bootstrap_stream(AR.resolve("XS1")["base"], "ou_csd", n2)
            m1 = d1[r1.integers(0, len(d1), size=(B_BOOT, len(d1)))].mean(axis=1)
            m2 = d2[r2.integers(0, len(d2), size=(B_BOOT, len(d2)))].mean(axis=1)
            lo, hi = np.percentile(m1 - m2, [2.5, 97.5])
            res[ind] = {"F1_escalation": float(d1.mean()), "XS1_escalation": float(d2.mean()),
                        "difference": float(d1.mean() - d2.mean()), "ci95": [float(lo), float(hi)],
                        "state_dependence_may_be_claimed": bool(lo > 0 or hi < 0)}
        out[test] = res
    return out


def dose_response(blocks, p1_rows):
    ms = {k: float(P2.BLOCKS[k.replace("X", "")]["m"]) for k in ("XR5", "XR6", "XR7", "XR8")}
    ms["P1"] = float(P2.BLOCKS["P1"]["m"])
    srcs = {k: (p1_rows if k == "P1" else blocks.get(k)) for k in ms}
    if any(v is None for v in srcs.values()):
        return "not all blocks available"
    out = {"note": "total loss uses XR5, P1, XR6-XR8 (ids 0-999, shared seeds). The escalation trend uses XR5-XR8 only: "
                   "P1 has no poisson_matched arm and F1 cannot be paired with the XR blocks."}
    for comp in ("total", "escalation"):
        use = sorted([k for k in ms if comp == "total" or k != "P1"], key=lambda k: ms[k])
        for ind in INDS:
            per = {}
            for k in use:
                idx = index_rows(srcs[k])
                a = idx.get(("informative_reduced_N", "index", "ou_ss"))
                b = idx.get(("regular_N" if comp == "total" else "poisson_matched", "index", "ou_ss"))
                if not a or not b:
                    per = None; break
                per[k] = (a, b)
            if per is None:
                continue
            ids = [i for i in common_ids(*[c for ab in per.values() for c in ab]) if i < 1000]
            x = np.log(np.array([ms[k] for k in use]))
            Y = np.column_stack([det_vec(per[k][0], ids, ind) - det_vec(per[k][1], ids, ind) for k in use])
            xc = x - x.mean()
            slopes = (Y - Y.mean(axis=1, keepdims=True)) @ xc / (xc @ xc)
            est, lo, hi = boot(slopes, P2.BASE, "ou_csd", f"A1|DOSE|{comp}|ou_ss|{ind}")
            out[f"{comp}|ou_ss|{ind}"] = {"m_values": [ms[k] for k in use], "n": len(ids),
                                          "mean_per_patient_slope_per_log_m": est, "ci95": [lo, hi],
                                          "mean_by_m": {str(ms[k]): float(Y[:, j].mean()) for j, k in enumerate(use)}}
    return out


# ------------------------------------------------------------------ Q2
def q2_p5(p5_rows, p1_rows, calib):
    r = AR.resolve("P5")
    base, model = r["base"], r["model"]
    idx, p1 = index_rows(p5_rows), index_rows(p1_rows)
    fam, heads = [], []
    r_ou, i_ou = p1[("regular_N", "index", "ou_ss")], p1[("informative_reduced_N", "index", "ou_ss")]
    for an in r["design"]["analyses"]:
        for test, frozen_equiv in (("ar1_rep", "ar1"), ("shuffle_rep", "shuffle")):
            ci, cr = idx.get(("informative_reduced_N", an, test)), idx.get(("regular_N", an, test))
            i_idx = p1.get(("informative_reduced_N", "index", frozen_equiv))
            if not all([ci, cr, i_idx]):
                continue
            ids = common_ids(ci, cr, i_idx, r_ou, i_ou)
            for ind in INDS:
                inf, tinf = det_vec(ci, ids, ind), tau_vec(ci, ids, ind)
                rec = contrast(inf, det_vec(i_idx, ids, ind), base, model, f"A1|P5|rec|{an}|{test}|{ind}", tinf, tau_vec(i_idx, ids, ind))
                gap = contrast(inf, det_vec(r_ou, ids, ind), base, model, f"A1|P5|gap|{an}|{test}|{ind}", tinf, tau_vec(r_ou, ids, ind))
                within = contrast(inf, det_vec(cr, ids, ind), base, model, f"A1|P5|within|{an}|{test}|{ind}", tinf, tau_vec(cr, ids, ind))
                lab = label_of(calib, "P5FA", "informative_reduced_N", test, ind, analysis=an)
                cls = classify(rec[cikey(0.95)][0], lab)
                fam.append({"analysis": an, "test": test, "indicator": ind, "recovery": rec, "gap_to_even_index_ou_ss": gap,
                            "informative_minus_regular": within, "fa_label_informative": lab,
                            "fa_label_regular": label_of(calib, "P5FA", "regular_N", test, ind, analysis=an),
                            "class": cls, "material": cls == "GENUINE_RECOVERY" and rec[cikey(0.95)][0] > 0.05,
                            "mcnemar_p": rec["mcnemar_p"]})
                if test == "ar1_rep":
                    hh = contrast(inf, det_vec(i_ou, ids, ind), base, model, f"A1|P5|h2h|{an}|{ind}", tinf, tau_vec(i_ou, ids, ind))
                    lo, hi = hh[cikey(0.95)]
                    heads.append({"analysis": an, "indicator": ind, "head_to_head_vs_index_ou_ss": hh,
                                  "fix_data_or_test_flag": cls == "GENUINE_RECOVERY" and lo >= -0.03 and hi <= 0.03})
    return holm(fam), heads


def q2_e2(e2_rows, p1_rows, calib):
    r = AR.resolve("E2")
    idx, p1 = index_rows(e2_rows), index_rows(p1_rows)
    ci, cr = idx.get(("informative_reduced_N", "ct|W30", "ou_ss")), idx.get(("regular_N", "ct|W30", "ou_ss"))
    i_ou, r_ou = p1[("informative_reduced_N", "index", "ou_ss")], p1[("regular_N", "index", "ou_ss")]
    if not ci or not cr:
        return []
    ids = common_ids(ci, cr, i_ou, r_ou)
    out = []
    for ind in INDS:
        inf, tinf = det_vec(ci, ids, ind), tau_vec(ci, ids, ind)
        rec = contrast(inf, det_vec(i_ou, ids, ind), r["base"], r["model"], f"A1|E2|rec|{ind}", tinf, tau_vec(i_ou, ids, ind))
        gap = contrast(inf, det_vec(r_ou, ids, ind), r["base"], r["model"], f"A1|E2|gap|{ind}", tinf, tau_vec(r_ou, ids, ind))
        within = contrast(inf, det_vec(cr, ids, ind), r["base"], r["model"], f"A1|E2|within|{ind}", tinf, tau_vec(cr, ids, ind))
        lab = label_of(calib, "E2FA", "informative_reduced_N", "ou_ss", ind, analysis="ct|W30")
        out.append({"analysis": "ct|W30", "test": "ou_ss", "indicator": ind, "recovery": rec,
                    "gap_to_even_index_ou_ss": gap, "informative_minus_regular": within, "fa_label_informative": lab,
                    "class": classify(rec[cikey(0.95)][0], lab), "mcnemar_p": rec["mcnemar_p"],
                    "precision_note": "E2 has 500 patients: paired 95% CI half-width about +/-5.5 pp (A1 6.7)."})
    return out


def q2_explore(e1_rows, e3_rows, p1_rows):
    """Four-way rule applied to the E1/E3 (ou_ss) representations (A1 6.5, last sentence). E1/E3 remain exploratory."""
    if e1_rows is None or e3_rows is None:
        return "E1/E3 outputs not found"
    e1, e3, p1 = index_rows(e1_rows), index_rows(e3_rows), index_rows(p1_rows)
    i_ou, r_ou = p1[("informative_reduced_N", "index", "ou_ss")], p1[("regular_N", "index", "ou_ss")]
    out = []
    for (arm, an, test), cell in sorted(e1.items()):
        if arm != "informative_reduced_N":
            continue
        ids = common_ids(cell, i_ou, r_ou)
        fa = e3.get(("informative_reduced_N", an, test))
        for ind in INDS:
            rec = contrast(det_vec(cell, ids, ind), det_vec(i_ou, ids, ind), P2.BASE, "ou_csd", f"A1|E1|rec|{an}|{ind}",
                           tau_vec(cell, ids, ind), tau_vec(i_ou, ids, ind))
            if fa:
                x = sum(1 for v in fa.values() if v[f"det_{ind}"]); lab = calib_label(x, len(fa))
            else:
                lab = "NOT_AVAILABLE"
            out.append({"analysis": an, "indicator": ind, "recovery": rec, "fa_label_informative_E3": lab,
                        "class": classify(rec[cikey(0.95)][0], lab), "status": "exploratory (frozen E1/E3)"})
    return out


# ------------------------------------------------------------------ main
def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if not math.isfinite(f) else f
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def main():
    STREAMS_USED.clear()
    blocks, status = {}, {}
    for entry in [e for e in AR.block_list() if not e.startswith("FOLDP")]:
        rows, st = load_block(entry)
        blocks[entry] = rows; status[entry] = st
    p1_rows = AR._frozen_rows("P1")
    devs = AR.read_deviations()
    e2fa_n500 = False                       # n < 1000 in the E2FA rows triggers the descriptive label (A1 6.7)
    nulls = {b: blocks[b] for b in NULL_BLOCKS + ["P5FA", "E2FA"] if blocks.get(b) is not None}
    calib, s_fa, g1a_fa = calibration(nulls, e2fa_n500)
    summ = {"written": dt.datetime.now().isoformat(timespec="seconds"), "a1_sha256": A.A1_SHA256,
            "config_sha256": S.CONFIG_HASH, "block_status": status, "bootstrap_resamples": B_BOOT,
            "calibration": calib, "S_FA_family": s_fa, "G1a_informative_minus_regular_FA (gate, no multiplicity)": g1a_fa,
            "evaluability": evaluability({k: v for k, v in blocks.items() if v is not None})}
    gate_ok = None
    if blocks.get("G1a") is not None:
        summ["gate_G1"] = gate(calib); gate_ok = summ["gate_G1"]["passed"]
        summ["scope_statements"] = scope_statements(calib)
    if blocks.get("F1") is not None and blocks.get("G1a") is not None:
        conf = confirmatory(blocks["F1"], calib, nulls)
        if not gate_ok:
            conf["decision"]["verdict"] = "GATE G1 FAILED: no Q3 claim; H1/H2 are reported as descriptive only."
        summ["confirmatory_F1"] = conf
        summ["Q1_replication_F1_vs_P1"] = q1_replication(blocks["F1"], p1_rows)
    q1_all, q3_all = [], []
    for bid in Q1_BLOCKS:
        if blocks.get(bid) is not None:
            q1, q3 = q1_q3_block(bid, blocks[bid])
            q1_all += q1; q3_all += q3
    holm([e for e in q1_all if e["family"] == "S-Q1"])
    holm([e for e in q3_all if e["family"] == "S-Q3"])
    for e in q3_all:
        ou = {f"{ref}|{arm}": label_of(calib, ref, arm, "ou_ss", e["indicator"])
              for ref in ("G1a", "XN2") for arm in ("informative_reduced_N", "regular_N")}
        e["ou_ss_labels"] = ou
        e["comparator_labels"] = {f"{ref}|{arm}": label_of(calib, ref, arm, e["comparator"], e["indicator"])
                                  for ref in ("G1a", "XN2") for arm in ("informative_reduced_N", "regular_N")}
        e["class"] = classify(e[cikey(0.95)][0], worst_label(v for k, v in ou.items() if "informative" in k))
        sp = e["specificity_gain_inf_minus_gain_reg"]
        sp["class"] = classify(sp[cikey(0.95)][0], worst_label(ou.values()))
        e["material"] = e["class"] == "GENUINE_RECOVERY" and e[cikey(0.95)][0] > 0.05
        if gate_ok is False:
            e["gate_failed_descriptive_only"] = True
            e["class"] = "DESCRIPTIVE_GATE_FAILED"; sp["class"] = "DESCRIPTIVE_GATE_FAILED"; e["material"] = False
    summ["S_Q1_family"] = q1_all
    summ["S_Q3_family"] = q3_all
    xs2 = [e for e in q3_all if e["block"] == "XS2" and e["comparator"] == "ar1" and e["indicator"] == "var"]
    if xs2:
        g = xs2[0]["diff"]
        summ["XS2_restriction_rule"] = {"gain_point_estimate": g, "rule": "point estimate of the XS2 ou_ss-vs-ar1 variance gain (A1 6.4)",
                                        "restrict_Q3_to_constant_mean_processes": bool(g < 0.05)}
    if blocks.get("F1") is not None and blocks.get("XS1") is not None:
        summ["state_dependence_rule"] = state_dependence(blocks["F1"], blocks["XS1"])
    summ["dose_response"] = dose_response(blocks, p1_rows)
    if blocks.get("P5") is not None:
        fam, heads = q2_p5(blocks["P5"], p1_rows, calib)
        summ["S_Q2_P5_family"] = fam; summ["P5_head_to_head"] = heads
    if blocks.get("E2") is not None:
        summ["S_Q2_E2"] = q2_e2(blocks["E2"], p1_rows, calib)
    if summ.get("S_Q2_P5_family") or summ.get("S_Q2_E2"):
        joint = list(summ.get("S_Q2_P5_family", [])) + list(summ.get("S_Q2_E2", []))
        holm(joint, key="holm_p_joint_P5_E2")
        summ["S_Q2_holm_note"] = ("holm_p: over the 52 P5 cells as A1 6.5 specifies; holm_p_joint_P5_E2: P5 and E2 together "
                                  "(A1 Section 5 lists S-Q2 as P5 and E2), reported as sensitivity.")
    summ["E1_E3_four_way (exploratory)"] = q2_explore(load_explore(["E1__all"]), load_explore(["E3__all", "E3__noct"]), p1_rows)
    if blocks.get("FOLD2") is not None:
        n_tip = len({int(x["patient_id"]) for x in blocks["FOLD2"] if x.get("tipped")})
        summ["FOLD2_note"] = {"tipped_patients": n_tip, "enrolled": len(AR.resolve("FOLD2")["ids"]),
                              "statement": "FOLD2 results are conditional on non-tipping (A1 6.6)."}
    if os.path.exists(AR.FOLD_SELECTION):
        with open(AR.FOLD_SELECTION, encoding="utf-8") as f:
            summ["fold_selection"] = json.load(f)
    summ["deviations"] = devs
    import collections
    dup = [k for k, c in collections.Counter(STREAMS_USED).items() if c > 1]
    if dup:
        raise RuntimeError(f"bootstrap stream names used more than once: {dup[:5]}")
    R.check_unique(set(STREAMS_USED))
    summ["bootstrap_streams_used_and_checked_unique"] = sorted(set(STREAMS_USED))
    import platform, scipy
    summ["software"] = {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__}
    summ["code_sha256"] = AR.code_hashes()
    summ["summariser_sha256"] = RUN._sha(os.path.abspath(__file__))
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(AR.OUT_A1, exist_ok=True)
    p = os.path.join(AR.OUT_A1, f"a1_summary_{stamp}.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(_clean(summ), f, indent=2, allow_nan=False)
    print("A1 summary written to", p)
    print(json.dumps(status, indent=1))
    if "gate_G1" in summ:
        print("Gate G1 passed:", summ["gate_G1"]["passed"])
    if "confirmatory_F1" in summ:
        print("Decision:", json.dumps(_clean(summ["confirmatory_F1"]["decision"]), indent=1))


if __name__ == "__main__":
    main()
