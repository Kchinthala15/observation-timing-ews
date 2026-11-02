# -*- coding: utf-8 -*-
import os
from si_common import *

S = load(A1F)
D = Doc("S3", "Full simulation results")
D.para("Every number in this document is taken directly from the frozen simulation summary (a1_summary_20261031_220543.json, archived at "
       f"{ZEN}), which was computed from the patient-level rows of all production blocks. Detection and false-alarm proportions are in percent and differences in "
       "percentage points, rounded half away from zero from the stored values. Arms: even (evenly spaced), uneven (escalating, the informative scheme), random-time "
       "(uniform times, no escalation) and thinned (the uneven scheme thinned by 1/m after escalation; reported for completeness). Tests: timing-aware state-space "
       "test, timestamp-blind AR(1) test and shuffle test; for the preprocessing blocks, AR(1) and shuffle surrogates are generated on the regularised series. "
       "Categories use the pre-specified ±5-point margin (Supporting Information S2).")
D.para("Contents: false-alarm calibration in every null block (Tables S3.1–S3.3); paired false-alarm comparisons (Table S3.4); confirmatory hypotheses "
       "(Tables S3.5–S3.6); the cost of uneven timing and its decomposition in every scenario block (Table S3.7); timing-aware versus timestamp-blind tests in every "
       "scenario block (Table S3.8); state dependence, decline shape and dose–response (Tables S3.9–S3.10); preprocessing (Tables S3.11–S3.12); the continuous-time "
       "estimator (Table S3.13); and the fold-model pilot (Table S3.14).")

cal = S["calibration"]
assert all(l == "CALIBRATED" for b in ("G1b","G1c","G1d") for l in S["scope_statements"][b]["labels"].values())
assert all(not e["fix_data_or_test_flag"] for e in S["P5_head_to_head"])
assert all(not S["state_dependence_rule"][t][i]["state_dependence_may_be_claimed"] for t in ("ou_ss","ar1","shuffle") for i in ("var","ac"))
assert S["XS2_restriction_rule"]["gain_point_estimate"] > 0.05 and not S["XS2_restriction_rule"]["restrict_Q3_to_constant_mean_processes"]
assert cal["G1a|informative_reduced_N|index|ar1|var"]["label"] == "LOW"
assert cal["G1a|informative_reduced_N|index|ou_ss|var"]["label"] == "CALIBRATED" and cal["XN2|informative_reduced_N|index|ou_ss|var"]["label"] == "CALIBRATED"

def calrow(v, with_analysis=False):
    r = [v["block"]] + ([rep_name(v["analysis"])] if with_analysis else []) + [
        ARM[v["arm"]], TEST[v["test"]], IND[v["indicator"]], f"{v['x']:,}/{v['n']:,}", pp(v["rate"]),
        ppci(v["cp90"]), LAB[v["label"]]]
    return r

# ---------- calibration, raw-series tests
D.landscape(True)
D.head("1. False-alarm calibration")
order = ["G1a", "G1b", "G1c", "G1d", "XN2", "XN1"]
rows = []
for b in order:
    rows.append(f"{b}: {NULLB[b]}")
    for v in cal.values():
        if v["block"] == b and v["analysis"] == "index":
            rows.append(calrow(v)[1:])
D.table("Calibration of the three tests on the raw series in every null block",
        ["Arm", "Test", "Indicator", "False alarms", "Rate (%)", "90% CI (%)", "Label"], rows,
        [1.3, 1.3, 1.1, 1.3, 1.0, 1.6, 1.6], size=8,
        note="90% Clopper–Pearson intervals. Labels: Calibrated (2.5% ≤ L and U ≤ 7.5%), Low (U ≤ 7.5%, L < 2.5%), High not excluded (U > 7.5%, L ≤ 7.5%), High (L > 7.5%). "
             f"Gate (G1a): the timing-aware test was {'CALIBRATED in all six cells' if S['gate_G1']['passed'] else 'not calibrated'}; "
             "in G1b–G1d it was CALIBRATED in every arm and indicator, so the Q3 statements extend to θ = 0.05 and to sixfold escalation. "
             "XN1 (level drop without slowing) is reported for every test as a limitation, not a finding.")
rows = []
reps = []
for v in cal.values():
    if v["block"] == "P5FA" and v["analysis"] not in reps:
        reps.append(v["analysis"])
for a in reps:
    rows.append(rep_name(a))
    for v in cal.values():
        if v["block"] == "P5FA" and v["analysis"] == a:
            rows.append(calrow(v)[1:])
D.table("Calibration of AR(1) and shuffle surrogates on each regularised representation (block P5FA: stable process, θ = 1.2, 2,000 patients)",
        ["Arm", "Test", "Indicator", "False alarms", "Rate (%)", "90% CI (%)", "Label"], rows,
        [1.3, 1.3, 1.1, 1.3, 1.0, 1.6, 1.6], size=8)
rows = [calrow(v)[1:] for v in cal.values() if v["block"] == "E2FA"]
D.table("Calibration of the continuous-time estimator (block E2FA: stable process, θ = 1.2, 1,000 patients; width-30 windows, timing-aware test)",
        ["Arm", "Test", "Indicator", "False alarms", "Rate (%)", "90% CI (%)", "Label"], rows,
        [1.3, 1.3, 1.1, 1.3, 1.0, 1.6, 1.6], size=8,
        note="The block was run with 1,000 rather than 2,000 patients (deviation 1, Supporting Information S2).")

# ---------- paired false-alarm comparisons
rows = []
fa = list(S["G1a_informative_minus_regular_FA (gate, no multiplicity)"]) + list(S["S_FA_family"])
cur = None
for e in fa:
    b = e["block"]
    if b != cur:
        rows.append(f"{b}: {NULLB[b]}" + ("  (gate block; no multiplicity adjustment)" if b == "G1a" else "")); cur = b
    rows.append([rep_name(e.get("analysis", "index")), TEST[e["test"]], IND[e["indicator"]], pp(e["rate_A"]), pp(e["rate_B"]),
                 f"{pp(e['diff'], sign=True)} ({ppci(e['ci95_bootstrap'])})", pfmt(e.get("holm_p")), e["categories"]["pm5"]])
D.table("Paired false-alarm comparisons, uneven minus even, in every null block",
        ["Analysis", "Test", "Indicator", "Uneven (%)", "Even (%)", "Difference (95% CI)", "Holm p", "Category (±5)"], rows,
        [1.9, 0.9, 0.85, 0.8, 0.75, 1.75, 0.65, 1.6], size=7.5,
        note="Paired bootstrap 95% intervals over patients; Holm-adjusted exact McNemar p-values within the false-alarm family (G1a excluded as the gate).")

# ---------- confirmatory
D.head("2. Confirmatory hypotheses (block F1, 2,000 fresh patients)")
cf = S["confirmatory_F1"]
rows = []
for comp in ("ar1", "shuffle"):
    for ind in ("var", "ac"):
        v = cf[comp][ind]
        h1 = v["H1"]; h2 = v["H2"]
        k1 = "ci97.5_bootstrap" if "ci97.5_bootstrap" in h1 else "ci95_bootstrap"
        k2 = "ci97.5_bootstrap" if "ci97.5_bootstrap" in h2 else "ci95_bootstrap"
        lvl = "97.5%" if k1.startswith("ci97") else "95%"
        rows.append([TEST[comp], IND[ind], lvl, pp(h1["rate_A"]), pp(h1["rate_B"]),
                     f"{pp(h1['diff'], sign=True)} ({ppci(h1[k1])})", h1["categories"]["pm5"],
                     f"{pp(h2['diff_in_diff'], sign=True)} ({ppci(h2[k2])})", h2["categories"]["pm5"]])
D.table("Raw comparisons: timing-aware test versus each timestamp-blind test on the uneven raw series",
        ["Comparator", "Indicator", "Interval", "Timing-aware (%)", "Comparator (%)", "H1 gain (CI)", "Category (±5)", "H2 specificity (CI)", "Category (±5)"],
        rows, [0.9, 0.8, 0.6, 0.9, 0.9, 1.5, 1.35, 1.5, 1.35], size=7.5,
        note="H1 and H2 are the variance comparisons against the AR(1) test (two-sided 97.5% intervals, Bonferroni). All other rows are secondary (95% intervals). "
             "H2 is the gain under uneven sampling minus the gain under even sampling.")
rows = []
for comp in ("ar1", "shuffle"):
    for ind in ("var", "ac"):
        sa = cf[comp][ind]["size_adjusted"]
        for ref in ("G1a", "XN2"):
            s_ = sa[ref]; a = s_["alpha_star"]
            k1 = "ci97.5_bootstrap" if "ci97.5_bootstrap" in s_["H1"] else "ci95_bootstrap"
            k2 = "ci97.5_bootstrap" if "ci97.5_bootstrap" in s_["H2"] else "ci95_bootstrap"
            rows.append([TEST[comp], IND[ind], f"{ref} ({'primary' if ref == 'G1a' else 'sensitivity'})", "97.5%" if k1.startswith("ci97") else "95%",
                         num(a[f"informative_reduced_N|ou_ss"], 4), num(a[f"informative_reduced_N|{comp}"], 4),
                         num(a["regular_N|ou_ss"], 4), num(a[f"regular_N|{comp}"], 4),
                         f"{pp(s_['H1']['diff'], sign=True)} ({ppci(s_['H1'][k1])})", "Yes" if s_["H1"]["supported"] else "No",
                         f"{pp(s_['H2']['diff_in_diff'], sign=True)} ({ppci(s_['H2'][k2])})", "Yes" if s_["H2"]["supported"] else "No"])
D.table("Size-adjusted analysis of the gain (H1) and its specificity (H2), every comparator and indicator",
        ["Comparator", "Indicator", "Reference block", "Interval", "α* timing-aware, uneven", "α* comparator, uneven", "α* timing-aware, even", "α* comparator, even",
         "Gain (CI)", "Lower bound > 0", "Specificity (CI)", "Lower bound > 0"], rows,
        [0.7, 0.7, 1.05, 0.55, 0.75, 0.75, 0.75, 0.75, 1.2, 0.6, 1.2, 0.6], size=7,
        note="α* is the largest threshold on the grid k/201 giving a false-alarm rate of at most 5% in the reference block for the same test, arm and indicator; detection is "
             "reclassified as τ > 0 and p ≤ α*. The bootstrap does not re-estimate α*. The confirmatory H1 and H2 are the variance rows against the AR(1) test (97.5% intervals); "
             "all other rows are secondary (95% intervals). Because the AR(1) comparator was LOW (conservative) in the uneven arm of G1a, the pre-specified wording is that "
             "the timing-aware test detects more than a conservative timestamp-blind test, judged on this analysis.")

# ---------- S-Q1
D.head("3. Cost of uneven timing and its decomposition (Q1), every scenario block")
comp_lab = {"total(inf-reg)": "Total: uneven − even", "irregularity(pm-reg)": "Irregularity: random-time − even",
            "escalation(inf-pm)": "Escalation: uneven − random-time", "thinned-reg": "Thinned − even"}
rows = []; cur = None
for e in S["S_Q1_family"]:
    b = e["block"]
    if b != cur:
        rows.append(f"{b}: {SCEN[b]} ({e['n']:,} patients)"); cur = b
    rows.append([TEST[e["test"]], IND[e["indicator"]], comp_lab[e["component"]], pp(e["rate_A"]), pp(e["rate_B"]),
                 f"{pp(e['diff'], sign=True)} ({ppci(e['ci95_bootstrap'])})", pfmt(e.get("holm_p"), 2), e["categories"]["pm5"]])
D.table("Paired detection differences by block, test, indicator and component",
        ["Test", "Indicator", "Component", "First arm (%)", "Second arm (%)", "Difference (95% CI)", "Holm p", "Category (±5)"], rows,
        [0.9, 0.8, 2.1, 0.85, 0.85, 1.7, 0.65, 1.6], size=7.5,
        note="First and second arm follow the order in the component name. Paired bootstrap 95% intervals over patients; Holm-adjusted exact McNemar p-values within family S-Q1. "
             "The smooth oracle schedule (XR10) has no thinned arm. Scenario blocks share patients with each other (common random numbers); F1 is independent.")

# ---------- S-Q3
D.head("4. Timing-aware versus timestamp-blind tests (Q3), every scenario block")
rows = []; cur = None
for e in S["S_Q3_family"]:
    b = e["block"]
    if b != cur:
        rows.append(f"{b}: {SCEN[b]} ({e['n']:,} patients)"); cur = b
    sp = e["specificity_gain_inf_minus_gain_reg"]
    rows.append([TEST[e["comparator"]], IND[e["indicator"]], pp(e["rate_A"]), pp(e["rate_B"]),
                 f"{pp(e['diff'], sign=True)} ({ppci(e['ci95_bootstrap'])})",
                 f"{pp(sp['diff_in_diff'], sign=True)} ({ppci(sp['ci95_bootstrap'])})", CLS[e["class"]],
                 f"{LAB[e['comparator_labels']['G1a|informative_reduced_N']]} / {LAB[e['comparator_labels']['XN2|informative_reduced_N']]}"])
D.table("Gain of the timing-aware test over each timestamp-blind test on the uneven raw series",
        ["Comparator", "Indicator", "Timing-aware (%)", "Comparator (%)", "Gain (95% CI)", "Specificity (95% CI)", "Classification", "Comparator label, uneven (G1a / XN2)"], rows,
        [0.85, 0.8, 0.9, 0.9, 1.55, 1.55, 1.6, 1.4], size=7.5,
        note="Raw comparisons with 95% intervals (the F1 variance rows against AR(1) are tested confirmatorily with 97.5% intervals in Table S3.5). Specificity is the gain under "
             "uneven sampling minus the gain under even sampling. Classification applies the pre-specified four-way rule to the gain, using the calibration of both tests. "
             "The timing-aware test was CALIBRATED in the uneven arm of G1a and XN2.")

# ---------- rules and dose response
D.head("5. State dependence, decline shape and escalation size")
sd = S["state_dependence_rule"]
rows = []
for t in ("ou_ss", "ar1", "shuffle"):
    for i in ("var", "ac"):
        r = sd[t][i]
        rows.append([TEST[t], IND[i], pp(r["F1_escalation"], sign=True), pp(r["XS1_escalation"], sign=True),
                     f"{pp(r['difference'], sign=True)} ({ppci(r['ci95'])})", "Yes" if r["state_dependence_may_be_claimed"] else "No"])
D.table("Pre-specified state-dependence rule: escalation component under the primary trigger (F1) versus the independent trigger (XS1)",
        ["Test", "Indicator", "Escalation, F1", "Escalation, XS1", "Difference (95% CI)", "State dependence may be claimed"], rows,
        [1.1, 1.0, 1.2, 1.2, 1.9, 1.9], size=8,
        note=f"No interval excluded zero, so the paper uses the pre-specified wording 'a rise in sampling density'. Decline-shape rule: the gain of the timing-aware over the AR(1) test "
             f"for variance in XS2 (slowing with level drop) was {pp(S['XS2_restriction_rule']['gain_point_estimate'], sign=True)} points, above the 5-point threshold, so the Q3 "
             "statements were not restricted to constant-mean processes.")
dr = S["dose_response"]
rows = []
for k, lab in (("total|ou_ss|var", "Total loss, variance"), ("total|ou_ss|ac", "Total loss, lag-1 AC"),
               ("escalation|ou_ss|var", "Escalation component, variance"), ("escalation|ou_ss|ac", "Escalation component, lag-1 AC")):
    r = dr[k]
    mb = r["mean_by_m"]
    cells = [pp(mb[m], sign=True) if m in mb else "–" for m in ("1.5", "2.0", "3.0", "4.0", "6.0")]
    rows.append([lab] + cells + [f"{pp(r['mean_per_patient_slope_per_log_m'], sign=True)} ({ppci(r['ci95'])})"])
D.table("Dose–response in the escalation factor m (timing-aware test; 1,000 patients, shared seeds)",
        ["Quantity", "m = 1.5", "m = 2", "m = 3", "m = 4", "m = 6", "Slope per unit log m (95% CI)"], rows,
        [2.3, 0.8, 0.8, 0.8, 0.8, 0.8, 2.1], size=8,
        note="Total loss uses XR5, the Phase 2 primary block P1 (m = 2) and XR6–XR8, all with patients 0–999 and shared seeds. The escalation trend uses XR5–XR8 only, because "
             "P1 has no random-time arm and F1 cannot be paired with the scenario blocks. Mean per-patient slope with bootstrap 95% interval.")

# ---------- P5
D.head("6. Preprocessing with standard tests (Q2; block P5, 1,000 patients)")
rows = []; cur = None
for e in S["S_Q2_P5_family"]:
    a = e["analysis"]
    if a != cur:
        rows.append(rep_name(a)); cur = a
    rc, gp, wi = e["recovery"], e["gap_to_even_index_ou_ss"], e["informative_minus_regular"]
    rows.append([TEST[e["test"]], IND[e["indicator"]], pp(rc["rate_A"]), pp(rc["rate_B"]),
                 f"{pp(rc['diff'], sign=True)} ({ppci(rc['ci95_bootstrap'])})",
                 f"{pp(gp['diff'], sign=True)} ({ppci(gp['ci95_bootstrap'])})",
                 f"{pp(wi['diff'], sign=True)} ({ppci(wi['ci95_bootstrap'])})",
                 CLS[e["class"]], f"{LAB[e['fa_label_informative']]} / {LAB[e['fa_label_regular']]}", pfmt(e["holm_p"])])
D.table("Detection after regularisation, by representation, surrogate and indicator",
        ["Surrogate", "Indicator", "Uneven (%)", "Same test, raw uneven (%)", "Recovery vs raw, same test (95% CI)", "Gap to even, timing-aware (95% CI)", "Uneven − even, same pipeline (95% CI)",
         "Classification", "False-alarm label (uneven / even)", "Holm p"], rows,
        [0.65, 0.7, 0.6, 0.7, 1.35, 1.35, 1.35, 1.3, 1.05, 0.5], size=7,
        note="Recovery compares the regularised pipeline with the same test on the raw uneven series. The gap compares it with the timing-aware test on even raw data. "
             "Classification uses the false-alarm label of the same representation, surrogate, arm and indicator in P5FA (Table S3.2). Holm adjustment across the 52 cells "
             "(family S-Q2).")
rows = []
for e in S["P5_head_to_head"]:
    h = e["head_to_head_vs_index_ou_ss"]
    rows.append([rep_name(e["analysis"]), IND[e["indicator"]], pp(h["rate_A"]), pp(h["rate_B"]),
                 f"{pp(h['diff'], sign=True)} ({ppci(h['ci95_bootstrap'])})", "Yes" if e["fix_data_or_test_flag"] else "No"])
D.table("Head to head on the uneven series: AR(1) test after regularisation versus the timing-aware test on the raw series",
        ["Representation", "Indicator", "AR(1) after regularisation (%)", "Timing-aware, raw (%)", "Difference (95% CI)", "Equivalent within ±3 and genuine recovery"], rows,
        [2.5, 0.9, 1.4, 1.2, 1.7, 1.7], size=7.5,
        note="The pre-specified 'fix the data or fix the test' message required genuine recovery and a 95% interval within ±3 points; no representation met it.")

# ---------- E2
D.head("7. Continuous-time estimator (Q2; block E2, 500 patients)")
rows = []
for e in S["S_Q2_E2"]:
    rc, gp, wi = e["recovery"], e["gap_to_even_index_ou_ss"], e["informative_minus_regular"]
    rows.append([IND[e["indicator"]], pp(rc["rate_A"]), pp(rc["rate_B"]), pp(wi["rate_B"]), pp(gp["rate_B"]),
                 f"{pp(rc['diff'], sign=True)} ({ppci(rc['ci95_bootstrap'])})",
                 f"{pp(gp['diff'], sign=True)} ({ppci(gp['ci95_bootstrap'])})",
                 f"{pp(wi['diff'], sign=True)} ({ppci(wi['ci95_bootstrap'])})", CLS[e["class"]], LAB[e["fa_label_informative"]], pfmt(e["holm_p_joint_P5_E2"], 4)])
D.table("Continuous-time estimator (width-30 windows) on the uneven series",
        ["Indicator", "Continuous-time, uneven (%)", "Timing-aware, raw uneven (%)", "Continuous-time, even (%)", "Timing-aware, raw even (%)", "Recovery (95% CI)", "Gap to even, timing-aware (95% CI)", "Uneven − even, continuous-time (95% CI)",
         "Classification", "False-alarm label, uneven (E2FA)", "Holm p (joint)"], rows,
        [0.7, 0.8, 0.8, 0.8, 0.8, 1.2, 1.3, 1.3, 1.0, 0.9, 0.55], size=7,
        note="With 500 patients the paired 95% interval half-width is about ±5.5 points, as stated in the protocol. Holm p (joint): adjustment over the 52 preprocessing "
             "cells and the two continuous-time cells together, reported as a sensitivity analysis.")

# ---------- fold pilot
D.head("8. Fold-model pilot")
fs = S["fold_selection"]
rows = [[num(g["sigma_f"], 2), num(g["r_end"], 2), f"{g['n_enrolled']}", f"{g['tipped']}", pp(g["tipping"]),
         pp(g["regular_N_variance_detection"]), "Yes" if g["tipping"] <= 0.05 and 0.35 <= g["regular_N_variance_detection"] <= 0.65 else "No"]
        for g in fs["grid_results"]]
assert fs["selected_model"] is None
D.table("Pilot grid for the fold model (300 patients per point; even arm only; timing-aware test)",
        ["σf", "r_end", "Patients", "Tipped", "Tipping (%)", "Even-arm variance detection (%)", "Eligible and in 35–65%"], rows,
        [0.8, 0.8, 0.9, 0.8, 1.0, 2.0, 1.9], size=8,
        note="Rule fixed in advance: eligible if tipping ≤ 5%; choose the eligible point with even-arm detection closest to 50% and inside 35–65%. No point qualified, so the main "
             "fold block was not run and the fold model is reported as uninformative. The uneven arm was never computed in the pilot.")
out = os.path.join(OUT, "SiM_Supporting_Information_S3_Full_Results.docx")
D.save(out); print("saved", out, "numbers", len(USED))

