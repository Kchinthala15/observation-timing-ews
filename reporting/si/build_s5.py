# -*- coding: utf-8 -*-
import os
from si_common import *
from docx.shared import Inches

M = load(M1F); F = load(M1F0)
mt = M["matching"]; co = M["cohort"]
assert all(M[w]["old"]["did_ci95"][0] > 0 for w in ("primary_24h","sensitivity_12h"))
assert F["primary_24h"]["draws"]["did_ci95"][0] > 0
assert M["script_sha256"].startswith("1b869172")
D = Doc("S5", "MIMIC-IV analysis: full methods, matching diagnostics, clock-hour distribution and record of the first run")
D.para("Every number in this document is taken from the aggregate output of the frozen analysis script (mimic_M1_v122_full_summary.json; script SHA-256 "
       f"{M['script_sha256']}) or, for Section 6, from the aggregate output of the first full-data run. Both files, the script, its version history, its self-test "
       f"output and a run on the open MIMIC-IV demonstration data are archived at {ZEN} (folder m1_mimic). No patient-level MIMIC-IV data are included.")

D.head("1. Data and governance")
D.para("MIMIC-IV version 3.1 (hosp and icu modules) was accessed under the PhysioNet credentialed data use agreement. The analysis ran on the author's own computer; it "
       "writes a single file of aggregate results containing no patient rows, identifiers or dates, and any patient count below 10 is suppressed. Only that aggregate "
       "file left the local computing environment, and no MIMIC-IV data were uploaded to any online service.")

D.head("2. Methods in full")
D.bullets([
    "Care units. Each transfer record was classified from its care-unit name as intensive care (names containing ICU, intensive care, cardiac vascular or coronary care), "
    "emergency department, procedural (post-anaesthesia care, operating room, procedure units, catheterisation laboratory, endoscopy, radiology), ward (medicine, surgery, "
    "medical/surgical, oncology, obstetrics, psychiatry, neurology, haematology, transplant, vascular, cardiology, observation, step-down, intermediate, trauma, gynaecology) "
    "or other. Discharge records were ignored.",
    "Continuous ward runs. Consecutive ward records separated by gaps of at most one minute form one run; any non-ward record, including one of zero length, ends a run.",
    "Cases. The first intensive-care entry of an admission whose preceding record is a ward record and whose entire preceding 48 hours lie within one continuous ward run. "
    "Each patient contributes at most one case (the earliest).",
    "Candidate controls. Admissions with no intensive-care record, from patients who are never cases, containing at least one continuous ward run of 48 hours or more.",
    "Risk-set matching, using transfer records only and never laboratory data. Cases are processed in seeded random order. For each case, every candidate admission receives "
    "a pseudo-index time with exactly the case's clock hour and minute, nearest to the candidate's admission time plus the case's elapsed time since admission (so the "
    "elapsed-time difference is within ±12 hours by construction). A candidate is eligible if the 48 hours before its pseudo-index time lie within one continuous ward run. "
    "The search is exhaustive over all candidates; up to two eligible controls are drawn at random from patients not already used, without replacement. Reasons for any case "
    "receiving fewer than two controls are recorded.",
    "Windows. Baseline: 48 to 24 hours before the index time; pre-event: the final 24 hours. Both windows span all 24 clock hours, and controls are matched on the clock time "
    "of the index, so time-of-day patterns cannot create a difference. Sensitivity: 12-hour windows (24 to 12 and 12 to 0 hours before the index).",
    "Outcomes per hour of window. Primary: specimen draws, the number of distinct specimen identifiers (specimen_id). Secondary: distinct collection times (one venepuncture "
    "can yield several specimens), laboratory test records, test records per draw, and draws counted by an earlier rule in which collection times more than 60 seconds apart "
    "start a new draw.",
    "Estimand. For each matched set, (case pre-event − case baseline) − mean over that set's controls of (control pre-event − control baseline); the estimate is the mean of "
    "these set contrasts, so every set carries equal weight whether it has one or two controls. Control means and ratios are averages of set-level control means. The ratio "
    "of ratios is (case pre/baseline) ÷ (control pre/baseline).",
    f"Inference. Percentile bootstrap over matched sets ({M['bootstrap_resamples']:,} resamples, seed {M['seed']}), recomputing the same estimand in every resample.",
    "Decision rule fixed before the full-data analysis: increased sampling before escalation is supported if the 95% interval for the primary difference-in-differences lies above zero.",
    "Descriptive contrast. Escalations from the emergency department after at least 48 hours since admission are summarised descriptively.",
])

D.head("3. Cohort and matching diagnostics")
q = mt["elapsed_diff_hours_control_minus_case"]["abs_quantiles_10_25_50_75_90"]
rows = [["Ward-to-intensive-care cases", f"{co['n_ward_escalations']:,}"],
        ["Matched controls", f"{co['n_controls']:,}"],
        ["Candidate control admissions examined (exhaustive search)", f"{mt['candidate_control_admissions']:,}"],
        ["Cases with 2 / 1 / 0 controls", f"{mt['cases_with_2_controls']:,} / {mt['cases_with_1_control']} / {mt['cases_with_no_control']}"],
        ["Cases with no eligible control; eligible but all already assigned", f"{mt['cases_no_eligible_control']}; {mt['cases_eligible_but_all_already_assigned']}"],
        ["Eligible control patients per case, minimum and median", f"{int(mt['eligible_patients_per_case_quantiles_0_1_5_10_25_50'][0]):,} and {mt['eligible_patients_per_case_quantiles_0_1_5_10_25_50'][5]:,.0f}"],
        ["Exact match on clock hour and minute (share of pairs)", pp(mt["clock_hour_and_minute_exact_match_share"]) + "%"],
        ["Elapsed-time difference, control minus case (hours): mean; median; range",
         f"{num(mt['elapsed_diff_hours_control_minus_case']['mean'], 2)}; {num(mt['elapsed_diff_hours_control_minus_case']['median'], 2)}; "
         f"{num(mt['elapsed_diff_hours_control_minus_case']['min'], 1)} to {num(mt['elapsed_diff_hours_control_minus_case']['max'], 1)}"],
        ["Absolute elapsed-time difference (hours): 10th, 25th, 50th, 75th, 90th percentiles", ", ".join(num(x, 1) for x in q)],
        ["Pairs within the ±12-hour caliper", pp(mt["elapsed_diff_hours_control_minus_case"]["share_within_12h"]) + "%"],
        ["Median hours since admission at the index, cases and controls", f"{num(co['case_hours_since_admit_median'], 1)} and {num(co['control_hours_since_admit_median'], 1)}"],
        ["48 hours of continuous ward occupancy, verified independently: cases; controls", f"{pp(mt['cases_48h_continuous_ward_share'])}%; {pp(mt['controls_48h_continuous_ward_share'])}%"],
        ["One admission per patient (cases; controls); no patient both case and control",
         f"{'Yes' if mt['one_admission_per_case_patient'] else 'No'}; {'Yes' if mt['one_admission_per_control_patient'] else 'No'}; {'Yes' if mt['no_patient_both_case_and_control'] else 'No'}"],
        ["Emergency-department escalations (descriptive contrast)", f"{co['n_ed_escalations']}"]]
D.table("Cohort and matching diagnostics (final analysis)", ["Diagnostic", "Value"], rows, [4.4, 2.1], size=9)

D.head("4. Full results")
def blockrows(key):
    out = []
    for k, lb in (("draws", "Specimen draws"), ("times", "Collection times"), ("tests", "Test records"), ("old", "Draws, 60-second rule")):
        r = M[key][k]; dp = 2 if k == "tests" else 3
        out.append([lb, num(r["case_base_per_h"], dp), num(r["case_pre_per_h"], dp), num(r["control_base_per_h"], dp), num(r["control_pre_per_h"], dp),
                    f"{num(r['did_per_h'], dp, True)} ({num(r['did_ci95'][0], dp)} to {num(r['did_ci95'][1], dp)})",
                    f"{num(r['case_ratio'], 2)} ({num(r['case_ratio_ci95'][0], 2)} to {num(r['case_ratio_ci95'][1], 2)})",
                    num(r["control_ratio"], 2),
                    f"{num(r['ratio_of_ratios'], 2)} ({num(r['ratio_of_ratios_ci95'][0], 2)} to {num(r['ratio_of_ratios_ci95'][1], 2)})",
                    f"{pp(r['share_sets_case_increase'])} / {pp(r['share_sets_control_mean_increase'])} / {pp(r['share_sets_contrast_positive'])}"])
    return out
D.landscape(True)
rows = ["24-hour windows (primary)"] + blockrows("primary_24h") + ["12-hour windows (sensitivity)"] + blockrows("sensitivity_12h")
D.table("Sampling before ward-to-intensive-care escalation: all outcomes, both window lengths",
        ["Outcome (per hour)", "Cases, baseline", "Cases, pre-event", "Controls, baseline", "Controls, pre-event", "Difference-in-differences (95% CI)",
         "Case ratio (95% CI)", "Control ratio", "Ratio of ratios (95% CI)", "Sets with increase (%): case / control mean / contrast > 0"], rows,
        [1.3, 0.7, 0.7, 0.75, 0.75, 1.5, 1.2, 0.7, 1.2, 1.4], size=7.5,
        note=f"{co['n_ward_escalations']:,} matched sets ({co['n_controls']:,} controls). Equal-set-weighted estimates; percentile bootstrap over matched sets; all "
             f"{M['primary_24h']['draws']['valid_bootstrap_replicates']['of']:,} replicates valid for every estimate. The 24-hour specimen-draw row is the pre-specified primary analysis; "
             "its interval lies above zero, so the decision rule is met.")
D.landscape(False)
rows = []
for key, lab in (("primary_24h", "24-hour windows"), ("sensitivity_12h", "12-hour windows")):
    t = M[key]["tests_per_draw"]; o = M[key]["old_vs_specimen"]
    rows.append([lab, num(t["case_base"], 2), num(t["case_pre"], 2), num(t["control_base"], 2), num(t["control_pre"], 2),
                 num(o["mean_draws_specimen_per_h"], 3), num(o["mean_draws_60s_per_h"], 3), pp(o["share_windows_equal"]), num(o["correlation"], 2)])
D.table("Test records per draw, and agreement between the specimen-identifier and 60-second definitions of a draw",
        ["Windows", "Tests per draw: cases, baseline", "Cases, pre-event", "Controls, baseline", "Controls, pre-event",
         "Draws per hour, specimen_id", "Draws per hour, 60-second rule", "Windows with equal counts (%)", "Correlation"], rows,
        [0.9, 0.75, 0.65, 0.7, 0.7, 0.75, 0.75, 0.65, 0.65], size=7.5,
        note="The 60-second rule counts roughly half as many draws as the specimen identifier, so the definition matters for absolute rates; the difference-in-differences is "
             "positive under both definitions.")
e = M["ed_origin_descriptive"]
D.table("Escalations from the emergency department (descriptive)",
        ["Patients", "Draws per hour, baseline", "Draws per hour, pre-event", "Tests per hour, baseline", "Tests per hour, pre-event"],
        [[f"{e['n']}", num(e["draws_base_per_h"], 3), num(e["draws_pre_per_h"], 3), num(e["tests_base_per_h"], 2), num(e["tests_pre_per_h"], 2)]],
        [1.0, 1.4, 1.4, 1.4, 1.4], size=9, note="No controls; reported for contrast only.")

D.head("5. Hour-by-hour profile and clock-hour distribution")
pr = M["profile"]
c = pr["cases_mean_draws_by_hour_before_event"]; k = pr["controls_mean_draws_by_hour_before_event"]
rows = []
for i in range(24):
    rows.append([f"{i}–{i + 1}", num(c[i], 3), num(k[i], 3), f"{i + 24}–{i + 25}", num(c[i + 24], 3), num(k[i + 24], 3)])
D.table("Mean specimen draws per patient in each hour before the index time (data for Figure 3 of the main text)",
        ["Hours before index", "Cases", "Controls", "Hours before index", "Cases", "Controls"], rows, [1.2, 0.9, 0.9, 1.2, 0.9, 0.9], size=8,
        note="Hour 0–1 is the final hour before intensive-care entry (cases) or the matched pseudo-index time (controls).")
D.d.add_picture(FIG_S1, width=Inches(6.3))
D.para(f"Figure S1. Share of specimen draws by clock hour of collection in the 48 hours before the index time, MIMIC-IV, for ward-to-intensive-care cases "
       f"({co['n_ward_escalations']:,}) and matched ward controls ({co['n_controls']:,}). Draws cluster in the early-morning hours in both groups, which is why both windows "
       "span the full day and controls are matched on the clock time of the index.", size=10, after=10)
cs = pr["cases_share_of_draws_by_clock_hour"]; ks = pr["controls_share_of_draws_by_clock_hour"]
rows = [[f"{h:02d}:00", pp(cs[h]), pp(ks[h]), f"{h + 12:02d}:00", pp(cs[h + 12]), pp(ks[h + 12])] for h in range(12)]
D.table("Share of draws by clock hour (%), data for Figure S1", ["Hour", "Cases", "Controls", "Hour", "Cases", "Controls"], rows,
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], size=8)

D.head("6. Record of the first full-data run and the corrections made")
D.para("The analysis plan (cohort definitions, matching criteria, windows, outcomes, estimand, seed, bootstrap count and decision rule) was fixed and tested on the open "
       "demonstration data before any full-data run. The first full-data run (script version 1.2) completed, but its built-in validation and an inspection of its "
       "matching showed two implementation defects. Both were corrected before the final run, without changing any element of the plan. All results in the paper come from "
       "the corrected version (script 1.2.2).")
D.table("Defects found in the first full-data run and their correction",
        ["Defect", "Evidence in the first run", "Correction", "Verification"],
        [["Ward runs were not always ended by a short non-ward record, so a small fraction of controls passed matching but failed the independent check of 48-hour ward continuity.",
          f"Independent continuity check: controls {pp(F['matching']['controls_48h_continuous_ward_share'], 2)}%, cases {pp(F['matching']['cases_48h_continuous_ward_share'])}%.",
          "Version 1.2.1: every non-ward record, including zero-length and overlapping records, ends a ward run.",
          "Unit tests on constructed transfer sequences; agreement with the independent checker in 12,000 random checks."],
         ["The control search examined at most 5,000 randomly ordered candidates per case, so a case could be left with fewer than two controls although eligible controls existed.",
          f"Cases with 2 / 1 / 0 controls: {F['matching']['cases_with_2_controls']:,} / {F['matching']['cases_with_1_control']} / fewer than 10 (suppressed).",
          "Version 1.2.2: exhaustive search over all candidates, with the reason recorded for any case receiving fewer than two controls.",
          "Eligible sets equal a brute-force search on synthetic data; exact clock match and caliper confirmed for every selected control."]],
        [1.8, 1.6, 1.6, 1.5], size=8,
        note="Self-test output (m1_mimic/results/M1_v122_selftest_output.txt): all checks passed, including the hand-worked example for equal-set weighting (correct answer 2.75; "
             "pooled answer 2.0). The corrected script was also run on the open demonstration data before the final run.")
f0 = F["primary_24h"]["draws"]; f1 = M["primary_24h"]["draws"]
rows = [["Cases", f"{F['cohort']['n_ward_escalations']:,}", f"{co['n_ward_escalations']:,}"],
        ["Controls", f"{F['cohort']['n_controls']:,}", f"{co['n_controls']:,}"],
        ["Controls passing the independent 48-hour ward check", f"{pp(F['matching']['controls_48h_continuous_ward_share'], 2)}%", f"{pp(mt['controls_48h_continuous_ward_share'], 2)}%"],
        ["Cases with two controls", f"{F['matching']['cases_with_2_controls']:,}", f"{mt['cases_with_2_controls']:,}"],
        ["Specimen draws, cases (baseline → pre-event)", f"{num(f0['case_base_per_h'], 3)} → {num(f0['case_pre_per_h'], 3)}", f"{num(f1['case_base_per_h'], 3)} → {num(f1['case_pre_per_h'], 3)}"],
        ["Specimen draws, controls (baseline → pre-event)", f"{num(f0['control_base_per_h'], 3)} → {num(f0['control_pre_per_h'], 3)}", f"{num(f1['control_base_per_h'], 3)} → {num(f1['control_pre_per_h'], 3)}"],
        ["Difference-in-differences (95% CI)", f"{num(f0['did_per_h'], 3, True)} ({num(f0['did_ci95'][0], 3)} to {num(f0['did_ci95'][1], 3)})",
         f"{num(f1['did_per_h'], 3, True)} ({num(f1['did_ci95'][0], 3)} to {num(f1['did_ci95'][1], 3)})"],
        ["Ratio of ratios (95% CI)", f"{num(f0['ratio_of_ratios'], 2)} ({num(f0['ratio_of_ratios_ci95'][0], 2)} to {num(f0['ratio_of_ratios_ci95'][1], 2)})",
         f"{num(f1['ratio_of_ratios'], 2)} ({num(f1['ratio_of_ratios_ci95'][0], 2)} to {num(f1['ratio_of_ratios_ci95'][1], 2)})"]]
D.table("First full-data run (version 1.2) and final run (version 1.2.2), primary analysis",
        ["Quantity", "First run (v1.2)", "Final run (v1.2.2)"], rows, [3.0, 1.75, 1.75], size=9,
        note="The first run is disclosed for transparency only; it is not used for any inference. The conclusion is the same in both runs.")

out = os.path.join(OUT, "SiM_Supporting_Information_S5_MIMIC_IV.docx")
D.save(out); print("saved", out)
