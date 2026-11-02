# -*- coding: utf-8 -*-
import hashlib, os
import os
from si_common import *

A1 = load(A1J); S = load(A1F)
cfg = DEP + "/a1_simulation/rebuild_v322/config/"
D = Doc("S2", "Pre-specified simulation protocol, configuration fingerprints and deviation record")

D.para("This document summarises every pre-specified element of the simulation study that the main text relies on. "
       "The protocol, its machine-readable configuration, the code, the unit tests and all outputs are archived at "
       f"{ZEN} (folder a1_simulation/rebuild_v322/config for the protocol and configuration files). Table S2.1 gives the fingerprints "
       "of the frozen files used for the runs and of their archived copies.")

D.head("1. How the protocol was fixed")
D.bullets([
    "The study was specified in three nested, hash-locked layers. A base configuration (Phase 1) fixed the data-generating models, "
    "observation schemes, analyses, tests and random-number streams and was used to select the null method by a calibration run. "
    "A Phase 2 configuration added the production blocks without changing any Phase 1 value. Addendum A1, version 1.2, added the "
    "models, arm, tests and blocks reported in the main text without changing any Phase 2 value.",
    "Addendum A1 v1.2 was frozen on 30 October 2026, before any A1 production run. The SHA-256 of its machine-readable file is "
    "hard-coded in the simulation code, which refuses to run if the file differs; the parent and grandparent configuration hashes "
    "are checked in the same way.",
    "Production runs started only after a Stage 0 check: unit tests, a discarded plumbing run, identity checks against completed "
    "blocks, and a timing pilot whose projected runtime was approved before Stage 1.",
    "The protocol fixed in advance that no null method, margin or decision rule would be changed after any A1 result was seen, "
    "that completed outputs would never be overwritten, and that no model would be tuned on the uneven (informative) arm.",
])

files = [("Simulation_Rebuild_Config_v3.2.2_Phase1.json", "Base configuration (Phase 1)", "9a245eb092a47bc5e2851f0b681ec439c1a72ab34ed2fe3dedf63b9b29bff482"),
         ("Simulation_Rebuild_Protocol_v3.2.2_Phase1.pdf", "Base implementation specification (Phase 1)", "160e7d36ce86900bca29c11c3944302515f30da27efc57c7b7bab63064a8341e"),
         ("Simulation_Rebuild_Config_v3.2.2_Phase2.json", "Phase 2 configuration (parent of A1)", "0473d4830407f1fde68988d32fb1d2ad14537b815c0fd34492b8391d084950d3"),
         ("Simulation_Addendum_A1_v1.2_FROZEN.json", "Addendum A1 v1.2, machine-readable", "9aff4ef745296e32b6b888b3cc32d08861aec3affe71a8153a88409922b4d7e5"),
         ("Protocol_Addendum_A1_v1.2_FROZEN.pdf", "Addendum A1 v1.2, protocol text", "a736da98f83c09f0255963e138e01ef406c71c05e0b612f3ae495498f51b937b")]
rows = []
for f, role, orig in files:
    h = hashlib.sha256(open(cfg + f, "rb").read()).hexdigest()
    rows.append([f, role, orig, "Identical" if h == orig else h])
assert files[3][2] == S["a1_sha256"] and files[2][2] == S["config_sha256"] == A1["config_lineage"]["parent_config_sha256"]
assert files[0][2] == A1["config_lineage"]["grandparent_config_sha256"]
assert rows[0][3] == rows[1][3] == rows[2][3] == "Identical"
D.table("Frozen protocol and configuration files (archived in a1_simulation/rebuild_v322/config)",
        ["File", "Role", "SHA-256 of the frozen file used for the runs", "SHA-256 of the archived copy"], rows, [1.6, 1.25, 1.8, 1.85], size=7,
        note="The run records, manifests and the summary used for every reported number carry the fingerprints of the frozen files. In the archived copies, "
             "administrative remarks (approval and process notes) were removed from the A1 configuration file and protocol text, so their fingerprints differ; "
             "no value, rule or definition changed. Every such change is listed with both fingerprints in REDACTIONS.md in the archive, rerunning the archived "
             "summariser reproduces every reported result (Supporting Information S4), and the unaltered files are available from the author on request. "
             "The SHA-256 of every simulation code module is listed in Table S2.10.")

D.head("2. Questions and estimands")
D.para("All estimands are paired within simulated patient (the same latent path observed under different arms) and expressed as "
       "differences in detection proportions in percentage points; a patient who is not evaluable is counted as not detected.")
D.table("Scientific questions, estimands and the blocks that answer them",
        ["Question", "Estimand (each indicator: variance, lag-1 autocorrelation)", "Blocks"],
        [["Q1. Does timing change detection at a matched number of observations?",
          "Total: uneven − even (raw series, timing-aware test). Irregularity: random-time − even. Escalation: uneven − random-time.",
          "F1 (fresh-seed replication); XS1, XR1–XR10, XS2"],
         ["Q2. Can preprocessing or a continuous-time estimator recover detection?",
          "Recovery: detection with representation r and test n minus detection on the raw series with the same test; gap to even sampling: "
          "minus detection of the timing-aware test on even raw data; each read against the false-alarm rate of (r, n).",
          "P5 and P5FA (AR(1) and shuffle on 13 representations); E2 and E2FA (continuous-time estimator)"],
         ["Q3. Does a timing-aware test detect better?",
          "Gain: timing-aware minus timestamp-blind detection on the uneven raw series; specificity: gain under uneven minus gain under even "
          "sampling; read against the false-alarm rates of every test.",
          "G1a–G1d, F1 (confirmatory), XN1, XN2, all scenario blocks"]],
        [2.0, 3.0, 1.5])

D.head("3. Components added by Addendum A1")
D.table("New models, arm and tests",
        ["Component", "Definition", "Purpose"],
        [["Stable slow process, θ = 0.2", "Stationary OU, θ fixed at 0.2, μ = 0, σ = 0.3, X(0) from the stationary distribution; same measurement error, baseline and schedules as the primary model.",
          "False alarms in the slow regime, where spacing matters most (the fitted θ of the timing-aware test in transition series was about 0.15–0.19)."],
         ["Stable slow process, θ = 0.05", "As above with θ = 0.05, the value at the end of the transition.", "Bounds the slow regime from below."],
         ["Fold model (revised)", "Saddle-node model with noise σf and end-point r_end chosen from {0.05, 0.07, 0.10} × {0.05, 0.10} by a pre-specified pilot rule.",
          "Bifurcation-type robustness check, run only if the pilot finds an adequately powered setting."],
         ["Random-time arm", "The 8 shared baseline observations, then Ni − 8 times drawn independently and uniformly over [15, 100) and sorted, with its own time and error streams.",
          "Exactly count-matched timing that is irregular but does not escalate; separates irregularity from escalation."],
         ["AR(1) and shuffle on a representation", "Regularise first (z = R(t, y)), then fit the AR(1) surrogate to z or permute z, and compute the window statistics directly on the surrogate series.",
          "Evaluates the common practice of regularising the data before applying a standard test."]],
        [1.6, 2.7, 2.2])

D.head("4. Hierarchy of evidence, multiplicity and margins")
D.table("Inference rules fixed before any A1 result",
        ["Level", "What", "Rule"],
        [["Gate", "Timing-aware test CALIBRATED for variance and autocorrelation in all three arms of the stable slow process (θ = 0.2, m = 2; 2,000 patients).",
          "Failure means no Q3 claim; H1 and H2 would then be descriptive only."],
         ["Confirmatory", "H1: under uneven sampling, variance-based detection is higher with the timing-aware test than with the AR(1) test. H2: that gain is larger under uneven than under even sampling. Block F1: 2,000 patients never analysed before.",
          "Family-wise α 0.05 by Bonferroni: each judged on its two-sided 97.5% paired bootstrap interval (10,000 resamples of patients). Supported if the lower bound exceeds 0; material if it exceeds +5 points."],
         ["Pre-specified secondary", "Families S-Q1 (replication and decomposition, scenario blocks), S-Q2 (preprocessing and continuous-time), S-Q3 (gains by test across scenario blocks), S-FA (false-alarm comparisons).",
          "Unadjusted 95% intervals, with Holm-adjusted exact McNemar p-values within each family."],
         ["Exploratory", "Anything not listed above.", "Estimates and intervals only; never the basis of a claim."]],
        [1.2, 3.0, 2.3],
        note="Practical margin: ±5 percentage points, kept as frozen in Phase 2 (at detection near 50%, a 10% relative change; close to the resolution of a 1,000-patient paired design). "
             "Interpretation categories are also reported at ±3 and ±10 points. Mean Kendall τ is reported next to detection for every comparison. "
             "Scenario blocks share patients and latent paths with each other by design (common random numbers); agreement across them is reported as consistency, "
             "not as independent replication. Only F1 is independent.")

D.head("5. Calibration labels, size adjustment and classification of detection changes")
D.para("Every test was labelled in every null-process block, arm and indicator from the 90% Clopper–Pearson interval [L, U] of its false-alarm rate, and the labels governed the wording of every comparative statement.")
D.table("Calibration labels", ["Label", "Rule", "Consequence for claims"],
        [["CALIBRATED", "L ≥ 2.5% and U ≤ 7.5% (two one-sided tests against 2.5–7.5%)", "May enter a claim at controlled false-alarm rates."],
         ["LOW", "U ≤ 7.5% and L < 2.5%", "Conservative; comparisons against it are worded as such."],
         ["HIGH_NOT_EXCLUDED", "U > 7.5% and L ≤ 7.5%", "False-alarm control not established; cannot support a superiority or recovery claim."],
         ["HIGH", "L > 7.5%", "False-alarm rate demonstrably above the bound; the only label that permits 'at the cost of excess false alarms'."]],
        [1.5, 2.6, 2.4],
        note="Reference blocks: for Q3, G1a (primary) and XN2 (θ = 1.2); for Q2, P5FA for the AR(1) and shuffle representations and E2FA for the continuous-time estimator.")
D.para("Size-adjusted analysis (pre-specified, and the basis of interpretation whenever a compared test was not CALIBRATED): for each test and arm, α* is the largest threshold on the Monte Carlo grid k/201 that gives a false-alarm rate of at most 5% in G1a for the same test, arm and indicator; F1 detection is reclassified as τ > 0 and p ≤ α*, and H1 and H2 are recomputed with their 97.5% intervals. The analysis is repeated with XN2 as reference. The bootstrap does not re-estimate α*.")
D.table("Four-way classification of any change in detection (Q2 recovery and Q3 gains)",
        ["Class", "Condition", "Permitted wording"],
        [["No recovery", "95% interval of the change includes 0 or lies below it", "No change, or loss"],
         ["Improvement, false-alarm control not established", "Lower bound > 0; method HIGH_NOT_EXCLUDED, or false-alarm result unavailable", "Detection improvement with false-alarm control not established; never 'recovery'"],
         ["Genuine recovery", "Lower bound > 0; method CALIBRATED or LOW in the matching null block (same arm and indicator)", "Recovers detection (material if the lower bound exceeds +5)"],
         ["Improvement at false-alarm cost", "Lower bound > 0; method HIGH", "Detection improvement at the cost of excess false alarms"]],
        [1.9, 2.6, 2.0])

D.landscape(True)
D.head("6. Blocks")
mod = {"stable_slow_0.2": "Stable slow, θ = 0.2", "stable_slow_0.05": "Stable slow, θ = 0.05", "ou_csd": "OU with slowing (primary)",
       "ou_combined": "Slowing with level drop", "level_drop_only": "Level drop, no slowing", "stable": "Stable, θ = 1.2",
       "fold_v2_grid": "Fold model, pilot grid", "fold_v2": "Fold model, selected setting"}
trig = {"primary": "primary", "indep": "independent", "single": "single crossing", "k2.0": "threshold 2.0 SD", "k2.5": "threshold 2.5 SD",
        "oracle": "oracle", "none": "none (smooth)"}
P2 = {b["id"]: b for b in load(P2J)["phase2_matrix"]["blocks"]}
role = {"gate": "Gate", "secondary_scope": "Scope (S-FA)", "confirmatory": "Confirmatory", "secondary": "Secondary",
        "secondary_false_alarm": "False alarms (S-FA)", "pilot": "Pilot (selection only)"}
rows = []
for b in A1["blocks"]:
    base = P2.get(b.get("base_block")) or P2.get(b.get("frozen_block")) or {}
    m_ = b.get("model", base.get("model")); sch = b.get("schedule", base.get("schedule"))
    mm = b.get("m", base.get("m")); tr = b.get("trigger", base.get("trigger"))
    ids = b.get("patient_ids", base.get("patient_ids"))
    n = ids[1] - ids[0] + 1
    design = f"{mod.get(m_, m_)}; {sch}" + (f", m = {mm:g}" if mm else "") + f"; trigger {trig.get(tr, tr)}"
    arms = ", ".join(ARM.get(a, a) for a in b.get("arms", base.get("arms", ["regular_N", "informative_reduced_N"]))) if b.get("arms") else "Even, Uneven"
    if b["id"] == "E2":
        tests = "Continuous-time estimator, width 30; timing-aware test"
    elif b.get("analyses") and len(b["analyses"]) > 1:
        tests = "AR(1), shuffle on 13 representations"
    elif b["id"] == "E2FA":
        tests = "Continuous-time estimator, width 30; timing-aware test"
    else:
        tests = ", ".join(TEST[t] for t in b["nulls"])
    seed = b.get("seed_base", 3000000)
    status = S["block_status"].get(b["id"], "selection record")
    if b["id"] == "FOLDP":
        n_txt = "300 per grid point (6 points)"
    else:
        n_txt = f"{n:,} (ids {ids[0]}–{ids[1]})"
    rows.append([b["id"], role.get(b["role"], b["role"]), design, arms, tests, n_txt, f"{seed:,}",
                 "Run" if status == "ok" else ("Not run (pilot rule)" if status == "not run" else "Run")])
D.table("Every block in Addendum A1, as frozen",
        ["Block", "Role", "Design", "Arms", "Tests", "Patients", "Seed base", "Status"], rows,
        [0.55, 1.05, 2.5, 1.35, 1.45, 1.15, 0.75, 0.75], size=7.5,
        note="Random numbers: entropy = seed base + patient id, with named streams; blocks with the same seed base and patient ids share latent paths. "
             "Scenario blocks XS1, XR1–XR10, XS2, XN1 and XN2 extend completed Phase 2 blocks (S1, R1–R10, S2, N1, N2) and had to reproduce their completed rows exactly (Supporting Information S4). "
             "Thinned arm: the uneven scheme with post-escalation events thinned stochastically by 1/m (a Phase 2 secondary arm; not inverse-intensity weighting). "
             "E2 was run exactly as frozen in Phase 2 (patients 0–499). The fold pilot found no eligible setting, so FOLD2 was not run (Supporting Information S3).")
D.landscape(False)

D.head("7. What each result was allowed to support")
D.table("Claim rules fixed before any A1 result",
        ["Result", "Permitted", "Not permitted"],
        [["Gate fails", "Q1 and Q2 findings only; timing-aware test described as uncalibrated in the slow regime.", "Any recommendation of the timing-aware test."],
         ["Gate passes; H1 and H2 supported; timing-aware and comparator tests CALIBRATED in G1a and XN2", "Timestamp-blind tests lose variance detection under irregular sampling and a timing-aware test recovers it at controlled false-alarm rates.", "That the timing-aware test rescues autocorrelation (unless shown); that the effect is state-dependent (unless the rule below is met)."],
         ["H1 and H2 supported; comparator LOW", "The timing-aware test detects more than a conservative timestamp-blind test, judged on the size-adjusted analysis.", "'At controlled false-alarm rates' for the raw comparison."],
         ["H1 and H2 supported; comparator HIGH_NOT_EXCLUDED or HIGH", "Only what the size-adjusted analysis supports.", "Any superiority claim from the raw comparison."],
         ["H1 only", "The timing-aware test is more powerful in this setting.", "That the gain is specific to irregular sampling."],
         ["Escalation component under the independent trigger similar to the primary design", "'A rise in sampling density'.", "'State-dependent timing'. State dependence may be claimed only if the escalation components differ with a 95% interval excluding 0."],
         ["Slowing with level drop: gain below 5 points", "Q3 claim restricted to constant-mean processes.", "An unrestricted Q3 claim."],
         ["Preprocessing or continuous-time estimator", "Wording set by the four-way classification (Table S2.6).", "'Recovers detection' without genuine recovery."]],
        [2.0, 2.5, 2.0])

D.head("8. Reproducibility procedures fixed in the protocol")
D.bullets([
    "Inputs: the A1 file hash is checked by the code at start, together with the parent and grandparent hashes; the SHA-256 of every code module is written to each block's manifest.",
    "Seeds and streams: seed base and patient ids per block as listed; every stream name planned and checked unique before the run; a run that uses an unplanned stream aborts without a record.",
    "Outputs: per block, patient-level rows (every patient × arm × analysis × test, τ and p for both indicators, evaluability, counts, escalation time), a manifest (command, software versions, platform, hashes), an execution log, and an immutable completion record written last.",
    "Identity checks: before summarising, every block that extends a completed block must reproduce the completed rows exactly, and the preprocessing blocks must reproduce the earlier exploratory runs on the same patients; any mismatch stops the analysis.",
    "Code checks before production: unit tests for every new component, a discarded plumbing run, and an independent code review.",
    "Independent numerical verification: the summary is recomputed from the patient-level rows by a separately written script.",
    "Deviations: an append-only record, one dated entry per departure from the protocol, written before the affected run.",
])

D.head("9. Deviation record")
dv = S["deviations"]
assert len(dv) == 1
D.table("Deviations from the frozen protocol (complete record)",
        ["No.", "Block", "Recorded", "Deviation"],
        [["1", "E2FA", "31 October 2026, before the run",
          "The frozen exploratory configuration listed the continuous-time estimator for 2,000 null patients (about 69 million window fits). "
          "The false-alarm block was run on patients 0–999. At n = 1,000 the equivalence test passes for observed rates of 3.4–6.1% "
          "(probability 0.94 if the true rate is 5%). The protocol allowed a further reduction to 500 only with recorded approval and only as a strictly descriptive result; "
          "no such reduction was made."]],
        [0.4, 0.6, 1.3, 4.2],
        note="This is the only entry in the archived deviations.json. The main text reports the continuous-time false-alarm block with 1,000 patients.")

code = S["code_sha256"]
rows = [[k, v] for k, v in sorted(code.items())]
rows.append(["a1_summary/summarize.py (summariser)", S["summariser_sha256"]])
D.table("SHA-256 of the simulation code modules (from the A1 summary file)", ["Module", "SHA-256"], rows, [2.4, 4.1], size=7.5,
        note=f"Fingerprints of the code that produced the results. Software recorded in the summary: Python {S['software']['python']}, NumPy {S['software']['numpy']}, SciPy {S['software']['scipy']}. "
             "In the archived copies of a1_run.py, a1_spec.py, phase2.py, spec.py and the summariser, comment lines and administrative notes were removed, one file path was "
             "updated to a renamed configuration file, and the check of run records was pointed at the frozen A1 fingerprint; no computation changed (REDACTIONS.md).")

out = os.path.join(OUT, "SiM_Supporting_Information_S2_Protocol.docx")
D.save(out); print("saved", out)
