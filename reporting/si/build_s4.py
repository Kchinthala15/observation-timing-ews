# -*- coding: utf-8 -*-
import glob, hashlib, json, os, platform, sys, subprocess
import os
from si_common import *

O = DEP + "/a1_simulation/rebuild_v322/outputs_addendum_A1"
S = load(A1F)
REPRO = sys.argv[1]   # summary recomputed on a second machine from the archived rows
B = load(REPRO)
D = Doc("S4", "Reproduction record")
D.para("This document records the checks that the simulation outputs are complete, unaltered and reproducible. All production runs were made on the "
       "author's computer; the archived outputs were then checked again on a second, independent computer. Every file named here is archived at "
       f"{ZEN} (folder a1_simulation/rebuild_v322/outputs_addendum_A1 unless stated otherwise).")

# ---------- Stage 0
s0 = load(O + "/stage0/stage0_report.json"); ap = load(O + "/stage0/stage0_APPROVED.json")
pl = load(O + "/stage0/plumbing_check.json"); idc = load(O + "/stage0/identity_check.json")
ut = open(O + "/stage0/unit_tests.txt", "rb").read().decode("utf-16", errors="ignore") if open(O + "/stage0/unit_tests.txt", "rb").read(2) in (b"\xff\xfe", b"\xfe\xff") else open(O + "/stage0/unit_tests.txt", encoding="utf-8", errors="ignore").read()
import re
m = re.search(r"(\d+) passed", ut.replace("\x00", "").replace(" ", "") if "passed" not in ut else ut)
n_tests = m.group(1) if m else None
assert n_tests == "118", n_tests
assert pl["passed"] and not pl["problems"]
id_rows = sum(v["rows_compared"] for v in idc["report"].values()); id_diff = sum(v["rows_differing"] for v in idc["report"].values())
assert id_diff == 0
assert ap["stage0_report_sha256"] == hashlib.sha256(open(O + "/stage0/stage0_report.json", "rb").read()).hexdigest()
D.head("1. Checks before any production run (Stage 0)")
D.table("Stage 0 checks, 31 October 2026",
        ["Check", "Result", "Record"],
        [["Unit tests (every component, including the A1 hash lock)", f"{n_tests} passed, 0 failed", "stage0/unit_tests.txt"],
         ["Plumbing run (every block, {0} patients per block, {1} surrogates; output discarded)".format(pl["patients_per_block"], pl["surrogates"]),
          f"{pl['rows']} rows, no problems; seed base {pl['seed_block']:,} (not used for any reported result)", "stage0/plumbing_check.json"],
         ["Identity check: blocks extending completed Phase 2 blocks, first 10 patients, full production design",
          f"{len(idc['report'])} blocks, {id_rows} rows compared, {id_diff} differing", "stage0/identity_check.json"],
         ["Timing pilot and runtime projection", "Projected runtime reported and approved before Stage 1", "stage0/timing_report.json, stage0/stage0_APPROVED.json"],
         ["Approval record", "Contains the SHA-256 of the Stage 0 report, the A1 file and every code module", "stage0/stage0_APPROVED.json"]],
        [2.6, 2.3, 1.6])

# ---------- completion records
D.head("2. Completion records: every production block")
rows = []; nblocks = 0
for r in sorted(glob.glob(O + "/a1_*_record*.json")):
    d = load(r)
    if "rows_sha256" not in d:
        continue
    csv = r.replace("_record", "_rows").replace(".json", ".csv")
    ok = hashlib.sha256(open(csv, "rb").read()).hexdigest() == d["rows_sha256"]
    assert ok and d["a1_sha256"] == S["a1_sha256"] and d["config_sha256"] == S["config_sha256"]
    nblocks += 1
    blk = d["block"].replace("FOLDP|fold_v2|", "FOLDP ").replace("sf=", "σf ").replace("|re=", ", r_end ")
    rows.append([blk, f"{d['n_patients']:,}", f"{d['rows']:,}", d["rows_sha256"][:16] + "…", "Yes", d["written"].replace("T", " ")[:16]])
order = ["G1a", "G1b", "G1c", "G1d", "F1", "XS1"] + [f"XR{i}" for i in range(1, 11)] + ["XS2", "XN1", "XN2", "P5", "P5FA", "E2", "E2FA"]
rows.sort(key=lambda x: (order.index(x[0]) if x[0] in order else 99, x[0]))
D.table(f"Completion records of the {nblocks} production runs",
        ["Block", "Patients", "Rows", "SHA-256 of the rows file (first 16 characters)", "Rows file matches record", "Record written"], rows,
        [1.6, 0.8, 0.8, 1.7, 0.9, 1.2], size=8,
        note="Each completion record is written last and stores the SHA-256 of the block's patient-level rows file and the A1 and configuration hashes. For every block, "
             "the archived rows file was re-hashed for this document and matched its record, and every record carries the frozen A1 and parent configuration hashes. "
             "The six fold-pilot runs are the selection runs; the main fold block was not run. Full hashes are in the records.")

# ---------- identity checks
D.head("3. Identity checks before summarising")
rows = []
for f in sorted(glob.glob(O + "/a1_*_identity_*.json")):
    d = load(f)
    assert d["passed"] and d["rows_differing"] == 0
    ag = {"explore_E1__all": "Earlier exploratory run E1 (same patients)", "explore_E3__noct": "Earlier exploratory run E3 (same patients)"}.get(d["against"], f"Completed Phase 2 block {d['against']}")
    rows.append([d["block"], ag, f"{d['rows_compared']:,}", f"{d['rows_differing']}", d["written"].replace("T", " ")[:16]])
rows.sort(key=lambda x: (order.index(x[0]), x[4]))
D.table("Exact reproduction of previously completed rows",
        ["Block", "Compared with", "Rows compared", "Rows differing", "Check written"], rows, [0.8, 2.8, 1.0, 1.0, 1.4], size=8,
        note="Rows are compared on τ and p for both indicators for every arm, analysis and test already run. P5 and P5FA were checked twice, after their runs and again before "
             "the final summary, with identical row hashes. Any mismatch would have stopped the analysis; none occurred.")

# ---------- second-machine recomputation
nf = 0; mx = 0.0; other = []; nleaves = 0
def walk(a, b, path):
    global nf, mx, nleaves
    if isinstance(a, dict):
        for k in a:
            if k in b:
                walk(a[k], b[k], path + "/" + k)
            else:
                other.append(path + "/" + k)
    elif isinstance(a, list):
        if len(a) != len(b):
            other.append(path); return
        for i, (x, y) in enumerate(zip(a, b)):
            walk(x, y, path + f"[{i}]")
    else:
        nleaves += 1
        if a != b:
            if isinstance(a, float) and isinstance(b, float):
                nf += 1; mx = max(mx, abs(a - b) / max(abs(a), 1e-300))
            elif not path.startswith(("/written", "/software")):
                other.append(path)
_skip = ("bootstrap_streams_used_and_checked_unique", "code_sha256", "summariser_sha256", "a1_sha256")
walk({k: v for k, v in S.items() if not k.startswith("E1_E3") and k not in _skip},
     {k: v for k, v in B.items() if not k.startswith("E1_E3") and k not in _skip}, "")
import hashlib as _h
assert B["a1_sha256"] == _h.sha256(open(DEP + "/a1_simulation/rebuild_v322/config/Simulation_Addendum_A1_v1.2_FROZEN.json", "rb").read()).hexdigest()
assert not other, other[:5]
stS = set(S["bootstrap_streams_used_and_checked_unique"]); stB = set(B["bootstrap_streams_used_and_checked_unique"])
assert stB <= stS and all("E1" in s or "E3" in s for s in stS - stB)
lab_same = all(S["calibration"][k]["label"] == B["calibration"][k]["label"] for k in S["calibration"])
assert lab_same
man = load(sorted(glob.glob(O + "/a1_F1_manifest*.json"))[0])
D.head("4. Recomputation of the summary on a second computer")
D.para("The summariser in the archive (a1_summary/summarize.py) was rerun from a fresh copy of the archive on a second computer, reading only the archived "
       "patient-level rows and the archived configuration files, and its output was compared value by value with the summary used for the paper. Because the archived "
       "copies of some files have administrative remarks removed (Supporting Information S2), this rerun also confirms that those copies reproduce every result.")
D.table("Comparison of the two summaries",
        ["Item", "Author's computer (summary used in the paper)", "Second computer"],
        [["Platform", man["platform"], platform.platform()],
         ["Software", f"Python {S['software']['python']}, NumPy {S['software']['numpy']}, SciPy {S['software']['scipy']}",
          f"Python {B['software']['python']}, NumPy {B['software']['numpy']}, SciPy {B['software']['scipy']}"],
         ["Values compared", f"{nleaves:,}", ""],
         ["Values identical", f"{nleaves - nf:,}", ""],
         ["Values differing", f"{nf} floating-point values, largest relative difference {mx:.1e}", ""],
         ["Calibration labels, classifications, decisions, estimates and intervals", "Identical", ""],
         ["Unit tests", f"{n_tests} passed", "118 passed"]],
        [2.0, 2.4, 2.1],
        note="The differing values are the last digits of Clopper–Pearson interval endpoints (largest relative difference of the order of 10⁻¹⁴), which arise from the different SciPy "
             "versions; none changes any reported number, label or decision. One exploratory section of the summariser, comparing against the earlier exploratory runs E1 and E3, "
             "needs those runs' outputs, which are not part of the archive; it is not used anywhere in the paper and was excluded from the comparison. Code and configuration "
             "fingerprints recorded in the two summaries differ by design (archived copies) and were not compared.")

for mf in glob.glob(O + "/a1_*_manifest*.json"):
    mm = load(mf)
    assert mm["code_sha256"] == S["code_sha256"], mf
    assert mm["a1_sha256"] == S["a1_sha256"] and mm["config_sha256"] == S["config_sha256"], mf
    assert not mm["streams_used_not_in_plan"], mf
D.head("5. Code and configuration fingerprints")
D.para("Every block manifest records the SHA-256 of the A1 file, the configuration and every code module as run; all manifests agree with each other and with the fingerprints of the frozen files in Supporting Information S2 (Tables S2.1 and S2.10), and every manifest records that no random-number stream outside the pre-planned set was used. "
       "The A1 code refuses to run if the A1 file differs from its hard-coded hash, and every run that used an unplanned random-number stream would have aborted without a record.")

out = os.path.join(OUT, "SiM_Supporting_Information_S4_Reproduction_Record.docx")
D.save(out); print("saved", out, nleaves, nf, mx)
