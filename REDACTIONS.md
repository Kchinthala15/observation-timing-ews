# Changes from the frozen files (version 3.1.0)

Version 3.1.0 replaces version 3.0.0. Administrative remarks (approval and process notes) were removed from some files, two configuration files were renamed, and two documents were rebuilt without such remarks. No configuration value, rule, definition, computation or output changed. Rerunning the summariser in this archive reproduces every reported estimate, interval, calibration label, classification and decision.

Run records, manifests and the frozen summary (`a1_simulation/results/a1_summary_20261031_220543.json`) were written at run time and therefore carry the fingerprints of the frozen files listed below in the column *Frozen SHA-256*. The unaltered frozen files are available from the author on request.

| File | Frozen SHA-256 | Archived SHA-256 | Change |
|---|---|---|---|
| `config/Simulation_Rebuild_Config_v3.2.2_Phase1.json` | `9a245eb092a47bc5e2851f0b681ec439c1a72ab34ed2fe3dedf63b9b29bff482` | `9a245eb092a47bc5e2851f0b681ec439c1a72ab34ed2fe3dedf63b9b29bff482` | Renamed from its earlier file name; content unchanged |
| `config/Simulation_Rebuild_Protocol_v3.2.2_Phase1.pdf` | `160e7d36ce86900bca29c11c3944302515f30da27efc57c7b7bab63064a8341e` | `160e7d36ce86900bca29c11c3944302515f30da27efc57c7b7bab63064a8341e` | Renamed from its earlier file name; content unchanged |
| `ewsv322/spec.py` | `bfa7f5a41ea48a5a867485c3c08cdfdedd74258fa65a4f8c7ddc251460860e02` | `11ddec2075ba9bb273ae19cb9576d78b011f23ef34cf49cf7f70e81e92a65952` | Docstring and file path updated to the renamed Phase 1 configuration |
| `ewsv322/phase2.py` | `3f72e6deaa058638099f8a979af5e605e4cbb07f5084c61fc90f8b3c8d39409d` | `8e86b35bab03f3b6105561a855596af1a69287c3ac9c3bf1a97d240212102de3` | Comment wording only |
| `ewsv322/a1_run.py` | `e9c7888f99cff91c03bce55573b75ba9be559e0586a13cd91c3fa79ae7cd8854` | `d8898ac43b371e213ed4db32e28af10352602570bb75c7fc9caea221b8ebd3cf` | Comment and note text only; Run-record check compares against the frozen A1 hash |
| `outputs_addendum_A1/stage0/stage0_APPROVED.json` | `f283b247024c796fb07fcec74cba629a44434c3807bc2e7620b6f9cc6d0a99f5` | `b8551f94bfe2a0245858927778afefed9ff127038512382f68d32bf501463d30` | Note text only |
| `config/Simulation_Addendum_A1_v1.2_FROZEN.json` | `9aff4ef745296e32b6b888b3cc32d08861aec3affe71a8153a88409922b4d7e5` | `07e896f9bc70eb1356efbfcbfce08b0e6e06d470c28485f5ed592d66d0bffe4b` | Administrative text fields only (revision_history, status, config_lineage.rule); every key and every scientific value unchanged |
| `ewsv322/a1_spec.py` | `ff9e43d73887ca442fe50d86f4395493040e06b6b2b659791c2badf5d3c39a5a` | `fefaa8770923093c41f92ec097a2958bbcee4cf376dc91af5a69a56d8e59ab1e` | Hash of this archived A1 copy set as the hash lock; the frozen A1 hash added for checking run records |
| `a1_summary/summarize.py` | `97df495d8b23a66aa3933353b0032b25f534aa9999fe8330e3bd07048b7c94fd` | `d6aa1ed6499a0f644e33a521df9f33cc75a82c10f1a9bee9f8d5a11592ca135c` | Run-record check compares against the frozen A1 hash |
| `tests/test_phase2.py` | `587d329236e68493d9a1414da5eabd03a4279070e29823ce83bf8f1e921f31f1` | `e52e79b6de0ba82ce8806d46452ddc8c126fe6bc6c3d48b7fb2e841a9fe8562f` | Comment wording only |
| `config/Protocol_Addendum_A1_v1.2_FROZEN.pdf` | `a736da98f83c09f0255963e138e01ef406c71c05e0b612f3ae495498f51b937b` | `d44e1aaa70b533a77ef90e076dfdc1b4484a9c5297b5d9d8096d6a8f09e701b4` | Protocol text rebuilt from the same source with administrative remarks removed and the archived A1 fingerprint shown; every specification unchanged |
| `m1_mimic/plan/MIMIC_M1_Analysis_Plan.pdf` | `5fbde1a9b4937b68922dd8181ccdc267adde975723faf6448bc4e8dd518c4bba` | `225ef00ed41a7c727e0f872590b75ad4893bea7d5a8d6d5197564a0f030d276f` | Plan rebuilt from the same source with process remarks removed; every specification unchanged |

Paths in the table are relative to `a1_simulation/rebuild_v322/` except the MIMIC-IV plan.

## Other changes in version 3.1.0

- The consolidated results report was removed; every result it contained is in the frozen summary and the results tables in `a1_simulation/results/`, and in Supporting Information S3 of the manuscript.
- The manuscript text-building scripts were removed. The folder `reporting/` contains the number registry, the figure script, the analytical-account checks and the scripts that build Supporting Information S2–S5, all reading from this archive.
- `reporting/analytic/analytical_account.md` is the analytical account as published in Supporting Information S1.
