"""Governing specification loader for Simulation Rebuild Plan v3.2.2 (Phase 2 configuration).

Structured numeric values are read from the governing JSON
(config/Simulation_Rebuild_Config_v3.2.2_Phase2.json, hash-locked below; every parent value is
unchanged from the frozen Phase 1 file Simulation_Rebuild_Config_v3.2.2_Phase1.json). Values the
JSON states only inside text (e.g. the 1.5 trigger multiplier, the 52 expected
smooth-schedule events, N cap 60, 6 valid windows, alpha 0.05, phase seed bases,
2000 calibration patients, 10 subsampling draws) are written here as constants,
each next to the JSON key it comes from. Choices the JSON does not state are listed in
IMPLEMENTATION_NOTES.md and are marked "IMPL-NOTE n" in the code.
"""
import hashlib
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
# Phase 2 configuration = frozen parent (used for Phase 1a/1b) + phase2_matrix + config_lineage.
CONFIG_PATH = os.path.join(HERE, "..", "config", "Simulation_Rebuild_Config_v3.2.2_Phase2.json")
EXPECTED_SHA256 = "0473d4830407f1fde68988d32fb1d2ad14537b815c0fd34492b8391d084950d3"
# Parent configuration, kept unchanged for the audit trail (Phase 1a/1b records point to it).
PARENT_CONFIG_PATH = os.path.join(HERE, "..", "config", "Simulation_Rebuild_Config_v3.2.2_Phase1.json")
PARENT_SHA256 = "9a245eb092a47bc5e2851f0b681ec439c1a72ab34ed2fe3dedf63b9b29bff482"


def config_sha256(path=CONFIG_PATH):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def load(path=CONFIG_PATH, enforce_hash=True):
    h = config_sha256(path)
    if enforce_hash and h != EXPECTED_SHA256:
        raise RuntimeError(f"Config hash {h} does not match the governing v3.2.2 file {EXPECTED_SHA256}")
    with open(path, encoding="utf-8") as f:
        return json.load(f), h


CFG, CONFIG_HASH = load()
if config_sha256(PARENT_CONFIG_PATH) != PARENT_SHA256:
    raise RuntimeError("Parent configuration file has changed; it must stay identical to the frozen Phase 1 file")
if CFG.get("config_lineage", {}).get("parent_config_sha256") != PARENT_SHA256:
    raise RuntimeError("Phase 2 config does not record the frozen parent hash")

# ---- time --------------------------------------------------------------
T = float(CFG["time"]["T"])                       # 100
DT = float(CFG["time"]["dt"])                     # 0.02
T_START = float(CFG["time"]["transition_start"])  # 15
T_END = float(CFG["time"]["transition_end"])      # 75
NGRID = int(round(T / DT)) + 1                    # grid t_j = j*dt, j = 0..5000 (includes t=0 and t=T)

# ---- latent ------------------------------------------------------------
MU0 = float(CFG["latent"]["ou"]["mu"])
SIGMA = float(CFG["latent"]["ou"]["sigma"])
THETA0 = float(CFG["latent"]["ou"]["theta0"])
THETA_MIN = float(CFG["latent"]["ou"]["theta_min"])
LEVEL_DROP = float(CFG["latent"]["combined_level_drop"])
FOLD_R0 = float(CFG["latent"]["fold"]["r0"])
FOLD_REND = float(CFG["latent"]["fold"]["r_end"])
FOLD_SIGMA = float(CFG["latent"]["fold"]["sigma_f"])
FOLD_STOP_T = float(CFG["latent"]["fold"]["stop_t"])          # 100 in v3.2.2
FOLD_R_AFTER = float(CFG["latent"]["fold"]["r_after_transition"])
MODELS = ["ou_csd", "ou_combined", "fold", "stable", "level_drop_only", "noise_only"]

# ---- measurement / baseline -------------------------------------------
MEAS_SD = float(CFG["measurement_error"]["sd"])
N_BASE = int(CFG["baseline"]["n"])
BASELINE_TIMES = tuple(T_START * i / (N_BASE + 1) for i in range(1, N_BASE + 1))  # 15*i/9, i=1..8

# ---- trigger / schedules ----------------------------------------------
TRIG_K = 1.5                       # trigger.primary: baseline_mean - 1.5*baseline_sd
TRIG_K_SENS = (2.0, 2.5)           # trigger.sensitivity
LAMBDA0 = float(CFG["schedules"]["lambda0"])
M_PRIMARY = float(CFG["schedules"]["step"]["m_primary"])
M_SWEEP = tuple(float(x) for x in CFG["schedules"]["step"]["m_sweep"])
RAMP_A = float(CFG["schedules"]["ramp"]["a"])
SMOOTH_EXPECTED_POST = 52.0        # schedules.smooth.formula
SMOOTH_GAIN = 1.5                  # schedules.smooth.formula exp(1.5*(ref - X))

# ---- arms ----------------------------------------------------------------
N_CAP = 60                         # arms.count_convention N_i = min(60, informative total)
N_SUBSAMPLE_DRAWS = 10             # arms.subsampling variability sub-study

# ---- analyses --------------------------------------------------------------
A = CFG["analyses"]
WIN_FRAC = float(A["index"]["win_frac"])
FT_W = tuple(float(w) for w in A["fixed_time"]["W"])
FT_POS = int(A["fixed_time"]["positions"])
FT_MIN_OBS = int(A["fixed_time"]["min_obs_per_window"])
NBINS = tuple(int(b) for b in A["binning"]["nbins"])
EMPTY_RULES = tuple(A["binning"]["empty_bin_rules"])
INTERP_POINTS = int(A["interpolation"]["grid_points"])
CT_W = float(A["continuous_time"]["W"])
CT_POS = int(A["continuous_time"]["positions"])
CT_MIN_OBS = int(A["continuous_time"]["min_obs_per_window"])
_m = re.match(r"(\d+) log-spaced ([0-9.]+)-([0-9.]+)", A["continuous_time"]["theta_grid"])
THETA_GRID_N, THETA_GRID_LO, THETA_GRID_HI = int(_m.group(1)), float(_m.group(2)), float(_m.group(3))
RHO_GRID = tuple(float(r) for r in A["continuous_time"]["rho_grid"])
PRE_MIN_OBS = int(A["pre_min_obs"])
MIN_VALID_WINDOWS = 6              # evaluability.rule

# ---- inference -------------------------------------------------------------
N_SURR = int(CFG["nulls"]["surrogates"])
assert N_SURR == int(CFG["surrogates"]["replicates_per_test"])
NULLS = ("ou_ss", "ar1", "shuffle")  # nulls.candidates_in_order, same order
ALPHA_DET = 0.05
TOST_ALPHA = float(CFG["tost"]["alpha"])
TOST_LO = float(CFG["tost"]["lower"])
TOST_HI = float(CFG["tost"]["upper"])
N_BOOT = int(CFG["estimand_ci"]["bootstrap"])
REPS = {k: int(v) for k, v in CFG["replicates"].items()}
PHASE_BASE = {"phase1a": 1_000_000, "phase1b": 2_000_000, "phase2": 3_000_000}
PLUMBING_BASE = 9_000_000          # IMPL-NOTE (see IMPLEMENTATION_NOTES.md): plumbing test seed block, outside every phase block
N_CALIB = 2000                     # nulls.selection / validation: 2000 patients per control
