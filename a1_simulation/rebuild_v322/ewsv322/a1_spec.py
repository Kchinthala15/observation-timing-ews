"""Protocol Addendum A1 v1.2 (frozen) - loader and hash lock.

A1 only ADDS models, an arm, two tests and blocks on top of the frozen Phase 2 package. Every frozen
module is used unchanged; A1 code lives in the a1_*.py files.
"""
import hashlib
import json
import os

from . import spec as S

A1_PATH = os.path.join(S.HERE, "..", "config", "Simulation_Addendum_A1_v1.2_FROZEN.json")
A1_SHA256 = "07e896f9bc70eb1356efbfcbfce08b0e6e06d470c28485f5ed592d66d0bffe4b"
# SHA-256 of the frozen original A1 file, as written into every run record and manifest. The public copy differs
# only in administrative text fields (see REDACTIONS.md); records are checked against the original value.
A1_SHA256_FROZEN_ORIGINAL = "9aff4ef745296e32b6b888b3cc32d08861aec3affe71a8153a88409922b4d7e5"


def _sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


if _sha(A1_PATH) != A1_SHA256:
    raise RuntimeError(f"A1 file hash {_sha(A1_PATH)} does not match the frozen A1 v1.2 hash {A1_SHA256}")
with open(A1_PATH, encoding="utf-8") as f:
    A1 = json.load(f)
if A1["config_lineage"]["parent_config_sha256"] != S.CONFIG_HASH:
    raise RuntimeError("A1 parent hash does not match the Phase 2 configuration in use")
if A1["config_lineage"]["grandparent_config_sha256"] != S.PARENT_SHA256:
    raise RuntimeError("A1 grandparent hash does not match the frozen Phase 1 configuration")

BLOCKS = {b["id"]: b for b in A1["blocks"]}
FROZEN_NULLS = ("ou_ss", "ar1", "shuffle")
REP_NULLS = ("ar1_rep", "shuffle_rep")
NEW_ARMS = ("poisson_matched",)
STABLE_SLOW = {"stable_slow_0.2": 0.2, "stable_slow_0.05": 0.05}
FOLD_GRID = [(sf, re) for sf in A1["new_models"]["fold_v2"]["grid"]["sigma_f"]
             for re in A1["new_models"]["fold_v2"]["grid"]["r_end"]]
MODEL_STREAM_NEW = ("pm_times", "err_pm")


def fold_model_name(sf, re):
    return f"fold_v2|sf={sf:.2f}|re={re:.2f}"


def parse_fold(model):
    """'fold_v2|sf=0.07|re=0.10' -> (0.07, 0.10)."""
    parts = dict(p.split("=") for p in model.split("|")[1:])
    return float(parts["sf"]), float(parts["re"])
