"""Run-gate tests: config hash, deterministic streams, key uniqueness, nested surrogate keys, cohort bootstrap."""
import hashlib
import os
import subprocess
import sys

import numpy as np
import pytest

from ewsv322 import spec as S
from ewsv322 import rng as R

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def test_config_hash_is_the_governing_file():
    assert S.config_sha256() == S.EXPECTED_SHA256 == S.CONFIG_HASH


def test_fixed_substream_keys_match_config():
    fixed = S.CFG["seeds"]["streams"]["fixed_substreams"]
    for name, K in fixed.items():
        assert R.key(name) == K
        assert R.key(name) == int(hashlib.sha256(name.encode()).hexdigest()[:16], 16)


def test_same_seed_and_name_reproduce_across_fresh_processes():
    here = R.gen(3000001, "latent").random(5).tolist()
    code = ("import sys; sys.path.insert(0, r'%s'); from ewsv322.rng import gen; "
            "print(repr(gen(3000001, 'latent').random(5).tolist()))" % ROOT)
    for _ in range(2):
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
        assert eval(out.strip()) == here


def test_generator_equals_documented_construction():
    K = R.key("err_baseline")
    manual = np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy=1000007, spawn_key=(K,))))
    assert np.array_equal(R.gen(1000007, "err_baseline").random(4), manual.random(4))


def test_surrogate_nested_key():
    k1, k2 = R.key("surrogates"), R.key("regular_N|index|ou_ss")
    manual = np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy=42, spawn_key=(k1, k2, 7))))
    assert np.array_equal(R.surrogate_stream(42, "regular_N", "index", "ou_ss", 7).random(3), manual.random(3))
    a = R.surrogate_stream(42, "regular_N", "index", "ou_ss", 0).random(3)
    b = R.surrogate_stream(42, "regular_N", "index", "ou_ss", 1).random(3)
    assert not np.array_equal(a, b)


def test_bootstrap_is_cohort_level():
    K = R.key("bootstrap|ou_csd|regular_N-vs-informative_reduced_N")
    manual = np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy=3_000_000, spawn_key=(K,))))
    g = R.bootstrap_stream(3_000_000, "ou_csd", "regular_N-vs-informative_reduced_N")
    assert np.array_equal(g.random(3), manual.random(3))


def test_distinct_names_give_distinct_streams():
    a = R.gen(5, "err_reg_N").random(3)
    b = R.gen(5, "err_reg_matched").random(3)
    assert not np.array_equal(a, b)


def test_negative_or_bad_components_rejected():
    with pytest.raises(ValueError):
        R.gen(5, "latent", -1)
    with pytest.raises(ValueError):
        R.gen(-5, "latent")
    with pytest.raises(TypeError):
        R.gen(5, 1.5)


def test_planned_stream_keys_unique_and_m_formatting():
    from ewsv322 import run
    d1 = run.calibration_design("indep")
    d2 = run.calibration_design("primary")
    keys = run.planned_streams([d1, d2], ["stable", "noise_only"])
    assert len(set(keys.values())) == len(keys)
    assert "err_inf|step|2.00|primary" in keys and "schedule|step|2.00|indep" in keys
    assert R.tag("step", 2.0, "primary") == "step|2.00|primary"
    assert R.tag("smooth", None, "none") == "smooth|none|none"
