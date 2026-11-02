"""Random streams — config key seeds.streams (single SHA-256 method).

Every random draw in the simulator comes from a generator made here:
    numpy.random.Generator(PCG64(SeedSequence(entropy=E, spawn_key=(K1, K2, ...))))
where each K is either the SHA-256 key of a stream name (first 16 hex digits read as an
integer) or, for surrogate replicates only, the zero-based replicate index k appended last.
Python's built-in hash() and SeedSequence.spawn() are never used. No global RNG is used.
"""
import hashlib
import numpy as np

# Every stream-name component requested in this process: name -> SHA-256 key
_REGISTRY = {}


def key(name: str) -> int:
    """First 16 hex digits of SHA-256(UTF-8 name), read as an integer."""
    return int(hashlib.sha256(name.encode("utf-8")).hexdigest()[:16], 16)


def spawn_key(*parts):
    out = []
    for p in parts:
        if isinstance(p, str):
            out.append(key(p))
        elif isinstance(p, (int, np.integer)) and not isinstance(p, bool):
            if int(p) < 0:
                raise ValueError("spawn_key components must be non-negative integers")
            out.append(int(p))
        else:
            raise TypeError(f"stream component {p!r} must be a name (str) or a non-negative int")
    return tuple(out)


def gen(entropy: int, *parts) -> np.random.Generator:
    """Generator for stream `parts` with integer entropy (patient seed or cohort block base)."""
    if not isinstance(entropy, (int, np.integer)) or int(entropy) < 0:
        raise ValueError("entropy must be a non-negative integer")
    sk = spawn_key(*parts)
    # record every named component (replicate indices are not recorded, to keep the registry small)
    for p in parts:
        if isinstance(p, str):
            _REGISTRY[p] = key(p)
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy=int(entropy), spawn_key=sk)))


# ---- stream names ----------------------------------------------------------------------
def tag(schedule: str, m, trigger: str) -> str:
    """'<schedule>|<m>|<trigger>' with m written to 2 decimals (templated_substreams).
    IMPL-NOTE (see IMPLEMENTATION_NOTES.md): the smooth schedule has no multiplier or trigger; it uses 'smooth|none|none'."""
    if schedule == "smooth":
        return "smooth|none|none"
    return f"{schedule}|{float(m):.2f}|{trigger}"


def patient_stream(seed, name):
    return gen(seed, name)


def surrogate_stream(seed, arm, analysis, null, k):
    """spawn_key = (K('surrogates'), K('<arm>|<analysis>|<null>'), k)."""
    return gen(seed, "surrogates", f"{arm}|{analysis}|{null}", int(k))


def bootstrap_stream(block_base, model, comparison):
    """Cohort-level: entropy = phase-block base seed; spawn_key = (K('bootstrap|<model>|<comparison>'),)."""
    return gen(block_base, f"bootstrap|{model}|{comparison}")


def registry_snapshot():
    """name -> SHA-256 key for every stream-name component used in this process."""
    return dict(_REGISTRY)


def registry_clear():
    _REGISTRY.clear()


def check_unique(names):
    """All distinct names must have distinct SHA-256 keys (run_gate)."""
    names = sorted(set(names))
    keys = [key(n) for n in names]
    if len(set(keys)) != len(keys):
        raise RuntimeError("stream-key collision among: " + ", ".join(names))
    return {n: k for n, k in zip(names, keys)}
