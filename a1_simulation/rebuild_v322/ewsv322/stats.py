"""Inference helpers — config keys tost, estimand_ci, nulls.phase1a_rule, nulls.phase1b_rule."""
import numpy as np
from scipy.stats import binom, beta
from . import spec as S

Z95 = 1.959963984540054


def clopper_pearson(x, n, level=0.95):
    a = (1 - level) / 2
    lo = 0.0 if x == 0 else float(beta.ppf(a, x, n - x + 1))
    hi = 1.0 if x == n else float(beta.ppf(1 - a, x + 1, n - x))
    return lo, hi


def tost(x, n, lo=S.TOST_LO, hi=S.TOST_HI, alpha=S.TOST_ALPHA):
    """Exact binomial TOST of equivalence of the false-alarm rate to [lo, hi].
    H0a: p <= lo rejected if P(X >= x | lo) < alpha;  H0b: p >= hi rejected if P(X <= x | hi) < alpha."""
    if n <= 0:
        raise ValueError("TOST needs at least one patient")
    p_lo = float(binom.sf(x - 1, n, lo))
    p_hi = float(binom.cdf(x, n, hi))
    return {"x": int(x), "n": int(n), "rate": x / n, "p_lower": p_lo, "p_upper": p_hi,
            "cp90": clopper_pearson(x, n, 1 - 2 * alpha), "pass": bool(p_lo < alpha and p_hi < alpha)}


def wilson(x, n, z=Z95):
    p = x / n
    c = p + z * z / (2 * n)
    r = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    d = 1 + z * z / n
    return (c - r) / d, (c + r) / d


def newcombe_paired(e, f, g, h, z=Z95):
    """Newcombe (1998) matched-pair score interval (method 10) for delta = (f - g)/n, where
    e = both arms positive, f = arm A only, g = arm B only, h = neither (delta = p_A - p_B).
    phi numerator: max(eh - fg - n/2, 0) if eh > fg else eh - fg; phi = 0 if any margin is zero."""
    n = e + f + g + h
    p1, p2 = (e + f) / n, (e + g) / n
    l2, u2 = wilson(e + f, n, z)
    l3, u3 = wilson(e + g, n, z)
    dl2, du2, dl3, du3 = p1 - l2, u2 - p1, p2 - l3, u3 - p2
    den = (e + f) * (g + h) * (e + g) * (f + h)
    if den == 0:
        phi = 0.0
    else:
        num = e * h - f * g
        num = max(num - n / 2, 0.0) if num > 0 else num
        phi = num / np.sqrt(den)
    delta = (f - g) / n
    L = delta - np.sqrt(max(dl2 ** 2 - 2 * phi * dl2 * du3 + du3 ** 2, 0.0))
    U = delta + np.sqrt(max(du2 ** 2 - 2 * phi * du2 * dl3 + dl3 ** 2, 0.0))
    return float(delta), float(max(L, -1.0)), float(min(U, 1.0))


def paired_bootstrap(dA, dB, rng, B=S.N_BOOT, chunk=500):
    """Patient-level paired bootstrap of mean(dA) - mean(dB); percentile 95% interval.
    dA, dB: per-patient 0/1 outcomes for the two arms (same patient order)."""
    dA = np.asarray(dA, float); dB = np.asarray(dB, float)
    n = len(dA)
    diff = dA - dB
    est = float(diff.mean())
    boots = np.empty(B)
    done = 0
    while done < B:
        b = min(chunk, B - done)
        idx = rng.integers(0, n, size=(b, n))
        boots[done:done + b] = diff[idx].mean(axis=1)
        done += b
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return est, float(lo), float(hi)


MARGIN = 0.05   # phase2_matrix.primary_estimand.interpretation_rule: +/- 5 percentage points


def interpret(L, U, margin=MARGIN):
    """Mutually exclusive categories for d = detection(informative_reduced_N) - detection(regular_N)
    with 95% CI [L, U] (proportions). Also returns whether U < 0 (evidence of any loss)."""
    # compare in percentage points rounded to 1e-9 so floating-point noise cannot flip a boundary case
    L, U, mg = round(L * 100, 9), round(U * 100, 9), round(margin * 100, 9)
    if U < -mg:
        cat = "Material detection loss"
    elif L > mg:
        cat = "Material detection gain"
    elif L >= -mg and U <= mg:
        cat = "No material difference"
    else:
        cat = "Inconclusive"
    return {"category": cat, "upper_ci_below_zero_any_loss": bool(U < 0)}


def select_null(calib):
    """Phase 1a rule. calib[null][(control, arm, indicator)] -> tost dict. Returns first candidate (in
    nulls.candidates_in_order) passing every required TOST, or None (stop: no calibrated workflow)."""
    for null in S.NULLS:
        cells = calib.get(null, {})
        if len(cells) == 8 and all(c["pass"] for c in cells.values()):
            return null
    return None


def phase1b_decision(calib_b, selected):
    """Phase 1b rule: pass only if the selected null passes all 8 held-out TOSTs."""
    if selected is None:
        return {"calibrated": False, "reason": "no null selected in Phase 1a"}
    cells = calib_b.get(selected, {})
    ok = len(cells) == 8 and all(c["pass"] for c in cells.values())
    return {"calibrated": bool(ok),
            "reason": "selected null passed all Phase 1b TOSTs" if ok else
                      "selected null failed a Phase 1b TOST: Phase 2 may run only descriptively, labelled uncalibrated"}
