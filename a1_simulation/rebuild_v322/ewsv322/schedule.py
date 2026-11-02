"""Observation schedules, triggers and measurement — config keys baseline, trigger, schedules,
measurement_error, nulls.controls.

Measurement rule: one independent N(0, 0.1^2) error per observation EVENT. A stream hands out
errors in event order (the k-th event of a stream receives the k-th draw); the grid plays no part.
"""
import numpy as np
from . import spec as S
from .latent import value_at, reference_level, T_GRID
from .rng import patient_stream, tag as stream_tag

TRIGGERS = ("primary", "single", "k2.0", "k2.5", "oracle", "indep")


def measure(X, times, rng_err):
    """Return (x, e, y) for the given times; the k-th time receives the k-th error draw."""
    times = np.asarray(times, float)
    x = value_at(X, times) if len(times) else np.empty(0)
    e = rng_err.normal(0.0, S.MEAS_SD, size=len(times))
    return x, e, x + e


def baseline(X, seed):
    """8 fixed times 15*i/9; errors from 'err_baseline'; shared by every arm."""
    tb = np.array(S.BASELINE_TIMES)
    xb, eb, yb = measure(X, tb, patient_stream(seed, "err_baseline"))
    # IMPL-NOTE (see IMPLEMENTATION_NOTES.md): baseline SD uses the sample SD (ddof=1) of the 8 measured baseline values.
    return {"t": tb, "x": xb, "e": eb, "y": yb, "mean": float(yb.mean()), "sd": float(yb.std(ddof=1)),
            "x_mean": float(xb.mean()), "x_sd": float(xb.std(ddof=1))}


def trigger_rule(trigger):
    """(k, consecutive_required, uses_true_state)."""
    return {"primary": (S.TRIG_K, 2, False),
            "single": (S.TRIG_K, 1, False),
            "k2.0": (S.TRIG_K_SENS[0], 2, False),
            "k2.5": (S.TRIG_K_SENS[1], 2, False),
            # IMPL-NOTE (see IMPLEMENTATION_NOTES.md): oracle = same two-consecutive rule applied to the true state X at the
            # event times, threshold from the true state at the 8 baseline times (comparison only).
            "oracle": (S.TRIG_K, 2, True)}[trigger]


def informative(X, seed, base, schedule, m=None, trigger="primary", t_esc_indep=None, model=None):
    """Event-driven post-baseline sampling on [15, T).

    step:   rate lambda0 until escalation, then m*lambda0.
    ramp:   before escalation rate = min(m*lambda0, lambda0*exp(a*max(0,(base_mean - last_measured)/base_sd)));
            after escalation m*lambda0. IMPL-NOTE (see IMPLEMENTATION_NOTES.md): before the first post-baseline event,
            'last_measured' is the last baseline measurement (t = 13.33).
    smooth: oracle intensity c*exp(1.5*(reference_level(t) - X(t))), no trigger; c set per patient so the
            expected post-baseline count is 52 (IMPL-NOTE (see IMPLEMENTATION_NOTES.md): time-rescaling with a trapezoidal cumulative
            intensity on the latent grid, inverted by linear interpolation).
    Triggers: escalation happens at the time of the event that completes the rule; under 'indep' the
            rate switches exactly at the externally drawn t_esc (np.inf = never).
    Returns dict(t, x, e, y, t_esc, trig_inputs) for post-baseline events only, in time order.
    """
    tg = stream_tag(schedule, m, trigger)
    rng_s = patient_stream(seed, f"schedule|{tg}")
    rng_e = patient_stream(seed, f"err_inf|{tg}")
    ts, xs, es, ys, trig_inputs = [], [], [], [], []

    if schedule == "smooth":
        ref = reference_level(model, T_GRID)
        j0 = int(round(S.T_START / S.DT))
        g = np.exp(S.SMOOTH_GAIN * (ref[j0:] - X[j0:]))
        tt = T_GRID[j0:]
        cum = np.concatenate([[0.0], np.cumsum(0.5 * (g[1:] + g[:-1]) * S.DT)])
        c = S.SMOOTH_EXPECTED_POST / cum[-1]
        cum_c = c * cum                       # expected count from t=15 up to each grid time; ends at 52
        s = 0.0
        while True:
            s += rng_s.exponential(1.0)
            if s >= cum_c[-1]:
                break
            t = float(np.interp(s, cum_c, tt))
            if t >= S.T:
                break
            x = float(value_at(X, t)); e = float(rng_e.normal(0.0, S.MEAS_SD))
            ts.append(t); xs.append(x); es.append(e); ys.append(x + e)
        return _pack(ts, xs, es, ys, np.inf, trig_inputs, smooth_c=c)

    if schedule not in ("step", "ramp"):
        raise ValueError(schedule)
    m = float(m)
    lam_hi = m * S.LAMBDA0
    if trigger == "indep":
        if t_esc_indep is None:
            raise ValueError("independent trigger needs a drawn escalation time")
        t_esc = float(t_esc_indep)
        rule = None
    else:
        t_esc = np.inf
        k, need, true_state = trigger_rule(trigger)
        thr = (base["x_mean"] - k * base["x_sd"]) if true_state else (base["mean"] - k * base["sd"])
        run = 0
    escalated = False
    last_y = base["y"][-1]

    def rate_now():
        if escalated:
            return lam_hi
        if schedule == "ramp":
            z = max(0.0, (base["mean"] - last_y) / base["sd"])
            return min(lam_hi, S.LAMBDA0 * np.exp(S.RAMP_A * z))
        return S.LAMBDA0

    t = S.T_START
    while True:
        gap = rng_s.exponential(1.0 / rate_now())
        if trigger == "indep" and not escalated and t + gap >= t_esc:
            # memoryless switch exactly at the drawn escalation time (no event at t_esc)
            if t_esc >= S.T:
                break
            t = t_esc
            escalated = True
            continue
        t = t + gap
        if t >= S.T:
            break
        x = float(value_at(X, t)); e = float(rng_e.normal(0.0, S.MEAS_SD)); y = x + e
        ts.append(t); xs.append(x); es.append(e); ys.append(y)
        last_y = y
        if trigger != "indep" and not escalated:
            v = x if true_state else y        # the trigger sees the measured value later analysed
            trig_inputs.append(v)
            run = run + 1 if v < thr else 0
            if run >= need:
                escalated = True
                t_esc = t
    return _pack(ts, xs, es, ys, t_esc, trig_inputs)


def _pack(ts, xs, es, ys, t_esc, trig_inputs, smooth_c=None):
    return {"t": np.array(ts, float), "x": np.array(xs, float), "e": np.array(es, float),
            "y": np.array(ys, float), "t_esc": float(t_esc), "trig_inputs": np.array(trig_inputs, float),
            "smooth_c": smooth_c}


def regular_times(n, t_hi=None):
    """n evenly spaced post-baseline times over [15, t_hi): 15 + (t_hi-15)*j/n, j=0..n-1 (IMPL-NOTE (see IMPLEMENTATION_NOTES.md))."""
    t_hi = S.T if t_hi is None else t_hi
    if n <= 0:
        return np.empty(0)
    return S.T_START + (t_hi - S.T_START) * np.arange(n) / n


def draw_indep(seed, outcomes):
    """trigger.independent_implementation: one with-replacement draw from the cohort's empirical
    primary-trigger outcomes (np.inf = no escalation), using only the patient's 'indep_trigger' stream."""
    outcomes = np.asarray(outcomes, float)
    if len(outcomes) == 0:
        raise ValueError("independent-trigger source distribution is empty (no non-tipped patients)")
    i = int(patient_stream(seed, "indep_trigger").integers(len(outcomes)))
    return float(outcomes[i])
