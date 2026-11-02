"""Early-warning analyses — config keys analyses.*, evaluability, detection.

Each analysis maps an arm's series (t, y) to two sequences of window statistics (variance, lag-1
autocorrelation) and an x-position per window; Kendall tau-a of statistic vs position is the test
statistic (prespecified increasing direction). Window statistics that cannot be computed are NaN.
A series is evaluable for an indicator if it has >= 6 valid window statistics (evaluability.rule).
"""
import numpy as np
from . import spec as S
from . import kalman

ANALYSES_BASE = ["index", "elapsed"] + [f"fixed_time|W{int(w)}" for w in S.FT_W] + \
                [f"binned|{b}|{r}" for b in S.NBINS for r in S.EMPTY_RULES] + \
                [f"interp|{S.INTERP_POINTS}", f"ct|W{int(S.CT_W)}"]


def kendall_tau_a(x, y):
    """tau-a over pairs with both statistics finite: sum sign(dx)sign(dy) / (n(n-1)/2).
    Returns NaN when fewer than MIN_VALID_WINDOWS valid statistics. y may be 2-D (rows = series)."""
    x = np.asarray(x, float)
    Y = np.atleast_2d(np.asarray(y, float))
    ok = np.isfinite(Y) & np.isfinite(x)[None, :]
    nv = ok.sum(axis=1)
    sx = np.sign(x[:, None] - x[None, :])                       # (w, w)
    Yz = np.where(ok, Y, 0.0)
    sy = np.sign(Yz[:, :, None] - Yz[:, None, :])                # (r, w, w)
    pair_ok = ok[:, :, None] & ok[:, None, :]
    iu = np.triu(np.ones_like(sx, dtype=bool), 1)
    s = np.sum(np.where(pair_ok & iu[None], sx[None] * sy, 0.0), axis=(1, 2))
    with np.errstate(invalid="ignore", divide="ignore"):
        tau = s / (nv * (nv - 1) / 2.0)
    tau = np.where(nv >= S.MIN_VALID_WINDOWS, tau, np.nan)
    return tau if np.ndim(y) == 2 else float(tau[0])


def _detrended_stats(tt, yy):
    """Linear detrend of yy on tt; variance (ddof=1) and lag-1 Pearson autocorrelation of residuals,
    in observation order (IMPL-NOTE (see IMPLEMENTATION_NOTES.md): ddof=1 and Pearson lag-1, as in the original repository)."""
    n = len(yy)
    tc = tt - tt.mean()
    den = np.sum(tc * tc)
    if n < 3 or den <= 0:
        return np.nan, np.nan
    slope = np.sum(tc * (yy - yy.mean())) / den
    r = yy - yy.mean() - slope * tc
    v = r.var(ddof=1)
    a, b = r[:-1], r[1:]
    ac_den = np.sqrt(np.sum((a - a.mean()) ** 2) * np.sum((b - b.mean()) ** 2))
    ac = np.sum((a - a.mean()) * (b - b.mean())) / ac_den if ac_den > 1e-12 else np.nan
    return v, ac


def index_window(n):
    """IMPL-NOTE (see IMPLEMENTATION_NOTES.md): window length = max(5, floor(0.30 n)) (original repository convention)."""
    return max(5, int(S.WIN_FRAC * n))


def _index_stats(y):
    y = np.asarray(y, float)
    n = len(y)
    w = index_window(n)
    nw = n - w + 1
    if nw < 1:
        return np.empty(0), np.empty(0), w
    W = np.lib.stride_tricks.sliding_window_view(y, w)           # (nw, w)
    pos = np.arange(w, dtype=float); pc = pos - pos.mean()
    Wc = W - W.mean(axis=1, keepdims=True)
    slope = (Wc @ pc) / np.sum(pc * pc)
    R = Wc - np.outer(slope, pc)
    var = R.var(axis=1, ddof=1)
    a, b = R[:, :-1], R[:, 1:]
    ac_c = a - a.mean(axis=1, keepdims=True); bc = b - b.mean(axis=1, keepdims=True)
    den = np.sqrt(np.sum(ac_c ** 2, axis=1) * np.sum(bc ** 2, axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        ac = np.where(den > 1e-12, np.sum(ac_c * bc, axis=1) / den, np.nan)
    return var, ac, w


def window_starts(W, npos):
    """analyses.fixed_time.start_positions: start_j = j*(T-W)/(npos-1), j = 0..npos-1.
    IMPL-NOTE (see IMPLEMENTATION_NOTES.md): the continuous-time analysis uses the same start rule with W=30, 43 positions."""
    return np.arange(npos) * (S.T - W) / (npos - 1)


def _bin(t, y, nb, rule):
    edges = np.linspace(0.0, S.T, nb + 1)
    idx = np.clip(np.floor(t / (S.T / nb)).astype(int), 0, nb - 1)
    cnt = np.bincount(idx, minlength=nb)
    sm = np.bincount(idx, weights=y, minlength=nb)
    filled = cnt > 0
    vals = np.full(nb, np.nan)
    vals[filled] = sm[filled] / cnt[filled]
    centres = 0.5 * (edges[:-1] + edges[1:])
    if not filled.any():
        return np.empty(0)
    if rule == "drop":
        return vals[filled]
    if rule == "linear_interp":
        # IMPL-NOTE (see IMPLEMENTATION_NOTES.md): empty bins outside the first/last filled bin take the nearest filled value.
        vals[~filled] = np.interp(centres[~filled], centres[filled], vals[filled])
        return vals
    if rule == "locf":
        # IMPL-NOTE (see IMPLEMENTATION_NOTES.md): leading empty bins (before any observation) take the first filled value.
        first = np.argmax(filled)
        last = vals[first]
        for i in range(nb):
            if filled[i]:
                last = vals[i]
            else:
                vals[i] = last
        return vals
    raise ValueError(rule)


def window_stats(analysis, t, y, want_flags=False):
    """Return (var_stats, ac_stats, xpos, info)."""
    t = np.asarray(t, float); y = np.asarray(y, float)
    info = {}
    if analysis in ("index", "elapsed"):
        var, ac, w = _index_stats(y)
        if analysis == "index":
            x = np.arange(len(var), dtype=float)
        else:
            x = (np.lib.stride_tricks.sliding_window_view(t, w).mean(axis=1)
                 if len(var) else np.empty(0))
        return var, ac, x, info
    if analysis.startswith("fixed_time|W"):
        W = float(analysis.split("W")[1])
        st = window_starts(W, S.FT_POS)
        var = np.full(len(st), np.nan); ac = np.full(len(st), np.nan)
        for j, s0 in enumerate(st):
            msk = (t >= s0) & (t < s0 + W)
            if msk.sum() >= S.FT_MIN_OBS:
                var[j], ac[j] = _detrended_stats(t[msk], y[msk])
        return var, ac, st, info
    if analysis.startswith("binned|"):
        _, nb, rule = analysis.split("|")
        yb = _bin(t, y, int(nb), rule)
        var, ac, _ = _index_stats(yb)
        return var, ac, np.arange(len(var), dtype=float), info
    if analysis.startswith("interp|"):
        if len(t) < 2:
            return np.empty(0), np.empty(0), np.empty(0), info
        grid = np.linspace(0.0, S.T, int(analysis.split("|")[1]))
        yi = np.interp(grid, t, y)          # IMPL-NOTE (see IMPLEMENTATION_NOTES.md): constant extrapolation beyond first/last observation
        var, ac, _ = _index_stats(yi)
        return var, ac, np.arange(len(var), dtype=float), info
    if analysis.startswith("ct|W"):
        W = float(analysis.split("W")[1])
        st = window_starts(W, S.CT_POS)
        var = np.full(len(st), np.nan); ac = np.full(len(st), np.nan)
        th_edge = rho_edge = 0
        for j, s0 in enumerate(st):
            msk = (t >= s0) & (t < s0 + W)
            if msk.sum() >= S.CT_MIN_OBS:
                f = kalman.fit(t[msk], y[msk])          # mu = window mean
                if f is not None:
                    # IMPL-NOTE (see IMPLEMENTATION_NOTES.md): variance indicator = fitted latent stationary variance V_inf;
                    # autocorrelation indicator = exp(-theta_hat) (lag-1 time unit; any lag gives the
                    # same Kendall tau because exp(-theta*lag) is monotone in theta).
                    var[j] = f["V"]; ac[j] = np.exp(-f["theta"])
                    th_edge += f["theta_edge"]; rho_edge += f["rho_edge"]
        info = {"ct_theta_edge_windows": th_edge, "ct_rho_edge_windows": rho_edge}
        return var, ac, st, info
    raise ValueError(analysis)


def taus(analysis, t, y):
    var, ac, x, info = window_stats(analysis, t, y)
    nv_var = int(np.isfinite(var).sum()); nv_ac = int(np.isfinite(ac).sum())
    tv = kendall_tau_a(x, var) if len(var) else np.nan
    ta = kendall_tau_a(x, ac) if len(ac) else np.nan
    return tv, ta, nv_var, nv_ac, info
