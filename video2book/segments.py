"""Detection of stable segments (one per shown page) in the motion signal.

Algorithm
---------
1. **Smooth** the raw frame-difference signal with a short running median
   (default 0.15 s). This removes one- or two-frame spikes caused by
   compression (keyframe "pops") without touching real transitions, which last
   several frames.
2. **Adaptive threshold.** Two levels are estimated from the signal itself:

   * ``base``: the median level, i.e. what the video looks like while a page
     is shown (compression noise, or a slow zoom/pan "Ken Burns" effect);
   * ``peak``: the typical height of transition peaks (median height of the
     local maxima that stand well above ``base``).

   The threshold is interpolated geometrically between them,
   ``T = base * (peak / base) ** alpha``, where ``alpha`` is derived from the
   user-facing *sensitivity* (0..1, higher = lower threshold = more pages).
   A manual absolute threshold overrides all of this.
3. **Stable segments** are maximal runs with ``smoothed <= T`` lasting at least
   ``min_duration`` seconds. Everything else is a transition.
4. Inside each segment we find the **calm core**: the stretch around the
   lowest-motion point whose motion stays within a few times that minimum.
   This matters for slideshows with slow zooms (the page is still moving a
   little) and for slide transitions with ease-in/ease-out tails: the page
   image is built from the core only, never from the tails.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np

from .motion import MotionAnalysis

NOISE_FLOOR = 0.02      # minimum "base" level (0-255 mean abs diff)
MIN_THRESHOLD = 0.15    # never threshold below this
DETAIL_KEEP = 0.7       # calm frame must show >= 70% of the segment's max detail
BLANK_DETAIL = 1.5      # mean |Laplacian| (small frame) below this = blank frame
SPLIT_MIN = 0.08        # minimum bump height for a hidden-transition candidate
MAX_SPLITS = 4          # candidates examined per segment


@dataclass
class SegmentParams:
    sensitivity: float = 0.5          # 0..1; higher -> lower threshold -> more pages
    threshold: Optional[float] = None  # manual absolute threshold (overrides sensitivity)
    min_duration: float = 0.3         # seconds a page must stay still
    smooth: float = 0.15              # running-median window, seconds
    short_flag: float = 0.6           # segments shorter than this get a "short" flag

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Segment:
    start: int          # first sample index (inclusive)
    end: int            # last sample index (inclusive)
    t_start: float
    t_end: float
    core_start: int
    core_end: int
    calm: int           # sample index with the least motion
    motion: float       # median smoothed motion inside the core
    flags: list = field(default_factory=list)
    splits: list = field(default_factory=list)  # candidate hidden transitions (i0, i1)

    @property
    def duration(self) -> float:
        return self.t_end - self.t_start

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SegmentResult:
    segments: list
    threshold: float
    base: float
    peak: float
    alpha: float
    smoothed: np.ndarray
    rate: float = 30.0
    detail_ref: float = 0.0
    params: SegmentParams = field(default_factory=SegmentParams)

    @property
    def count(self) -> int:
        return len(self.segments)


# -- small signal helpers (numpy only, no scipy dependency) -----------------

def running_median(x: np.ndarray, k: int) -> np.ndarray:
    x = np.asarray(x, np.float64)
    if k <= 1 or x.size < 3:
        return x.copy()
    k = int(k) | 1
    pad = k // 2
    xp = np.pad(x, pad, mode="edge")
    win = np.lib.stride_tricks.sliding_window_view(xp, k)
    return np.median(win, axis=1)


def running_mean(x: np.ndarray, k: int) -> np.ndarray:
    x = np.asarray(x, np.float64)
    if k <= 1 or x.size < 2:
        return x.copy()
    k = int(k) | 1
    pad = k // 2
    xp = np.pad(x, pad, mode="edge")
    c = np.cumsum(np.concatenate([[0.0], xp]))
    return (c[k:] - c[:-k]) / k


def local_peaks(x: np.ndarray, min_height: float, min_distance: int) -> np.ndarray:
    """Indices of local maxima >= min_height, non-max-suppressed within min_distance."""
    x = np.asarray(x, np.float64)
    if x.size < 3:
        return np.array([], int)
    left = np.concatenate([[-np.inf], x[:-1]])
    right = np.concatenate([x[1:], [-np.inf]])
    cand = np.where((x >= left) & (x > right) & (x >= min_height))[0]
    if cand.size == 0:
        return cand
    order = cand[np.argsort(-x[cand])]
    taken = np.zeros(x.size, bool)
    keep = []
    for i in order:
        lo, hi = max(0, i - min_distance), min(x.size, i + min_distance + 1)
        if not taken[lo:hi].any():
            keep.append(i)
            taken[i] = True
    return np.sort(np.asarray(keep, int))


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive (start, end) index pairs of True runs."""
    m = np.concatenate([[False], np.asarray(mask, bool), [False]])
    d = np.diff(m.astype(np.int8))
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0] - 1
    return list(zip(starts.tolist(), ends.tolist()))


# -- threshold ------------------------------------------------------------

def sensitivity_to_alpha(sensitivity: float) -> float:
    """Position of the threshold between the still level (0) and the typical
    transition peak (1), on a log scale. 0 -> 0.90, 0.5 -> 0.625, 1 -> 0.20.
    The upper half reaches lower so weak transitions (slow fades mixed with
    zoom effects) can still be separated when the user asks for more pages."""
    s = float(np.clip(sensitivity, 0.0, 1.0))
    if s <= 0.5:
        return 0.9 - 0.55 * s
    return 0.625 - 0.85 * (s - 0.5)


def estimate_levels(smoothed: np.ndarray, rate: float) -> tuple[float, float]:
    """Return ``(base, peak)`` levels of a smoothed motion signal."""
    s = np.asarray(smoothed, np.float64)
    base = max(float(np.median(s)), NOISE_FLOOR)
    top = float(np.percentile(s, 99.5)) if s.size else base
    top = max(top, base * 4)
    min_h = float(np.sqrt(base * top))
    pk = local_peaks(s, min_h, max(1, int(round(0.5 * rate))))
    peak = float(np.median(s[pk])) if pk.size >= 2 else top
    peak = max(peak, base * 4)
    return base, peak


def compute_threshold(smoothed: np.ndarray, rate: float, sensitivity: float = 0.5,
                      manual: Optional[float] = None) -> tuple[float, float, float, float]:
    """Return ``(threshold, base, peak, alpha)``."""
    base, peak = estimate_levels(smoothed, rate)
    alpha = sensitivity_to_alpha(sensitivity)
    if manual is not None and manual > 0:
        return float(manual), base, peak, alpha
    t = base * (peak / base) ** alpha
    t = min(max(t, 2.5 * base, MIN_THRESHOLD), 0.8 * peak)
    return float(t), base, peak, alpha


# -- segments -------------------------------------------------------------

def _calm_core(s: np.ndarray, detail: np.ndarray, a: int, b: int, threshold: float,
               base: float, rate: float):
    seg = s[a:b + 1]
    det = detail[a:b + 1]
    w = max(1, int(round(0.2 * rate)))
    ma = running_mean(seg, w)
    n = seg.size
    # Fades (from white/black, or cross-dissolves) lower the image detail: only
    # frames showing close to the segment's full detail may be the calm point.
    sharp_ok = det >= DETAIL_KEEP * float(np.percentile(det, 95))
    # Least motion, with a slight preference for the middle of the segment.
    centre_pen = 1e-3 * np.abs(np.arange(n) - (n - 1) / 2.0) / max(n, 1)
    score = ma + centre_pen + np.where(sharp_ok, 0.0, 1e6)
    m = int(np.argmin(score))
    lim = max(3.0 * ma[m], 0.12 * threshold, base)
    lo = m
    while lo > 0 and ma[lo - 1] <= lim and sharp_ok[lo - 1]:
        lo -= 1
    hi = m
    while hi < n - 1 and ma[hi + 1] <= lim and sharp_ok[hi + 1]:
        hi += 1
    core_motion = float(np.median(seg[lo:hi + 1]))
    return a + lo, a + hi, a + m, core_motion


def _make_segment(ctx: "SegmentResult", analysis: MotionAnalysis, a: int, b: int) -> Segment:
    s, thr, rate = ctx.smoothed, ctx.threshold, ctx.rate
    t0 = float(analysis.times[a])
    t1 = float(analysis.times[b]) + 1.0 / rate
    cs, ce, calm, motion = _calm_core(s, analysis.detail, a, b, thr, ctx.base, rate)
    flags = []
    if t1 - t0 < ctx.params.short_flag:
        flags.append("short")
    if float(np.median(analysis.detail[a:b + 1])) < max(BLANK_DETAIL, 0.05 * ctx.detail_ref):
        flags.append("blank")
    return Segment(a, b, t0, t1, cs, ce, calm, motion, flags)


def _split_candidates(ctx: "SegmentResult", analysis: MotionAnalysis, seg: Segment) -> list:
    """Possible hidden transitions inside a stable segment.

    A gentle transition between two very similar pages (e.g. a cross-fade
    where only the page number changes) stays below the global threshold.
    It still shows up as a bump far above the segment's own noise level.
    Candidates must be verified on the actual frames (see
    :func:`video2book.project.refine_segments`).
    """
    s = ctx.smoothed[seg.start:seg.end + 1]
    lvl = float(np.median(s))
    bump = max(6.0 * lvl, 0.06 * ctx.threshold, SPLIT_MIN)
    min_len = max(1, int(round(ctx.params.min_duration * ctx.rate)))
    out = []
    for lo, hi in runs(s > bump):
        if lo < min_len or (s.size - 1 - hi) < min_len:
            continue  # not enough still frames on one side to be a page
        out.append((float(s[lo:hi + 1].max()), seg.start + lo, seg.start + hi))
    out.sort(reverse=True)
    return [(i0, i1) for _, i0, i1 in out[:MAX_SPLITS]]


def split_segment(result: "SegmentResult", analysis: MotionAnalysis, seg: Segment,
                  bump: tuple) -> tuple[Segment, Segment]:
    """Split ``seg`` around the bump ``(i0, i1)`` into two segments."""
    i0, i1 = bump
    return (_make_segment(result, analysis, seg.start, i0 - 1),
            _make_segment(result, analysis, i1 + 1, seg.end))


def detect_segments(analysis: MotionAnalysis, params: SegmentParams | None = None) -> SegmentResult:
    params = params or SegmentParams()
    rate = analysis.sample_rate
    k = max(3, int(round(params.smooth * rate)) | 1)
    s = running_median(analysis.diff, k)
    if s.size:
        s[0] = s[1] if s.size > 1 else 0.0
    thr, base, peak, alpha = compute_threshold(s, rate, params.sensitivity, params.threshold)
    detail_ref = float(np.median(analysis.detail)) if analysis.n else 0.0
    result = SegmentResult([], thr, base, peak, alpha, s.astype(np.float32), rate=rate,
                           detail_ref=detail_ref, params=params)
    dt = 1.0 / rate if rate > 0 else 0.0
    for a, b in runs(s <= thr):
        if float(analysis.times[b]) + dt - float(analysis.times[a]) < params.min_duration:
            continue
        seg = _make_segment(result, analysis, a, b)
        seg.splits = _split_candidates(result, analysis, seg)
        result.segments.append(seg)
    return result


def count_for_sensitivity(analysis: MotionAnalysis, params: SegmentParams, sensitivity: float) -> int:
    p = SegmentParams(**{**params.to_dict(), "sensitivity": sensitivity, "threshold": None})
    return detect_segments(analysis, p).count


def suggest_sensitivity(analysis: MotionAnalysis, params: SegmentParams, target_segments: int,
                        steps: int = 41) -> tuple[float, int]:
    """Sensitivity whose segment count is closest to ``target_segments``.

    Only the (cached) motion signal is used, so this is instantaneous.
    Ties are broken towards the current sensitivity.
    """
    best = (params.sensitivity, count_for_sensitivity(analysis, params, params.sensitivity))
    best_err = abs(best[1] - target_segments)
    for sens in np.linspace(0.0, 1.0, steps):
        c = count_for_sensitivity(analysis, params, float(sens))
        err = abs(c - target_segments)
        if err < best_err or (err == best_err and abs(sens - params.sensitivity) < abs(best[0] - params.sensitivity)):
            best, best_err = (float(round(sens, 3)), c), err
    return best
