"""Segment detection on hand-made motion signals (no video needed)."""

import numpy as np
import pytest

from video2book.motion import MotionAnalysis
from video2book.segments import (SegmentParams, detect_segments, running_median, runs,
                                 suggest_sensitivity)
from video2book.video import VideoInfo

FPS = 30.0


def fake_analysis(diff, detail=None):
    n = len(diff)
    info = VideoInfo(path="x", width=100, height=100, coded_width=100, coded_height=100, fps=FPS,
                     duration=n / FPS, frame_count=n, rotation=0, codec="h264", pix_fmt="yuv420p",
                     time_base_num=1, time_base_den=int(FPS), start_pts=0, file_size=0)
    return MotionAnalysis(
        info=info, step=1, width=160, index=np.arange(n), pts=np.arange(n),
        times=np.arange(n) / FPS, diff=np.asarray(diff, np.float32),
        brightness=np.full(n, 128, np.float32),
        detail=np.asarray(detail if detail is not None else np.full(n, 40.0), np.float32),
    )


def slideshow_signal(holds, trans=0.4, noise=0.03, peak=8.0, seed=0, hold_level=None):
    """Holds (seconds) separated by bell-shaped transitions."""
    rng = np.random.default_rng(seed)
    sig, truth = [], []
    for k, h in enumerate(holds):
        n = int(h * FPS)
        start = len(sig)
        level = noise if hold_level is None else hold_level[k]
        sig += list(np.abs(rng.normal(level, level * 0.5, n)))
        truth.append((start / FPS, (start + n) / FPS))
        if k < len(holds) - 1:
            m = int(trans * FPS)
            x = np.linspace(-2.5, 2.5, m)
            sig += list(peak * np.exp(-x ** 2) + rng.uniform(0, noise, m))
    return np.array(sig), truth


def test_counts_irregular_holds():
    holds = [1.5, 0.5, 3.0, 0.8, 2.2, 0.45, 4.0, 1.0]
    diff, truth = slideshow_signal(holds)
    res = detect_segments(fake_analysis(diff))
    assert res.count == len(holds)
    for seg, (t0, t1) in zip(res.segments, truth):
        assert abs(seg.t_start - t0) < 0.25 and abs(seg.t_end - t1) < 0.25
        assert seg.core_start >= seg.start and seg.core_end <= seg.end


def test_single_frame_spikes_do_not_split():
    diff, _ = slideshow_signal([3.0, 3.0])
    diff[30] = 15.0      # keyframe "pop" inside the first hold
    diff[60:62] = 12.0   # two-frame glitch
    res = detect_segments(fake_analysis(diff))
    assert res.count == 2


def test_threshold_is_scale_invariant():
    diff, _ = slideshow_signal([1.0, 2.0, 1.5, 0.6, 2.5])
    a = detect_segments(fake_analysis(diff))
    b = detect_segments(fake_analysis(diff * 8))
    assert a.count == b.count == 5
    assert b.threshold == pytest.approx(a.threshold * 8, rel=0.25)


def test_slow_zoom_holds_are_pages():
    # "Ken Burns" holds: sustained small motion (0.3) far below transition peaks.
    holds = [2.0, 2.0, 2.0, 2.0]
    diff, _ = slideshow_signal(holds, hold_level=[0.3, 0.02, 0.3, 0.02])
    res = detect_segments(fake_analysis(diff))
    assert res.count == 4
    assert res.threshold > 0.3


def test_min_duration_filters_brief_stops():
    # A 0.15 s pause mid-swipe; with the transitions' calm tails it is still
    # for ~0.35 s in total.
    diff, _ = slideshow_signal([2.0, 0.15, 2.0])
    assert detect_segments(fake_analysis(diff), SegmentParams(min_duration=0.5)).count == 2
    assert detect_segments(fake_analysis(diff), SegmentParams(min_duration=0.1)).count == 3


def test_short_segments_are_flagged():
    diff, _ = slideshow_signal([2.0, 0.25, 2.0])
    res = detect_segments(fake_analysis(diff))
    assert [("short" in s.flags) for s in res.segments] == [False, True, False]


def test_sensitivity_and_manual_threshold():
    diff, _ = slideshow_signal([1.0] * 6)
    lo = detect_segments(fake_analysis(diff), SegmentParams(sensitivity=0.0))
    hi = detect_segments(fake_analysis(diff), SegmentParams(sensitivity=1.0))
    assert hi.threshold < lo.threshold
    man = detect_segments(fake_analysis(diff), SegmentParams(threshold=100.0))
    assert man.threshold == 100.0 and man.count == 1  # everything is "still"


def test_blank_segments_flagged():
    diff, _ = slideshow_signal([2.0, 2.0, 2.0])
    detail = np.full(len(diff), 40.0)
    detail[-60:] = 0.2  # last hold is a black frame
    res = detect_segments(fake_analysis(diff, detail))
    assert "blank" in res.segments[-1].flags
    assert all("blank" not in s.flags for s in res.segments[:-1])


def test_hidden_transition_candidate():
    # Two holds joined by a *gentle* change (cross-fade between near-identical
    # pages): stays below the threshold but is a clear bump above the noise.
    diff, _ = slideshow_signal([2.0, 2.0, 2.0], noise=0.01)
    mid = int(1.0 * FPS)
    diff[mid:mid + 10] = 0.5
    res = detect_segments(fake_analysis(diff))
    assert res.count == 3
    assert len(res.segments[0].splits) == 1
    i0, i1 = res.segments[0].splits[0]
    assert mid - 3 <= i0 <= mid + 3


def test_suggest_sensitivity_reaches_target():
    # Weak transitions (fades) mixed with strong ones: a higher sensitivity
    # is needed to see all pages.
    diff, _ = slideshow_signal([1.5] * 8, peak=8.0)
    trans = [i for i in range(len(diff)) if diff[i] > 1]
    first = trans[: len(trans) // 3]
    diff[first] *= 0.1   # the first transitions are much weaker
    params = SegmentParams(sensitivity=0.0)
    base = detect_segments(fake_analysis(diff), params).count
    sens, count = suggest_sensitivity(fake_analysis(diff), params, 8)
    assert count == 8 and base < 8 and sens > 0.0


def test_helpers():
    x = np.array([0, 0, 9, 0, 0, 1, 1, 1, 0], float)
    assert running_median(x, 3)[2] == 0
    assert runs(np.array([1, 1, 0, 1, 0, 0, 1], bool)) == [(0, 1), (3, 3), (6, 6)]
