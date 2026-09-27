"""Video reading: metadata, rotation, random access."""

import numpy as np

from conftest import identify
from video2book.motion import analyze_motion
from video2book.video import FrameReader, fingerprint, iter_luma, probe


def test_probe_and_fingerprint(video_factory):
    path, cfg, truth = video_factory("base")
    info = probe(path)
    assert (info.width, info.height) == (cfg.width, cfg.height)
    assert info.rotation == 0
    assert abs(info.fps - cfg.fps) < 0.01
    assert abs(info.duration - truth.holds[-1][1]) < 0.2
    assert len(fingerprint(path)) == 16


def test_rotation_metadata_is_honoured(video_factory):
    path, cfg, truth = video_factory("rot90", rotation=90)
    info = probe(path)
    assert info.rotation == 90
    assert (info.coded_width, info.coded_height) == (cfg.height, cfg.width)  # stored landscape
    assert (info.width, info.height) == (cfg.width, cfg.height)              # shown portrait
    t = sum(truth.holds[0]) / 2
    with FrameReader(path, info) as r:
        _, frame = r.get_at_time(t)
    assert frame.shape[:2] == (cfg.height, cfg.width)
    assert identify(frame, cfg) == truth.sequence[0]


def test_random_access_matches_sequential_decode(video_factory):
    path, cfg, _ = video_factory("base")
    a = analyze_motion(path)
    wanted = [int(a.pts[i]) for i in (0, 5, 100, 101, 400, a.n - 1)]
    with FrameReader(path) as r:
        got = r.get_by_pts(list(reversed(wanted)))  # order must not matter
    assert set(got) == set(wanted)
    # compare luma of the random-access frames with the sequential pass
    small = {p: g for _, p, _, g in iter_luma(path, target_width=64)}
    import cv2
    for p in wanted:
        y = cv2.cvtColor(got[p], cv2.COLOR_RGB2GRAY)
        y = cv2.resize(y, small[p].shape[::-1], interpolation=cv2.INTER_AREA).astype(float)
        y = 16 + y * 219 / 255  # the analysis pass reads raw (limited-range) luma
        assert np.abs(y - small[p]).mean() < 3.0


def test_motion_signal_shape(video_factory):
    path, cfg, truth = video_factory("base")
    a = analyze_motion(path, step=2)
    assert a.step == 2
    assert np.all(np.diff(a.times) > 0)
    assert a.diff[0] == 0 and a.diff.max() > 5
    assert abs(a.sample_rate - cfg.fps / 2) < 1
