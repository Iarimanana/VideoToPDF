import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from video2book.synthetic import SynthConfig, generate  # noqa: E402

# Small but complete videos: 12 pages, 360x640, two near-identical pairs,
# one swipe back (2 repeated pages), slides + fades, phone UI, H.264.
BASE = dict(n_pages=12, width=360, height=640, seed=11)


@pytest.fixture(scope="session")
def video_factory(tmp_path_factory):
    root = tmp_path_factory.mktemp("videos")
    made = {}

    def make(name: str, **overrides):
        if name not in made:
            cfg = SynthConfig(**{**BASE, **overrides})
            path = str(root / f"{name}.mp4")
            truth = generate(path, cfg)
            made[name] = (path, cfg, truth)
        return made[name]

    return make


@pytest.fixture(scope="session")
def cache_dir(tmp_path_factory):
    return str(tmp_path_factory.mktemp("cache"))


_REFS = {}


def identify(img: np.ndarray, cfg: SynthConfig) -> int:
    """Which synthetic page does a (full-frame, upright) image show?"""
    import cv2
    from video2book.synthetic import _Composer

    key = repr(cfg)
    if key not in _REFS:
        comp = _Composer(cfg)
        zooms = [0.0] if not cfg.ken_burns else list(np.linspace(0, cfg.ken_burns, 4))
        # refs[i] = list of frames of page i at the zoom levels it goes through
        _REFS[key] = [[comp.frame([(i, 0, 1.0, z)]) for z in zooms] for i in range(cfg.n_pages)]
    refs = _REFS[key]

    def small(x):
        # low resolution + blur: robust to a few % of zoom (Ken Burns), still
        # separates pages by their big number / bar code / text layout
        h, w = refs[0][0].shape[:2]
        x = cv2.resize(x.astype(np.float32), (w // 4, h // 4), interpolation=cv2.INTER_AREA)
        return cv2.GaussianBlur(x, (0, 0), 1.0)

    if key + "small" not in _REFS:
        _REFS[key + "small"] = [[small(z) for z in r] for r in refs]
    s = small(img)
    return int(np.argmin([min(np.abs(s - z).mean() for z in r) for r in _REFS[key + "small"]]))
