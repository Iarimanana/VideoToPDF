"""Building the best image for each stable segment.

* ``median`` (default): per-pixel temporal median of up to ``max_frames``
  frames spread over the segment's calm core. This removes compression noise
  and flicker. Frames that do not line up with the calm frame (slow zoom/pan,
  end of a transition) are left out first; if too few aligned frames remain
  the segment is treated as *moving* and the sharpest frame is used instead.
* ``sharpest``: the single frame with the highest Laplacian variance.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

from .motion import MotionAnalysis
from .segments import Segment
from .video import FrameReader

METHODS = ("median", "sharpest")


@dataclass
class ExtractParams:
    method: str = "median"
    max_frames: int = 9       # frames combined by the median
    candidates: int = 12      # frames examined by "sharpest"
    align_tol: float = 1.6    # max drift (x noise) for a frame to join the median

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Representative:
    image: np.ndarray          # RGB uint8, upright, full resolution
    pts: list                  # pts of the frames used
    t: float                   # representative time (calm point), seconds
    sharpness: float
    moving: bool
    method: str

    def meta(self) -> dict:
        return {"pts": [int(p) for p in self.pts], "t": self.t, "sharpness": self.sharpness,
                "moving": self.moving, "method": self.method}


def sharpness(img: np.ndarray, max_side: int = 1000) -> float:
    """Variance of the Laplacian on a grayscale copy (higher = sharper)."""
    g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h, w = g.shape[:2]
    s = max_side / float(max(h, w))
    if s < 1:
        g = cv2.resize(g, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def _small_gray(img: np.ndarray, width: int = 256) -> np.ndarray:
    h, w = img.shape[:2]
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
    return cv2.resize(g, (width, max(1, int(round(h * width / w)))), interpolation=cv2.INTER_AREA).astype(np.float32)


def temporal_median(frames: list[np.ndarray]) -> np.ndarray:
    """Per-pixel median of uint8 frames without converting to float."""
    if len(frames) == 1:
        return frames[0].copy()
    stack = np.stack(frames, axis=0)
    k = (len(frames) - 1) // 2
    return np.partition(stack, k, axis=0)[k]


def candidate_samples(seg: Segment, n: int) -> list[int]:
    """Sample indices spread evenly over the calm core, including the calm point."""
    lo, hi = seg.core_start, seg.core_end
    idx = set(np.round(np.linspace(lo, hi, max(1, min(n, hi - lo + 1)))).astype(int).tolist())
    idx.add(int(seg.calm))
    return sorted(idx)


def combine(frames: list[np.ndarray], calm_pos: int, method: str = "median",
            max_frames: int = 9, align_tol: float = 1.6) -> tuple[np.ndarray, list[int], bool]:
    """Combine candidate frames. Returns ``(image, used_positions, moving)``."""
    if method == "sharpest":
        scores = [sharpness(f) for f in frames]
        best = int(np.argmax(scores))
        return frames[best], [best], False
    ref = _small_gray(frames[calm_pos])
    drift = np.array([float(np.abs(_small_gray(f) - ref).mean()) for f in frames])
    others = np.sort(drift[np.arange(len(frames)) != calm_pos])
    # Noise level = drift to the nearest-looking other frame. For a still page
    # every frame is at this level; for a moving shot drift grows with time.
    noise = float(others[0]) if others.size else 0.0
    tol = max(align_tol * noise, noise + 0.35, 0.5)
    ok = np.where(drift <= tol)[0].tolist()
    if len(ok) < 3 and len(frames) >= 3:
        # Moving (zoom/pan) shot: a median would blur, take the sharpest
        # among the frames closest to the calm point.
        near = np.argsort(drift)[:3].tolist()
        best = max(near, key=lambda i: sharpness(frames[i]))
        return frames[best], [best], True
    if len(ok) > max_frames:
        # keep the ones closest in time to the calm frame
        ok = sorted(ok, key=lambda i: abs(i - calm_pos))[:max_frames]
        ok.sort()
    return temporal_median([frames[i] for i in ok]), ok, False


def extract_segment(reader: FrameReader, analysis: MotionAnalysis, seg: Segment,
                    params: ExtractParams | None = None) -> Representative:
    params = params or ExtractParams()
    n = params.candidates if params.method == "sharpest" else params.max_frames * 2
    samples = candidate_samples(seg, n)
    pts = [int(analysis.pts[i]) for i in samples]
    got = reader.get_by_pts(pts)
    frames = [got[p] for p in pts]
    calm_pos = samples.index(int(seg.calm))
    img, used, moving = combine(frames, calm_pos, params.method, params.max_frames, params.align_tol)
    return Representative(
        image=img, pts=[pts[i] for i in used], t=float(analysis.times[seg.calm]),
        sharpness=sharpness(img), moving=moving, method=params.method,
    )


def extract_frame(reader: FrameReader, pts: int) -> Representative:
    """A single, user-chosen frame."""
    img = reader.get_by_pts([pts])[pts]
    return Representative(image=img, pts=[pts], t=reader.pts_to_time(pts),
                          sharpness=sharpness(img), moving=False, method="frame")
