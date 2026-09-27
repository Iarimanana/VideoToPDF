"""Frame-to-frame motion signal.

For every analysed frame we store the mean absolute difference (0-255 scale)
from the previous analysed frame, computed on a small grayscale copy
(default 160 px wide). A few cheap per-frame statistics are stored too
(brightness, detail) so later stages can reason about blank frames without
re-reading the video.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from .video import ProgressFn, VideoInfo, iter_luma, probe


@dataclass
class MotionAnalysis:
    info: VideoInfo
    step: int
    width: int                 # analysis width in px
    index: np.ndarray          # decoded frame index of each sample (int64)
    pts: np.ndarray            # presentation timestamps (int64)
    times: np.ndarray          # seconds (float64)
    diff: np.ndarray           # motion vs previous sample (float32), diff[0] = 0
    brightness: np.ndarray     # mean luma (float32)
    detail: np.ndarray         # mean |Laplacian| of the small frame (float32)
    elapsed: float = 0.0
    extra: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return int(self.diff.shape[0])

    @property
    def sample_rate(self) -> float:
        """Analysed samples per second."""
        if self.n > 1 and self.times[-1] > self.times[0]:
            return (self.n - 1) / float(self.times[-1] - self.times[0])
        return self.info.fps / max(1, self.step)

    def nearest(self, t: float) -> int:
        """Index of the sample closest to time ``t``."""
        i = int(np.searchsorted(self.times, t))
        if i <= 0:
            return 0
        if i >= self.n:
            return self.n - 1
        return i if abs(self.times[i] - t) < abs(self.times[i - 1] - t) else i - 1

    # -- persistence -----------------------------------------------------
    def save(self, path: str) -> None:
        import json
        np.savez_compressed(
            path, index=self.index, pts=self.pts, times=self.times, diff=self.diff,
            brightness=self.brightness, detail=self.detail,
            meta=np.array(json.dumps({
                "info": self.info.to_dict(), "step": self.step, "width": self.width,
                "elapsed": self.elapsed, "extra": self.extra,
            })),
        )

    @classmethod
    def load(cls, path: str) -> "MotionAnalysis":
        import json
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            return cls(
                info=VideoInfo.from_dict(meta["info"]), step=meta["step"], width=meta["width"],
                index=z["index"], pts=z["pts"], times=z["times"], diff=z["diff"],
                brightness=z["brightness"], detail=z["detail"],
                elapsed=meta.get("elapsed", 0.0), extra=meta.get("extra", {}),
            )


def analyze_motion(path: str, step: int = 1, width: int = 160,
                   progress: ProgressFn = None, info: VideoInfo | None = None) -> MotionAnalysis:
    """Stream the video once and compute the motion signal."""
    t0 = time.time()
    info = info or probe(path)
    step = max(1, int(step))
    idx, pts, times, diff, bright, detail = [], [], [], [], [], []
    prev = None
    for i, p, t, small in iter_luma(path, target_width=width, step=step, progress=progress):
        g = small.astype(np.float32)
        d = 0.0 if prev is None else float(cv2.absdiff(g, prev).mean())
        prev = g
        idx.append(i)
        pts.append(p)
        times.append(t)
        diff.append(d)
        bright.append(float(g.mean()))
        detail.append(float(np.abs(cv2.Laplacian(small, cv2.CV_16S, ksize=3)).mean()))
    if not diff:
        raise ValueError("Could not decode any frame from the video.")
    return MotionAnalysis(
        info=info, step=step, width=width,
        index=np.asarray(idx, np.int64), pts=np.asarray(pts, np.int64),
        times=np.asarray(times, np.float64), diff=np.asarray(diff, np.float32),
        brightness=np.asarray(bright, np.float32), detail=np.asarray(detail, np.float32),
        elapsed=time.time() - t0,
    )
