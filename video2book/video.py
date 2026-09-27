"""Reading videos with PyAV (FFmpeg is bundled in the ``av`` wheel).

Two access patterns are needed:

* a fast sequential pass producing small grayscale frames for motion analysis
  (:func:`iter_luma`), which reads the Y plane directly and never converts the
  full-resolution frame;
* random access to a handful of full-colour frames by presentation timestamp
  (:class:`FrameReader`), used to build page images, neighbour previews and
  "add page from timestamp".

Frames are identified by their ``pts`` (presentation timestamp in stream
time-base units), which is stable between passes even for variable-frame-rate
phone recordings. Rotation metadata (display matrix) is honoured: returned
colour frames are upright. The motion signal does not depend on rotation, so
the analysis pass skips it.
"""

from __future__ import annotations

import hashlib
import itertools
import os
from dataclasses import asdict, dataclass
from fractions import Fraction
from typing import Callable, Iterable, Iterator, Optional

import av
import cv2
import numpy as np

ProgressFn = Optional[Callable[[float, str], None]]

# Pixel formats whose first plane is 8-bit luma.
_LUMA8 = {
    "yuv420p", "yuvj420p", "yuv422p", "yuvj422p", "yuv444p", "yuvj444p",
    "nv12", "nv21", "yuv411p", "yuv410p", "yuv440p", "yuvj440p", "gray",
}
# 10/12/16-bit little-endian planar YUV (HEVC Main10 from iPhones, etc.).
_LUMA16 = {
    "yuv420p10le", "yuv422p10le", "yuv444p10le", "yuv420p12le", "yuv422p12le",
    "yuv444p12le", "p010le", "p016le", "gray10le", "gray12le", "gray16le",
}


@dataclass
class VideoInfo:
    path: str
    width: int          # displayed (upright) width
    height: int         # displayed (upright) height
    coded_width: int
    coded_height: int
    fps: float
    duration: float     # seconds
    frame_count: int    # container estimate (may be 0 if unknown)
    rotation: int       # display-matrix rotation in degrees (counter-clockwise)
    codec: str
    pix_fmt: str
    time_base_num: int
    time_base_den: int
    start_pts: int
    file_size: int

    @property
    def time_base(self) -> Fraction:
        return Fraction(self.time_base_num, self.time_base_den)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "VideoInfo":
        return cls(**d)

    def describe(self) -> str:
        rot = f", rotation {self.rotation}°" if self.rotation else ""
        return (f"{self.width}x{self.height} {self.codec} @ {self.fps:.2f} fps, "
                f"{self.duration:.1f} s, ~{self.frame_count} frames{rot}")


def rotation_k(rotation: int) -> int:
    """Number of counter-clockwise quarter turns (for ``np.rot90``)."""
    return int(round(rotation / 90.0)) % 4


def apply_rotation(img: np.ndarray, rotation: int) -> np.ndarray:
    k = rotation_k(rotation)
    return np.ascontiguousarray(np.rot90(img, k)) if k else img


def _open(path: str):
    container = av.open(str(path))
    if not container.streams.video:
        container.close()
        raise ValueError(f"No video stream found in {path}")
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    return container, stream


def probe(path: str) -> VideoInfo:
    """Read metadata (decodes only the first frame, to get rotation)."""
    path = str(path)
    container, stream = _open(path)
    try:
        rate = stream.average_rate or stream.guessed_rate or stream.base_rate
        fps = float(rate) if rate else 30.0
        tb = stream.time_base or Fraction(1, 1000)
        if stream.duration:
            duration = float(stream.duration * tb)
        elif container.duration:
            duration = container.duration / 1_000_000
        else:
            duration = 0.0
        rotation = 0
        pix_fmt = stream.codec_context.pix_fmt or ""
        first_pts = stream.start_time or 0
        for frame in container.decode(stream):
            rotation = int(getattr(frame, "rotation", 0) or 0)
            pix_fmt = frame.format.name
            if frame.pts is not None:
                first_pts = frame.pts
            break
        cw, ch = stream.codec_context.width, stream.codec_context.height
        w, h = (ch, cw) if rotation_k(rotation) % 2 else (cw, ch)
        frames = int(stream.frames or 0) or int(round(duration * fps))
        return VideoInfo(
            path=os.path.abspath(path), width=w, height=h, coded_width=cw,
            coded_height=ch, fps=fps, duration=duration, frame_count=frames,
            rotation=rotation, codec=stream.codec_context.name, pix_fmt=pix_fmt,
            time_base_num=tb.numerator, time_base_den=tb.denominator,
            start_pts=int(first_pts), file_size=os.path.getsize(path),
        )
    finally:
        container.close()


def fingerprint(path: str) -> str:
    """Content-based id of a video file (size + three 1 MB chunks).

    Independent of the file name/location, so re-opening the same video from a
    different folder (or re-uploading it) reuses the cache.
    """
    size = os.path.getsize(path)
    h = hashlib.sha1(str(size).encode())
    chunk = 1 << 20
    with open(path, "rb") as f:
        for off in (0, max(0, size // 2 - chunk // 2), max(0, size - chunk)):
            f.seek(off)
            h.update(f.read(chunk))
    return h.hexdigest()[:16]


def _luma(frame: "av.VideoFrame") -> np.ndarray:
    """Full-resolution 8-bit luma of a frame, without colour conversion."""
    name = frame.format.name
    w, h = frame.width, frame.height
    plane = frame.planes[0]
    if name in _LUMA8:
        buf = np.frombuffer(plane, np.uint8)
        return buf.reshape(-1, plane.line_size)[:h, :w]
    if name in _LUMA16:
        buf = np.frombuffer(plane, "<u2")
        arr = buf.reshape(-1, plane.line_size // 2)[:h, :w]
        shift = 8 if name.startswith(("p016", "gray16")) else (4 if "12" in name else 2)
        if name.startswith("p010"):
            shift = 8  # p010 stores 10 bits in the high bits
        return (arr >> shift).astype(np.uint8)
    return frame.to_ndarray(format="gray")


def analysis_size(width: int, height: int, target_width: int) -> tuple[int, int]:
    """Downscaled size used for motion analysis (keeps aspect ratio of the coded frame)."""
    scale = min(1.0, target_width / float(max(width, 1)))
    return max(8, int(round(width * scale))), max(8, int(round(height * scale)))


def iter_luma(path: str, target_width: int = 160, step: int = 1,
              progress: ProgressFn = None) -> Iterator[tuple[int, int, float, np.ndarray]]:
    """Yield ``(index, pts, time_s, small_gray)`` for every ``step``-th frame.

    Frames are *not* rotated (the motion signal is rotation invariant). Memory
    use is constant: one small frame at a time.
    """
    container, stream = _open(path)
    try:
        tb = float(stream.time_base or Fraction(1, 1000))
        total = stream.frames or 0
        if not total and stream.duration and stream.average_rate:
            total = int(stream.duration * float(stream.time_base) * float(stream.average_rate))
        size = None
        last_report = -1.0
        fallback_dt = 1.0 / float(stream.average_rate or 30)
        for idx, frame in enumerate(container.decode(stream)):
            if idx % step:
                continue
            if size is None:
                size = analysis_size(frame.width, frame.height, target_width)
            y = _luma(frame)
            small = cv2.resize(y, size, interpolation=cv2.INTER_AREA)
            pts = frame.pts if frame.pts is not None else idx
            t = frame.pts * tb if frame.pts is not None else idx * fallback_dt
            yield idx, int(pts), float(t), small
            if progress and total:
                frac = min(1.0, idx / total)
                if frac - last_report >= 0.01:
                    last_report = frac
                    progress(frac, f"Analysing motion: frame {idx}/{total}")
        if progress:
            progress(1.0, "Motion analysis done")
    finally:
        container.close()


def _to_rgb(frame: "av.VideoFrame", rotation: int) -> np.ndarray:
    return apply_rotation(frame.to_ndarray(format="rgb24"), rotation)


class FrameReader:
    """Random access to full-colour, upright frames by pts or time.

    Requests are served by decoding forward. The decoder stays positioned
    between calls, so asking for frames in increasing time order (as the
    pipeline does, page after page) never re-decodes the same stretch; a seek
    only happens to go backwards or to skip more than ``SEEK_GAP_S`` seconds.
    """

    SEEK_GAP_S = 3.0

    def __init__(self, path: str, info: Optional[VideoInfo] = None):
        self.path = str(path)
        self.info = info or probe(path)
        self._container, self._stream = _open(self.path)
        self._tb = self._stream.time_base or Fraction(1, 1000)
        self._gen = None          # live decoding iterator
        self._last_pts = None     # pts of the last decoded frame
        self._last_frame = None

    def close(self) -> None:
        try:
            self._container.close()
        except Exception:  # pragma: no cover - best effort
            pass

    def __enter__(self) -> "FrameReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def time_to_pts(self, t: float) -> int:
        return int(round(self.info.start_pts + t / float(self._tb)))

    def pts_to_time(self, pts: int) -> float:
        return float(pts * self._tb)

    def _start(self, target: int) -> None:
        """Seek to the keyframe before ``target`` and start a new decoder."""
        backoff = 0
        one_s = int(1.0 / float(self._tb))
        while True:
            self._container.seek(max(0, int(target - backoff)), stream=self._stream, backward=True,
                                 any_frame=False)
            gen = self._container.decode(self._stream)
            first = None
            for f in gen:
                if f.pts is not None:
                    first = f
                    break
            if first is None:
                self._gen = iter(())
                return
            # Seek landed after the target (imprecise index): back off further.
            if first.pts > target and target - backoff > self.info.start_pts:
                backoff = backoff * 2 + one_s
                continue
            self._gen = itertools.chain([first], gen)
            self._last_pts = None
            return

    def get_by_pts(self, pts_list: Iterable[int], progress: ProgressFn = None) -> dict[int, np.ndarray]:
        """Return ``{requested_pts: rgb_frame}``.

        A requested pts that does not exist exactly is served by the first
        frame at or after it (or the last frame of the video).
        """
        wanted = sorted({int(p) for p in pts_list})
        result: dict[int, np.ndarray] = {}
        gap = int(self.SEEK_GAP_S / float(self._tb))
        rotation = self.info.rotation
        i = 0
        while i < len(wanted):
            target = wanted[i]
            if self._last_frame is not None and target == self._last_pts:
                result[target] = _to_rgb(self._last_frame, rotation)
                i += 1
                continue
            live = self._gen is not None and self._last_pts is not None
            if not (live and self._last_pts < target <= self._last_pts + gap):
                self._start(target)
            exhausted = True
            for frame in self._gen:
                if frame.pts is None:
                    continue
                self._last_pts, self._last_frame = frame.pts, frame
                if frame.pts < target:
                    continue
                rgb = None
                while i < len(wanted) and wanted[i] <= frame.pts:
                    if rgb is None:
                        rgb = _to_rgb(frame, rotation)
                    result[wanted[i]] = rgb
                    i += 1
                if progress:
                    progress(i / len(wanted), f"Reading frames {i}/{len(wanted)}")
                if i >= len(wanted):
                    exhausted = False
                    break
                target = wanted[i]
                if target - frame.pts > gap:
                    exhausted = False
                    break  # seek instead of decoding a long stretch
            if exhausted:
                # End of stream: serve the remaining requests with the last frame.
                self._gen = None
                if self._last_frame is None:
                    break
                rgb = _to_rgb(self._last_frame, rotation)
                while i < len(wanted):
                    result[wanted[i]] = rgb
                    i += 1
        return result

    def get_at_time(self, t: float) -> tuple[int, np.ndarray]:
        """Frame shown at time ``t`` (seconds). Returns ``(pts, rgb)``."""
        pts = self.time_to_pts(t)
        frames = self.get_by_pts([pts])
        return pts, frames[pts]
