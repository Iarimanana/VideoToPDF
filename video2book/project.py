"""The complete pipeline, its disk cache and the editable project state.

Typical use::

    pipe = Pipeline("book.mp4")
    project = pipe.run()                   # analyse + detect + extract + dedup + crop
    pipe.export_pdf(project, "book.pdf")

The expensive steps are cached on disk (per video *content*, not file name):

* the motion signal (one full decode of the video), so changing sensitivity,
  minimum duration or threshold never re-reads the whole video;
* every page image (keyed by the exact frames it was built from), so
  re-detecting with other settings only decodes frames it has not seen yet.

A :class:`Project` is plain data (JSON-serialisable): the page list with
per-page edits (deleted, rotated, reordered, replaced frame...). The UI keeps
one in memory and saves it next to the cache so work can be resumed.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterator, Optional

import cv2
import numpy as np

from . import __version__
from .crop import CropResult, apply_crop, auto_crop, manual_crop, static_border
from .dedup import DedupParams, classify, compare, describe, find_duplicates
from .enhance import apply as apply_enhance
from .export import ExportOptions, best_aspect, export_pdf, export_zip
from .extract import ExtractParams, Representative, candidate_samples, extract_frame, extract_segment
from .motion import MotionAnalysis, analyze_motion
from .segments import (Segment, SegmentParams, SegmentResult, _split_candidates, detect_segments,
                       split_segment, suggest_sensitivity)
from .video import FrameReader, VideoInfo, fingerprint, probe

ProgressFn = Optional[Callable[[float, str], None]]
THUMB_WIDTH = 320


# -- settings / state ---------------------------------------------------------

@dataclass
class Settings:
    # analysis (changing these re-reads the video once)
    step: int = 1
    analysis_width: int = 160
    # detection (instant: works on the cached motion signal)
    sensitivity: float = 0.5
    threshold: Optional[float] = None
    min_duration: float = 0.3
    # page image
    method: str = "median"             # "median" | "sharpest"
    # duplicates
    dedup: bool = True
    # cropping
    crop_mode: str = "auto"            # "auto" | "manual" | "none"
    manual_crop: Optional[list] = None  # [x0, y0, x1, y1] relative to the frame
    perspective: bool = False
    # enhancement
    enhance: str = "original"          # "original" | "clean" | "bw"
    # export
    page_size: str = "uniform"         # "uniform" | "fit" | "a4" | "letter"
    fill: str = "auto"                 # padding colour: "auto" (page edge) | "white" | "black"
    dpi: int = 200
    jpeg_quality: int = 88
    ocr_lang: Optional[str] = None
    # review
    expected_pages: Optional[int] = None

    def segment_params(self) -> SegmentParams:
        return SegmentParams(sensitivity=self.sensitivity, threshold=self.threshold,
                             min_duration=self.min_duration)

    def extract_params(self) -> ExtractParams:
        return ExtractParams(method=self.method)

    def export_options(self) -> ExportOptions:
        return ExportOptions(page_size=self.page_size, dpi=self.dpi, jpeg_quality=self.jpeg_quality,
                             ocr_lang=self.ocr_lang, fill=self.fill)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Settings":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)


@dataclass
class Page:
    id: str
    image_key: str                 # cache key of the (uncropped, upright) page image
    t: float                       # representative time (s)
    t_start: float
    t_end: float
    source: str = "auto"           # "auto" | "manual" (added from timestamp)
    status: str = "keep"           # "keep" | "duplicate" | "blank" | "deleted"
    flags: list = field(default_factory=list)  # "possible_duplicate", "short", "moving", "picked"
    dup_of: Optional[str] = None   # id of the page this one duplicates / resembles
    dup_score: Optional[dict] = None
    rotation: int = 0              # user rotation, degrees clockwise
    crop: Optional[dict] = None    # automatic CropResult as dict
    sharpness: float = 0.0
    pts: list = field(default_factory=list)
    segment: Optional[int] = None

    @property
    def included(self) -> bool:
        return self.status == "keep"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Page":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)


@dataclass
class Project:
    video: str
    fingerprint: str
    info: dict
    settings: Settings
    pages: list = field(default_factory=list)
    threshold: float = 0.0
    base: float = 0.0
    peak: float = 0.0
    segment_count: int = 0
    static_box: Optional[list] = None
    timings: dict = field(default_factory=dict)
    version: str = __version__

    # -- queries --
    def page(self, pid: str) -> Page:
        for p in self.pages:
            if p.id == pid:
                return p
        raise KeyError(pid)

    def index(self, pid: str) -> int:
        for i, p in enumerate(self.pages):
            if p.id == pid:
                return i
        raise KeyError(pid)

    def included(self) -> list:
        return [p for p in self.pages if p.included]

    def page_number(self, pid: str) -> Optional[int]:
        """1-based number in the output PDF (None if not included)."""
        n = 0
        for p in self.pages:
            if p.included:
                n += 1
                if p.id == pid:
                    return n
        return None

    def summary(self) -> dict:
        c = lambda st: sum(1 for p in self.pages if p.status == st)
        return {
            "segments": self.segment_count,
            "pages": len(self.included()),
            "duplicates_merged": c("duplicate"),
            "possible_duplicates": sum(1 for p in self.pages if p.included and "possible_duplicate" in p.flags),
            "blank": c("blank"),
            "deleted": c("deleted"),
            "short": sum(1 for p in self.pages if p.included and "short" in p.flags),
            "moving": sum(1 for p in self.pages if p.included and "moving" in p.flags),
            "manual": sum(1 for p in self.pages if p.source == "manual"),
        }

    # -- edits --
    def set_status(self, pid: str, status: str) -> None:
        self.page(pid).status = status

    def rotate(self, pid: str, degrees: int) -> None:
        p = self.page(pid)
        p.rotation = (p.rotation + degrees) % 360

    def move(self, pid: str, new_index: int) -> None:
        i = self.index(pid)
        p = self.pages.pop(i)
        new_index = max(0, min(len(self.pages), new_index))
        self.pages.insert(new_index, p)

    def sort_by_time(self) -> None:
        self.pages.sort(key=lambda p: p.t)

    # -- persistence --
    def to_dict(self) -> dict:
        d = asdict(self)
        d["settings"] = self.settings.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Project":
        d = dict(d)
        d["settings"] = Settings.from_dict(d.get("settings", {}))
        d["pages"] = [Page.from_dict(p) for p in d.get("pages", [])]
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    def save(self, path: str) -> None:
        tmp = str(path) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=1)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str) -> "Project":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


# -- cache ----------------------------------------------------------------------

def default_cache_root() -> Path:
    env = os.environ.get("VIDEO2BOOK_CACHE")
    if env:
        return Path(env)
    if sys.platform.startswith("win"):
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / "video2book" / "cache"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "video2book"
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "video2book"


class Cache:
    """Per-video cache directory: motion signal, page images, project state."""

    def __init__(self, video_fingerprint: str, root: Optional[Path] = None, enabled: bool = True):
        self.enabled = enabled
        self.root = Path(root) if root else default_cache_root()
        self.dir = self.root / video_fingerprint
        self._mem: dict = {}
        if enabled:
            (self.dir / "frames").mkdir(parents=True, exist_ok=True)

    def analysis_path(self, step: int, width: int) -> Path:
        return self.dir / f"motion_s{step}_w{width}.npz"

    @property
    def project_path(self) -> Path:
        return self.dir / "project.json"

    def _paths(self, key: str) -> tuple[Path, Path, Path]:
        d = self.dir / "frames"
        return d / f"{key}.png", d / f"{key}.json", d / f"{key}_thumb.jpg"

    def has(self, key: str) -> bool:
        if key in self._mem:
            return True
        return self.enabled and self._paths(key)[0].exists()

    def put(self, key: str, rep: Representative) -> None:
        self._mem[key] = (rep.image, rep.meta())
        if not self.enabled:
            return
        img_p, meta_p, thumb_p = self._paths(key)
        cv2.imwrite(str(img_p), cv2.cvtColor(rep.image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 1])
        with open(meta_p, "w") as f:
            json.dump(rep.meta(), f)
        cv2.imwrite(str(thumb_p), cv2.cvtColor(make_thumb(rep.image), cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 85])

    def image(self, key: str) -> np.ndarray:
        if key in self._mem:
            return self._mem[key][0]
        img = cv2.imread(str(self._paths(key)[0]), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Page image {key} missing from cache")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def meta(self, key: str) -> dict:
        if key in self._mem:
            return self._mem[key][1]
        with open(self._paths(key)[1]) as f:
            return json.load(f)

    def thumb(self, key: str) -> np.ndarray:
        if self.enabled:
            p = self._paths(key)[2]
            if p.exists():
                img = cv2.imread(str(p), cv2.IMREAD_COLOR)
                if img is not None:
                    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return make_thumb(self.image(key))

    def forget_memory(self) -> None:
        self._mem.clear()


def make_thumb(img: np.ndarray, width: int = THUMB_WIDTH) -> np.ndarray:
    h, w = img.shape[:2]
    if w <= width:
        return img
    return cv2.resize(img, (width, max(1, int(round(h * width / w)))), interpolation=cv2.INTER_AREA)


def _rep_key(kind: str, params: dict, pts: list) -> str:
    h = hashlib.sha1(json.dumps([kind, params, [int(p) for p in pts]], sort_keys=True).encode())
    return h.hexdigest()[:20]


def _stage(progress: ProgressFn, lo: float, hi: float) -> ProgressFn:
    if progress is None:
        return None
    return lambda f, msg="": progress(lo + (hi - lo) * max(0.0, min(1.0, f)), msg)


def rotate_cw(img: np.ndarray, degrees: int) -> np.ndarray:
    k = (int(degrees) // 90) % 4
    return np.ascontiguousarray(np.rot90(img, -k)) if k else img


# -- pipeline -------------------------------------------------------------------

class Pipeline:
    """Runs and caches the processing of one video."""

    def __init__(self, video: str, settings: Optional[Settings] = None,
                 cache_root: Optional[str] = None, use_cache: bool = True):
        self.video = os.path.abspath(str(video))
        if not os.path.exists(self.video):
            raise FileNotFoundError(self.video)
        self.settings = settings or Settings()
        self.info: VideoInfo = probe(self.video)
        self.fingerprint = fingerprint(self.video)
        self.cache = Cache(self.fingerprint, Path(cache_root) if cache_root else None, enabled=use_cache)
        self._analysis: Optional[MotionAnalysis] = None
        self._analysis_key = None
        self._desc_cache: dict = {}

    # -- stage 1: motion ------------------------------------------------------
    def analysis(self, progress: ProgressFn = None) -> MotionAnalysis:
        s = self.settings
        key = (s.step, s.analysis_width)
        if self._analysis is not None and self._analysis_key == key:
            return self._analysis
        path = self.cache.analysis_path(*key)
        a = None
        if self.cache.enabled and path.exists():
            try:
                a = MotionAnalysis.load(str(path))
            except Exception:
                a = None
        if a is None:
            a = analyze_motion(self.video, step=s.step, width=s.analysis_width, progress=progress, info=self.info)
            if self.cache.enabled:
                a.save(str(path))
        elif progress:
            progress(1.0, "Motion analysis loaded from cache")
        self._analysis, self._analysis_key = a, key
        return a

    # -- stage 2: segments ----------------------------------------------------
    def segments(self) -> SegmentResult:
        return detect_segments(self.analysis(), self.settings.segment_params())

    def refine_segments(self, result: SegmentResult, reader: FrameReader) -> list:
        """Verify candidate hidden transitions on real frames; split where the
        page content actually changes (e.g. a cross-fade between two nearly
        identical pages)."""
        a = self.analysis()
        dp = DedupParams()
        min_d = result.params.min_duration
        out = []

        def refine(seg: Segment, depth: int = 0) -> list:
            if depth > 6 or not seg.splits:
                return [seg]
            for bump in seg.splits:
                left, right = split_segment(result, a, seg, bump)
                if left.duration < min_d or right.duration < min_d:
                    continue
                pl, pr = int(a.pts[left.calm]), int(a.pts[right.calm])
                fr = reader.get_by_pts([pl, pr])
                m = compare(describe(fr[pl]), describe(fr[pr]))
                if classify(m, dp) != "duplicate":
                    left.splits = _split_candidates(result, a, left)
                    right.splits = _split_candidates(result, a, right)
                    return refine(left, depth + 1) + refine(right, depth + 1)
            seg.splits = []
            return [seg]

        for seg in result.segments:
            out.extend(refine(seg))
        return out

    # -- stage 3: page images ---------------------------------------------------
    def _extract(self, reader: FrameReader, seg: Segment) -> tuple[str, dict]:
        a = self.analysis()
        ep = self.settings.extract_params()
        n = ep.candidates if ep.method == "sharpest" else ep.max_frames * 2
        cand_pts = [int(a.pts[i]) for i in candidate_samples(seg, n)] + [int(a.pts[seg.calm])]
        key = _rep_key("seg", ep.to_dict(), cand_pts)
        if not self.cache.has(key):
            self.cache.put(key, extract_segment(reader, a, seg, ep))
        return key, self.cache.meta(key)

    def representative(self, key: str) -> np.ndarray:
        return self.cache.image(key)

    def thumbnail(self, key: str) -> np.ndarray:
        return self.cache.thumb(key)

    # -- full run -------------------------------------------------------------
    def run(self, progress: ProgressFn = None, keep_manual_from: Optional[Project] = None) -> Project:
        t0 = time.time()
        timings = {}
        cached = self._analysis is not None or (self.cache.enabled and self.cache.analysis_path(
            self.settings.step, self.settings.analysis_width).exists())
        a = self.analysis(_stage(progress, 0.0, 0.55))
        timings["analysis"] = time.time() - t0   # actual time of this run (small when cached)
        t1 = time.time()
        result = self.segments()
        pages: list = []
        with FrameReader(self.video, self.info) as reader:
            segs = self.refine_segments(result, reader)
            timings["segments"] = time.time() - t1
            t2 = time.time()
            prog = _stage(progress, 0.55, 0.85)
            for i, seg in enumerate(segs):
                key, meta = self._extract(reader, seg)
                flags = list(seg.flags)
                if meta.get("moving"):
                    flags.append("moving")
                pages.append(Page(
                    id=uuid.uuid4().hex[:8], image_key=key, t=float(meta["t"]),
                    t_start=seg.t_start, t_end=seg.t_end, source="auto",
                    status="blank" if "blank" in flags else "keep",
                    flags=[f for f in flags if f != "blank"] + (["blank"] if "blank" in flags else []),
                    sharpness=float(meta["sharpness"]), pts=list(meta["pts"]), segment=i,
                ))
                if prog:
                    prog((i + 1) / max(1, len(segs)), f"Building page images {i + 1}/{len(segs)}")
        timings["extract"] = time.time() - t2
        if keep_manual_from is not None:
            pages += [p for p in keep_manual_from.pages if p.source == "manual"]
            pages.sort(key=lambda p: p.t)
        project = Project(
            video=self.video, fingerprint=self.fingerprint, info=self.info.to_dict(),
            settings=self.settings, pages=pages, threshold=result.threshold, base=result.base,
            peak=result.peak, segment_count=len(segs),
        )
        t3 = time.time()
        self.update_crops(project, _stage(progress, 0.85, 0.92))
        timings["crop"] = time.time() - t3
        t4 = time.time()
        if self.settings.dedup:
            self.deduplicate(project, _stage(progress, 0.92, 1.0))
        timings["dedup"] = time.time() - t4
        timings["total"] = time.time() - t0
        project.timings = {k: round(v, 2) for k, v in timings.items()}
        project.timings["analysis_cached"] = bool(cached)
        project.timings["first_analysis"] = round(a.elapsed, 2)  # full video read, first time
        if progress:
            progress(1.0, "Done")
        return project

    # -- cropping ---------------------------------------------------------------
    def update_crops(self, project: Project, progress: ProgressFn = None) -> None:
        """(Re)compute the static border and each page's automatic crop."""
        thumbs = [self.thumbnail(p.image_key) for p in project.pages if p.status != "blank"]
        box = None
        if len(thumbs) >= 3 and len({t.shape for t in thumbs}) == 1:
            small = static_border(thumbs, width=THUMB_WIDTH)
            if small is not None:
                sx = self.info.width / float(thumbs[0].shape[1])
                sy = self.info.height / float(thumbs[0].shape[0])
                box = [int(small[0] * sx), int(small[1] * sy),
                       min(self.info.width, int(np.ceil(small[2] * sx))),
                       min(self.info.height, int(np.ceil(small[3] * sy)))]
        project.static_box = box
        n = len(project.pages)
        for i, p in enumerate(project.pages):
            img = self.representative(p.image_key)
            b = tuple(box) if box and img.shape[1] == self.info.width and img.shape[0] == self.info.height else None
            p.crop = auto_crop(img, b).to_dict()
            if progress:
                progress((i + 1) / max(1, n), f"Finding page area {i + 1}/{n}")

    # -- duplicates ---------------------------------------------------------------
    def _descriptor(self, key: str, box):
        ck = (key, tuple(box) if box else None)
        d = self._desc_cache.get(ck)
        if d is None:
            img = self.representative(key)
            b = box if box and img.shape[1] >= box[2] and img.shape[0] >= box[3] else None
            d = describe(img, box=b)
            self._desc_cache[ck] = d
        return d

    def deduplicate(self, project: Project, progress: ProgressFn = None) -> None:
        """Mark duplicates (status) and uncertain look-alikes (flag). Pages the
        user deleted or that are blank are ignored; previous automatic
        decisions are recomputed."""
        pages = project.pages
        for p in pages:
            if p.status == "duplicate":
                p.status = "keep"
            if "possible_duplicate" in p.flags:
                p.flags.remove("possible_duplicate")
            p.dup_of, p.dup_score = None, None
        active = [p.status == "keep" for p in pages]
        descs = []
        for i, p in enumerate(pages):
            descs.append(self._descriptor(p.image_key, project.static_box) if active[i] else None)
            if progress:
                progress(0.5 * (i + 1) / max(1, len(pages)), "Comparing pages")
        dummy = next((d for d in descs if d is not None), None)
        if dummy is None:
            return
        descs = [d if d is not None else dummy for d in descs]
        res = find_duplicates(descs, DedupParams(), active)
        for p, r in zip(pages, res):
            if r.status == "duplicate":
                p.status = "duplicate"
            elif r.status == "possible":
                p.flags.append("possible_duplicate")
            if r.status != "unique":
                p.dup_of = pages[r.ref].id
                p.dup_score = r.match.to_dict()
        if progress:
            progress(1.0, "Duplicates checked")

    # -- rendering ------------------------------------------------------------------
    def crop_for(self, project: Project, page: Page, img: np.ndarray) -> Optional[CropResult]:
        s = project.settings
        if s.crop_mode == "manual" and s.manual_crop:
            return manual_crop(img, s.manual_crop)
        if s.crop_mode == "auto" and page.crop:
            return CropResult.from_dict(page.crop)
        return None

    def render(self, project: Project, page: Page, enhance: bool = True) -> np.ndarray:
        img = self.representative(page.image_key)
        img = apply_crop(img, self.crop_for(project, page, img), project.settings.perspective)
        img = rotate_cw(img, page.rotation)
        if enhance:
            img = apply_enhance(img, project.settings.enhance)
        return img

    def render_preview(self, project: Project, page: Page, enhance: bool = True) -> np.ndarray:
        """Like :meth:`render` but on the small cached thumbnail (fast, for the UI)."""
        s = project.settings
        th = self.thumbnail(page.image_key)
        scale = th.shape[1] / float(self.info.width)
        crop = None
        if s.crop_mode == "manual" and s.manual_crop:
            crop = manual_crop(th, s.manual_crop)
        elif s.crop_mode == "auto" and page.crop:
            c = CropResult.from_dict(page.crop)
            crop = CropResult(tuple(int(round(v * scale)) for v in c.box), c.method,
                              None if c.quad is None else c.quad * scale)
        img = apply_crop(th, crop, s.perspective)
        img = rotate_cw(img, page.rotation)
        if enhance:
            img = apply_enhance(img, s.enhance, preview=True)
        return img

    def iter_rendered(self, project: Project) -> Iterator[np.ndarray]:
        for p in project.included():
            yield self.render(project, p)

    # -- export -----------------------------------------------------------------------
    def page_aspect(self, project: Project) -> float:
        """Common page shape (w/h) for 'same size' PDFs: the one needing the
        least filling, measured on the small previews (fast)."""
        shapes = (self.render_preview(project, p, enhance=False).shape for p in project.included())
        return best_aspect(w / h for h, w in (sh[:2] for sh in shapes))

    def export_pdf(self, project: Project, out_path: str, progress: ProgressFn = None) -> str:
        pages = project.included()
        opts = project.settings.export_options()
        aspect = self.page_aspect(project) if opts.page_size == "uniform" else None
        return export_pdf(self.iter_rendered(project), out_path, opts, total=len(pages), progress=progress,
                          aspect=aspect)

    def export_zip(self, project: Project, out_path: str, fmt: str = "png", progress: ProgressFn = None) -> str:
        pages = project.included()
        return export_zip(self.iter_rendered(project), out_path, fmt, project.settings.export_options(),
                          total=len(pages), progress=progress)

    # -- review helpers ---------------------------------------------------------------
    def neighbor_times(self, page: Page, n: int = 8) -> list:
        """Candidate frame times for replacing a page image: spread over its
        segment, plus a little before and after it."""
        a = self.analysis()
        lo = max(0.0, page.t_start - 0.4)
        hi = min(float(a.times[-1]), page.t_end + 0.4)
        ts = np.linspace(lo, hi, n)
        idx = sorted({a.nearest(float(t)) for t in ts})
        return [(int(a.pts[i]), float(a.times[i])) for i in idx]

    def frames(self, pts_list: list) -> dict:
        with FrameReader(self.video, self.info) as reader:
            return reader.get_by_pts(pts_list)

    def use_frame(self, project: Project, page: Page, pts: int) -> None:
        """Replace a page's image with one specific frame."""
        key = _rep_key("frame", {}, [pts])
        if not self.cache.has(key):
            with FrameReader(self.video, self.info) as reader:
                self.cache.put(key, extract_frame(reader, int(pts)))
        meta = self.cache.meta(key)
        page.image_key = key
        page.t = float(meta["t"])
        page.pts = [int(pts)]
        page.sharpness = float(meta["sharpness"])
        if "picked" not in page.flags:
            page.flags.append("picked")
        img = self.representative(key)
        b = project.static_box
        page.crop = auto_crop(img, tuple(b) if b else None).to_dict()

    def add_page_at(self, project: Project, t: float) -> Page:
        """Insert a page built from the frame at time ``t`` (in time order)."""
        a = self.analysis()
        i = a.nearest(t)
        pts = int(a.pts[i])
        page = Page(id=uuid.uuid4().hex[:8], image_key="", t=float(a.times[i]),
                    t_start=float(a.times[i]), t_end=float(a.times[i]), source="manual")
        self.use_frame(project, page, pts)
        page.flags = ["manual"]
        pos = len(project.pages)
        for k, p in enumerate(project.pages):
            if p.t > page.t:
                pos = k
                break
        project.pages.insert(pos, page)
        return page

    def check_expected(self, project: Project) -> Optional[dict]:
        """Compare with the expected page count; suggest a sensitivity that
        gets closer (only if one exists)."""
        exp = project.settings.expected_pages
        if not exp:
            return None
        got = len(project.included())
        if got == exp:
            return {"ok": True, "expected": exp, "detected": got, "message": f"{got} pages, as expected."}
        params = project.settings.segment_params()
        auto_excluded = sum(1 for p in project.pages if p.source == "auto" and not p.included)
        manual = sum(1 for p in project.pages if p.source == "manual" and p.included)
        memo_key = (exp, json.dumps(params.to_dict(), sort_keys=True), project.segment_count, auto_excluded, manual)
        memo = getattr(self, "_expect_memo", {})
        if memo_key not in memo:
            memo[memo_key] = self._suggest_for(exp, params, auto_excluded, manual)
            self._expect_memo = memo
        sens, predicted = memo[memo_key]
        if got < exp:
            msg = f"Found {got} pages but you expect {exp} ({exp - got} missing)."
        else:
            msg = f"Found {got} pages but you expect {exp} ({got - exp} extra)."
        out = {"ok": False, "expected": exp, "detected": got, "message": msg}
        if predicted is not None and abs(predicted - exp) < abs(got - exp) and abs(sens - params.sensitivity) > 1e-6:
            direction = "higher" if sens > params.sensitivity else "lower"
            out["message"] += f" Try a {direction} sensitivity: {sens:.2f} should give about {predicted} pages."
            out["suggested_sensitivity"] = sens
            out["suggested_count"] = predicted
        else:
            out["message"] += (" No sensitivity setting gets closer: check the 'Worth a look' pages, "
                               "delete extras or add missing pages from their time in the video.")
        return out

    def _suggest_for(self, expected: int, params: SegmentParams, auto_excluded: int, manual: int,
                     max_checks: int = 6) -> tuple:
        """Sensitivity whose page count is closest to ``expected``.

        The motion signal gives segment counts instantly for every
        sensitivity; the few most promising values are then checked with the
        frame-verified hidden-transition step (a handful of frame reads each).
        Returns ``(sensitivity, predicted_pages)``.
        """
        a = self.analysis()
        target = expected - manual + auto_excluded
        by_count: dict = {}
        for sv in np.round(np.linspace(0.0, 1.0, 41), 3):
            p = SegmentParams(**{**params.to_dict(), "sensitivity": float(sv), "threshold": None})
            c = detect_segments(a, p).count
            # for each count keep the sensitivity closest to the current one
            if c not in by_count or abs(sv - params.sensitivity) < abs(by_count[c] - params.sensitivity):
                by_count[c] = float(sv)
        order = sorted(by_count.items(), key=lambda kv: (abs(kv[0] - target), abs(kv[1] - params.sensitivity)))
        best = None
        with FrameReader(self.video, self.info) as reader:
            for _, sv in order[:max_checks]:
                p = SegmentParams(**{**params.to_dict(), "sensitivity": sv, "threshold": None})
                pages = len(self.refine_segments(detect_segments(a, p), reader)) - auto_excluded + manual
                key = (abs(pages - expected), abs(sv - params.sensitivity))
                if best is None or key < best[0]:
                    best = (key, sv, pages)
                if pages == expected:
                    break
        return (best[1], best[2]) if best else (params.sensitivity, None)

    # -- persistence helpers ------------------------------------------------------------
    def save_project(self, project: Project) -> Optional[str]:
        if not self.cache.enabled:
            return None
        project.save(str(self.cache.project_path))
        return str(self.cache.project_path)

    def load_project(self) -> Optional[Project]:
        p = self.cache.project_path
        if self.cache.enabled and p.exists():
            try:
                proj = Project.load(str(p))
                if all(self.cache.has(pg.image_key) for pg in proj.pages):
                    return proj
            except Exception:
                return None
        return None
