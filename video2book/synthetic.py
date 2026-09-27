"""Synthetic test videos: "photographed book pages" shown as a phone slideshow.

Every generated page has a large unique page number and a bar-code marker.
Some pages are *nearly identical* (same body text, only the number differs),
which must never be merged by duplicate detection. The video has:

* slide transitions (plus occasional cross-fades) with ease-in/ease-out,
* irregular hold times (including a very short and a long hold),
* a swipe back (two pages shown again),
* a fake phone UI (status bar and gallery toolbar) over black bars,
* each page photographed on a dark table, slightly rotated, with vignetting,
* H.264 compression, optional rotation metadata, optional slow zoom.

Used by the tests and by ``scripts/make_synthetic_video.py``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Optional

import av
import cv2
import numpy as np

WORDS = ("the of and to in is was that for it with as his on be at by had not are but from "
         "or have an they which one you were her all she there would their we him been has "
         "when who will more no if out so said what up its about into than them can only "
         "other new some could time these two may then do first any my now such like our "
         "over man me even most made after also did many before must through back years "
         "where much your way well down should because each just those people how too "
         "little state good very make world still own see men work long get here between "
         "both life being under never day same another know while last might us great old "
         "year off come since against go came right used take three").split()


@dataclass
class SynthConfig:
    n_pages: int = 14
    width: int = 480
    height: int = 854
    fps: int = 30
    seed: int = 7
    hold_range: tuple = (0.5, 2.0)
    transition_range: tuple = (0.25, 0.55)
    fade_every: int = 4                  # every Nth transition is a cross-fade
    repeat_after: Optional[int] = 6      # after this page, swipe back one page then forward again
    near_identical: tuple = ((3, 4), (9, 10))  # (a, b): b reuses a's body text
    crf: int = 30
    phone_ui: bool = True
    rotation: int = 0                    # display-matrix rotation to store
    ken_burns: float = 0.0               # zoom amount during holds (e.g. 0.04)
    codec: str = "libx264"


@dataclass
class SynthTruth:
    n_pages: int
    sequence: list                  # page index shown in each hold, in order
    holds: list                     # (t_start, t_end) of each hold
    page_rects: list                # paper bbox (x0, y0, x1, y1) in the upright frame, per page
    photo_rect: tuple               # where the photo sits in the frame
    duplicates: int                 # number of holds that repeat an earlier page
    config: dict = field(default_factory=dict)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=1)

    @classmethod
    def load(cls, path: str) -> "SynthTruth":
        with open(path) as f:
            return cls(**json.load(f))


# -- page / photo rendering ------------------------------------------------

def _body_seed(cfg: SynthConfig, i: int) -> int:
    for a, b in cfg.near_identical:
        if i == b:
            return _body_seed(cfg, a)
    return cfg.seed * 1000 + i


def render_page(cfg: SynthConfig, i: int, size: tuple = (600, 840)) -> np.ndarray:
    """A book page (RGB) with a unique big number and bar code."""
    pw, ph = size
    img = np.full((ph, pw, 3), (238, 234, 222), np.uint8)
    rng = np.random.default_rng(_body_seed(cfg, i))
    font = cv2.FONT_HERSHEY_SIMPLEX
    ink = (40, 38, 36)
    cv2.putText(img, "THE SYNTHETIC BOOK", (int(pw * 0.3), int(ph * 0.05)), font, pw / 1400, ink, 1, cv2.LINE_AA)
    # Unique marker: big number and a 10-bit bar code.
    num = str(i + 1)
    scale = pw / 170
    (tw, th), _ = cv2.getTextSize(num, cv2.FONT_HERSHEY_DUPLEX, scale, max(2, pw // 60))
    cv2.putText(img, num, (int(pw * 0.93 - tw), int(ph * 0.08 + th)), cv2.FONT_HERSHEY_DUPLEX, scale,
                (20, 20, 90), max(2, pw // 60), cv2.LINE_AA)
    bw = int(pw * 0.045)
    for b in range(10):
        x = int(pw * 0.08) + b * bw
        if ((i + 1) >> b) & 1:
            cv2.rectangle(img, (x, int(ph * 0.08)), (x + bw - 3, int(ph * 0.08 + th)), (30, 30, 30), -1)
        else:
            cv2.rectangle(img, (x, int(ph * 0.08)), (x + bw - 3, int(ph * 0.08 + th)), (30, 30, 30), 1)
    # Body text (same for near-identical pairs).
    y = int(ph * 0.08 + th + ph * 0.07)
    line_h = max(12, int(ph * 0.03))
    fs = pw / 1500
    while y < ph * 0.9:
        if rng.random() < 0.12:
            y += line_h  # paragraph break
            continue
        x = int(pw * 0.08)
        indent = rng.random() < 0.2
        if indent:
            x += int(pw * 0.05)
        while True:
            w = WORDS[int(rng.integers(len(WORDS)))]
            (ww, _), _ = cv2.getTextSize(w, font, fs, 1)
            if x + ww > pw * 0.92:
                break
            cv2.putText(img, w, (x, y), font, fs, ink, 1, cv2.LINE_AA)
            x += ww + int(pw * 0.012)
        y += line_h
    cv2.putText(img, f"- {i + 1} -", (int(pw * 0.46), int(ph * 0.96)), font, pw / 1300, ink, 1, cv2.LINE_AA)
    return img


def _table_background(w: int, h: int, rng) -> np.ndarray:
    """Dark wooden table: stretched grain streaks plus fine noise."""
    base = np.zeros((h, w, 3), np.float32)
    base[:] = (72, 50, 34)
    streaks = rng.normal(0, 1, (max(4, h // 3), 6)).astype(np.float32)
    streaks = cv2.resize(streaks, (w, h), interpolation=cv2.INTER_CUBIC)
    yy = np.arange(h, dtype=np.float32)[:, None]
    grain = 0.6 * streaks + 0.5 * np.sin(yy / max(2.0, h / 90.0) + 3 * streaks)
    fine = rng.normal(0, 1, (h, w)).astype(np.float32)
    base += (grain * 16 + fine * 5)[..., None] * np.array([1.0, 0.75, 0.55], np.float32)
    return base


def render_photo(cfg: SynthConfig, i: int, pw: int, ph: int) -> tuple[np.ndarray, tuple]:
    """Photo of page ``i`` on a table: returns (RGB photo, paper bbox in photo)."""
    rng = np.random.default_rng(cfg.seed * 7919 + i)
    photo = _table_background(pw, ph, rng)
    page_w = int(pw * rng.uniform(0.78, 0.84))
    page_h = int(page_w * 1.4)
    page_h = min(page_h, int(ph * 0.92))
    page = render_page(cfg, i, (page_w, page_h)).astype(np.float32)
    angle = rng.uniform(-1.8, 1.8)
    cx = pw / 2 + rng.uniform(-0.03, 0.03) * pw
    cy = ph / 2 + rng.uniform(-0.02, 0.02) * ph
    m = cv2.getRotationMatrix2D((page_w / 2, page_h / 2), angle, 1.0)
    m[0, 2] += cx - page_w / 2
    m[1, 2] += cy - page_h / 2
    warped = cv2.warpAffine(page, m, (pw, ph), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    mask = cv2.warpAffine(np.ones((page_h, page_w), np.float32), m, (pw, ph), flags=cv2.INTER_LINEAR)
    photo = photo * (1 - mask[..., None]) + warped * mask[..., None]
    # Lighting: vignette + slight warm cast.
    yy, xx = np.mgrid[0:ph, 0:pw].astype(np.float32)
    r = np.sqrt(((xx - pw * 0.45) / pw) ** 2 + ((yy - ph * 0.4) / ph) ** 2)
    photo *= (1.05 - 0.35 * r)[..., None]
    photo = np.clip(photo, 0, 255).astype(np.uint8)
    ys, xs = np.where(mask > 0.5)
    return photo, (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def _draw_phone_ui(frame: np.ndarray) -> None:
    h, w = frame.shape[:2]
    white = (235, 235, 235)
    sb = max(14, int(h * 0.035))
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "9:41", (int(w * 0.05), int(sb * 0.8)), font, sb / 32, white, 1, cv2.LINE_AA)
    for k in range(4):  # signal bars
        x = int(w * 0.72) + k * max(3, w // 90)
        cv2.rectangle(frame, (x, int(sb * (0.8 - 0.15 * k))), (x + max(2, w // 130), int(sb * 0.8)), white, -1)
    bx = int(w * 0.84)
    cv2.rectangle(frame, (bx, int(sb * 0.3)), (bx + int(w * 0.09), int(sb * 0.8)), white, 1)
    cv2.rectangle(frame, (bx + 2, int(sb * 0.3) + 2), (bx + int(w * 0.06), int(sb * 0.8) - 2), white, -1)
    # Gallery toolbar: back arrow + title on top, 4 icons at the bottom.
    ty = int(sb * 2.1)
    cv2.putText(frame, "<  Book photos", (int(w * 0.04), ty), font, sb / 30, white, 1, cv2.LINE_AA)
    by = h - int(h * 0.05)
    r = max(6, int(w * 0.03))
    for k, cx in enumerate(np.linspace(w * 0.15, w * 0.85, 4)):
        cx = int(cx)
        if k == 0:
            cv2.circle(frame, (cx, by), r, white, 1)
        elif k == 1:
            cv2.rectangle(frame, (cx - r, by - r), (cx + r, by + r), white, 1)
        elif k == 2:
            pts = np.array([[cx, by + r], [cx - r, by - r // 2], [cx + r, by - r // 2]], np.int32)
            cv2.polylines(frame, [pts], True, white, 1)
        else:
            cv2.line(frame, (cx - r, by - r), (cx + r, by + r), white, 1)
            cv2.line(frame, (cx - r, by + r), (cx + r, by - r), white, 1)


def _ease(u: float) -> float:
    u = min(1.0, max(0.0, u))
    return u * u * (3 - 2 * u)


class _Composer:
    def __init__(self, cfg: SynthConfig):
        self.cfg = cfg
        w, h = cfg.width, cfg.height
        self.pw = w
        self.ph = min(h, int(round(w * 4 / 3)))
        self.py = (h - self.ph) // 2
        self.photos = {}
        self.rects = {}
        ui = np.zeros((h, w, 3), np.uint8)
        if cfg.phone_ui:
            _draw_phone_ui(ui)
        self.ui = ui
        self.ui_mask = ui.max(axis=2) > 0

    def photo(self, i: int) -> np.ndarray:
        if i not in self.photos:
            p, r = render_photo(self.cfg, i, self.pw, self.ph)
            self.photos[i] = p
            self.rects[i] = (r[0], r[1] + self.py, r[2], r[3] + self.py)
        return self.photos[i]

    def _zoomed(self, i: int, z: float) -> np.ndarray:
        p = self.photo(i)
        if z <= 1e-4:
            return p
        m = cv2.getRotationMatrix2D((self.pw / 2, self.ph / 2), 0, 1 + z)
        return cv2.warpAffine(p, m, (self.pw, self.ph), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    def frame(self, layers: list[tuple[int, float, float, float]]) -> np.ndarray:
        """layers: (page, x_offset_px, alpha, zoom)."""
        cfg = self.cfg
        f = np.zeros((cfg.height, cfg.width, 3), np.float32)
        for page, dx, alpha, zoom in layers:
            p = self._zoomed(page, zoom).astype(np.float32)
            dx = int(round(dx))
            x0, x1 = max(0, dx), min(cfg.width, dx + self.pw)
            if x1 <= x0:
                continue
            region = f[self.py:self.py + self.ph, x0:x1]
            src = p[:, x0 - dx:x1 - dx]
            region[:] = region * (1 - alpha) + src * alpha if alpha < 1 else src
        out = np.clip(f, 0, 255).astype(np.uint8)
        out[self.ui_mask] = self.ui[self.ui_mask]
        return out


def build_schedule(cfg: SynthConfig) -> tuple[list[int], list[float], list[tuple[str, float]]]:
    """Sequence of shown pages, hold durations and transitions between them."""
    rng = np.random.default_rng(cfg.seed)
    seq = []
    for i in range(cfg.n_pages):
        seq.append(i)
        if cfg.repeat_after is not None and i == cfg.repeat_after and i >= 1:
            seq += [i - 1, i]  # swipe back, then forward again
    holds = [float(rng.uniform(*cfg.hold_range)) for _ in seq]
    if len(holds) > 3:
        holds[2] = 0.4    # very short hold
        holds[-2] = 3.0   # long hold
    trans = []
    for k in range(len(seq) - 1):
        kind = "fade" if cfg.fade_every and (k + 1) % cfg.fade_every == 0 else "slide"
        trans.append((kind, float(rng.uniform(*cfg.transition_range))))
    return seq, holds, trans


def generate(path: str, cfg: SynthConfig | None = None, truth_path: Optional[str] = None) -> SynthTruth:
    """Write the synthetic video to ``path`` and return the ground truth."""
    cfg = cfg or SynthConfig()
    comp = _Composer(cfg)
    seq, holds, trans = build_schedule(cfg)
    container = av.open(str(path), mode="w")
    stream = container.add_stream(cfg.codec, rate=cfg.fps)
    k = int(round(cfg.rotation / 90)) % 4
    coded_w, coded_h = (cfg.height, cfg.width) if k % 2 else (cfg.width, cfg.height)
    stream.width, stream.height = coded_w, coded_h
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": str(cfg.crf), "preset": "veryfast"}
    if cfg.rotation:
        stream.set_display_rotation(cfg.rotation)

    def emit(frame: np.ndarray) -> None:
        if k:
            frame = np.ascontiguousarray(np.rot90(frame, -k))
        for pkt in stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
            container.mux(pkt)

    t = 0.0
    fi = 0
    hold_times = []
    dt = 1.0 / cfg.fps
    for n, page in enumerate(seq):
        comp.photo(page)
        nf = max(1, int(round(holds[n] * cfg.fps)))
        hold_times.append((fi * dt, (fi + nf) * dt))
        for j in range(nf):
            z = cfg.ken_burns * j / max(1, nf - 1)
            emit(comp.frame([(page, 0, 1.0, z)]))
        fi += nf
        if n < len(trans):
            kind, dur = trans[n]
            nxt = seq[n + 1]
            back = nxt < page
            zf = cfg.ken_burns
            nt = max(2, int(round(dur * cfg.fps)))
            for j in range(1, nt + 1):
                u = _ease(j / (nt + 1))
                if kind == "fade":
                    layers = [(page, 0, 1.0, zf), (nxt, 0, u, 0.0)]
                else:
                    off = u * cfg.width
                    if back:
                        layers = [(page, off, 1.0, zf), (nxt, off - cfg.width, 1.0, 0.0)]
                    else:
                        layers = [(page, -off, 1.0, zf), (nxt, cfg.width - off, 1.0, 0.0)]
                emit(comp.frame(layers))
            fi += nt
        t = fi * dt
    for pkt in stream.encode():
        container.mux(pkt)
    container.close()
    seen, dups = set(), 0
    for p in seq:
        dups += p in seen
        seen.add(p)
    truth = SynthTruth(
        n_pages=cfg.n_pages, sequence=seq, holds=hold_times,
        page_rects=[comp.rects[i] for i in range(cfg.n_pages)],
        photo_rect=(0, comp.py, comp.pw, comp.py + comp.ph), duplicates=dups,
        config={k2: (list(v) if isinstance(v, tuple) else v) for k2, v in asdict(cfg).items()},
    )
    if truth_path:
        truth.save(truth_path)
    return truth


def hold_frame(cfg: SynthConfig, page: int) -> np.ndarray:
    """The exact (uncompressed) frame shown while ``page`` is held."""
    comp = _Composer(cfg)
    return comp.frame([(page, 0, 1.0, 0.0)])
