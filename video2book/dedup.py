"""Duplicate page detection.

Each page gets a descriptor: a 64-bit DCT perceptual hash and a grayscale copy
at a comparison resolution (default 512 px wide - enough to tell apart two
text pages whose only difference is a page number).

Two pages are compared after aligning them with phase correlation (a page
shown again after swiping back can sit a few pixels off). We compute the SSIM
map and look at two numbers:

* ``ssim``    - mean SSIM (after a light blur that removes most compression
  noise), overall similarity;
* ``local``   - the *minimum* of the SSIM map averaged over small windows
  (about 4% of the width): a changed word or paragraph drives it down;
* ``outlier`` - the largest local difference divided by the 95th percentile
  of local differences. Compression noise is spread out (ratio ~1.5-2.5), a
  changed page number is one isolated spot (ratio >> 4).

Decision:

* duplicate   ``ssim >= same_ssim``, ``local >= same_local`` and no isolated
              differing spot                                     -> merged
* uncertain   identical except for one spot, or merely similar   -> flagged
* different   otherwise

Limit: in very heavily compressed video a single changed digit can drown in
the noise; such pages would be merged. Reviewing merged duplicates (they are
kept, only hidden) takes one click in the UI.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class DedupParams:
    compare_width: int = 512
    same_ssim: float = 0.88
    same_local: float = 0.50
    flag_ssim: float = 0.80
    flag_local: float = 0.30
    outlier: float = 4.0       # peak/spread ratio above which a spot "really differs"
    min_peak: float = 6.0      # ...if that spot differs by at least this many gray levels
    window: int = 20           # always compare with this many previous pages
    hash_candidates: int = 10  # ...and with any earlier page within this hash distance
    max_hash_dist: int = 26    # skip the (slower) SSIM check above this hash distance (of 64)
    prefilter_ssim: float = 0.65  # skip it too when SSIM at 128 px is below this (repeats: > 0.9)
    max_shift: float = 0.06    # max alignment shift (fraction of size)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Descriptor:
    gray: np.ndarray    # float32, compare resolution
    phash: np.ndarray   # 64 bools
    small: Optional[np.ndarray] = None  # float32, 128 px wide (fast pre-check)


@dataclass
class Match:
    ssim: float
    local: float
    shift: tuple
    hash_dist: int
    peak: float = 0.0     # largest local mean abs difference (gray levels)
    spread: float = 0.0   # 95th percentile of the same

    @property
    def outlier(self) -> float:
        """How much the worst spot stands out from the overall difference."""
        return self.peak / max(self.spread, 1.0)

    def to_dict(self) -> dict:
        return {"ssim": round(self.ssim, 4), "local": round(self.local, 4),
                "peak": round(self.peak, 2), "spread": round(self.spread, 2),
                "shift": [round(float(s), 2) for s in self.shift], "hash_dist": int(self.hash_dist)}


@dataclass
class DedupResult:
    status: str                  # "unique" | "duplicate" | "possible"
    ref: Optional[int] = None    # index of the page it duplicates / resembles
    match: Optional[Match] = None


def phash(gray: np.ndarray) -> np.ndarray:
    small = cv2.resize(gray.astype(np.float32), (32, 32), interpolation=cv2.INTER_AREA)
    d = cv2.dct(small)[:8, :8].flatten()
    return d > np.median(d[1:])


def describe(img: np.ndarray, width: int = 512, box: Optional[tuple] = None) -> Descriptor:
    if box is not None:
        x0, y0, x1, y1 = box
        img = img[y0:y1, x0:x1]
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
    h, w = g.shape[:2]
    if w > width:
        g = cv2.resize(g, (width, max(8, int(round(h * width / w)))), interpolation=cv2.INTER_AREA)
    sw = 128
    small = cv2.resize(g, (sw, max(8, int(round(g.shape[0] * sw / g.shape[1])))), interpolation=cv2.INTER_AREA)
    return Descriptor(gray=g.astype(np.float32), phash=phash(g), small=small.astype(np.float32))


def quick_similarity(a: Descriptor, b: Descriptor) -> float:
    """Unaligned SSIM at 128 px: a cheap upper-bound-ish pre-check."""
    if a.small is None or b.small is None:
        return 1.0
    sb = b.small
    if sb.shape != a.small.shape:
        sb = cv2.resize(sb, (a.small.shape[1], a.small.shape[0]), interpolation=cv2.INTER_AREA)
    return float(ssim_map(a.small, sb).mean())


def ssim_map(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Standard SSIM (Wang et al. 2004), Gaussian 7x7 window, 8-bit range."""
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    blur = lambda x: cv2.GaussianBlur(x, (7, 7), 1.5)
    mu_a, mu_b = blur(a), blur(b)
    aa, bb, ab = blur(a * a) - mu_a ** 2, blur(b * b) - mu_b ** 2, blur(a * b) - mu_a * mu_b
    num = (2 * mu_a * mu_b + c1) * (2 * ab + c2)
    den = (mu_a ** 2 + mu_b ** 2 + c1) * (aa + bb + c2)
    return num / den


def compare(a: Descriptor, b: Descriptor, max_shift: float = 0.06, blur: float = 1.0) -> Match:
    """Align ``b`` to ``a`` (translation) and measure how alike they are."""
    ga, gb = a.gray, b.gray
    if gb.shape != ga.shape:
        gb = cv2.resize(gb, (ga.shape[1], ga.shape[0]), interpolation=cv2.INTER_AREA)
    h, w = ga.shape
    win = cv2.createHanningWindow((w, h), cv2.CV_32F)
    # Copies: some OpenCV versions (5.0) multiply the inputs by the window in place.
    (dx, dy), _resp = cv2.phaseCorrelate(ga.copy(), gb.copy(), win)
    if abs(dx) > max_shift * w or abs(dy) > max_shift * h:
        dx = dy = 0.0
    if abs(dx) > 0.25 or abs(dy) > 0.25:
        m = np.float32([[1, 0, -dx], [0, 1, -dy]])
        gb = cv2.warpAffine(gb, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    # A light blur removes most compression noise but keeps text-sized detail.
    if blur > 0:
        ga = cv2.GaussianBlur(ga, (0, 0), blur)
        gb = cv2.GaussianBlur(gb, (0, 0), blur)
    mx, my = int(np.ceil(abs(dx))) + 3, int(np.ceil(abs(dy))) + 3
    sa, sb = ga[my:h - my, mx:w - mx], gb[my:h - my, mx:w - mx]
    if sa.size == 0:
        sa, sb = ga, gb
    smap = ssim_map(sa, sb)
    k = max(5, int(round(0.04 * w)) | 1)
    edge = k // 2

    def inner(x: np.ndarray) -> np.ndarray:
        y = x[edge:x.shape[0] - edge, edge:x.shape[1] - edge]
        return y if y.size else x

    local = inner(cv2.blur(smap, (k, k)))
    # Localised differences: mean abs difference in small windows. Compression
    # noise is spread out (peak close to the 95th percentile); a changed word
    # or number is a single outlier.
    k2 = max(3, int(round(0.015 * w)) | 1)
    diff = inner(cv2.blur(np.abs(sa - sb), (k2, k2)))
    peak = float(diff.max())
    p95 = float(np.percentile(diff, 95))
    hd = int(np.count_nonzero(a.phash != b.phash))
    return Match(ssim=float(smap.mean()), local=float(local.min()), shift=(dx, dy), hash_dist=hd,
                 peak=peak, spread=p95)


def classify(m: Match, p: Optional[DedupParams] = None) -> str:
    p = p or DedupParams()
    localized = m.peak >= p.min_peak and m.outlier >= p.outlier
    if m.ssim >= p.same_ssim:
        if localized:
            return "possible"   # identical except for one spot (page number, a word...)
        if m.local >= p.same_local:
            return "duplicate"
        return "possible"       # very similar overall, one weaker area: let the user decide
    if m.ssim >= p.flag_ssim and m.local >= p.flag_local:
        return "possible"
    return "unique"


def find_duplicates(descs: list[Descriptor], params: DedupParams | None = None,
                    active: Optional[list[bool]] = None) -> list[DedupResult]:
    """Compare every page with earlier ones (in order).

    ``active[i] = False`` pages (e.g. deleted or blank) are skipped both as
    candidates and as references. The first occurrence of a page is kept; later
    occurrences become ``duplicate`` (identical) or ``possible`` (uncertain).
    """
    p = params or DedupParams()
    n = len(descs)
    active = active or [True] * n
    results = [DedupResult("unique") for _ in range(n)]
    hashes = np.array([d.phash for d in descs]) if n else np.zeros((0, 64), bool)
    for j in range(n):
        if not active[j]:
            continue
        refs = [i for i in range(max(0, j - p.window), j)]
        if j > p.window:
            dist = np.count_nonzero(hashes[:j - p.window] != hashes[j], axis=1)
            refs = np.where(dist <= p.hash_candidates)[0].tolist() + refs
        best: Optional[tuple] = None
        for i in refs:
            # Only compare with pages that are themselves kept originals.
            if not active[i] or results[i].status == "duplicate":
                continue
            if np.count_nonzero(hashes[i] != hashes[j]) > p.max_hash_dist:
                continue  # clearly different: repeats measured <= 6 even at heavy compression
            if quick_similarity(descs[i], descs[j]) < p.prefilter_ssim:
                continue
            m = compare(descs[i], descs[j], p.max_shift)
            c = classify(m, p)
            rank = {"duplicate": 2, "possible": 1, "unique": 0}[c]
            key = (rank, m.ssim)
            if best is None or key > best[0]:
                best = (key, i, m, c)
        if best is not None and best[3] != "unique":
            results[j] = DedupResult(best[3], best[1], best[2])
    return results
