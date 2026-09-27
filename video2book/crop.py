"""Finding the page inside each frame.

Automatic cropping works in three steps:

1. **Static border** (global, all pages at once): rows/columns at the frame
   edges that are identical on every page - phone UI (status bar, gallery
   buttons), black bars, watermarks in the margin - are removed.
2. **Uniform bars** (per page): remaining edge rows/columns of a single flat
   colour are trimmed.
3. **Page area** (per page), whichever fits first:

   * *paper*: the largest bright, roughly rectangular region that contains most
     of the image detail (a photographed page on a darker background). This
     also gives the 4 corners used by the optional perspective correction;
   * *detail*: the bounding box of the area with fine detail. This removes
     blurred/zoomed background fills used by slideshow apps and low-texture
     surroundings.

A manual crop (a rectangle in relative coordinates, drawn once) replaces the
automatic one for every page.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import cv2
import numpy as np

Box = tuple  # (x0, y0, x1, y1), pixels, x1/y1 exclusive


@dataclass
class CropResult:
    box: Box
    method: str                      # "paper" | "bars" | "detail" | "none" | "manual"
    quad: Optional[np.ndarray] = None  # 4x2 float32 corners (tl, tr, br, bl) for perspective

    def to_dict(self) -> dict:
        return {"box": [int(v) for v in self.box], "method": self.method,
                "quad": None if self.quad is None else self.quad.round(2).tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "CropResult":
        q = d.get("quad")
        return cls(tuple(d["box"]), d["method"], None if q is None else np.asarray(q, np.float32))


def _gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img


def _scaled(img: np.ndarray, width: int) -> tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    s = min(1.0, width / float(w))
    if s < 1:
        img = cv2.resize(img, (int(round(w * s)), max(1, int(round(h * s)))), interpolation=cv2.INTER_AREA)
    return img, s


def _box_scale(box, s: float, shape) -> Box:
    h, w = shape[:2]
    x0, y0, x1, y1 = box
    return (max(0, int(np.floor(x0 / s))), max(0, int(np.floor(y0 / s))),
            min(w, int(np.ceil(x1 / s))), min(h, int(np.ceil(y1 / s))))


def full_box(img: np.ndarray) -> Box:
    return (0, 0, img.shape[1], img.shape[0])


# -- 1. static border ---------------------------------------------------------

def static_border(images: Sequence[np.ndarray], tol: float = 12.0, frac: float = 0.9,
                  width: int = 256) -> Optional[Box]:
    """Box excluding edge rows/cols that are the same on every page.

    A pixel is static when its value (lightly blurred, to ignore compression
    noise) stays within ``tol`` on *all* pages; a row/column is static when at
    least ``frac`` of its pixels are (so noise around UI text is tolerated but
    a page reaching into the row on even one frame is not). Needs at least 3
    pages of identical size; returns None otherwise.
    """
    imgs = [im for im in images if im is not None]
    if len(imgs) < 3 or len({im.shape[:2] for im in imgs}) != 1:
        return None
    # Blur (sigma 1.5 px at ~256 px wide) so heavy compression noise around UI
    # text and icons doesn't make static rows look like they change.
    small = [cv2.GaussianBlur(_scaled(_gray(im), width)[0], (0, 0), 1.5) for im in imgs]
    s = small[0].shape[1] / float(imgs[0].shape[1])
    lo = np.minimum.reduce(small).astype(np.int16)
    hi = np.maximum.reduce(small).astype(np.int16)
    static = (hi - lo) < tol
    rows = static.mean(axis=1) >= frac
    cols = static.mean(axis=0) >= frac
    h, w = static.shape
    y0 = 0
    while y0 < h - 1 and rows[y0]:
        y0 += 1
    y1 = h
    while y1 > y0 + 1 and rows[y1 - 1]:
        y1 -= 1
    x0 = 0
    while x0 < w - 1 and cols[x0]:
        x0 += 1
    x1 = w
    while x1 > x0 + 1 and cols[x1 - 1]:
        x1 -= 1
    if (x1 - x0) < 0.2 * w or (y1 - y0) < 0.2 * h:
        return None  # nearly everything static: something is off, don't trust it
    return _box_scale((x0, y0, x1, y1), s, imgs[0].shape)


# -- 2. uniform bars ----------------------------------------------------------

def trim_uniform(img: np.ndarray, box: Optional[Box] = None, tol: float = 5.0,
                 color_tol: float = 12.0) -> Box:
    """Shrink ``box`` while its edge rows/cols are a flat bar of the edge colour.

    Only bars matching the colour found at the very edge are removed, so a
    page's own blank margin (flat too, but paper-coloured) is kept.
    """
    x0, y0, x1, y1 = box or full_box(img)
    g = _gray(img).astype(np.float32)

    def trim(get, lo: int, hi: int, step: int) -> int:
        first = get(lo if step > 0 else hi - 1)
        if first.size == 0 or float(first.std()) >= tol:
            return lo if step > 0 else hi
        ref = float(first.mean())
        i = lo if step > 0 else hi - 1
        while hi - lo > 8:
            v = get(i)
            if v.size == 0 or float(v.std()) >= tol or abs(float(v.mean()) - ref) > color_tol:
                break
            if step > 0:
                lo += 1
            else:
                hi -= 1
            i += step
        return lo if step > 0 else hi

    y0 = trim(lambda i: g[i, x0:x1], y0, y1, 1)
    y1 = trim(lambda i: g[i, x0:x1], y0, y1, -1)
    x0 = trim(lambda i: g[y0:y1, i], x0, x1, 1)
    x1 = trim(lambda i: g[y0:y1, i], x0, x1, -1)
    return (x0, y0, x1, y1)


# -- 3a. paper ----------------------------------------------------------------

def _order_quad(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]], np.float32)


def _detail_map(g: np.ndarray) -> np.ndarray:
    g = cv2.GaussianBlur(g, (3, 3), 0)
    return np.abs(cv2.Laplacian(g, cv2.CV_32F, ksize=3))


MIN_EDGE = 25.0  # median gradient magnitude along the outline (Sobel, 8-bit)


def _edge_sharpness(gray: np.ndarray, contour: np.ndarray) -> float:
    """Median gradient magnitude along a contour, ignoring the image border."""
    h, w = gray.shape
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.dilate(cv2.magnitude(gx, gy), np.ones((3, 3), np.uint8))  # tolerate 1 px offsets
    pts = contour.reshape(-1, 2)
    # densify: CHAIN_APPROX_SIMPLE keeps only corners of straight runs
    dense = []
    for a, b in zip(pts, np.roll(pts, -1, axis=0)):
        n = int(max(abs(b[0] - a[0]), abs(b[1] - a[1]))) + 1
        dense.append(np.stack([np.linspace(a[0], b[0], n), np.linspace(a[1], b[1], n)], 1))
    p = np.concatenate(dense).round().astype(int)
    inside = (p[:, 0] > 2) & (p[:, 0] < w - 3) & (p[:, 1] > 2) & (p[:, 1] < h - 3)
    if inside.sum() < 0.1 * len(p):
        return 1e9  # page fills the frame: nothing to judge, don't reject
    p = p[inside]
    return float(np.median(mag[p[:, 1], p[:, 0]])) / 4.0  # Sobel 3x3 gain is 4


def find_paper(img: np.ndarray, box: Optional[Box] = None, width: int = 480) -> Optional[CropResult]:
    """Largest bright, rectangular region holding most of the detail."""
    x0, y0, x1, y1 = box or full_box(img)
    region = _gray(img[y0:y1, x0:x1])
    small, s = _scaled(region, width)
    h, w = small.shape
    if h < 16 or w < 16:
        return None
    blur = cv2.GaussianBlur(small, (5, 5), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = max(3, int(round(min(h, w) * 0.02)) | 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return None
    lab = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = float(stats[lab, cv2.CC_STAT_AREA])
    if not (0.15 * h * w <= area <= 0.97 * h * w):
        return None
    l, t_, cw, ch = (int(v) for v in stats[lab, :4])
    tol = max(2, int(0.01 * max(h, w)))
    touching = (l <= tol) + (t_ <= tol) + (l + cw >= w - tol) + (t_ + ch >= h - tol)
    if touching >= 3:
        return None  # fills the area: not a sheet lying on a background
    comp = (labels == lab).astype(np.uint8)
    # Fill holes (text inside the page).
    contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(cnt)
    rect = cv2.minAreaRect(hull)
    rect_area = max(1.0, rect[1][0] * rect[1][1])
    if cv2.contourArea(cnt) / rect_area < 0.85:
        return None  # not rectangular: not a sheet of paper
    bx, by, bw, bh = cv2.boundingRect(hull)
    # A sheet of paper has a crisp outline; a blurred background fill (common
    # in slideshow apps) or a vignette does not.
    if _edge_sharpness(small, cnt) < MIN_EDGE:
        return None
    # The paper must contain most of the fine detail (text, pictures).
    det = _detail_map(small)
    det = np.where(det > max(10.0, float(np.percentile(det, 75))), det, 0)  # strong edges only
    total = float(det.sum()) + 1e-6
    inside = float(det[by:by + bh, bx:bx + bw].sum())
    if inside / total < 0.85:
        return None
    peri = cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, 0.02 * peri, True)
    quad = approx.reshape(-1, 2) if len(approx) == 4 else cv2.boxPoints(rect)
    quad = _order_quad(quad) / s + np.array([x0, y0], np.float32)
    b = _box_scale((bx, by, bx + bw, by + bh), s, region.shape)
    return CropResult((b[0] + x0, b[1] + y0, b[2] + x0, b[3] + y0), "paper", quad)


# -- 3b. detail -------------------------------------------------------------

def _profile_extent(p: np.ndarray, rel: float) -> tuple[int, int]:
    n = p.size
    win = max(3, int(round(n * 0.02)) | 1)
    ps = np.convolve(p, np.ones(win) / win, mode="same")
    ref = float(np.percentile(ps, 90))
    if ref <= 0:
        return 0, n
    idx = np.where(ps >= rel * ref)[0]
    if idx.size == 0:
        return 0, n
    return int(idx[0]), int(idx[-1]) + 1


def find_detail_box(img: np.ndarray, box: Optional[Box] = None, width: int = 480,
                    rel: float = 0.25, margin: float = 0.01) -> Box:
    x0, y0, x1, y1 = box or full_box(img)
    region = _gray(img[y0:y1, x0:x1])
    small, s = _scaled(region, width)
    det = _detail_map(small)
    # Ignore faint texture (compression noise, blur): keep real edges only.
    det = np.where(det > max(2.0, float(np.percentile(det, 60))), det, 0)
    ry0, ry1 = _profile_extent(det.mean(axis=1), rel)
    rx0, rx1 = _profile_extent(det.mean(axis=0), rel)
    h, w = small.shape
    mx, my = int(round(margin * w)), int(round(margin * h))
    b = (max(0, rx0 - mx), max(0, ry0 - my), min(w, rx1 + mx), min(h, ry1 + my))
    b = _box_scale(b, s, region.shape)
    return (b[0] + x0, b[1] + y0, b[2] + x0, b[3] + y0)


# -- putting it together ------------------------------------------------------

def _margin_is_flat(img: np.ndarray, inner: Box, outer: Box, tol: float = 12.0) -> bool:
    """True if the band between ``inner`` and ``outer`` is plain (paper margin),
    not blurred picture content."""
    g = _gray(img).astype(np.float32)
    x0, y0, x1, y1 = outer
    ix0, iy0, ix1, iy1 = inner
    bands = [g[y0:iy0, x0:x1], g[iy1:y1, x0:x1], g[iy0:iy1, x0:ix0], g[iy0:iy1, ix1:x1]]
    # Robust spread: a few pixels of a running header or page number in the
    # margin are fine, blurred picture content is not.
    spreads = [float(np.percentile(np.abs(b - np.median(b)), 90)) for b in bands if min(b.shape) >= 3]
    return not spreads or max(spreads) < tol


def auto_crop(img: np.ndarray, static_box: Optional[Box] = None) -> CropResult:
    start = static_box or full_box(img)
    base = trim_uniform(img, start)
    paper = find_paper(img, base)
    if paper is not None:
        return paper
    det = find_detail_box(img, base)
    area = lambda b: max(1, (b[2] - b[0]) * (b[3] - b[1]))
    trimmed = area(base) < 0.9 * area(start)
    if trimmed and area(det) >= 0.5 * area(base) and _margin_is_flat(img, det, base):
        # Bars were removed and what's left is essentially all content (e.g. a
        # scanned page letterboxed in black): keep its margins.
        return CropResult(base, "bars")
    bw, bh = det[2] - det[0], det[3] - det[1]
    if bw < 0.15 * img.shape[1] or bh < 0.15 * img.shape[0]:
        return CropResult(base, "none")  # implausibly small: keep the trimmed frame
    return CropResult(det, "detail")


def manual_crop(img: np.ndarray, rel: Sequence[float]) -> CropResult:
    """``rel`` = (x0, y0, x1, y1) in 0..1 of the (upright) frame."""
    h, w = img.shape[:2]
    x0, y0, x1, y1 = rel
    b = (int(round(min(x0, x1) * w)), int(round(min(y0, y1) * h)),
         int(round(max(x0, x1) * w)), int(round(max(y0, y1) * h)))
    b = (max(0, b[0]), max(0, b[1]), min(w, max(b[2], b[0] + 1)), min(h, max(b[3], b[1] + 1)))
    return CropResult(b, "manual")


def warp_quad(img: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """Perspective-correct the quadrilateral ``quad`` (tl, tr, br, bl) to a rectangle."""
    q = np.asarray(quad, np.float32)
    wa = np.linalg.norm(q[1] - q[0])
    wb = np.linalg.norm(q[2] - q[3])
    ha = np.linalg.norm(q[3] - q[0])
    hb = np.linalg.norm(q[2] - q[1])
    W, H = int(round(max(wa, wb))), int(round(max(ha, hb)))
    if W < 8 or H < 8:
        return img
    dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], np.float32)
    m = cv2.getPerspectiveTransform(q, dst)
    return cv2.warpPerspective(img, m, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def apply_crop(img: np.ndarray, crop: Optional[CropResult], perspective: bool = False) -> np.ndarray:
    if crop is None:
        return img
    if perspective and crop.quad is not None:
        return warp_quad(img, crop.quad)
    x0, y0, x1, y1 = crop.box
    return img[y0:y1, x0:x1]


def iou(a: Box, b: Box) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0
