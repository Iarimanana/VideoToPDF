"""Exporting pages: PDF (img2pdf), searchable PDF (OCRmyPDF), ZIP of images."""

from __future__ import annotations

import io
import os
import shutil
import subprocess
import tempfile
import zipfile
from dataclasses import asdict, dataclass
from typing import Callable, Iterable, Optional

import cv2
import img2pdf
import numpy as np
from PIL import Image

ProgressFn = Optional[Callable[[float, str], None]]

PAGE_SIZES_PT = {
    "uniform": None,                               # same size for every page, shape fitted to the pages
    "fit": None,                                   # page = image size at the given DPI (sizes vary)
    "a4": (img2pdf.mm_to_pt(210), img2pdf.mm_to_pt(297)),
    "letter": (img2pdf.in_to_pt(8.5), img2pdf.in_to_pt(11)),
}
PAGE_SIZE_LABELS = {"uniform": "Same size (least filling)", "fit": "Fit each image", "a4": "A4",
                    "letter": "Letter"}
FILLS = ("auto", "white", "black")
FILL_LABELS = {"auto": "Page edge colour", "white": "White", "black": "Black"}
UNIFORM_LONG_SIDE_PT = img2pdf.mm_to_pt(297)       # like A4's long side


@dataclass
class ExportOptions:
    page_size: str = "uniform"   # "uniform" | "fit" | "a4" | "letter"
    dpi: int = 200
    jpeg_quality: int = 88
    ocr_lang: Optional[str] = None  # e.g. "eng", "fra+eng"; None = no OCR
    fill: str = "auto"           # padding colour for fixed page sizes: "auto" | "white" | "black"

    def to_dict(self) -> dict:
        return asdict(self)


def _is_binary(img: np.ndarray) -> bool:
    if img.ndim != 2:
        return False
    vals = np.unique(img[:: max(1, img.shape[0] // 64), :: max(1, img.shape[1] // 64)])
    return vals.size <= 2 and set(vals.tolist()) <= {0, 255}


# -- same page size for every page ------------------------------------------------

def fill_fraction(aspect: float, page_aspect: float) -> float:
    """Share of the page that is filling when an image of ``aspect`` (w/h) is
    fitted into a page of ``page_aspect``."""
    return 1.0 - min(aspect, page_aspect) / max(aspect, page_aspect)


def best_aspect(aspects: Iterable[float]) -> float:
    """Page shape (w/h) needing the least filling in total over all pages."""
    a = np.array([x for x in aspects if x and x > 0], dtype=float)
    if a.size == 0:
        return 210 / 297
    cands = np.unique(np.round(a, 4)) if a.size <= 400 else np.quantile(a, np.linspace(0, 1, 201))
    lo, hi = np.minimum.outer(cands, a), np.maximum.outer(cands, a)
    total = (1.0 - lo / hi).sum(axis=1)
    return float(cands[int(np.argmin(total))])


def uniform_page_pt(aspect: float) -> tuple:
    if aspect <= 1:
        return (UNIFORM_LONG_SIDE_PT * aspect, UNIFORM_LONG_SIDE_PT)
    return (UNIFORM_LONG_SIDE_PT, UNIFORM_LONG_SIDE_PT / aspect)


def fill_color(img: np.ndarray, fill: str = "auto"):
    """Padding colour: the median colour of the image's outer edge ("auto"),
    white or black. Binary (B&W scan) pages stay binary."""
    binary = _is_binary(img)
    if fill == "white":
        v = 255
    elif fill == "black":
        v = 0
    else:
        h, w = img.shape[:2]
        k = max(1, int(round(0.02 * min(h, w))))
        edge = np.concatenate([img[:k].reshape(-1, *img.shape[2:]), img[-k:].reshape(-1, *img.shape[2:]),
                               img[:, :k].reshape(-1, *img.shape[2:]), img[:, -k:].reshape(-1, *img.shape[2:])])
        med = np.median(edge, axis=0)
        if binary:
            return 255 if float(med) >= 128 else 0
        return tuple(int(round(float(c))) for c in np.atleast_1d(med))
    if img.ndim == 3:
        return (v,) * img.shape[2]
    return v


def pad_to_aspect(img: np.ndarray, aspect: float, fill: str = "auto") -> np.ndarray:
    """Pad (centred, no scaling) so that width/height == ``aspect``."""
    h, w = img.shape[:2]
    if w / h < aspect:
        nw, nh = int(round(h * aspect)), h
    else:
        nw, nh = w, int(round(w / aspect))
    if (nw, nh) == (w, h):
        return img
    left, top = (nw - w) // 2, (nh - h) // 2
    return cv2.copyMakeBorder(img, top, nh - h - top, left, nw - w - left, cv2.BORDER_CONSTANT,
                              value=fill_color(img, fill))


def page_size_pt(opts: ExportOptions, aspect: Optional[float] = None) -> Optional[tuple]:
    """Physical page size for fixed-size modes (None for 'fit')."""
    if opts.page_size == "uniform":
        return uniform_page_pt(aspect if aspect else 210 / 297)
    return PAGE_SIZES_PT.get(opts.page_size)


def fit_for_page(img: np.ndarray, opts: ExportOptions, page_pt: Optional[tuple] = None) -> np.ndarray:
    """Downscale (never upscale) so the image is at most ``dpi`` on the page."""
    size = page_pt or PAGE_SIZES_PT.get(opts.page_size)
    if size is None:
        return img
    h, w = img.shape[:2]
    pw, ph = size
    max_w, max_h = pw / 72.0 * opts.dpi, ph / 72.0 * opts.dpi
    s = min(max_w / w, max_h / h)
    if s >= 1:
        return img
    return cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)


def prepare_page(img: np.ndarray, opts: ExportOptions, aspect: Optional[float] = None) -> np.ndarray:
    """What goes on a PDF page: padded to the page shape (fixed-size modes),
    then limited to ``dpi``."""
    page = page_size_pt(opts, aspect)
    if page is not None:
        img = pad_to_aspect(img, page[0] / page[1], opts.fill)
    return fit_for_page(img, opts, page)


def encode_image(img: np.ndarray, opts: ExportOptions, fmt: Optional[str] = None) -> tuple[bytes, str]:
    """Encode for embedding. Returns ``(bytes, extension)``."""
    if fmt is None:
        fmt = "png" if _is_binary(img) else "jpg"
    pil = Image.fromarray(img)
    buf = io.BytesIO()
    if fmt == "png":
        if _is_binary(img):
            pil = pil.convert("1")
        pil.save(buf, format="PNG", dpi=(opts.dpi, opts.dpi), optimize=True)
        return buf.getvalue(), "png"
    if pil.mode not in ("RGB", "L"):
        pil = pil.convert("RGB")
    pil.save(buf, format="JPEG", quality=int(opts.jpeg_quality), dpi=(opts.dpi, opts.dpi), optimize=True)
    return buf.getvalue(), "jpg"


def export_pdf(images: Iterable[np.ndarray], out_path: str, opts: ExportOptions | None = None,
               total: Optional[int] = None, progress: ProgressFn = None,
               aspect: Optional[float] = None) -> str:
    """Write a PDF, one page per image. Pages are staged on disk (low memory).

    With ``page_size="uniform"`` every page gets the same size; its shape is
    ``aspect`` (w/h) or, if not given, the shape needing the least filling for
    these images (then the images are held in memory once to measure them).
    """
    opts = opts or ExportOptions()
    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    if opts.page_size == "uniform" and aspect is None:
        images = list(images)
        aspect = best_aspect(im.shape[1] / im.shape[0] for im in images)
    page = page_size_pt(opts, aspect)
    with tempfile.TemporaryDirectory(prefix="video2book_") as tmp:
        files = []
        for i, img in enumerate(images):
            data, ext = encode_image(prepare_page(img, opts, aspect), opts)
            p = os.path.join(tmp, f"page_{i:05d}.{ext}")
            with open(p, "wb") as f:
                f.write(data)
            files.append(p)
            if progress:
                progress((i + 1) / total if total else 0.0, f"Preparing page {i + 1}" + (f"/{total}" if total else ""))
        if not files:
            raise ValueError("No pages to export.")
        if page is None:
            layout = img2pdf.get_fixed_dpi_layout_fun((opts.dpi, opts.dpi))
        else:
            # every page exactly this size (no auto-rotation of landscape pages)
            layout = img2pdf.get_layout_fun(pagesize=page, fit=img2pdf.FitMode.into, auto_orient=False)
        target = out_path if not opts.ocr_lang else os.path.join(tmp, "plain.pdf")
        with open(target, "wb") as f:
            img2pdf.convert(files, layout_fun=layout, outputstream=f)
        if opts.ocr_lang:
            if progress:
                progress(1.0, f"Running OCR ({opts.ocr_lang})...")
            ocr_pdf(target, out_path, opts.ocr_lang)
    return out_path


# -- OCR --------------------------------------------------------------------

def tesseract_languages() -> list[str]:
    exe = shutil.which("tesseract")
    if not exe:
        return []
    try:
        out = subprocess.run([exe, "--list-langs"], capture_output=True, text=True, timeout=30)
    except Exception:
        return []
    lines = (out.stdout or out.stderr).splitlines()
    return sorted(l.strip() for l in lines[1:] if l.strip() and " " not in l.strip())


def ocr_status() -> tuple[bool, str]:
    """``(available, human readable reason)``."""
    if not shutil.which("tesseract"):
        return False, "Tesseract is not installed (see README: 'Searchable PDF')."
    try:
        import ocrmypdf  # noqa: F401
    except Exception:
        if not shutil.which("ocrmypdf"):
            return False, "OCRmyPDF is not installed: run  pip install ocrmypdf"
    if not (shutil.which("gs") or shutil.which("gswin64c") or shutil.which("gswin32c")):
        return False, "Ghostscript is not installed (needed by OCRmyPDF, see README)."
    langs = tesseract_languages()
    return True, "OCR available; languages: " + (", ".join(l for l in langs if l != "osd") or "?")


def ocr_pdf(in_pdf: str, out_pdf: str, lang: str = "eng") -> str:
    ok, why = ocr_status()
    if not ok:
        raise RuntimeError(why)
    missing = [l for l in lang.split("+") if l not in tesseract_languages()]
    if missing:
        raise RuntimeError(f"Tesseract language(s) not installed: {', '.join(missing)}. "
                           f"Installed: {', '.join(tesseract_languages())}")
    kwargs = dict(language=lang.split("+"), output_type="pdf", progress_bar=False,
                  optimize=0, oversample=300, rotate_pages=False, deskew=False)
    try:
        import ocrmypdf
        ocrmypdf.ocr(in_pdf, out_pdf, **kwargs)
    except ImportError:
        cmd = ["ocrmypdf", "-l", lang, "--output-type", "pdf", "--optimize", "0",
               "--oversample", "300", in_pdf, out_pdf]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"OCRmyPDF failed: {res.stderr[-2000:]}")
    return out_pdf


# -- ZIP --------------------------------------------------------------------

def export_zip(images: Iterable[np.ndarray], out_path: str, fmt: str = "png",
               opts: ExportOptions | None = None, total: Optional[int] = None,
               progress: ProgressFn = None) -> str:
    opts = opts or ExportOptions()
    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_STORED) as z:
        for i, img in enumerate(images):
            data, ext = encode_image(img, opts, fmt)
            z.writestr(f"page_{i + 1:04d}.{ext}", data)
            if progress:
                progress((i + 1) / total if total else 0.0, f"Adding page {i + 1}")
    return out_path
