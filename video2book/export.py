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
    "fit": None,                                   # page = image size at the given DPI
    "a4": (img2pdf.mm_to_pt(210), img2pdf.mm_to_pt(297)),
    "letter": (img2pdf.in_to_pt(8.5), img2pdf.in_to_pt(11)),
}
PAGE_SIZE_LABELS = {"fit": "Fit to image", "a4": "A4", "letter": "Letter"}


@dataclass
class ExportOptions:
    page_size: str = "a4"        # "fit" | "a4" | "letter"
    dpi: int = 200
    jpeg_quality: int = 88
    ocr_lang: Optional[str] = None  # e.g. "eng", "fra+eng"; None = no OCR

    def to_dict(self) -> dict:
        return asdict(self)


def _is_binary(img: np.ndarray) -> bool:
    if img.ndim != 2:
        return False
    vals = np.unique(img[:: max(1, img.shape[0] // 64), :: max(1, img.shape[1] // 64)])
    return vals.size <= 2 and set(vals.tolist()) <= {0, 255}


def fit_for_page(img: np.ndarray, opts: ExportOptions) -> np.ndarray:
    """Downscale (never upscale) so the image is at most ``dpi`` on the page."""
    size = PAGE_SIZES_PT.get(opts.page_size)
    if size is None:
        return img
    h, w = img.shape[:2]
    pw, ph = size
    if (w > h) != (pw > ph):  # img2pdf auto-orients landscape images
        pw, ph = ph, pw
    max_w, max_h = pw / 72.0 * opts.dpi, ph / 72.0 * opts.dpi
    s = min(max_w / w, max_h / h)
    if s >= 1:
        return img
    return cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)


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
               total: Optional[int] = None, progress: ProgressFn = None) -> str:
    """Write a PDF, one page per image. Pages are staged on disk (low memory)."""
    opts = opts or ExportOptions()
    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="video2book_") as tmp:
        files = []
        for i, img in enumerate(images):
            data, ext = encode_image(fit_for_page(img, opts), opts)
            p = os.path.join(tmp, f"page_{i:05d}.{ext}")
            with open(p, "wb") as f:
                f.write(data)
            files.append(p)
            if progress:
                progress((i + 1) / total if total else 0.0, f"Preparing page {i + 1}" + (f"/{total}" if total else ""))
        if not files:
            raise ValueError("No pages to export.")
        size = PAGE_SIZES_PT.get(opts.page_size)
        if size is None:
            layout = img2pdf.get_fixed_dpi_layout_fun((opts.dpi, opts.dpi))
        else:
            layout = img2pdf.get_layout_fun(pagesize=size, fit=img2pdf.FitMode.into, auto_orient=True)
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
