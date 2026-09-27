"""Enhancement presets.

* ``original`` - untouched pixels.
* ``clean``    - white balance on the paper colour, auto-contrast, light sharpening.
* ``bw``       - "black & white scan": lighting flattened, then adaptive
  threshold. Small images are upscaled first so the 1-bit result stays smooth.
"""

from __future__ import annotations

import cv2
import numpy as np

PRESETS = ("original", "clean", "bw")
PRESET_LABELS = {
    "original": "Original",
    "clean": "Clean (auto-contrast, white balance, sharpen)",
    "bw": "Black & white scan",
}


def white_balance(img: np.ndarray) -> np.ndarray:
    """Make the brightest area (usually the paper) neutral grey/white."""
    if img.ndim != 3:
        return img
    f = img.astype(np.float32)
    lum = f.mean(axis=2)
    sel = f[lum >= np.percentile(lum, 90)]
    if sel.size == 0:
        return img
    white = np.median(sel, axis=0)
    if float(white.min()) < 80:
        return img  # no reliable bright reference
    gain = np.clip(white.mean() / np.maximum(white, 1.0), 0.75, 1.35)
    return np.clip(f * gain, 0, 255).astype(np.uint8)


def auto_contrast(img: np.ndarray, low: float = 0.5, high: float = 99.5) -> np.ndarray:
    """Stretch luminance between two percentiles (same gain on all channels)."""
    f = img.astype(np.float32)
    lum = f.mean(axis=2) if f.ndim == 3 else f
    lo, hi = np.percentile(lum, [low, high])
    lo = min(float(lo), 80.0)    # don't crush pictures that are simply light
    hi = max(float(hi), 150.0)   # ...or dark
    if hi - lo < 20:
        return img
    return np.clip((f - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)


def sharpen(img: np.ndarray, amount: float = 0.6) -> np.ndarray:
    h, w = img.shape[:2]
    sigma = max(0.8, min(h, w) / 1200.0)
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    return cv2.addWeighted(img, 1 + amount, blur, -amount, 0)


def clean(img: np.ndarray) -> np.ndarray:
    return sharpen(auto_contrast(white_balance(img)))


def bw_scan(img: np.ndarray, target_long_side: int = 2000) -> np.ndarray:
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
    h, w = g.shape
    scale = min(3.0, target_long_side / float(max(h, w)))
    if scale > 1.2:
        g = cv2.resize(g, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
        h, w = g.shape
    # Flatten uneven lighting: divide by an estimate of the paper background.
    k = max(15, int(min(h, w) * 0.03) | 1)
    bg = cv2.morphologyEx(g, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    bg = cv2.GaussianBlur(bg, (0, 0), k / 3.0)
    norm = cv2.divide(g, np.maximum(bg, 1), scale=255)
    block = max(15, int(min(h, w) / 40) | 1)
    bw = cv2.adaptiveThreshold(norm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, 15)
    return cv2.medianBlur(bw, 3)


def apply(img: np.ndarray, preset: str = "original", preview: bool = False) -> np.ndarray:
    """Apply a preset. ``preview=True`` skips the B&W upscaling (thumbnails)."""
    if preset == "clean":
        return clean(img)
    if preset == "bw":
        return bw_scan(img, target_long_side=0 if preview else 2000)
    return img
