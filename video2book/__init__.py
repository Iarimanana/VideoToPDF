"""Video2Book: turn a video of book pages into a clean PDF.

The package is split into small, UI-independent modules:

- :mod:`video2book.video`     reading videos (rotation aware, streaming)
- :mod:`video2book.motion`    frame-to-frame motion signal
- :mod:`video2book.segments`  adaptive detection of stable (page) segments
- :mod:`video2book.extract`   best image per segment (median / sharpest)
- :mod:`video2book.dedup`     duplicate detection (perceptual hash + SSIM)
- :mod:`video2book.crop`      automatic / manual cropping, perspective
- :mod:`video2book.enhance`   enhancement presets
- :mod:`video2book.export`    PDF / OCR / ZIP export
- :mod:`video2book.project`   the whole pipeline, cache and editable state
"""

__version__ = "1.0.0"
