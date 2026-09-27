"""Command line interface: fully automatic video -> PDF.

Example::

    video2book input.mp4 -o book.pdf --expected-pages 240 --ocr eng
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from . import __version__


def _fmt_time(t: float) -> str:
    m, s = divmod(max(0.0, t), 60)
    return f"{int(m):02d}:{s:04.1f}"


class _Bar:
    def __init__(self, quiet: bool):
        self.quiet = quiet
        self.last = ""
        self.tty = sys.stderr.isatty()

    def __call__(self, frac: float, msg: str = "") -> None:
        if self.quiet:
            return
        n = int(frac * 30)
        line = f"[{'#' * n}{'.' * (30 - n)}] {frac * 100:5.1f}%  {msg}"[:110]
        if self.tty:
            sys.stderr.write("\r" + line.ljust(len(self.last)))
            sys.stderr.flush()
        elif msg.split(" ")[0:2] != self.last.split("%  ")[-1].split(" ")[0:2]:
            sys.stderr.write(line + "\n")
        self.last = line

    def done(self) -> None:
        if not self.quiet and self.tty and self.last:
            sys.stderr.write("\n")
        self.last = ""


def _parse_crop(value: str):
    if value in ("auto", "none"):
        return value, None
    try:
        parts = [float(x) for x in value.split(",")]
        assert len(parts) == 4 and all(0 <= p <= 1 for p in parts)
        return "manual", parts
    except Exception:
        raise argparse.ArgumentTypeError("--crop must be 'auto', 'none' or 'x0,y0,x1,y1' with values in 0..1")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="video2book",
        description="Turn a video of book pages (slideshow / gallery swipe) into a clean PDF.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("video", help="input video (MP4, MOV, MKV, ...)")
    p.add_argument("-o", "--output", help="output PDF (default: <video name>.pdf next to the video)")
    p.add_argument("--expected-pages", type=int, help="warn if the number of pages found differs")
    p.add_argument("--auto-adjust", action="store_true",
                   help="with --expected-pages: automatically retry with the suggested sensitivity")
    p.add_argument("--ocr", metavar="LANG", help="make a searchable PDF with Tesseract, e.g. 'eng' or 'fra+eng'")
    g = p.add_argument_group("detection")
    g.add_argument("--sensitivity", type=float, default=0.5, help="0..1, higher finds more pages")
    g.add_argument("--threshold", type=float, help="manual motion threshold (overrides --sensitivity)")
    g.add_argument("--min-duration", type=float, default=0.3, help="seconds a page must stay still")
    g.add_argument("--step", type=int, default=1, help="analyse every Nth frame (faster, less precise)")
    g.add_argument("--method", choices=("median", "sharpest"), default="median",
                   help="page image: temporal median of the still frames, or the single sharpest frame")
    g.add_argument("--no-dedup", action="store_true", help="keep repeated pages")
    g = p.add_argument_group("page processing")
    g.add_argument("--crop", type=_parse_crop, default=("auto", None), metavar="auto|none|x0,y0,x1,y1",
                   help="page crop; manual values are fractions of the frame (0..1)")
    g.add_argument("--perspective", action="store_true", help="straighten photographed pages (4-corner warp)")
    g.add_argument("--enhance", choices=("original", "clean", "bw"), default="original",
                   help="enhancement preset")
    g = p.add_argument_group("export")
    g.add_argument("--page-size", choices=("uniform", "fit", "a4", "letter"), default="uniform",
                   help="uniform: every page the same size, shaped to need the least filling; "
                        "fit: each page the size of its image")
    g.add_argument("--fill", choices=("auto", "white", "black"), default="auto",
                   help="colour used to fill pages to the common size (auto: each page's edge colour)")
    g.add_argument("--dpi", type=int, default=200)
    g.add_argument("--quality", type=int, default=88, help="JPEG quality (1-100)")
    g.add_argument("--zip", metavar="FILE", help="also save the page images as a ZIP")
    g.add_argument("--zip-format", choices=("png", "jpg"), default="png")
    g.add_argument("--report", metavar="FILE", help="write a JSON report (pages, timestamps, flags)")
    g.add_argument("--plot", metavar="FILE", help="save the motion graph as PNG (needs matplotlib)")
    g = p.add_argument_group("misc")
    g.add_argument("--cache-dir", help="cache folder (default: per-user cache folder)")
    g.add_argument("--no-cache", action="store_true", help="don't read or write the cache")
    g.add_argument("-q", "--quiet", action="store_true")
    g.add_argument("--version", action="version", version=f"video2book {__version__}")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    from .export import ocr_status
    from .project import Pipeline, Settings

    if not os.path.exists(args.video):
        print(f"error: video not found: {args.video}", file=sys.stderr)
        return 2
    out = args.output or os.path.splitext(args.video)[0] + ".pdf"
    if args.ocr:
        ok, why = ocr_status()
        if not ok:
            print(f"error: --ocr requested but OCR is unavailable: {why}", file=sys.stderr)
            return 2
    crop_mode, crop_rel = args.crop
    settings = Settings(
        step=args.step, sensitivity=args.sensitivity, threshold=args.threshold,
        min_duration=args.min_duration, method=args.method, dedup=not args.no_dedup,
        crop_mode=crop_mode, manual_crop=crop_rel, perspective=args.perspective,
        enhance=args.enhance, page_size=args.page_size, fill=args.fill, dpi=args.dpi, jpeg_quality=args.quality,
        ocr_lang=args.ocr, expected_pages=args.expected_pages,
    )
    say = (lambda *a: None) if args.quiet else (lambda *a: print(*a))
    t0 = time.time()
    bar = _Bar(args.quiet)
    try:
        pipe = Pipeline(args.video, settings, cache_root=args.cache_dir, use_cache=not args.no_cache)
    except Exception as e:  # unreadable video etc.
        print(f"error: cannot open video: {e}", file=sys.stderr)
        return 2
    say(f"Video: {pipe.info.describe()}")
    project = pipe.run(progress=bar)
    bar.done()
    check = pipe.check_expected(project)
    if check and not check["ok"] and args.auto_adjust and "suggested_sensitivity" in check:
        sens = check["suggested_sensitivity"]
        if abs(sens - settings.sensitivity) > 1e-6:
            say(f"{check['message']}\nRetrying with sensitivity {sens:.2f} ...")
            settings.sensitivity = sens
            project = pipe.run(progress=bar)
            bar.done()
            check = pipe.check_expected(project)

    s = project.summary()
    say(f"Motion threshold {project.threshold:.2f} (typical still level {project.base:.2f}, "
        f"typical transition {project.peak:.1f})")
    say(f"Detected {s['segments']} still segments -> {s['pages']} pages "
        f"({s['duplicates_merged']} duplicates merged, {s['possible_duplicates']} flagged as possible "
        f"duplicates, {s['blank']} blank)")
    notes = []
    for p in project.pages:
        num = project.page_number(p.id)
        where = f"page {num}" if num else f"(excluded, {p.status})"
        if p.status == "duplicate":
            ref = project.page_number(p.dup_of)
            notes.append(f"  {_fmt_time(p.t)}  duplicate of page {ref} - merged")
        if "possible_duplicate" in p.flags and p.included:
            ref = project.page_number(p.dup_of)
            notes.append(f"  {_fmt_time(p.t)}  {where}: looks like page {ref} "
                         f"(similarity {p.dup_score['ssim']:.2f}) - kept, please check")
        if "short" in p.flags and p.included:
            notes.append(f"  {_fmt_time(p.t)}  {where}: shown only {p.t_end - p.t_start:.2f}s - check it's a real page")
        if "moving" in p.flags and p.included:
            notes.append(f"  {_fmt_time(p.t)}  {where}: page was moving (zoom/pan) - sharpest frame used")
        if p.status == "blank":
            notes.append(f"  {_fmt_time(p.t)}  blank/black frame - skipped")
    if notes:
        say("Review notes:")
        for n in notes:
            say(n)
    if check:
        say(("OK: " if check["ok"] else "WARNING: ") + check["message"])

    if not project.included():
        print("error: no pages to export", file=sys.stderr)
        return 1
    pdf = pipe.export_pdf(project, out, progress=bar)
    bar.done()
    size = os.path.getsize(pdf) / 1e6
    say(f"Wrote {pdf} ({s['pages']} pages, {size:.1f} MB{', searchable' if args.ocr else ''})")
    if args.zip:
        z = pipe.export_zip(project, args.zip, fmt=args.zip_format, progress=bar)
        bar.done()
        say(f"Wrote {z}")
    if args.plot:
        from .plot import motion_png
        res = pipe.segments()
        if motion_png(pipe.analysis(), res.smoothed, res.threshold, project, args.plot):
            say(f"Wrote {args.plot}")
        else:
            say("Could not write the plot (pip install matplotlib)")
    if args.report:
        rep = {
            "video": pipe.video, "info": project.info, "settings": project.settings.to_dict(),
            "threshold": project.threshold, "summary": s, "timings": project.timings,
            "expected_check": check, "output": pdf,
            "pages": [{"number": project.page_number(p.id), "time": round(p.t, 3),
                       "start": round(p.t_start, 3), "end": round(p.t_end, 3), "status": p.status,
                       "flags": p.flags, "duplicate_of": project.page_number(p.dup_of) if p.dup_of else None,
                       "similarity": p.dup_score, "crop": p.crop} for p in project.pages],
        }
        with open(args.report, "w") as f:
            json.dump(rep, f, indent=1)
        say(f"Wrote {args.report}")
    pipe.save_project(project)
    say(f"Total time {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
