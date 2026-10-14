"""``yt2book``: YouTube channel or playlist -> one PDF per chapter / video.

Examples::

    yt2book https://www.youtube.com/@RuiNemesys "Miss Forensics" --latest 5
    yt2book @RuiNemesys "Miss Forensics" --chapters 140-145 -o ~/Comics
    yt2book "https://www.youtube.com/playlist?list=PL..."             # every video of a playlist
    yt2book "https://www.youtube.com/playlist?list=PL..." --latest 3  # its 3 latest videos
    yt2book @RuiNemesys "Miss Forensics" --list          # just show what's there
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from . import __version__


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="yt2book", formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Download the chapter videos of a series from a YouTube channel or playlist at 720p "
                    "and turn each one into a PDF with Video2Book.",
        epilog="Chapters are recognised by titles like 'Miss Forensics (Chapter 142)'. With a playlist and no "
               "series title, every video of the playlist is converted. "
               "Existing PDFs are skipped, so an interrupted run can simply be restarted.")
    p.add_argument("source", metavar="channel_or_playlist",
                   help="channel URL or @handle (e.g. https://www.youtube.com/@RuiNemesys), or a playlist link "
                        "(https://www.youtube.com/playlist?list=..., or a video link containing &list=...)")
    p.add_argument("series", nargs="?", default=None,
                   help='series title as written in the video titles, e.g. "Miss Forensics" '
                        "(required for a channel, optional for a playlist)")
    sel = p.add_mutually_exclusive_group()
    sel.add_argument("--chapters", metavar="SPEC", help="which chapters: 140-145, 12, or 3,5,9-11 (default: all)")
    sel.add_argument("--latest", type=int, metavar="N",
                     help="only the N latest: the N highest chapter numbers (for playlist videos without "
                          "chapter numbers: the last N of the playlist)")
    p.add_argument("--name", help="name for the output folder and PDF files (default: the series title, "
                                  "or the playlist's title)")
    p.add_argument("-o", "--output", default=os.path.join("~", "Video2Book"),
                   help="output folder; PDFs go to <output>/<name>/ (default: %(default)s)")
    p.add_argument("--list", action="store_true", help="only list what was found, download nothing")
    p.add_argument("--height", type=int, default=720,
                   help="video quality to download (default: 720). Without it: the closest quality above, "
                        "or if there is none above, the closest below")
    p.add_argument("--keep-videos", action="store_true",
                   help="keep the downloaded videos (needed to fix pages later in the Video2Book app)")
    p.add_argument("--force", action="store_true", help="redo chapters whose PDF already exists")
    p.add_argument("--refresh", action="store_true", help="re-read the channel / playlist instead of using the "
                                                          "listing cached in the last hours")
    p.add_argument("--cookies-from-browser", metavar="BROWSER",
                   help="use your browser's YouTube login (firefox, chrome, ...) if YouTube asks to sign in")
    g = p.add_argument_group("PDF options (see `video2book --help`)")
    g.add_argument("--sensitivity", type=float, default=0.5)
    g.add_argument("--enhance", choices=("original", "clean", "bw"), default="original")
    g.add_argument("--page-size", choices=("uniform", "fit", "a4", "letter"), default="uniform",
                   help="uniform (default): every page the same size, shaped to need the least filling; "
                        "fit: each page the size of its panel")
    g.add_argument("--fill", choices=("auto", "white", "black"), default="auto",
                   help="colour used to fill pages to the common size (auto: each page's edge colour)")
    g.add_argument("--dpi", type=int, default=150)
    g.add_argument("--ocr", metavar="LANG", help="searchable PDFs, e.g. eng")
    p.add_argument("-q", "--quiet", action="store_true")
    p.add_argument("--version", action="version", version=f"yt2book (video2book {__version__})")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    from .project import Settings
    from .youtube import gather, is_playlist, process_chapter, series_folder, write_report

    say = (lambda *a: None) if args.quiet else (lambda *a: print(*a, flush=True))
    t0 = time.time()
    playlist = is_playlist(args.source)
    what = "playlist" if playlist else "channel"
    try:
        if playlist:
            say(f"Reading the playlist {args.source} ...")
        elif args.latest and not args.chapters:
            say(f"Scanning {args.source} for the {args.latest} latest '{args.series}' chapters...")
        else:
            say(f"Listing the videos of {args.source} ...")
        sel = gather(args.source, args.series, args.chapters, args.latest, args.name, args.refresh,
                     args.cookies_from_browser, log=say)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"error: could not read the {what}: {e}", file=sys.stderr)
        if "Sign in" in str(e) or "bot" in str(e):
            print("hint: try --cookies-from-browser firefox (or chrome)", file=sys.stderr)
        return 1
    if sel.listed is not None:
        say(f"{sel.listed} videos in the {what}.")
    chapters = sel.items
    if not chapters:
        if args.series:
            msg = f"No videos titled like '{args.series} (Chapter N)' found"
        else:
            msg = "No videos found in the playlist"
        print(msg + (f" for chapters {args.chapters}" if args.chapters else "") + ".", file=sys.stderr)
        return 1
    name = sel.name
    folder = series_folder(args.output, name)
    if all(c.numbered for c in chapters):
        say(f"{len(chapters)} chapter(s): " + ", ".join(c.label.lstrip("0") or "0" for c in chapters))
    else:
        say(f"{len(chapters)} video(s) - PDFs named '{name} - <position> - <title>'")
    if args.list:
        for c in chapters:
            done = "PDF exists" if (folder / c.pdf_name(name)).exists() else ""
            alt = f" (+{len(c.alternates)} other upload(s))" if c.alternates else ""
            say(f"  {c.display}: {c.video.title}  {c.video.url}{alt}  {done}")
        return 0

    settings = Settings(sensitivity=args.sensitivity, enhance=args.enhance, page_size=args.page_size, fill=args.fill,
                        dpi=args.dpi, ocr_lang=args.ocr)
    results = []
    for i, c in enumerate(chapters, 1):
        say(f"[{i}/{len(chapters)}] {c.display}: {c.video.title}")
        last = [-1.0]

        def progress(f: float, msg: str = "") -> None:
            if not args.quiet and sys.stderr.isatty() and f - last[0] >= 0.01:
                last[0] = f
                sys.stderr.write(f"\r    {f * 100:5.1f}%  {msg[:60]:<60}")
                sys.stderr.flush()

        r = process_chapter(c, name, args.output, settings, keep_videos=args.keep_videos,
                            force=args.force, height=args.height,
                            cookies_from_browser=args.cookies_from_browser, progress=progress)
        if not args.quiet and sys.stderr.isatty():
            sys.stderr.write("\r" + " " * 75 + "\r")
        if r.status == "exists":
            say("    already done (use --force to redo)")
        elif r.status == "done":
            say(f"    {r.pages} pages, {r.height}p {r.vcodec or ''}, {r.seconds:.0f}s -> {r.pdf}"
                + ("".join(f"\n    note: {n}" for n in r.notes)))
        else:
            say(f"    FAILED: {r.error}")
        results.append(r)
    report = write_report(results, folder, name, args.source)
    done = sum(r.status == "done" for r in results)
    failed = [r for r in results if r.status == "failed"]
    say(f"\n{done} PDF(s) made, {sum(r.status == 'exists' for r in results)} already existed, "
        f"{len(failed)} failed - {folder}  ({time.time() - t0:.0f}s)")
    say(f"Report: {report}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
