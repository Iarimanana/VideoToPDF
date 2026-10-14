"""Turn the chapter videos of a series - from a YouTube channel or a playlist -
into PDFs.

Workflow (``yt2book`` on the command line, or the "YouTube series" page of
the web app):

1. list the videos with yt-dlp, without downloading: a channel's Videos tab
   (newest first), or a playlist (in playlist order);
2. keep the ones titled like ``<Series> (Chapter N)``. For a playlist the
   series title is optional: without it every video of the playlist is used,
   ordered by its chapter number when the title has one ("Chapter 12",
   "Ep. 3"...), otherwise by its position in the playlist;
3. select: the latest N, a chapter range / list, or all;
4. download each one at 720p (video stream only - the audio is not
   needed); without a 720p version, the closest quality above 720p, and only
   if there is none above, the closest below;
5. convert it with the Video2Book pipeline into ``<Name> - Chapter NNN.pdf``
   (or ``<Name> - NNN - <video title>.pdf`` for videos without a chapter
   number, NNN being the playlist position).

Videos whose PDF already exists are skipped, so an interrupted run can
simply be started again. Downloaded videos are deleted after a successful
conversion unless ``keep_videos`` is set.
"""

from __future__ import annotations

import json
import os
import re
import time
import unicodedata
from urllib.parse import parse_qs, urlparse
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional

from .project import Pipeline, Settings, default_cache_root

Log = Callable[[str], None]
ProgressFn = Optional[Callable[[float, str], None]]

TARGET_HEIGHT = 720
LISTING_TTL_S = 6 * 3600


# -- titles ---------------------------------------------------------------------

def normalize(text: str) -> str:
    """Case-folded, accent-free, punctuation-simplified text for matching."""
    t = unicodedata.normalize("NFKD", text)
    t = "".join(c for c in t if not unicodedata.combining(c)).casefold()
    t = re.sub(r"[‐-―−]", "-", t)          # dashes
    t = re.sub(r"[‘’“”`´]", "'", t)   # quotes
    t = re.sub(r"[()\[\]{}【】「」『』〈〉《》|:·•,;/_#]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def chapter_pattern(series: str) -> re.Pattern:
    words = [re.escape(w) for w in normalize(series).replace("'", " ' ").split()]
    series_rx = r"\s*".join(words)
    return re.compile(
        r"(?:^|[^a-z0-9])" + series_rx +
        r"[\s\-'.!?]*(?:chapter|chap|ch|episode|ep)\.?\s*(\d+(?:\.\d+)?)"
        r"(?:\s*(?:-|~|to|&|and|\+)\s*(?:chapter|ch\.?)?\s*(\d+(?:\.\d+)?))?(?![0-9])")


def parse_chapter(title: str, series: str) -> Optional[tuple]:
    """``(first, last)`` chapter numbers if ``title`` is a chapter of ``series``.

    Accepts e.g. "Miss Forensics (Chapter 142)", "Miss Forensics - Ch. 142",
    "【Miss Forensics】Chapter 142", "Miss Forensics (Chapter 12-13)".
    A different series with a longer name ("Miss Forensics 2 (Chapter 5)")
    does not match.
    """
    m = chapter_pattern(series).search(normalize(title))
    if not m:
        return None
    a = float(m.group(1))
    b = float(m.group(2)) if m.group(2) else a
    if b < a or b - a > 20:  # "Chapter 5 & 2024" etc.: keep the first number only
        b = a
    return (a, b)


_ANY_CHAPTER = re.compile(
    r"(?:^|[^a-z0-9])(?:chapter|chap|ch|episode|ep)\.?\s*(\d+(?:\.\d+)?)"
    r"(?:\s*(?:-|~|to|&|and|\+)\s*(?:chapter|ch\.?)?\s*(\d+(?:\.\d+)?))?(?![0-9])")


def parse_any_chapter(title: str) -> Optional[tuple]:
    """``(first, last)`` chapter numbers from any title ("... (Chapter 12)",
    "Ep. 3", "Ch 10-11"), whatever the series name. None if there is none."""
    m = _ANY_CHAPTER.search(normalize(title))
    if not m:
        return None
    a = float(m.group(1))
    b = float(m.group(2)) if m.group(2) else a
    if b < a or b - a > 20:
        b = a
    return (a, b)


def fmt_num(x: float) -> str:
    return str(int(x)) if float(x).is_integer() else str(x)


def chapter_label(first: float, last: float, pad: int = 3) -> str:
    def p(x: float) -> str:
        s = fmt_num(x)
        whole, _, frac = s.partition(".")
        return whole.zfill(pad) + ("." + frac if frac else "")
    return p(first) if first == last else f"{p(first)}-{p(last)}"


def safe_filename(name: str) -> str:
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', " ", name)
    return re.sub(r"\s+", " ", name).strip(" .") or "untitled"


# -- channel listing ----------------------------------------------------------------

def channel_videos_url(channel: str) -> str:
    """Accepts '@handle', a channel URL or a channel URL with a tab."""
    c = channel.strip()
    if c.startswith("@"):
        c = "https://www.youtube.com/" + c
    elif not c.startswith("http"):
        c = "https://www.youtube.com/" + c.lstrip("/")
    c = c.split("?")[0].rstrip("/")
    c = re.sub(r"/(videos|shorts|streams|featured|playlists|community|about|search)$", "", c)
    return c + "/videos"


@dataclass
class ChannelVideo:
    id: str
    title: str
    url: str
    duration: Optional[float] = None
    index: int = 0            # position on the channel's Videos tab (0 = newest)


def _ydl_base_opts(cookies_from_browser: Optional[str] = None) -> dict:
    opts = {"quiet": True, "no_warnings": True, "noprogress": True, "retries": 5,
            "fragment_retries": 10, "extractor_retries": 3}
    if cookies_from_browser:
        opts["cookiesfrombrowser"] = (cookies_from_browser,)
    return opts


UNAVAILABLE_TITLES = {"[deleted video]", "[private video]", "[unavailable video]"}


def _iter_flat(url: str, cookies_from_browser: Optional[str] = None,
               meta: Optional[dict] = None) -> Iterator[ChannelVideo]:
    """Entries of a channel tab or playlist, fetched page by page (lazily).
    The list's title is stored in ``meta["title"]`` when given."""
    from yt_dlp import YoutubeDL

    opts = {**_ydl_base_opts(cookies_from_browser), "extract_flat": "in_playlist", "skip_download": True,
            "lazy_playlist": True}
    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False, process=False)
        if meta is not None:
            meta["title"] = info.get("title")
        for i, e in enumerate(info.get("entries") or []):
            if not e or not e.get("id"):
                continue
            vid = e["id"]
            yield ChannelVideo(id=vid, title=e.get("title") or "", duration=e.get("duration"), index=i,
                               url=e.get("url") if str(e.get("url", "")).startswith("http")
                               else f"https://www.youtube.com/watch?v={vid}")


def iter_channel_videos(channel: str, cookies_from_browser: Optional[str] = None) -> Iterator[ChannelVideo]:
    """Videos of a channel, newest first, fetched page by page (lazily)."""
    return _iter_flat(channel_videos_url(channel), cookies_from_browser)


def iter_playlist_videos(playlist: str, cookies_from_browser: Optional[str] = None,
                         meta: Optional[dict] = None) -> Iterator[ChannelVideo]:
    """Videos of a playlist, in playlist order (``index`` = position, 0-based)."""
    return _iter_flat(playlist_url(playlist), cookies_from_browser, meta)


def usable(v: ChannelVideo) -> bool:
    return v.title.strip().lower() not in UNAVAILABLE_TITLES


# -- playlists -------------------------------------------------------------------------

_PLAYLIST_ID = re.compile(r"^(PL|UU|OL|FL|LL|RD|UL|PU)[A-Za-z0-9_-]{10,}$")


def playlist_id(source: str) -> Optional[str]:
    """The playlist id in a playlist link (or a video link with ``&list=``),
    or a bare playlist id; None for anything else (e.g. a channel)."""
    s = source.strip()
    if _PLAYLIST_ID.match(s):
        return s
    if not re.match(r"^https?://", s):
        if "youtube.com/" in s or "youtu.be/" in s:
            s = "https://" + s
        else:
            return None
    q = parse_qs(urlparse(s).query)
    ids = q.get("list")
    return ids[0] if ids and ids[0] else None


def is_playlist(source: str) -> bool:
    return playlist_id(source) is not None


def playlist_url(source: str) -> str:
    pid = playlist_id(source)
    if not pid:
        raise ValueError(f"Not a playlist link: {source}")
    return f"https://www.youtube.com/playlist?list={pid}"


# -- listing cache ----------------------------------------------------------------------

def _source_url(source: str) -> str:
    return playlist_url(source) if is_playlist(source) else channel_videos_url(source)


def _listing_cache(source: str) -> Path:
    key = re.sub(r"[^a-z0-9]+", "_", _source_url(source).lower())[-80:]
    return default_cache_root() / "youtube" / f"{key}.json"


def list_source(source: str, refresh: bool = False, cookies_from_browser: Optional[str] = None,
                log: Optional[Log] = None) -> tuple:
    """``(title, videos)`` of a channel or playlist (cached for a few hours).
    ``title`` is the playlist title (None for a channel)."""
    path = _listing_cache(source)
    if not refresh and path.exists() and time.time() - path.stat().st_mtime < LISTING_TTL_S:
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list):  # cache written by an older version
            data = {"title": None, "videos": data}
        return data.get("title"), [ChannelVideo(**d) for d in data["videos"]]
    meta: dict = {}
    it = iter_playlist_videos(source, cookies_from_browser, meta) if is_playlist(source) \
        else iter_channel_videos(source, cookies_from_browser)
    videos = []
    for v in it:
        if usable(v):
            videos.append(v)
        if log and len(videos) % 100 == 0 and videos:
            log(f"  ...{len(videos)} videos listed")
    title = meta.get("title")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"title": title, "videos": [asdict(v) for v in videos]}, f)
    return title, videos


def list_channel_videos(channel: str, refresh: bool = False, cookies_from_browser: Optional[str] = None,
                        log: Optional[Log] = None) -> list:
    """All videos of the channel (cached for a few hours)."""
    return list_source(channel, refresh, cookies_from_browser, log)[1]


# -- chapters -------------------------------------------------------------------------

@dataclass
class Chapter:
    first: Optional[float]   # chapter numbers; None for a playlist video without one
    last: Optional[float]
    video: ChannelVideo
    alternates: list = field(default_factory=list)  # other uploads of the same chapter

    @property
    def numbered(self) -> bool:
        return self.first is not None

    @property
    def position(self) -> int:
        """1-based position in the playlist / channel list."""
        return self.video.index + 1

    @property
    def label(self) -> str:
        return chapter_label(self.first, self.last) if self.numbered else f"#{self.position:03d}"

    @property
    def display(self) -> str:
        return f"Chapter {self.label}" if self.numbered else f"Video {self.position}"

    def pdf_name(self, name: str) -> str:
        if self.numbered:
            return safe_filename(f"{name} - Chapter {self.label}") + ".pdf"
        title = safe_filename(self.video.title)[:80].rstrip(" .")
        return safe_filename(f"{name} - {self.position:03d} - {title}") + ".pdf"


def sort_items(items: list) -> list:
    """Numbered items by chapter number. An item without a number (Prologue,
    Extra...) keeps its place after the numbered item it follows in the list
    (or first, if it comes before any numbered one), so one prologue doesn't
    turn the whole playlist into list order."""
    in_list = sorted(items, key=lambda c: c.video.index)
    numbered = sorted((c for c in items if c.numbered), key=lambda c: (c.first, c.last))
    if len(numbered) == len(items):
        return numbered
    after: dict = {}          # id(numbered item) or None -> unnumbered items following it
    prev = None
    for c in in_list:
        if c.numbered:
            prev = c
        else:
            after.setdefault(id(prev) if prev else None, []).append(c)
    out = list(after.get(None, []))
    for c in numbered:
        out.append(c)
        out.extend(after.get(id(c), []))
    return out


def find_chapters(videos: Iterable[ChannelVideo], series: str) -> list:
    """Chapters of ``series`` among ``videos`` (newest upload wins for repeats),
    sorted by chapter number."""
    by_key: dict = {}
    for v in sorted(videos, key=lambda v: v.index):
        rng = parse_chapter(v.title, series)
        if rng is None:
            continue
        if rng in by_key:
            by_key[rng].alternates.append(v)
        else:
            by_key[rng] = Chapter(rng[0], rng[1], v)
    return sorted(by_key.values(), key=lambda c: (c.first, c.last))


def parse_selection(spec: str) -> list:
    """'140-145', '12', '3,5,9-11' -> list of (lo, hi) ranges."""
    out = []
    for part in re.split(r"[,\s]+", spec.strip()):
        if not part:
            continue
        m = re.fullmatch(r"(\d+(?:\.\d+)?)(?:-(\d+(?:\.\d+)?))?", part)
        if not m:
            raise ValueError(f"Invalid chapter selection '{part}' (use e.g. 140-145 or 3,5,9-11)")
        lo = float(m.group(1))
        hi = float(m.group(2)) if m.group(2) else lo
        out.append((min(lo, hi), max(lo, hi)))
    if not out:
        raise ValueError("Empty chapter selection")
    return out


def select_chapters(chapters: list, spec: Optional[str] = None, latest: Optional[int] = None) -> list:
    """Filter by chapter ranges (``spec``) and/or keep the ``latest`` N: the N
    highest chapter numbers, or for videos without numbers the last N of the
    playlist."""
    if spec:
        ranges = parse_selection(spec)
        chapters = [c for c in chapters if c.numbered and any(c.first <= hi and c.last >= lo for lo, hi in ranges)]
    chapters = sort_items(chapters)
    if latest:
        chapters = chapters[-int(latest):]
    return chapters


def playlist_items(videos: Iterable[ChannelVideo], series: Optional[str] = None) -> list:
    """What to convert from a playlist: the chapters of ``series`` if given,
    otherwise every video (a chapter number is read from the title when there
    is one; the first occurrence of a repeated chapter is used)."""
    videos = [v for v in videos if usable(v)]
    if series:
        return find_chapters(videos, series)
    items, by_key = [], {}
    for v in sorted(videos, key=lambda v: v.index):
        rng = parse_any_chapter(v.title)
        if rng is None:
            items.append(Chapter(None, None, v))
        elif rng in by_key:
            by_key[rng].alternates.append(v)
        else:
            by_key[rng] = Chapter(rng[0], rng[1], v)
            items.append(by_key[rng])
    return sort_items(items)


def latest_chapters(channel: str, series: str, n: int, cookies_from_browser: Optional[str] = None,
                    log: Optional[Log] = None, extra: int = 30, scan_limit: int = 5000,
                    videos: Optional[Iterable[ChannelVideo]] = None) -> list:
    """The ``n`` highest chapters without listing the whole channel.

    The Videos tab is newest first: scanning stops ``extra`` videos after the
    ``n``-th distinct chapter was found (a margin for uploads out of order).
    """
    seen: list = []
    keys: set = set()
    found_at = None
    source = videos if videos is not None else iter_channel_videos(channel, cookies_from_browser)
    for v in source:
        if not usable(v):
            continue
        seen.append(v)
        rng = parse_chapter(v.title, series)
        if rng is not None:
            keys.add(rng)
        if found_at is None and len(keys) >= n:
            found_at = len(seen)
        if found_at is not None and len(seen) - found_at >= extra:
            break
        if len(seen) >= scan_limit:
            break
        if log and len(seen) % 100 == 0:
            log(f"  ...{len(seen)} videos scanned")
    return select_chapters(find_chapters(seen, series), latest=n)


@dataclass
class Selection:
    name: str                 # used for the output folder and the PDF names
    source: str
    kind: str                 # "channel" | "playlist"
    items: list               # Chapter objects, in output order
    listed: Optional[int] = None  # how many videos were looked at


def gather(source: str, series: Optional[str] = None, spec: Optional[str] = None,
           latest: Optional[int] = None, name: Optional[str] = None, refresh: bool = False,
           cookies_from_browser: Optional[str] = None, log: Optional[Log] = None) -> Selection:
    """Find what to convert, from a channel (series title required) or a
    playlist (series title optional)."""
    series = (series or "").strip() or None
    name = (name or "").strip() or None
    if is_playlist(source):
        title, videos = list_source(source, refresh, cookies_from_browser, log)
        items = select_chapters(playlist_items(videos, series), spec, latest)
        label = name or series or (title or "").strip() or f"Playlist {playlist_id(source)}"
        return Selection(safe_filename(label), source, "playlist", items, len(videos))
    if not series:
        raise ValueError("A series title is needed for a channel (or give a playlist link instead).")
    if latest and not spec:
        items = latest_chapters(source, series, latest, cookies_from_browser, log=log)
        return Selection(safe_filename(name or series), source, "channel", items)
    videos = list_channel_videos(source, refresh, cookies_from_browser, log)
    items = select_chapters(find_chapters(videos, series), spec, latest)
    return Selection(safe_filename(name or series), source, "channel", items, len(videos))


# -- download + convert ---------------------------------------------------------------------

@dataclass
class ChapterResult:
    chapter: str
    title: str
    url: str
    status: str                  # "done" | "exists" | "failed"
    height: Optional[int] = None
    vcodec: Optional[str] = None
    pages: Optional[int] = None
    notes: list = field(default_factory=list)
    pdf: Optional[str] = None
    video: Optional[str] = None  # downloaded video, if kept
    seconds: float = 0.0
    error: Optional[str] = None


def quality(fmt: dict) -> Optional[int]:
    """YouTube-style quality number ("720p"): the shorter side of the frame,
    so a portrait 720x1280 video counts as 720p."""
    w, h = fmt.get("width"), fmt.get("height")
    if w and h:
        return int(min(w, h))
    return int(h) if h else None


_CODEC_RANK = (("avc1", 3), ("h264", 3), ("vp09", 2), ("vp9", 2), ("av01", 1))


def choose_format(formats: list, target: int = TARGET_HEIGHT) -> Optional[dict]:
    """The format to download: exactly ``target`` if available, otherwise the
    closest quality above it, and only if there is none above, the closest
    below. Among equal qualities: video-only streams (the sound isn't needed),
    then H.264 (fastest to decode), then the highest bitrate."""
    video = [f for f in formats if f.get("vcodec") not in (None, "none") and quality(f)
             and f.get("format_note") != "storyboard" and not f.get("has_drm")]
    if not video:
        return None
    qualities = {quality(f) for f in video}
    above = [q for q in qualities if q > target]
    below = [q for q in qualities if q < target]
    q = target if target in qualities else (min(above) if above else max(below))

    def rank(f: dict) -> tuple:
        codec = str(f.get("vcodec") or "")
        return (f.get("acodec") in (None, "none"),
                next((r for prefix, r in _CODEC_RANK if codec.startswith(prefix)), 0),
                f.get("tbr") or f.get("vbr") or 0)

    return max((f for f in video if quality(f) == q), key=rank)


def download_video(url: str, folder: str, height: int = TARGET_HEIGHT,
                   cookies_from_browser: Optional[str] = None, progress: ProgressFn = None) -> dict:
    """Download the video stream (no audio) at ``height`` - or the closest
    quality above, or else below (see :func:`choose_format`).
    Returns ``{"path", "height", "vcodec", "format_id"}`` (height = quality, e.g. 720)."""
    from yt_dlp import YoutubeDL

    os.makedirs(folder, exist_ok=True)

    def hook(d: dict) -> None:
        if progress and d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            done = d.get("downloaded_bytes") or 0
            if total:
                progress(min(1.0, done / total), f"Downloading {done / 1e6:.0f}/{total / 1e6:.0f} MB")

    def selector(ctx: dict):
        f = choose_format(ctx.get("formats") or [], height)
        if f is not None:
            yield f

    opts = {**_ydl_base_opts(cookies_from_browser), "format": selector,
            "outtmpl": {"default": os.path.join(folder, "%(id)s.%(format_id)s.%(ext)s")},
            "progress_hooks": [hook], "concurrent_fragment_downloads": 4, "overwrites": False}
    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        req = (info.get("requested_downloads") or [info])[0]
        path = req.get("filepath") or req.get("_filename") or ydl.prepare_filename(info)
    return {"path": path, "height": quality(req) or quality(info),
            "vcodec": req.get("vcodec") or info.get("vcodec"), "format_id": req.get("format_id") or info.get("format_id")}


def series_folder(out_dir: str, series: str) -> Path:
    return Path(out_dir).expanduser() / safe_filename(series)


def process_chapter(ch: Chapter, name: str, out_dir: str, settings: Optional[Settings] = None,
                    keep_videos: bool = False, force: bool = False, height: int = TARGET_HEIGHT,
                    cookies_from_browser: Optional[str] = None, progress: ProgressFn = None,
                    downloader: Optional[Callable] = None) -> ChapterResult:
    """Download one chapter (or playlist video) and turn it into a PDF named
    after ``name`` in ``<out_dir>/<name>/``."""
    downloader = downloader or download_video
    t0 = time.time()
    folder = series_folder(out_dir, name)
    pdf = folder / ch.pdf_name(name)
    res = ChapterResult(chapter=ch.label, title=ch.video.title, url=ch.video.url, status="failed", pdf=str(pdf))
    if pdf.exists() and not force:
        res.status = "exists"
        return res
    sub = (lambda lo, hi: (lambda f, m="": progress(lo + (hi - lo) * f, m))) if progress else (lambda lo, hi: None)
    try:
        dl = downloader(ch.video.url, str(folder / "videos"), height=height,
                        cookies_from_browser=cookies_from_browser, progress=sub(0.0, 0.5))
        res.height, res.vcodec = dl.get("height"), dl.get("vcodec")
        if res.height and res.height != height:
            res.notes.append(f"no {height}p version: used {res.height}p")
        s = Settings(**(settings or default_settings()).to_dict())
        pipe = Pipeline(dl["path"], s)
        project = pipe.run(progress=sub(0.5, 0.95))
        summary = project.summary()
        res.pages = summary["pages"]
        if summary["possible_duplicates"]:
            res.notes.append(f"{summary['possible_duplicates']} possible duplicate page(s) kept")
        if summary["duplicates_merged"]:
            res.notes.append(f"{summary['duplicates_merged']} repeated page(s) merged")
        if summary["blank"]:
            res.notes.append(f"{summary['blank']} blank frame(s) skipped")
        if not project.included():
            raise RuntimeError("no pages detected")
        tmp = pdf.with_suffix(".part.pdf")
        pipe.export_pdf(project, str(tmp), progress=sub(0.95, 1.0))
        os.replace(tmp, pdf)
        pipe.save_project(project)
        if keep_videos:
            res.video = dl["path"]
        else:
            try:
                os.remove(dl["path"])
            except OSError:
                pass
        res.status = "done"
    except Exception as e:  # keep going with the other chapters
        res.error = f"{type(e).__name__}: {e}"
    res.seconds = round(time.time() - t0, 1)
    return res


def default_settings() -> Settings:
    """Video2Book defaults: every PDF page the same size (comic panels vary in
    shape; each is padded with its own edge colour to the common shape)."""
    return Settings(page_size="uniform", fill="auto", dpi=150)


def write_report(results: list, folder: Path, name: str, source: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "report.json"
    old = {}
    if path.exists():
        try:
            old = {r["chapter"]: r for r in json.load(open(path)).get("chapters", [])}
        except Exception:
            old = {}
    for r in results:
        if r.status != "exists" or r.chapter not in old:
            old[r.chapter] = asdict(r)
    with open(path, "w") as f:
        json.dump({"name": name, "source": source, "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "chapters": [old[k] for k in sorted(old)]}, f, indent=1)
    return path
