"""Video2Book - local web interface (Streamlit).

Start it with ``video2book-app`` (or ``streamlit run video2book/app.py``).
Everything runs on this computer; nothing is uploaded anywhere.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import altair as alt
import cv2
import numpy as np
import pandas as pd
import streamlit as st

from video2book import __version__
from video2book.crop import CropResult
from video2book.enhance import PRESET_LABELS, PRESETS
from video2book.export import PAGE_SIZE_LABELS, ocr_status, tesseract_languages
from video2book.project import Pipeline, Project, Settings, default_cache_root

st.set_page_config(page_title="Video2Book", page_icon="📖", layout="wide")

S = st.session_state
VIDEO_EXT = (".mp4", ".mov", ".m4v", ".mkv", ".avi", ".webm", ".3gp", ".mts", ".wmv")
GRID_COLS = 6
PER_PAGE = 36
# Reference palette (see video2book/plot.py)
C_SIGNAL, C_KEEP, C_DUP, C_MUTED, C_INK = "#2a78d6", "#1baf7a", "#eb6834", "#898781", "#52514e"
STATUS_LEGEND = {"keep": "Page", "duplicate": "Duplicate (hidden)", "blank": "Blank (hidden)",
                 "deleted": "Deleted"}


# -- helpers ------------------------------------------------------------------

def fmt_t(t: float) -> str:
    m, s = divmod(max(0.0, float(t)), 60)
    return f"{int(m):02d}:{s:04.1f}"


def parse_time(text: str):
    """'83.5', '1:23.5', '01:23' or '1:02:03' -> seconds (None if invalid)."""
    text = (text or "").strip().replace(",", ".")
    if not re.fullmatch(r"\d+(\.\d+)?(:\d+(\.\d+)?){0,2}", text):
        return None
    t = 0.0
    for part in text.split(":"):
        t = t * 60 + float(part)
    return t


def jpeg(img: np.ndarray, quality: int = 85, max_side: int = 0) -> bytes:
    if max_side and max(img.shape[:2]) > max_side:
        s = max_side / float(max(img.shape[:2]))
        img = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()


def show_image(data, caption=None) -> None:
    try:
        st.image(data, caption=caption, width="stretch")
    except TypeError:  # older Streamlit
        st.image(data, caption=caption, use_container_width=True)


def cache_root() -> str:
    return os.environ.get("VIDEO2BOOK_CACHE") or str(default_cache_root())


def list_videos(folder: str) -> list:
    try:
        p = Path(folder).expanduser()
        return sorted(str(f) for f in p.iterdir() if f.suffix.lower() in VIDEO_EXT and f.is_file())
    except Exception:
        return []


def default_folder() -> str:
    for cand in (Path.cwd() / "samples", Path.home() / "Videos", Path.home() / "Movies",
                 Path.home() / "Downloads", Path.home()):
        if cand.is_dir() and list_videos(str(cand)):
            return str(cand)
    return str(Path.home())


def output_path(project: Project, ext: str) -> str:
    video = Path(project.video)
    target = video.with_suffix(ext)
    if os.access(video.parent, os.W_OK):
        return str(target)
    out = Path(cache_root()) / "output"
    out.mkdir(parents=True, exist_ok=True)
    return str(out / target.name)


# -- state and actions -------------------------------------------------------------

def changed() -> None:
    """Persist edits; previous exports are now stale."""
    S.pipe.save_project(S.project)
    S.exported = {}


def act_status(pid: str, status: str) -> None:
    S.project.set_status(pid, status)
    changed()


def act_rotate(pid: str, deg: int) -> None:
    S.project.rotate(pid, deg)
    changed()


def act_move(pid: str, delta: int) -> None:
    S.project.move(pid, S.project.index(pid) + delta)
    changed()


def act_move_to(pid: str, key: str) -> None:
    """Move so the page gets output number S[key] (counting included pages)."""
    project = S.project
    target = int(S[key])
    included = [p for p in project.included() if p.id != pid]
    if target <= 1 or not included:
        idx = 0
    elif target > len(included):
        idx = len(project.pages)
    else:
        idx = project.index(included[target - 1].id)
    cur = project.index(pid)
    project.move(pid, idx if idx < cur else idx - 1 if idx > cur else idx)
    changed()


def act_use_frame(pid: str, pts: int) -> None:
    S.pipe.use_frame(S.project, S.project.page(pid), int(pts))
    changed()


def act_open_detail(pid: str) -> None:
    S.detail = pid


def act_setting(field: str, key: str, transform=None) -> None:
    v = S[key]
    setattr(S.project.settings, field, transform(v) if transform else v)
    changed()


def act_request(what: str, value=None) -> None:
    S[what] = value if value is not None else True


WIDGET_FIELDS = {  # widget key -> settings field
    "w_sensitivity": "sensitivity", "w_min_duration": "min_duration", "w_step": "step",
    "w_method": "method", "w_dedup": "dedup", "w_crop_mode": "crop_mode", "w_perspective": "perspective",
    "w_enhance": "enhance", "w_page_size": "page_size", "w_dpi": "dpi", "w_quality": "jpeg_quality",
}


def load_widgets(settings: Settings) -> None:
    for key, field in WIDGET_FIELDS.items():
        S[key] = getattr(settings, field)
    S.w_manual_thr = settings.threshold is not None
    S.w_threshold = float(settings.threshold or 1.0)
    S.w_expected = int(settings.expected_pages or 0)
    S.w_ocr = bool(settings.ocr_lang)
    S.w_ocr_lang = settings.ocr_lang or "eng"
    mc = settings.manual_crop or [0.05, 0.05, 0.95, 0.95]
    S.w_crop_l, S.w_crop_t, S.w_crop_r, S.w_crop_b = [round(v * 100, 1) for v in mc]


def detection_settings_from_widgets(base: Settings) -> Settings:
    d = base.to_dict()
    for key in ("w_sensitivity", "w_min_duration", "w_step", "w_method", "w_dedup"):
        if key in S:
            d[WIDGET_FIELDS[key]] = S[key]
    d["threshold"] = float(S.w_threshold) if S.get("w_manual_thr") else None
    return Settings.from_dict(d)


def progress_bar(label: str):
    bar = st.progress(0.0, text=label)

    def cb(frac: float, msg: str = "") -> None:
        bar.progress(float(min(1.0, max(0.0, frac))), text=msg or label)

    return bar, cb


def do_open(path: str) -> None:
    st.title("📖 Video2Book")
    st.subheader(f"Opening {os.path.basename(path)}")
    note = st.empty()
    note.info("The first time, the whole video is read once (a few seconds per minute of video). "
              "After that everything is cached.")
    bar, cb = progress_bar("Reading video...")
    try:
        pipe = Pipeline(path, Settings(), cache_root=cache_root())
    except Exception as e:  # unreadable / not a video
        bar.empty()
        S.open_error = f"Could not open {path}: {e}"
        return
    prev = pipe.load_project()
    if prev is not None:
        pipe.settings = prev.settings
        project = prev
        project.video = pipe.video  # same content, maybe opened from another folder
        S.notice = "Resumed your previous review of this video (edits kept). Use 'Re-detect pages' to start over."
    else:
        project = pipe.run(progress=cb)
        pipe.save_project(project)
        S.notice = None
    bar.empty()
    note.empty()
    S.pipe, S.project = pipe, project
    S.exported, S.grid_page, S.detail = {}, 0, None
    S.pop("open_error", None)
    load_widgets(project.settings)


def do_redetect() -> None:
    st.title("📖 Video2Book")
    st.subheader("Detecting pages...")
    bar, cb = progress_bar("Detecting pages...")
    pipe, old = S.pipe, S.project
    new_settings = detection_settings_from_widgets(old.settings)
    need_reread = (new_settings.step != old.settings.step)
    pipe.settings = new_settings
    if need_reread:
        st.info("Frame step changed: reading the video again.")
    S.project = pipe.run(progress=cb, keep_manual_from=old)
    pipe.save_project(S.project)
    bar.empty()
    S.exported, S.grid_page, S.detail = {}, 0, None
    S.notice = None
    load_widgets(S.project.settings)


# -- sidebar -----------------------------------------------------------------------

def sidebar() -> None:
    sb = st.sidebar
    sb.title("📖 Video2Book")
    sb.caption("Turn a video of book pages into a PDF. Runs entirely on this computer.")
    project = S.get("project")
    with sb.expander("1 · Open a video", expanded=project is None):
        st.text_input("Folder", key="w_folder", help="A folder on this computer that contains your video.")
        files = list_videos(S.w_folder)
        if files:
            choice = st.selectbox("Video", files, index=None, key="w_choice",
                                  format_func=lambda p: os.path.basename(p), placeholder="Choose a video...")
            st.button("Open this video", type="primary", disabled=choice is None,
                      on_click=act_request, args=("to_open", choice), width="stretch")
        else:
            st.caption("No videos in this folder.")
        up = st.file_uploader("...or drop a video here", type=[e[1:] for e in VIDEO_EXT], key="w_upload",
                              help="Large videos: better use the folder option above (no copy needed).")
        if up is not None:
            dest = Path(cache_root()) / "uploads" / up.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists() or dest.stat().st_size != up.size:
                with open(dest, "wb") as f:
                    f.write(up.getbuffer())
            st.button("Open uploaded video", type="primary", on_click=act_request, args=("to_open", str(dest)),
                      width="stretch")
    if project is None:
        return
    settings = project.settings

    with sb.expander("2 · Page detection", expanded=False):
        st.slider("Sensitivity", 0.0, 1.0, step=0.01, key="w_sensitivity",
                  help="Higher finds more pages (use it if pages are missing); lower merges more.")
        st.slider("Minimum still time (s)", 0.1, 3.0, step=0.05, key="w_min_duration",
                  help="How long a page must stay still to count as a page.")
        st.checkbox("Set the motion threshold by hand", key="w_manual_thr", on_change=_manual_thr_toggled)
        # Always drawn (disabled when unused): Streamlit forgets the value of hidden widgets.
        st.number_input("Motion threshold", min_value=0.01, max_value=100.0, step=0.1, key="w_threshold",
                        disabled=not S.w_manual_thr,
                        help="See the Motion graph tab: stretches below the dashed line are pages.")
        st.radio("Page image", ["median", "sharpest"], key="w_method", horizontal=True,
                 format_func=lambda m: {"median": "Median of still frames", "sharpest": "Sharpest frame"}[m],
                 help="The median removes video noise; 'sharpest' picks one single frame.")
        st.checkbox("Merge repeated pages", key="w_dedup")
        st.number_input("Analyse every Nth frame", min_value=1, max_value=10, step=1, key="w_step",
                        help="2-3 makes the first analysis faster on long, high frame-rate videos.")
        pending = detection_settings_from_widgets(settings)
        dirty = any(getattr(pending, f) != getattr(settings, f)
                    for f in ("sensitivity", "min_duration", "step", "method", "dedup", "threshold"))
        st.button("Re-detect pages", type="primary" if dirty else "secondary", width="stretch",
                  on_click=act_request, args=("redetect",),
                  help="Runs detection again with these settings. Your deletions/rotations are reset; "
                       "pages you added by hand are kept.")
        if dirty:
            st.caption("Settings changed - click 'Re-detect pages' to apply.")

    with sb.expander("3 · Look of the pages", expanded=False):
        st.radio("Enhancement", PRESETS, key="w_enhance", format_func=lambda p: PRESET_LABELS[p],
                 on_change=act_setting, args=("enhance", "w_enhance"))
        st.caption("Cropping is set in the **Crop** tab.")

    with sb.expander("4 · PDF options", expanded=False):
        st.radio("Page size", list(PAGE_SIZE_LABELS), key="w_page_size", horizontal=True,
                 format_func=lambda k: PAGE_SIZE_LABELS[k], on_change=act_setting, args=("page_size", "w_page_size"))
        st.slider("Resolution (DPI)", 72, 600, step=4, key="w_dpi", on_change=act_setting, args=("dpi", "w_dpi"),
                  help="Maximum image resolution on the page (images are never enlarged).")
        st.slider("JPEG quality", 40, 100, key="w_quality", on_change=act_setting, args=("jpeg_quality", "w_quality"))
        ok, why = ocr_status()
        if ok:
            langs = [l for l in tesseract_languages() if l != "osd"] or ["eng"]
            st.checkbox("Searchable PDF (OCR)", key="w_ocr", on_change=_ocr_changed)
            st.text_input("OCR language(s)", key="w_ocr_lang", on_change=_ocr_changed, disabled=not S.w_ocr,
                          help="Tesseract codes, e.g. eng, fra, deu, or several: fra+eng. "
                               "Installed: " + ", ".join(langs))
        else:
            st.checkbox("Searchable PDF (OCR)", value=False, disabled=True, help=why)
            st.caption(f"OCR unavailable: {why}")
    sb.caption(f"Video2Book {__version__} · cache: {cache_root()}")


def _manual_thr_toggled() -> None:
    if S.w_manual_thr and S.project.settings.threshold is None:
        S.w_threshold = max(0.01, round(float(S.project.threshold), 2))  # start from the automatic one


def _ocr_changed() -> None:
    S.project.settings.ocr_lang = S.w_ocr_lang.strip() if S.w_ocr and S.w_ocr_lang.strip() else None
    changed()


# -- main area ---------------------------------------------------------------------

def welcome() -> None:
    st.title("📖 Video2Book")
    st.markdown(
        "Turn a video that shows book pages one after another (a slideshow, or swiping through "
        "photos) into a clean PDF - one PDF page per book page, in order, without duplicates or "
        "blurry in-between frames.\n\n"
        "**How to use it**\n"
        "1. In the left panel, choose the folder with your video, pick it and click **Open this video** "
        "(or drop the file into the upload box).\n"
        "2. Wait for the analysis (progress is shown).\n"
        "3. Click **Create PDF**. Done - or first review the pages: delete, rotate, reorder, "
        "or pick a sharper frame.\n\n"
        "Nothing leaves your computer.")
    if S.get("open_error"):
        st.error(S.open_error)


def summary_and_export() -> None:
    pipe, project = S.pipe, S.project
    info = pipe.info
    st.title(f"📖 {Path(project.video).name}")
    st.caption(f"{info.describe()} · {project.video}")
    if S.get("notice"):
        st.info(S.notice)
    s = project.summary()
    look = sum(1 for p in project.included() if needs_look(p))
    cols = st.columns(5)
    cols[0].metric("Pages in PDF", s["pages"])
    cols[1].metric("Repeats merged", s["duplicates_merged"],
                   help="Pages shown twice in the video (e.g. swiped back). Hidden, not deleted - see the "
                        "'Hidden' filter.")
    cols[2].metric("Worth a look", look, help="Possible duplicates, very brief or moving pages.")
    cols[3].metric("Blank / deleted", s["blank"] + s["deleted"])
    cols[4].metric("Processing time", f"{project.timings.get('total', 0):.1f} s",
                   help="This run: analysis {:.1f}s{}, page images {:.1f}s. First full read of the video: {:.1f}s."
                   .format(project.timings.get("analysis", 0),
                           " (cached)" if project.timings.get("analysis_cached") else "",
                           project.timings.get("extract", 0), project.timings.get("first_analysis", 0)))

    c1, c2 = st.columns([1, 3], vertical_alignment="bottom")
    c1.number_input("Expected number of pages (optional)", min_value=0, max_value=100000, step=1,
                    help="If you know how many pages the book has. 0 = not set.",
                    key="w_expected", on_change=act_setting, args=("expected_pages", "w_expected", lambda v: v or None))
    chk = pipe.check_expected(project)
    with c2:
        if chk and chk["ok"]:
            st.success(chk["message"])
        elif chk:
            st.warning(chk["message"])
            sug = chk.get("suggested_sensitivity")
            if sug is not None and abs(sug - project.settings.sensitivity) > 1e-6:
                st.button(f"Use sensitivity {sug:.2f} and re-detect", on_click=_apply_suggestion, args=(sug,))

    e1, e2, e3 = st.columns([1, 1, 2], vertical_alignment="center")
    if e1.button("Create PDF", type="primary", width="stretch", disabled=not project.included()):
        out = output_path(project, ".pdf")
        bar, cb = progress_bar("Creating PDF...")
        try:
            S.exported = {"pdf": pipe.export_pdf(project, out, progress=cb)}
        except Exception as e:
            S.exported = {"error": str(e)}
        bar.empty()
    if e2.button("Save images as ZIP", width="stretch", disabled=not project.included()):
        out = output_path(project, "_pages.zip")
        bar, cb = progress_bar("Saving images...")
        S.exported = {"zip": pipe.export_zip(project, out, progress=cb)}
        bar.empty()
    ex = S.get("exported") or {}
    with e3:
        if "error" in ex:
            st.error(ex["error"])
        for kind, mime in (("pdf", "application/pdf"), ("zip", "application/zip")):
            if kind in ex and os.path.exists(ex[kind]):
                size = os.path.getsize(ex[kind]) / 1e6
                st.success(f"Saved {ex[kind]} ({size:.1f} MB)")
                with open(ex[kind], "rb") as f:
                    st.download_button(f"Download {kind.upper()}", f.read(), file_name=os.path.basename(ex[kind]),
                                       mime=mime, key=f"dl_{kind}")


def _apply_suggestion(sens: float) -> None:
    S.w_sensitivity = float(sens)
    S.redetect = True


def needs_look(p) -> bool:
    return p.included and any(f in p.flags for f in ("possible_duplicate", "short", "moving"))


def badges(project: Project, p) -> list:
    out = []
    ref = project.page_number(p.dup_of) if p.dup_of else None
    if p.status == "duplicate":
        out.append(f":material/repeat: repeat of p. {ref} (hidden)")
    elif p.status == "blank":
        out.append(":material/hide_image: blank frame (hidden)")
    elif p.status == "deleted":
        out.append(":material/delete: deleted")
    if p.included and "possible_duplicate" in p.flags:
        out.append(f":material/warning: looks like p. {ref}" if ref else ":material/warning: possible repeat")
    if p.included and "short" in p.flags:
        out.append(f":material/warning: brief ({p.t_end - p.t_start:.1f}s)")
    if p.included and "moving" in p.flags:
        out.append(":material/open_with: moving shot")
    if "picked" in p.flags and p.source != "manual":
        out.append(":material/edit: frame picked")
    if p.source == "manual":
        out.append(":material/add: added by you")
    return out


def thumb(p) -> bytes:
    s = S.project.settings
    key = (p.image_key, str(p.crop), p.rotation, s.crop_mode, str(s.manual_crop), s.perspective, s.enhance)
    cache = S.setdefault("thumbs", {})
    if key not in cache:
        cache[key] = jpeg(S.pipe.render_preview(S.project, p), 82)
    return cache[key]


def pages_tab() -> None:
    project = S.project
    top = st.columns([2, 1, 1], vertical_alignment="bottom")
    with top[0]:
        flt = st.segmented_control("Show", ["All", "Worth a look", "Hidden"], default="All", key="w_filter")
    top[1].button("Restore time order", on_click=_sort_time, width="stretch",
                  help="Put the pages back in the order they appear in the video.")
    with top[2].popover("Add page from time", icon=":material/add_photo_alternate:", width="stretch"):
        add_page_ui()

    if flt == "Worth a look":
        pages = [p for p in project.pages if needs_look(p)]
    elif flt == "Hidden":
        pages = [p for p in project.pages if not p.included]
    else:
        pages = list(project.pages)
    if not pages:
        st.info("Nothing to show here." if flt != "All" else "No pages detected - try a higher sensitivity.")
        return
    n_grid = max(1, (len(pages) + PER_PAGE - 1) // PER_PAGE)
    S.grid_page = min(S.get("grid_page", 0), n_grid - 1)
    if n_grid > 1:
        nav = st.columns([1, 3, 1], vertical_alignment="center")
        nav[0].button("◀ Previous", disabled=S.grid_page == 0, on_click=_grid, args=(-1,), width="stretch")
        nav[1].markdown(f"<div style='text-align:center'>Showing {S.grid_page * PER_PAGE + 1}-"
                        f"{min(len(pages), (S.grid_page + 1) * PER_PAGE)} of {len(pages)}</div>",
                        unsafe_allow_html=True)
        nav[2].button("Next ▶", disabled=S.grid_page >= n_grid - 1, on_click=_grid, args=(1,), width="stretch")
    chunk = pages[S.grid_page * PER_PAGE:(S.grid_page + 1) * PER_PAGE]
    for r in range(0, len(chunk), GRID_COLS):
        cols = st.columns(GRID_COLS)
        for c, p in zip(cols, chunk[r:r + GRID_COLS]):
            with c:
                page_card(project, p)


def page_card(project: Project, p) -> None:
    num = project.page_number(p.id)
    with st.container(border=True):
        show_image(thumb(p))
        title = f"**p. {num}**" if num else "~~hidden~~"
        st.markdown(f"{title} · `{fmt_t(p.t)}`")
        for b in badges(project, p):
            st.caption(b)
        row = st.columns(5, gap="small")
        icon_button(row[0], "zoom_in", f"o_{p.id}", "Open large: rotate, move, or pick a better frame",
                    act_open_detail, (p.id,))
        icon_button(row[1], "rotate_right", f"r_{p.id}", "Rotate 90° clockwise", act_rotate, (p.id, 90))
        icon_button(row[2], "arrow_back", f"m_{p.id}", "Move one place earlier", act_move, (p.id, -1))
        icon_button(row[3], "arrow_forward", f"n_{p.id}", "Move one place later", act_move, (p.id, 1))
        if p.included:
            icon_button(row[4], "delete", f"d_{p.id}", "Delete this page", act_status, (p.id, "deleted"))
        else:
            icon_button(row[4], "undo", f"k_{p.id}", "Restore this page", act_status, (p.id, "keep"))


def icon_button(where, icon: str, key: str, help_text: str, cb, args) -> None:
    """Compact icon-only button (Material icons ship with Streamlit, no emoji font needed)."""
    try:
        where.button("", icon=f":material/{icon}:", key=key, help=help_text, on_click=cb, args=args,
                     width="stretch")
    except Exception:  # older Streamlit without icons
        where.button(icon.split("_")[0][:3], key=key, help=help_text, on_click=cb, args=args)


def _grid(delta: int) -> None:
    S.grid_page = max(0, S.get("grid_page", 0) + delta)


def _sort_time() -> None:
    S.project.sort_by_time()
    changed()


def add_page_ui() -> None:
    pipe = S.pipe
    dur = pipe.info.duration
    st.markdown("Recover a page the detector missed: enter the time where it is visible.")
    txt = st.text_input("Time (seconds or mm:ss)", key="w_add_time", placeholder="e.g. 83.5 or 1:23.5")
    t = parse_time(txt)
    if txt and t is None:
        st.error("Use a number of seconds (83.5) or minutes:seconds (1:23.5).")
        return
    if t is None:
        return
    t = min(max(0.0, t), max(0.0, dur - 0.05))
    a = pipe.analysis()
    i = a.nearest(t)
    pts = int(a.pts[i])
    img = pipe.frames([pts])[pts]
    show_image(jpeg(img, 80, 480), caption=f"Frame at {fmt_t(float(a.times[i]))}")
    st.button("Insert this page", type="primary", on_click=_add_page, args=(float(a.times[i]),))


def _add_page(t: float) -> None:
    S.pipe.add_page_at(S.project, t)
    changed()
    S.notice = f"Added a page from {fmt_t(t)}."


@st.dialog("Page", width="large", on_dismiss="rerun")
def page_dialog(pid: str) -> None:
    pipe, project = S.pipe, S.project
    try:
        p = project.page(pid)
    except KeyError:
        st.write("This page no longer exists.")
        return
    num = project.page_number(pid)
    left, right = st.columns([3, 2])
    with left:
        show_image(jpeg(pipe.render(project, p), 88, 1400))
    with right:
        st.markdown(f"### {'Page %d' % num if num else 'Hidden page'}")
        st.write(f"Shown in the video at **{fmt_t(p.t)}** (still from {fmt_t(p.t_start)} to {fmt_t(p.t_end)}).")
        for b in badges(project, p):
            st.write(b)
        if p.dup_of:
            try:
                other = project.page(p.dup_of)
                sc = p.dup_score or {}
                st.caption(f"Compared with this page (similarity {sc.get('ssim', 0):.2f}):")
                show_image(thumb(other))
            except KeyError:
                pass
        r = st.columns(2)
        r[0].button("Rotate left", icon=":material/rotate_left:", on_click=act_rotate, args=(pid, -90),
                    width="stretch", key="dl_rl")
        r[1].button("Rotate right", icon=":material/rotate_right:", on_click=act_rotate, args=(pid, 90),
                    width="stretch", key="dl_rr")
        r = st.columns(2)
        r[0].button("Earlier", icon=":material/arrow_back:", on_click=act_move, args=(pid, -1), width="stretch",
                    key="dl_me")
        r[1].button("Later", icon=":material/arrow_forward:", on_click=act_move, args=(pid, 1), width="stretch",
                    key="dl_ml")
        if p.included:
            st.button("Delete this page", icon=":material/delete:", on_click=act_status, args=(pid, "deleted"),
                      width="stretch", key="dl_del")
            n = len(project.included())
            r = st.columns([2, 1], vertical_alignment="bottom")
            r[0].number_input("Move to page number", 1, max(1, n), value=num or 1, key="dl_pos")
            r[1].button("Move", on_click=act_move_to, args=(pid, "dl_pos"), width="stretch", key="dl_mv")
        else:
            st.button("Restore this page", icon=":material/undo:", on_click=act_status, args=(pid, "keep"),
                      width="stretch", type="primary", key="dl_res")
        if st.button("Done", type="primary", width="stretch", key="dl_done"):
            S.detail = None
            st.rerun()

    st.divider()
    st.markdown("#### Pick a better frame")
    st.caption("Frames around this page. Click **Use** under the sharpest / straightest one.")
    nb_key = ("nb", p.id, round(p.t_start, 3), round(p.t_end, 3))
    if S.get("nb_cache_key") != nb_key:
        times = pipe.neighbor_times(p, 10)
        frames = pipe.frames([pts for pts, _ in times])
        S.nb_cache = [(pts, t, jpeg(frames[pts], 80, 360)) for pts, t in times]
        S.nb_cache_key = nb_key
    cols = st.columns(5)
    for k, (pts, t, data) in enumerate(S.nb_cache):
        with cols[k % 5]:
            show_image(data, caption=fmt_t(t))
            st.button("Use", key=f"use_{pts}", on_click=act_use_frame, args=(pid, pts), width="stretch",
                      type="primary" if pts in (p.pts or []) and len(p.pts) == 1 else "secondary")
    a = pipe.analysis()
    lo = max(0.0, p.t_start - 1.0)
    hi = min(float(a.times[-1]), p.t_end + 1.0)
    if hi > lo:
        t = st.slider("...or scrub to any moment", lo, hi, value=float(min(max(p.t, lo), hi)),
                      step=1.0 / max(1.0, a.sample_rate), format="%.2f s", key=f"scrub_{pid}")
        i = a.nearest(t)
        pts = int(a.pts[i])
        img = pipe.frames([pts])[pts]
        c = st.columns([2, 1], vertical_alignment="center")
        with c[0]:
            show_image(jpeg(img, 80, 640), caption=f"Frame at {fmt_t(float(a.times[i]))}")
        c[1].button("Use this frame", key=f"use_scrub_{pid}", on_click=act_use_frame, args=(pid, pts),
                    type="primary", width="stretch")


def motion_tab() -> None:
    pipe, project = S.pipe, S.project
    a = pipe.analysis()
    res = pipe.segments()
    sm, t = res.smoothed, a.times
    n = len(sm)
    b = max(1, int(np.ceil(n / 2500)))
    m = (n // b) * b
    if m == 0:
        st.info("Video too short to plot.")
        return
    df = pd.DataFrame({"time": t[:m].reshape(-1, b)[:, 0],
                       "motion": np.maximum(sm[:m].reshape(-1, b).max(axis=1), 1e-3)})
    segs = pd.DataFrame([{
        "start": p.t_start, "end": p.t_end, "status": STATUS_LEGEND.get(p.status, p.status),
        "page": str(project.page_number(p.id) or "-"),
        "from": fmt_t(p.t_start), "to": fmt_t(p.t_end),
    } for p in project.pages if p.source == "auto"])
    domain = list(STATUS_LEGEND.values())
    rng = [C_KEEP, C_DUP, C_MUTED, "#c3c2b7"]
    x = alt.X("time:Q", title="Time (s)", scale=alt.Scale(domain=[float(t[0]), float(t[-1])], nice=False))
    layers = []
    if len(segs):
        layers.append(alt.Chart(segs).mark_rect(opacity=0.22).encode(
            x="start:Q", x2="end:Q",
            color=alt.Color("status:N", scale=alt.Scale(domain=domain, range=rng),
                            legend=alt.Legend(title=None, orient="top")),
            tooltip=[alt.Tooltip("page:N", title="Page"), alt.Tooltip("status:N", title="Status"),
                     alt.Tooltip("from:N", title="From"), alt.Tooltip("to:N", title="To")]))
    line = alt.Chart(df).mark_line(color=C_SIGNAL, strokeWidth=1.5).encode(
        x=x, y=alt.Y("motion:Q", title="Motion (log scale)", scale=alt.Scale(type="log")))
    layers.append(line)
    thr = project.threshold
    layers.append(alt.Chart(pd.DataFrame({"y": [thr], "label": [f"threshold {thr:.2f}"]}))
                  .mark_rule(strokeDash=[4, 3], color=C_INK).encode(y="y:Q", tooltip=["label:N"]))
    hover = alt.selection_point(nearest=True, on="pointerover", fields=["time"], empty=False)
    layers.append(alt.Chart(df).mark_point(opacity=0, size=60).encode(
        x="time:Q", y="motion:Q",
        tooltip=[alt.Tooltip("time:Q", title="Time (s)", format=".2f"),
                 alt.Tooltip("motion:Q", title="Motion", format=".3f")]).add_params(hover))
    layers.append(alt.Chart(df).mark_rule(color=C_MUTED).encode(x="time:Q").transform_filter(hover))
    chart = alt.layer(*layers).properties(height=340).interactive(bind_y=False)
    try:
        st.altair_chart(chart, width="stretch")
    except TypeError:
        st.altair_chart(chart, use_container_width=True)
    st.caption(
        f"Blue line: how much the picture changes from one frame to the next. Shaded spans: the still "
        f"stretches detected as pages (hover for details; drag to pan, scroll to zoom). Dashed line: the "
        f"threshold ({thr:.2f}), set automatically between the typical still level ({project.base:.2f}) and "
        f"the typical transition peak ({project.peak:.1f}). Pages missing? Raise the sensitivity. "
        f"Pages split in two? Lower it.")


def crop_tab() -> None:
    pipe, project = S.pipe, S.project
    s = project.settings
    c = st.columns([1, 2])
    with c[0]:
        st.radio("Cropping", ["auto", "manual", "none"], key="w_crop_mode", on_change=act_setting,
                 args=("crop_mode", "w_crop_mode"),
                 format_func=lambda m: {"auto": "Automatic (find the page)", "manual": "Same rectangle for every page",
                                        "none": "No cropping"}[m])
        st.checkbox("Straighten photographed pages (perspective)", key="w_perspective", on_change=act_setting,
                    args=("perspective", "w_perspective"),
                    help="Uses the 4 corners of the paper when they are found. Off by default.")
        manual = s.crop_mode == "manual"
        # Always drawn (disabled unless manual): Streamlit forgets the value of hidden widgets.
        st.caption("Your rectangle, in % of the video frame" + ("" if manual else
                   " (choose 'Same rectangle for every page' to use it)") + ":")
        st.slider("Left edge %", 0.0, 100.0, step=0.5, key="w_crop_l", on_change=_manual_crop, disabled=not manual)
        st.slider("Right edge %", 0.0, 100.0, step=0.5, key="w_crop_r", on_change=_manual_crop, disabled=not manual)
        st.slider("Top edge %", 0.0, 100.0, step=0.5, key="w_crop_t", on_change=_manual_crop, disabled=not manual)
        st.slider("Bottom edge %", 0.0, 100.0, step=0.5, key="w_crop_b", on_change=_manual_crop,
                  disabled=not manual)
        if manual and s.manual_crop is None:
            _manual_crop()
        st.button("Start from the automatic crop of this page", on_click=_crop_from_auto, disabled=not manual)
        pages = project.included() or project.pages
        if not pages:
            return
        k = st.slider("Preview page", 1, len(pages), value=1, key="w_crop_preview") if len(pages) > 1 else 1
    p = pages[k - 1]
    S.crop_preview_pid = p.id
    img = pipe.representative(p.image_key)
    vis = img.copy()
    lw = max(2, img.shape[1] // 250)
    if p.crop:
        cr = CropResult.from_dict(p.crop)
        x0, y0, x1, y1 = cr.box
        cv2.rectangle(vis, (x0, y0), (x1 - 1, y1 - 1), (27, 175, 122), lw)  # automatic: aqua
        if cr.quad is not None:
            cv2.polylines(vis, [cr.quad.astype(np.int32)], True, (27, 175, 122), max(1, lw // 2))
    if s.crop_mode == "manual" and s.manual_crop:
        h, w = img.shape[:2]
        l_, t_, r_, b_ = s.manual_crop
        cv2.rectangle(vis, (int(l_ * w), int(t_ * h)), (int(r_ * w) - 1, int(b_ * h) - 1), (235, 104, 52), lw)
    with c[1]:
        cc = st.columns(2)
        with cc[0]:
            show_image(jpeg(vis, 85, 900), caption="Video frame - green: automatic page area" +
                       (", orange: your rectangle" if s.crop_mode == "manual" else ""))
        with cc[1]:
            show_image(jpeg(pipe.render(project, p), 85, 900), caption="Result in the PDF")


def _manual_crop() -> None:
    l_, r_ = sorted((S.w_crop_l, S.w_crop_r))
    t_, b_ = sorted((S.w_crop_t, S.w_crop_b))
    if r_ - l_ < 2:
        r_ = min(100.0, l_ + 2)
    if b_ - t_ < 2:
        b_ = min(100.0, t_ + 2)
    S.project.settings.manual_crop = [l_ / 100, t_ / 100, r_ / 100, b_ / 100]
    changed()


def _crop_from_auto() -> None:
    p = S.project.page(S.crop_preview_pid) if S.get("crop_preview_pid") else None
    if not p or not p.crop:
        return
    x0, y0, x1, y1 = p.crop["box"]
    w, h = S.pipe.info.width, S.pipe.info.height
    S.w_crop_l, S.w_crop_t, S.w_crop_r, S.w_crop_b = (round(x0 / w * 100, 1), round(y0 / h * 100, 1),
                                                      round(x1 / w * 100, 1), round(y1 / h * 100, 1))
    _manual_crop()


def help_tab() -> None:
    st.markdown("""
**Workflow.** Open a video → the pages are detected automatically → click **Create PDF**.
Reviewing is optional but recommended for long books:

* **Worth a look** shows pages that might be a repeat of another page, that were on screen very
  briefly, or that were moving (zoom/pan effect).
* **Hidden** shows repeats that were merged automatically (a page shown twice, e.g. after swiping
  back), blank frames and pages you deleted. Click :material/undo: to bring one back.
* :material/zoom_in: opens a page: rotate, move it to another position, or pick a sharper frame from the
  neighbouring frames (or scrub to any moment).
* **Add page from time** recovers a page the detector missed: find the time in the video
  (or on the Motion graph) and insert it.
* **Expected number of pages**: if you know how many pages the book has, type it in. If the count
  differs you'll get a suggested sensitivity.

**Detection settings** (left panel → *Page detection*): *Sensitivity* moves the threshold on the
Motion graph. Missing pages → raise it; one page split in two → lower it. Changes apply when you
click **Re-detect pages** (the video is not read again - that's cached).

**Privacy.** Everything runs on this computer. Cached analysis and page images are stored in the
folder shown at the bottom of the left panel; delete it any time to free space.
""")


# -- page ----------------------------------------------------------------------------

def main() -> None:
    S.setdefault("w_folder", default_folder())
    # Long actions run before any widget is drawn (they may reset widget values).
    if S.get("to_open"):
        path = S.pop("to_open")
        do_open(path)
        st.rerun()
    if S.get("redetect") and S.get("project") is not None:
        S.redetect = False
        do_redetect()
        st.rerun()
    sidebar()
    project = S.get("project")
    if project is None:
        welcome()
        return
    summary_and_export()
    tab_pages, tab_motion, tab_crop, tab_help = st.tabs(["Pages", "Motion graph", "Crop", "Help"])
    with tab_pages:
        pages_tab()
    with tab_motion:
        motion_tab()
    with tab_crop:
        crop_tab()
    with tab_help:
        help_tab()
    if S.get("detail"):
        page_dialog(S.detail)


main()
