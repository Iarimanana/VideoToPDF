"""'YouTube series' page of the Video2Book web app: channel + series title ->
one PDF per chapter (see :mod:`video2book.youtube`)."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st

from video2book.enhance import PRESET_LABELS, PRESETS
from video2book.export import PAGE_SIZE_LABELS
from video2book.project import Settings
from video2book import youtube as yt

S = st.session_state
BROWSERS = ["(none)", "firefox", "chrome", "chromium", "edge", "brave", "opera", "vivaldi", "safari"]


def _find() -> None:
    """Search the channel (runs inside the script, shows a spinner)."""
    S.yt_results = None
    S.yt_error = None
    channel, series = S.yt_channel.strip(), S.yt_series.strip()
    if not channel or not series:
        S.yt_error = "Enter a channel and a series title."
        return
    cookies = None if S.yt_cookies == "(none)" else S.yt_cookies
    try:
        with st.spinner("Reading the channel's videos..."):
            if S.yt_mode == "Latest":
                chapters = yt.latest_chapters(channel, series, int(S.yt_latest), cookies)
            else:
                videos = yt.list_channel_videos(channel, S.yt_refresh, cookies)
                spec = S.yt_range if S.yt_mode == "Range" else None
                chapters = yt.select_chapters(yt.find_chapters(videos, series), spec)
    except ValueError as e:
        S.yt_error = str(e)
        return
    except Exception as e:
        S.yt_error = f"Could not read the channel: {e}"
        if "Sign in" in str(e) or "bot" in str(e):
            S.yt_error += "\n\nTip: under Options, choose the browser where you are logged in to YouTube."
        return
    S.yt_found = {"series": series, "channel": channel, "chapters": chapters}
    if not chapters:
        S.yt_error = f"No videos titled like '{series} (Chapter N)' found."


def _settings() -> Settings:
    return Settings(sensitivity=float(S.yt_sensitivity), enhance=S.yt_enhance, page_size=S.yt_page_size,
                    dpi=150)


def _convert(chapters: list, series: str, channel: str) -> None:
    out = S.yt_out
    cookies = None if S.yt_cookies == "(none)" else S.yt_cookies
    results = []
    overall = st.progress(0.0, text="Starting...")
    log = st.status(f"Processing {len(chapters)} chapter(s)...", expanded=True)
    for i, c in enumerate(chapters):
        base = i / len(chapters)

        def progress(f: float, msg: str = "", base=base, c=c) -> None:
            overall.progress(min(1.0, base + f / len(chapters)),
                             text=f"Chapter {c.label} ({i + 1}/{len(chapters)}): {msg}")

        r = yt.process_chapter(c, series, out, _settings(), keep_videos=S.yt_keep, force=S.yt_force,
                               height=int(S.yt_height), cookies_from_browser=cookies, progress=progress)
        results.append(r)
        if r.status == "done":
            log.write(f"✅ Chapter {c.label}: {r.pages} pages ({r.height}p, {r.seconds:.0f}s)"
                      + "".join(f" · {n}" for n in r.notes))
        elif r.status == "exists":
            log.write(f"⏭️ Chapter {c.label}: PDF already there")
        else:
            log.write(f"❌ Chapter {c.label}: {r.error}")
    overall.empty()
    failed = sum(r.status == "failed" for r in results)
    log.update(label=f"Finished: {sum(r.status == 'done' for r in results)} made, "
                     f"{sum(r.status == 'exists' for r in results)} already there, {failed} failed",
               state="error" if failed else "complete", expanded=bool(failed))
    yt.write_report(results, yt.series_folder(out, series), series, channel)
    S.yt_results = results


def _review(path: str) -> None:
    S.to_open = path
    S.mode = "Single video"


def render() -> None:
    S.setdefault("yt_channel", "")
    S.setdefault("yt_series", "")
    S.setdefault("yt_mode", "Latest")
    S.setdefault("yt_latest", 5)
    S.setdefault("yt_range", "")
    S.setdefault("yt_out", str(Path.home() / "Video2Book"))
    S.setdefault("yt_keep", False)
    S.setdefault("yt_force", False)
    S.setdefault("yt_refresh", False)
    S.setdefault("yt_height", 720)
    S.setdefault("yt_cookies", "(none)")
    S.setdefault("yt_sensitivity", 0.5)
    S.setdefault("yt_enhance", "original")
    S.setdefault("yt_page_size", "fit")

    st.title("📺 YouTube series → PDFs")
    st.caption("Finds the videos titled like “Series (Chapter N)” on a channel, downloads them at 720p "
               "(or the closest lower quality) and makes one PDF per chapter. Chapters already converted "
               "are skipped.")
    c = st.columns([3, 2])
    c[0].text_input("YouTube channel", key="yt_channel", placeholder="https://www.youtube.com/@ChannelName")
    c[1].text_input("Series title (as written in the video titles)", key="yt_series",
                    placeholder="Miss Forensics")
    c = st.columns([2, 1, 2], vertical_alignment="bottom")
    c[0].radio("Chapters", ["Latest", "Range", "All"], key="yt_mode", horizontal=True,
               format_func=lambda m: {"Latest": "Latest N", "Range": "Range / list", "All": "All"}[m])
    if S.yt_mode == "Latest":
        c[1].number_input("N", 1, 500, step=1, key="yt_latest")
    elif S.yt_mode == "Range":
        c[1].text_input("Chapters", key="yt_range", placeholder="140-145 or 3,5,9-11")
    c[2].text_input("Save the PDFs in", key="yt_out", help="A sub-folder named after the series is created.")
    with st.expander("Options"):
        o = st.columns(3)
        o[0].checkbox("Keep the downloaded videos", key="yt_keep",
                      help="Needed if you want to review / fix pages in Video2Book afterwards.")
        o[0].checkbox("Redo chapters that already have a PDF", key="yt_force")
        o[0].checkbox("Re-read the channel (ignore the list cached in the last hours)", key="yt_refresh")
        o[1].number_input("Video height (p)", 144, 2160, step=1, key="yt_height",
                          help="720 by default; if a video has no such version, the closest lower one is used.")
        o[1].selectbox("Use my YouTube login from", BROWSERS, key="yt_cookies",
                       help="Only needed if YouTube asks to sign in / confirm you're not a bot.")
        o[2].slider("Detection sensitivity", 0.0, 1.0, step=0.01, key="yt_sensitivity")
        o[2].selectbox("Page look", PRESETS, key="yt_enhance", format_func=lambda p: PRESET_LABELS[p])
        o[2].selectbox("PDF page size", list(PAGE_SIZE_LABELS), key="yt_page_size",
                       format_func=lambda k: PAGE_SIZE_LABELS[k] + (" (each page = its panel)" if k == "fit" else ""))

    if st.button("Find chapters", type="primary"):
        _find()
    if S.get("yt_error"):
        st.error(S.yt_error)
    found = S.get("yt_found")
    if not found or not found["chapters"]:
        return
    series, chapters = found["series"], found["chapters"]
    folder = yt.series_folder(S.yt_out, series)
    rows = [{"Chapter": ch.label, "Title": ch.video.title, "Video": ch.video.url,
             "PDF": "✔ done" if (folder / ch.pdf_name(series)).exists() else "",
             "Other uploads": len(ch.alternates) or ""} for ch in chapters]
    st.subheader(f"{len(chapters)} chapter(s) of {series}")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                 column_config={"Video": st.column_config.LinkColumn("Video", display_text="open")})
    todo = [ch for ch in chapters if S.yt_force or not (folder / ch.pdf_name(series)).exists()]
    label = f"Download & make {len(todo)} PDF(s)" if todo else "All PDFs already made"
    if st.button(label, type="primary", disabled=not todo):
        _convert(todo, series, found["channel"])
    results = S.get("yt_results")
    if results:
        st.subheader("Results")
        st.caption(f"Folder: {folder}")
        for r in results:
            cols = st.columns([1, 3, 2, 2], vertical_alignment="center")
            cols[0].markdown(f"**Chapter {r.chapter}**")
            if r.status == "failed":
                cols[1].error(r.error or "failed")
                continue
            cols[1].write(f"{r.pages} pages · {r.height}p" + "".join(f" · {n}" for n in r.notes)
                          if r.status == "done" else "already there")
            if r.pdf and os.path.exists(r.pdf):
                with open(r.pdf, "rb") as f:
                    cols[2].download_button("Download PDF", f.read(), file_name=os.path.basename(r.pdf),
                                            mime="application/pdf", key=f"yt_dl_{r.chapter}")
            if r.video and os.path.exists(r.video):
                cols[3].button("Review in Video2Book", key=f"yt_rev_{r.chapter}", on_click=_review,
                               args=(r.video,))
