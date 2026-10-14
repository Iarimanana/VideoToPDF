"""YouTube series tool, offline: titles, selection and the whole flow with a
fake downloader (it writes synthetic videos instead of downloading)."""

import os
import shutil

import pikepdf
import pytest

import video2book.youtube as yt
from video2book.youtube import (ChannelVideo, channel_videos_url, chapter_label, find_chapters,
                                latest_chapters, parse_chapter, parse_selection, process_chapter,
                                select_chapters)

S = "Miss Forensics"


@pytest.mark.parametrize("title,expected", [
    ("Miss Forensics (Chapter 142)", (142, 142)),
    ("Miss Forensics (Chapter 7) | English", (7, 7)),
    ("MISS FORENSICS - Ch. 12", (12, 12)),
    ("【Miss Forensics】Chapter 3", (3, 3)),
    ("Miss Forensics: chapter 12.5", (12.5, 12.5)),
    ("Miss Forensics (Chapter 10-11)", (10, 11)),
    ("Miss Forensics (Chapter 10 & 11)", (10, 11)),
    ("[Manhwa] Miss Forensics (Chapter 42) - she found the killer", (42, 42)),
    ("Miss  Forensics(Chapter 5)", (5, 5)),
])
def test_parse_chapter_matches(title, expected):
    assert parse_chapter(title, S) == expected


@pytest.mark.parametrize("title", [
    "Miss Forensics 2 (Chapter 5)",       # another series
    "Mister Forensics (Chapter 5)",
    "Miss Forensics - trailer",
    "Miss Forensics Q&A with the author",
    "Doctor Rebirth (Chapter 5)",
    "Miss Forensicsss (Chapter 5)",
])
def test_parse_chapter_rejects(title):
    assert parse_chapter(title, S) is None


def test_parse_chapter_accents_and_case():
    assert parse_chapter("L'Été Meurtrier (Chapitre 3)", "L'été meurtrier") is None  # 'chapitre' not a keyword
    assert parse_chapter("l'ÉTÉ meurtrier (chapter 3)", "L'été meurtrier") == (3, 3)


def test_channel_url_forms():
    want = "https://www.youtube.com/@RuiNemesys/videos"
    for c in ("@RuiNemesys", "https://www.youtube.com/@RuiNemesys", "https://www.youtube.com/@RuiNemesys/",
              "https://www.youtube.com/@RuiNemesys/videos", "https://www.youtube.com/@RuiNemesys/featured?x=1"):
        assert channel_videos_url(c) == want


def test_labels_and_selection():
    assert chapter_label(7, 7) == "007"
    assert chapter_label(12.5, 12.5) == "012.5"
    assert chapter_label(10, 11) == "010-011"
    assert parse_selection("140-145") == [(140, 145)]
    assert parse_selection("3, 5,9-11") == [(3, 3), (5, 5), (9, 11)]
    with pytest.raises(ValueError):
        parse_selection("abc")


def videos(titles):
    return [ChannelVideo(id=f"id{i}", title=t, url=f"https://www.youtube.com/watch?v=id{i}", index=i)
            for i, t in enumerate(titles)]


CHANNEL = videos([  # newest first, like the Videos tab
    "Miss Forensics (Chapter 12)",
    "Doctor Rebirth (Chapter 99)",
    "Miss Forensics (Chapter 11)",
    "Miss Forensics (Chapter 10) [reupload]",
    "Miss Forensics 2 (Chapter 1)",
    "Miss Forensics (Chapter 9)",
    "Miss Forensics (Chapter 10)",
    "Miss Forensics (Chapter 8)",
])


def test_find_and_select():
    chs = find_chapters(CHANNEL, S)
    assert [c.label for c in chs] == ["008", "009", "010", "011", "012"]
    ten = chs[2]
    assert ten.video.id == "id3" and [a.id for a in ten.alternates] == ["id6"]  # newest upload wins
    assert [c.label for c in select_chapters(chs, "9-11")] == ["009", "010", "011"]
    assert [c.label for c in select_chapters(chs, latest=2)] == ["011", "012"]


def test_latest_stops_scanning_early():
    pulled = []

    def gen():
        for v in CHANNEL + videos(["Other video"] * 500):
            pulled.append(v)
            yield v
    got = latest_chapters("@x", S, 3, videos=gen(), extra=5)
    assert [c.label for c in got] == ["010", "011", "012"]
    assert len(pulled) < 20


# -- end to end with a fake downloader ----------------------------------------------

@pytest.fixture
def fake_downloader(video_factory, tmp_path):
    path, cfg, truth = video_factory("base")
    calls = []

    def download(url, folder, height=720, cookies_from_browser=None, progress=None):
        os.makedirs(folder, exist_ok=True)
        dst = os.path.join(folder, url.rsplit("=", 1)[-1] + ".mp4")
        shutil.copy(path, dst)
        calls.append(url)
        if progress:
            progress(1.0, "done")
        return {"path": dst, "height": 480 if "id2" in url else 720, "vcodec": "avc1", "format_id": "x"}
    download.calls = calls
    download.n_pages = cfg.n_pages
    return download


def test_process_chapter_makes_pdf_and_resumes(fake_downloader, tmp_path, monkeypatch, cache_dir):
    monkeypatch.setenv("VIDEO2BOOK_CACHE", cache_dir)
    ch = find_chapters(CHANNEL, S)[-1]
    r = process_chapter(ch, S, str(tmp_path), downloader=fake_downloader)
    assert r.status == "done", r.error
    assert r.pages == fake_downloader.n_pages
    pdf = tmp_path / "Miss Forensics" / "Miss Forensics - Chapter 012.pdf"
    assert pdf.exists() and str(pdf) == r.pdf
    with pikepdf.open(pdf) as doc:
        assert len(doc.pages) == r.pages
    assert not any((tmp_path / "Miss Forensics" / "videos").iterdir())  # video removed
    again = process_chapter(ch, S, str(tmp_path), downloader=fake_downloader)
    assert again.status == "exists" and len(fake_downloader.calls) == 1


def test_process_chapter_notes_lower_resolution(fake_downloader, tmp_path, monkeypatch, cache_dir):
    monkeypatch.setenv("VIDEO2BOOK_CACHE", cache_dir)
    ch = [c for c in find_chapters(CHANNEL, S) if c.video.id == "id2"][0]
    r = process_chapter(ch, S, str(tmp_path), downloader=fake_downloader, keep_videos=True)
    assert r.status == "done" and any("480p" in n for n in r.notes)
    assert any((tmp_path / "Miss Forensics" / "videos").iterdir())  # kept


def test_process_chapter_failure_is_reported(tmp_path):
    def broken(url, folder, **kw):
        raise RuntimeError("HTTP Error 403")
    r = process_chapter(find_chapters(CHANNEL, S)[0], S, str(tmp_path), downloader=broken)
    assert r.status == "failed" and "403" in r.error


def test_cli_list_and_convert(fake_downloader, tmp_path, monkeypatch, cache_dir, capsys):
    from video2book import yt_cli
    monkeypatch.setenv("VIDEO2BOOK_CACHE", cache_dir)
    monkeypatch.setattr(yt, "iter_channel_videos", lambda channel, cookies=None: iter(CHANNEL))
    monkeypatch.setattr(yt, "download_video", fake_downloader)
    assert yt_cli.main(["@RuiNemesys", S, "--list", "--refresh"]) == 0
    out = capsys.readouterr().out
    assert "5 chapter(s)" in out and "Chapter 010" in out
    assert yt_cli.main(["@RuiNemesys", S, "--latest", "2", "-o", str(tmp_path)]) == 0
    folder = tmp_path / "Miss Forensics"
    assert sorted(p.name for p in folder.glob("*.pdf")) == ["Miss Forensics - Chapter 011.pdf",
                                                            "Miss Forensics - Chapter 012.pdf"]
    assert (folder / "report.json").exists()


# -- which quality is downloaded -----------------------------------------------------

from video2book.youtube import choose_format, quality  # noqa: E402


def fmt(fid, w, h, vcodec="avc1.4d401f", acodec="none", tbr=1000, **kw):
    return {"format_id": fid, "width": w, "height": h, "vcodec": vcodec, "acodec": acodec, "tbr": tbr, **kw}


STORYBOARD = {"format_id": "sb0", "width": 320, "height": 180, "vcodec": "none", "acodec": "none",
              "format_note": "storyboard"}


def test_exactly_720_preferred_h264_video_only():
    formats = [STORYBOARD, fmt("18", 640, 360, acodec="mp4a"), fmt("247", 1280, 720, "vp9", tbr=1500),
               fmt("136", 1280, 720, "avc1.4d401f", tbr=1200), fmt("22", 1280, 720, acodec="mp4a", tbr=2000),
               fmt("137", 1920, 1080)]
    assert choose_format(formats)["format_id"] == "136"


def test_no_720_takes_closest_above_not_best():
    formats = [fmt("135", 854, 480), fmt("137", 1920, 1080), fmt("271", 2560, 1440, "vp9"),
               fmt("313", 3840, 2160, "vp9")]
    assert choose_format(formats)["format_id"] == "137"


def test_nothing_above_takes_closest_below():
    formats = [fmt("160", 256, 144), fmt("134", 640, 360), fmt("135", 854, 480), STORYBOARD]
    assert choose_format(formats)["format_id"] == "135"


def test_quality_is_the_shorter_side():
    # portrait / 4:5 videos: YouTube's "720p" is 720 wide
    assert quality(fmt("x", 720, 1280)) == 720 and quality(fmt("y", 720, 900)) == 720
    formats = [fmt("a", 720, 900), fmt("b", 576, 720), fmt("c", 1080, 1350)]
    assert choose_format(formats)["format_id"] == "a"


def test_combined_streams_used_when_no_video_only():
    formats = [fmt("18", 640, 360, acodec="mp4a"), fmt("22", 1280, 720, acodec="mp4a")]
    assert choose_format(formats)["format_id"] == "22"
    assert choose_format([STORYBOARD]) is None


def test_other_target_height():
    formats = [fmt("135", 854, 480), fmt("136", 1280, 720), fmt("137", 1920, 1080)]
    assert choose_format(formats, 540)["format_id"] == "136"
    assert choose_format(formats, 2160)["format_id"] == "137"


def test_yt_dlp_uses_our_choice():
    """yt-dlp's own format-selection machinery calls our selector (offline:
    a fake video description, nothing is downloaded)."""
    from yt_dlp import YoutubeDL

    def selector(ctx):
        f = choose_format(ctx.get("formats") or [], 720)
        if f is not None:
            yield f

    formats = [dict(f, url=f"https://example.invalid/{f['format_id']}", ext="mp4", protocol="https")
               for f in (fmt("135", 854, 480), fmt("137", 1920, 1080), fmt("271", 2560, 1440, "vp9"))]
    info = {"id": "abc", "title": "Miss Forensics (Chapter 1)", "formats": formats, "extractor": "fake",
            "extractor_key": "Fake", "webpage_url": "https://example.invalid/abc"}
    with YoutubeDL({"format": selector, "quiet": True, "simulate": True}) as ydl:
        out = ydl.process_ie_result(info, download=False)
    assert out["format_id"] == "137" and quality(out) == 1080


# -- playlists --------------------------------------------------------------------------

from video2book.youtube import (gather, is_playlist, parse_any_chapter, playlist_id,  # noqa: E402
                                playlist_items, playlist_url)

PL = "PLx1y2z3a4b5c6d7e8f9g0h1"


def test_playlist_links():
    assert playlist_id(f"https://www.youtube.com/playlist?list={PL}") == PL
    assert playlist_id(f"https://www.youtube.com/watch?v=abc123&list={PL}&index=4") == PL
    assert playlist_id(f"youtube.com/playlist?list={PL}") == PL
    assert playlist_id(PL) == PL
    assert playlist_url(f"https://youtu.be/abc123?list={PL}") == f"https://www.youtube.com/playlist?list={PL}"
    for channel in ("@RuiNemesys", "https://www.youtube.com/@RuiNemesys", "https://www.youtube.com/@RuiNemesys/videos",
                    "https://www.youtube.com/watch?v=abc123"):
        assert not is_playlist(channel)


@pytest.mark.parametrize("title,expected", [
    ("Miss Forensics (Chapter 142)", (142, 142)),
    ("Some Other Series - Ep. 3", (3, 3)),
    ("Ch 10-11 | Doctor Rebirth", (10, 11)),
    ("Episode 4: the trial", (4, 4)),
    ("Behind the scenes", None),
    ("Next step 5", None),
])
def test_parse_any_chapter(title, expected):
    assert parse_any_chapter(title) == expected


def pl_videos(titles):
    return [ChannelVideo(id=f"p{i}", title=t, url=f"https://www.youtube.com/watch?v=p{i}", index=i)
            for i, t in enumerate(titles)]


def test_playlist_items_with_chapter_numbers():
    vids = pl_videos(["Miss Forensics (Chapter 3)", "Miss Forensics (Chapter 1)", "[Deleted video]",
                      "Miss Forensics (Chapter 2)", "Miss Forensics (Chapter 2) reupload", "[Private video]"])
    items = playlist_items(vids)
    assert [c.label for c in items] == ["001", "002", "003"]              # chapter order, not playlist order
    assert items[1].video.id == "p3" and [a.id for a in items[1].alternates] == ["p4"]
    assert [c.label for c in select_chapters(items, latest=2)] == ["002", "003"]
    assert items[0].pdf_name("Miss Forensics") == "Miss Forensics - Chapter 001.pdf"


def test_playlist_items_without_chapter_numbers():
    vids = pl_videos(["Intro: who is she?", "The first case", "[Deleted video]", "A new body", "Finale"])
    items = playlist_items(vids)
    assert [c.video.title for c in items] == ["Intro: who is she?", "The first case", "A new body", "Finale"]
    assert [c.position for c in items] == [1, 2, 4, 5]                    # positions in the playlist
    assert [c.video.title for c in select_chapters(items, latest=2)] == ["A new body", "Finale"]
    assert items[0].pdf_name("My list") == "My list - 001 - Intro who is she.pdf"
    assert items[0].display == "Video 1" and items[0].label == "#001"
    assert select_chapters(items, "1-3") == []                            # ranges need chapter numbers


def test_playlist_items_filtered_by_series():
    vids = pl_videos(["Miss Forensics (Chapter 5)", "Doctor Rebirth (Chapter 9)", "Miss Forensics (Chapter 6)"])
    assert [c.label for c in playlist_items(vids, "Miss Forensics")] == ["005", "006"]


def fake_playlist(titles, playlist_title="Miss Forensics - all chapters"):
    def it(source, cookies=None, meta=None):
        if meta is not None:
            meta["title"] = playlist_title
        return iter(pl_videos(titles))
    return it


def test_gather_playlist_and_channel_rules(monkeypatch, cache_dir):
    monkeypatch.setenv("VIDEO2BOOK_CACHE", cache_dir)
    monkeypatch.setattr(yt, "iter_playlist_videos",
                        fake_playlist(["Miss Forensics (Chapter 7)", "Miss Forensics (Chapter 8)",
                                       "Doctor Rebirth (Chapter 1)"]))
    url = f"https://www.youtube.com/playlist?list={PL}"
    sel = gather(url, refresh=True)
    assert sel.kind == "playlist" and sel.listed == 3
    assert sel.name == "Miss Forensics - all chapters"                    # playlist title
    assert [c.label for c in sel.items] == ["001", "007", "008"]           # every video, by chapter
    sel = gather(url, "Miss Forensics", latest=1)                         # cached listing, series filter
    assert [c.label for c in sel.items] == ["008"] and sel.name == "Miss Forensics"
    assert gather(url, name="Mine").name == "Mine"
    with pytest.raises(ValueError):
        gather("@RuiNemesys")                                              # a channel needs a series


def test_cli_playlist_without_series(fake_downloader, tmp_path, monkeypatch, cache_dir, capsys):
    from video2book import yt_cli
    monkeypatch.setenv("VIDEO2BOOK_CACHE", cache_dir)
    monkeypatch.setattr(yt, "iter_playlist_videos",
                        fake_playlist(["Prologue", "The lab", "The verdict"], "My Comic"))
    monkeypatch.setattr(yt, "download_video", fake_downloader)
    url = f"https://www.youtube.com/watch?v=zzz&list={PL}"
    assert yt_cli.main([url, "--list", "--refresh"]) == 0
    assert "3 video(s)" in capsys.readouterr().out
    assert yt_cli.main([url, "--latest", "2", "-o", str(tmp_path)]) == 0
    folder = tmp_path / "My Comic"
    assert sorted(p.name for p in folder.glob("*.pdf")) == ["My Comic - 002 - The lab.pdf",
                                                            "My Comic - 003 - The verdict.pdf"]
    assert yt_cli.main(["@RuiNemesys"]) == 2                               # channel without series
