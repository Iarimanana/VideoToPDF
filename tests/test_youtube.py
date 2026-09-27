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
