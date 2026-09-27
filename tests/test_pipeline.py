"""End to end: synthetic video -> pages -> PDF. Checks page count AND order."""

import json
import os
import zipfile

import pikepdf
import pytest

from conftest import identify
from video2book.cli import main as cli_main
from video2book.crop import iou
from video2book.export import ocr_status
from video2book.project import Pipeline, Project, Settings


def run(path, cache_dir, **settings):
    pipe = Pipeline(path, Settings(**settings), cache_root=cache_dir)
    return pipe, pipe.run()


def shown_pages(pipe, project, cfg):
    return [identify(pipe.representative(p.image_key), cfg) for p in project.included()]


@pytest.mark.parametrize("variant,overrides", [
    ("base", {}),
    ("rot90", {"rotation": 90}),
    ("heavy", {"crf": 38}),
    # Slow zoom during every hold (like the sample video), slide transitions.
    ("kenburns", {"ken_burns": 0.03, "repeat_after": None, "fade_every": 0}),
])
def test_count_and_order(video_factory, cache_dir, variant, overrides):
    path, cfg, truth = video_factory(variant, **overrides)
    pipe, project = run(path, cache_dir)
    s = project.summary()
    assert s["pages"] == cfg.n_pages, s
    assert shown_pages(pipe, project, cfg) == list(range(cfg.n_pages))
    assert s["duplicates_merged"] == truth.duplicates
    assert s["possible_duplicates"] == 0
    assert s["segments"] == len(truth.sequence)


def test_weak_fades_recovered_with_expected_pages(video_factory, cache_dir):
    # Slow zoom + a mix of strong slides and weak cross-fades: at the default
    # sensitivity the fades hide below the threshold. Giving the expected page
    # count lets the tool find the sensitivity that separates them (this is
    # what `--expected-pages N --auto-adjust` does).
    path, cfg, truth = video_factory("kenburns_fades", ken_burns=0.03, repeat_after=None, near_identical=())
    pipe = Pipeline(path, Settings(expected_pages=cfg.n_pages), cache_root=cache_dir)
    project = pipe.run()
    chk = pipe.check_expected(project)
    if not chk["ok"]:
        pipe.settings.sensitivity = chk["suggested_sensitivity"]
        project = pipe.run()
    assert len(project.included()) == cfg.n_pages
    assert shown_pages(pipe, project, cfg) == list(range(cfg.n_pages))


def test_duplicates_point_to_their_original(video_factory, cache_dir):
    path, cfg, truth = video_factory("base")
    pipe, project = run(path, cache_dir)
    dups = [p for p in project.pages if p.status == "duplicate"]
    assert len(dups) == 2
    for d in dups:
        orig = project.page(d.dup_of)
        assert identify(pipe.representative(d.image_key), cfg) == identify(pipe.representative(orig.image_key), cfg)
        assert orig.t < d.t


def test_auto_crop_finds_the_page(video_factory, cache_dir):
    path, cfg, truth = video_factory("base")
    pipe, project = run(path, cache_dir)
    for p in project.included():
        k = identify(pipe.representative(p.image_key), cfg)
        assert iou(tuple(p.crop["box"]), tuple(truth.page_rects[k])) > 0.93, (k, p.crop)
    # the phone UI and black bars are outside the static box
    x0, y0, x1, y1 = project.static_box
    assert y0 >= truth.photo_rect[1] - 8 and y1 <= truth.photo_rect[3] + 8


def test_sensitivity_change_uses_cache(video_factory, cache_dir):
    path, cfg, _ = video_factory("base")
    pipe, project = run(path, cache_dir)
    pipe2 = Pipeline(path, Settings(sensitivity=0.6), cache_root=cache_dir)
    p2 = pipe2.run()
    # loaded from the cache, not recomputed
    assert p2.timings["analysis_cached"] is True
    assert p2.timings["analysis"] < 0.5 * max(0.2, p2.timings["first_analysis"])
    assert pipe2.analysis() is pipe2.analysis()
    assert len(p2.included()) == cfg.n_pages


def test_review_edits_and_export(video_factory, cache_dir, tmp_path):
    path, cfg, truth = video_factory("base")
    pipe, project = run(path, cache_dir)
    first = project.included()[0]
    project.rotate(first.id, 90)
    project.set_status(project.included()[1].id, "deleted")
    n = len(project.included())
    # add a page from a timestamp (the middle of the 4th hold)
    t = sum(truth.holds[3]) / 2
    added = pipe.add_page_at(project, t)
    assert added.source == "manual" and len(project.included()) == n + 1
    # pick another frame for a page
    cand = pipe.neighbor_times(first, 6)
    assert len(cand) >= 3
    pipe.use_frame(project, first, cand[2][0])
    assert "picked" in first.flags
    # persistence round trip
    f = tmp_path / "p.json"
    project.save(str(f))
    project2 = Project.load(str(f))
    assert [p.id for p in project2.pages] == [p.id for p in project.pages]
    # PDF
    pdf = pipe.export_pdf(project2, str(tmp_path / "book.pdf"))
    with pikepdf.open(pdf) as doc:
        assert len(doc.pages) == len(project2.included())
        box = [float(v) for v in doc.pages[0].mediabox]
        w, h = box[2] - box[0], box[3] - box[1]
        assert {round(w), round(h)} == {595, 842}  # A4 (rotated page may be landscape)
    z = pipe.export_zip(project2, str(tmp_path / "pages.zip"))
    with zipfile.ZipFile(z) as zz:
        assert len(zz.namelist()) == len(project2.included())


def test_expected_pages_warning(video_factory, cache_dir):
    path, cfg, _ = video_factory("base")
    pipe = Pipeline(path, Settings(expected_pages=cfg.n_pages + 5), cache_root=cache_dir)
    project = pipe.run()
    chk = pipe.check_expected(project)
    assert chk["ok"] is False and "expect" in chk["message"]
    # A suggestion is only made if it really gets closer to the expected count.
    if "suggested_sensitivity" in chk:
        assert abs(chk["suggested_count"] - chk["expected"]) < abs(chk["detected"] - chk["expected"])
    project.settings.expected_pages = cfg.n_pages
    assert pipe.check_expected(project)["ok"] is True


def test_cli_end_to_end(video_factory, cache_dir, tmp_path, capsys):
    path, cfg, _ = video_factory("base")
    out = tmp_path / "cli.pdf"
    rep = tmp_path / "report.json"
    code = cli_main([path, "-o", str(out), "--expected-pages", str(cfg.n_pages), "--cache-dir", cache_dir,
                     "--report", str(rep), "--enhance", "clean", "--page-size", "fit", "-q"])
    assert code == 0 and out.exists()
    data = json.loads(rep.read_text())
    assert data["summary"]["pages"] == cfg.n_pages
    assert data["expected_check"]["ok"] is True
    with pikepdf.open(str(out)) as doc:
        assert len(doc.pages) == cfg.n_pages


@pytest.mark.skipif(not ocr_status()[0], reason="Tesseract/OCRmyPDF not installed")
def test_ocr_makes_searchable_pdf(video_factory, cache_dir, tmp_path):
    path, cfg, _ = video_factory("base")
    out = tmp_path / "ocr.pdf"
    code = cli_main([path, "-o", str(out), "--ocr", "eng", "--cache-dir", cache_dir, "-q"])
    assert code == 0
    with pikepdf.open(str(out)) as doc:
        assert len(doc.pages) == cfg.n_pages
        # OCRmyPDF adds an invisible text layer, i.e. at least one font object.
        fonts = [o for o in doc.objects if isinstance(o, pikepdf.Dictionary) and o.get("/Type") == "/Font"]
        assert fonts
