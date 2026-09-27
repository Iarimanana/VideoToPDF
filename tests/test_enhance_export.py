import numpy as np
import pikepdf

from video2book.enhance import apply, auto_contrast, bw_scan, white_balance
from video2book.export import ExportOptions, export_pdf, fit_for_page
from video2book.synthetic import SynthConfig, render_page

PAGE = render_page(SynthConfig(), 0, (300, 420))


def test_presets_shapes_and_types():
    assert apply(PAGE, "original") is PAGE
    c = apply(PAGE, "clean")
    assert c.shape == PAGE.shape and c.dtype == np.uint8
    b = bw_scan(PAGE)
    assert b.ndim == 2 and set(np.unique(b).tolist()) <= {0, 255}
    assert max(b.shape) >= max(PAGE.shape[:2])  # small pages are upscaled for B&W


def test_white_balance_neutralises_cast():
    cast = np.clip(PAGE.astype(int) * np.array([1.0, 0.9, 0.75]), 0, 255).astype(np.uint8)
    wb = white_balance(cast)
    paper = wb[PAGE.mean(axis=2) > 200].mean(axis=0)
    assert paper.max() - paper.min() < 12


def test_auto_contrast_stretches():
    dull = (PAGE.astype(float) * 0.5 + 60).astype(np.uint8)
    out = auto_contrast(dull)
    assert out.max() > 240 and out.min() < dull.min()


def test_pdf_page_sizes(tmp_path):
    imgs = [PAGE, np.ascontiguousarray(np.rot90(PAGE))]
    for size, expect in (("a4", {595, 842}), ("letter", {612, 792})):
        out = export_pdf(imgs, str(tmp_path / f"{size}.pdf"), ExportOptions(page_size=size))
        with pikepdf.open(out) as doc:
            for pg in doc.pages:
                b = [float(v) for v in pg.mediabox]
                assert {round(b[2] - b[0]), round(b[3] - b[1])} == expect
    out = export_pdf([PAGE], str(tmp_path / "fit.pdf"), ExportOptions(page_size="fit", dpi=100))
    with pikepdf.open(out) as doc:
        b = [float(v) for v in doc.pages[0].mediabox]
        assert round(b[2] - b[0]) == round(300 / 100 * 72) and round(b[3] - b[1]) == round(420 / 100 * 72)


def test_fit_for_page_only_downscales():
    big = np.zeros((6000, 4000, 3), np.uint8)
    small = fit_for_page(big, ExportOptions(page_size="a4", dpi=150))
    assert small.shape[0] <= 297 / 25.4 * 150 + 1
    assert fit_for_page(PAGE, ExportOptions(page_size="a4", dpi=300)) is PAGE


def test_bw_pdf_is_small(tmp_path):
    bw = bw_scan(PAGE)
    out = export_pdf([bw], str(tmp_path / "bw.pdf"), ExportOptions())
    with pikepdf.open(out) as doc:
        img = next(iter(doc.pages[0].get_images().values())) if hasattr(doc.pages[0], 'get_images') else next(iter(doc.pages[0].images.values()))
        assert int(img.BitsPerComponent) == 1
