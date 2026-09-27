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


# -- same page size for every page ---------------------------------------------------

from video2book.export import best_aspect, fill_color, fill_fraction, pad_to_aspect  # noqa: E402


def _sizes(pdf):
    with pikepdf.open(pdf) as doc:
        return [tuple(round(float(v), 2) for v in pg.mediabox) for pg in doc.pages]


def test_best_aspect_needs_least_filling():
    aspects = [0.7, 0.7, 0.71, 0.69, 1.5]
    best = best_aspect(aspects)
    assert abs(best - 0.7) < 0.02
    total = lambda r: sum(fill_fraction(a, r) for a in aspects)
    assert all(total(best) <= total(r) + 1e-9 for r in (0.5, 0.6, 0.8, 1.0, 1.5))


def test_pad_to_aspect_and_fill_colours():
    dark = np.full((100, 50, 3), 30, np.uint8)
    out = pad_to_aspect(dark, 1.0)                       # 50x100 -> 100x100
    assert out.shape == (100, 100, 3) and out[50, 5].tolist() == [30, 30, 30]   # auto = edge colour
    assert pad_to_aspect(dark, 1.0, "white")[50, 5].tolist() == [255, 255, 255]
    assert pad_to_aspect(dark, 1.0, "black")[50, 5].tolist() == [0, 0, 0]
    assert pad_to_aspect(dark, 0.5) is dark                                     # already that shape
    wide = pad_to_aspect(np.full((50, 200, 3), 200, np.uint8), 1.0)
    assert wide.shape[:2] == (200, 200) and wide[:75].min() == 200             # centred, top/bottom filled
    bw = bw_scan(PAGE)
    padded = pad_to_aspect(bw, 1.2)
    assert set(np.unique(padded).tolist()) <= {0, 255}                         # B&W stays 1-bit
    assert fill_color(bw) == 255


def test_uniform_pdf_pages_all_same_size(tmp_path):
    portrait = PAGE                                            # 300x420
    landscape = np.ascontiguousarray(np.rot90(PAGE))           # 420x300
    square = PAGE[:300]                                        # 300x300
    imgs = [portrait, portrait, landscape, square, portrait]
    out = export_pdf(imgs, str(tmp_path / "u.pdf"), ExportOptions(page_size="uniform"))
    sizes = _sizes(out)
    assert len(set(sizes)) == 1
    x0, y0, x1, y1 = sizes[0]
    assert abs((x1 - x0) / (y1 - y0) - 300 / 420) < 0.01       # the majority shape wins
    with pikepdf.open(out) as doc:                              # each image fills its page exactly
        for pg in doc.pages:
            img = pikepdf.PdfImage(next(iter(pg.get_images().values())))
            assert abs(img.width / img.height - 300 / 420) < 0.01


def test_a4_pages_are_never_rotated(tmp_path):
    landscape = np.ascontiguousarray(np.rot90(PAGE))
    out = export_pdf([PAGE, landscape], str(tmp_path / "a4.pdf"), ExportOptions(page_size="a4"))
    assert _sizes(out)[0] == _sizes(out)[1] == (0, 0, 595.28, 841.89)
