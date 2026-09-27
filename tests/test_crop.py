"""Cropping: phone UI + bars, paper on a table, blurred background, manual, perspective."""

import cv2
import numpy as np

from video2book.crop import (CropResult, apply_crop, auto_crop, find_paper, iou, manual_crop,
                             static_border, trim_uniform, warp_quad)
from video2book.synthetic import SynthConfig, _Composer, hold_frame, render_page

CFG = SynthConfig(n_pages=6, width=360, height=640, seed=5)


def frames_and_truth():
    comp = _Composer(CFG)
    frames = [comp.frame([(i, 0, 1.0, 0.0)]) for i in range(CFG.n_pages)]
    return frames, comp


def test_static_border_removes_phone_ui_and_bars():
    frames, comp = frames_and_truth()
    box = static_border(frames)
    px0, py0, px1, py1 = 0, comp.py, comp.pw, comp.py + comp.ph
    # the status bar / toolbar live outside the photo: they must be cut away,
    # and nothing of any page may be cut.
    assert box[1] >= py0 - 6 and box[3] <= py1 + 6
    for i in range(CFG.n_pages):
        r = comp.rects[i]
        assert box[0] <= r[0] + 1 and box[1] <= r[1] + 1 and box[2] >= r[2] - 1 and box[3] >= r[3] - 1


def test_auto_crop_finds_paper_on_table():
    frames, comp = frames_and_truth()
    box = static_border(frames)
    for i, f in enumerate(frames):
        c = auto_crop(f, box)
        assert c.method == "paper"
        assert iou(c.box, comp.rects[i]) > 0.95
        assert c.quad is not None and c.quad.shape == (4, 2)


def test_auto_crop_with_black_bars_only():
    pg = render_page(CFG, 2, (300, 420))
    frame = np.zeros((700, 420, 3), np.uint8)
    frame[140:560, 60:360] = pg
    assert trim_uniform(frame) == (60, 140, 360, 560)
    c = auto_crop(frame)
    assert iou(c.box, (60, 140, 360, 560)) > 0.97


def test_auto_crop_blurred_background_fill():
    # Slideshow apps often fill the frame with a blurred, enlarged copy.
    pg = render_page(CFG, 1, (300, 420))
    bg = cv2.GaussianBlur(cv2.resize(pg, (400, 700)), (0, 0), 18)
    frame = bg.copy()
    frame[140:560, 50:350] = pg
    c = auto_crop(frame)
    # The blurred fill is removed. (The page's blank paper margin has the same
    # colour as the blurred copy, so the crop may stop at the text block.)
    x0, y0, x1, y1 = c.box
    assert x0 >= 50 - 10 and y0 >= 140 - 10 and x1 <= 350 + 10 and y1 <= 560 + 10
    assert (x1 - x0) * (y1 - y0) >= 0.6 * 300 * 420


def test_manual_crop_relative():
    img = np.zeros((200, 100, 3), np.uint8)
    c = manual_crop(img, (0.1, 0.25, 0.9, 0.75))
    assert c.box == (10, 50, 90, 150) and c.method == "manual"
    assert apply_crop(img, c).shape == (100, 80, 3)
    assert manual_crop(img, (0.9, 0.75, 0.1, 0.25)).box == (10, 50, 90, 150)  # any corner order


def test_perspective_quad_and_warp():
    pg = render_page(CFG, 3, (300, 420))
    src = np.float32([[0, 0], [299, 0], [299, 419], [0, 419]])
    dst = np.float32([[70, 60], [330, 90], [345, 560], [50, 540]])  # keystoned photo
    m = cv2.getPerspectiveTransform(src, dst)
    frame = np.full((620, 400, 3), 45, np.uint8)
    warped = cv2.warpPerspective(pg, m, (400, 620), dst=frame, borderMode=cv2.BORDER_TRANSPARENT)
    c = find_paper(warped)
    assert c is not None
    err = np.abs(c.quad - dst).max()
    assert err < 8, err
    flat = apply_crop(warped, c, perspective=True)
    # The warp hugs the sheet: all four corners are paper, not table.
    k = 6
    for patch in (flat[:k, :k], flat[:k, -k:], flat[-k:, :k], flat[-k:, -k:]):
        assert patch.mean() > 150
    # (The true aspect ratio can't be recovered without camera data; the
    #  result is a rectangle sized from the quad's edges.)
    assert 0.5 < flat.shape[1] / flat.shape[0] < 0.8


def test_crop_result_roundtrip():
    c = CropResult((1, 2, 3, 4), "paper", np.float32([[0, 0], [1, 0], [1, 1], [0, 1]]))
    d = CropResult.from_dict(c.to_dict())
    assert d.box == (1, 2, 3, 4) and d.method == "paper" and d.quad.shape == (4, 2)
    assert warp_quad(np.zeros((10, 10, 3), np.uint8), np.float32([[0, 0], [2, 0], [2, 2], [0, 2]])).shape == (10, 10, 3)
