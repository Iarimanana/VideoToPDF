"""Duplicate detection on synthetic page images."""

import cv2
import numpy as np

from video2book.dedup import DedupParams, classify, compare, describe, find_duplicates
from video2book.synthetic import SynthConfig, render_page

CFG = SynthConfig(seed=3)
INK = (40, 38, 36)
PAPER = (238, 234, 222)


def page(i, size=(600, 840)):
    return render_page(CFG, i, size)


def jpeg(img, q):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)


def verdict(a, b):
    return classify(compare(describe(a), describe(b)))


def test_same_page_different_compression_is_duplicate():
    p = page(1)
    assert verdict(jpeg(p, 90), jpeg(p, 60)) == "duplicate"
    assert verdict(jpeg(p, 50), jpeg(p, 30)) == "duplicate"


def test_same_page_shifted_is_duplicate():
    p = page(2)
    m = np.float32([[1, 0, 4], [0, 1, -3]])
    shifted = cv2.warpAffine(p, m, (p.shape[1], p.shape[0]), borderMode=cv2.BORDER_REPLICATE)
    assert verdict(jpeg(p, 80), jpeg(shifted, 80)) == "duplicate"


def test_different_pages_are_unique():
    for i in range(3):
        assert verdict(jpeg(page(i), 80), jpeg(page(i + 5), 80)) == "unique"


def test_near_identical_pages_are_not_merged():
    # CFG.near_identical: page 4 reuses page 3's body text; only the big
    # number and bar code differ.
    assert CFG.near_identical[0] == (3, 4)
    assert verdict(jpeg(page(3), 80), jpeg(page(4), 80)) != "duplicate"


def test_changed_word_is_not_merged():
    p = page(6)
    q = p.copy()
    cv2.rectangle(q, (300, 400), (360, 420), PAPER, -1)
    cv2.putText(q, "house", (302, 415), cv2.FONT_HERSHEY_SIMPLEX, 0.4, INK, 1, cv2.LINE_AA)
    assert verdict(jpeg(p, 85), jpeg(q, 85)) != "duplicate"


def test_changed_page_number_only_is_flagged_not_merged():
    p = page(4)
    q = p.copy()
    cv2.rectangle(q, (250, 790), (360, 815), PAPER, -1)
    cv2.putText(q, "- 6 -", (276, 806), cv2.FONT_HERSHEY_SIMPLEX, 600 / 1300, INK, 1, cv2.LINE_AA)
    # Everything identical except a tiny footer number: uncertain -> flag.
    assert verdict(jpeg(p, 90), jpeg(q, 80)) == "possible"


def test_sequence_with_swipe_back():
    # Shown order: 0 1 2 1 3  -> the second "1" is a duplicate of index 1.
    imgs = [jpeg(page(i), 70 + 5 * k) for k, i in enumerate([0, 1, 2, 1, 3])]
    res = find_duplicates([describe(x) for x in imgs])
    assert [r.status for r in res] == ["unique", "unique", "unique", "duplicate", "unique"]
    assert res[3].ref == 1


def test_inactive_pages_are_ignored():
    imgs = [jpeg(page(i), 80) for i in [0, 1, 0]]
    descs = [describe(x) for x in imgs]
    res = find_duplicates(descs, DedupParams(), active=[False, True, True])
    assert res[2].status == "unique"  # its twin was deleted: keep this one


def test_repeat_far_back_is_found_via_hash():
    imgs = [jpeg(page(i), 80) for i in range(8)] + [jpeg(page(0), 70)]
    res = find_duplicates([describe(x) for x in imgs], DedupParams(window=3))
    assert res[-1].status == "duplicate" and res[-1].ref == 0
