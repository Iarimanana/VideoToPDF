#!/usr/bin/env python3
"""Generate a synthetic "book pages slideshow" test video + ground truth JSON.

Examples:
    python scripts/make_synthetic_video.py out/test.mp4
    python scripts/make_synthetic_video.py out/big.mp4 --pages 40 --width 720 --height 1280
    python scripts/make_synthetic_video.py out/rot.mp4 --rotation 90 --crf 36 --ken-burns 0.04
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from video2book.synthetic import SynthConfig, generate  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("output", help="output .mp4")
    p.add_argument("--pages", type=int, default=14)
    p.add_argument("--width", type=int, default=480)
    p.add_argument("--height", type=int, default=854)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--crf", type=int, default=30, help="H.264 quality (higher = more compression)")
    p.add_argument("--rotation", type=int, default=0, help="store with rotation metadata (90, 180, 270)")
    p.add_argument("--ken-burns", type=float, default=0.0, help="slow zoom during holds, e.g. 0.04")
    p.add_argument("--no-repeat", action="store_true", help="don't swipe back to earlier pages")
    p.add_argument("--no-ui", action="store_true", help="no fake phone UI")
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args()
    cfg = SynthConfig(n_pages=a.pages, width=a.width, height=a.height, fps=a.fps, crf=a.crf,
                      rotation=a.rotation, ken_burns=a.ken_burns, seed=a.seed,
                      repeat_after=None if a.no_repeat else min(6, a.pages - 1),
                      phone_ui=not a.no_ui)
    os.makedirs(os.path.dirname(os.path.abspath(a.output)), exist_ok=True)
    truth_path = os.path.splitext(a.output)[0] + ".truth.json"
    t = generate(a.output, cfg, truth_path)
    print(f"Wrote {a.output}: {t.n_pages} unique pages, {len(t.sequence)} holds "
          f"(sequence {t.sequence}), {t.duplicates} repeats")
    print(f"Ground truth: {truth_path}")


if __name__ == "__main__":
    main()
