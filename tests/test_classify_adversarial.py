#!/usr/bin/env python3
"""Adversarial regression test: the four boundary failures an independent
review surfaced must classify safely. Frames are CONSTRUCTED deterministically
from the seed captures + the default baseline, reproducing exactly the
reviewers' reductions, so a future tuning change that reopens any of these
fails here.

  python3 tests/test_classify_adversarial.py
"""
import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests"))

import test_classify as t                              # noqa: E402
from bathroom.classify import classify, DEFAULT_BASELINE  # noqa: E402
from bathroom.zones import State                        # noqa: E402

BASE = DEFAULT_BASELINE


def sparse_top_standing():
    """Finding 1: a standing body whose top is sparsely occluded (only one deep
    top cell). Must NOT be confidently called SITTING."""
    m = copy.deepcopy(t.STANDING)
    for c in range(8):
        for r in (0, 1):
            if (r, c) != (0, 2):       # keep the single deep top cell C1=720
                m[r][c] = BASE[r][c]
    return m


def toilet_noise():
    """Finding 2: only the 4 toilet cells obstructed (door swing / dust). Must
    NOT be called SITTING (no body)."""
    m = copy.deepcopy(t.EMPTY)
    for (r, c) in [(6, 1), (7, 1), (6, 2), (7, 2)]:      # B7,B8,C7,C8
        m[r][c] = 500
    return m


def walking_at_boundary():
    """Finding 3: a body crossing in front but reading right at the VNEAR
    cutoff (651mm) instead of <350. Must stay WALKING, not flip to STANDING."""
    m = copy.deepcopy(t.WALKING)
    for r in range(8):
        for c in range(8):
            if m[r][c] < 650:
                m[r][c] = 651
    return m


def upper_noise():
    """Finding 4: a few shallow noisy cells in the top rows. Must stay EMPTY."""
    m = copy.deepcopy(t.EMPTY)
    for (r, c) in [(0, 2), (0, 3), (1, 2)]:              # co-located 2x2 cluster
        m[r][c] = BASE[r][c] - 320                       # just past DELTA, shallow
    return m


CASES = [
    ("sparse-top standing", sparse_top_standing(), State.STANDING_AT_TOILET),
    ("toilet-only noise", toilet_noise(), State.EMPTY),
    ("walking at 651mm", walking_at_boundary(), State.WALKING),
    ("upper-row noise", upper_noise(), State.EMPTY),
]


def main():
    ok = True
    for name, frame, expected in CASES:
        got, f = classify(frame)
        flag = "PASS" if got == expected else "FAIL"
        if got != expected:
            ok = False
        print("%-4s %-22s expected=%-20s got=%-20s  occ=%d vnear=%d deepUp=%d "
              "low=%d nearCols=%d noise=%d"
              % (flag, name, expected.name, got.name, f["occ_count"],
                 f["vnear_count"], f["deep_upper"], f["lower_count"],
                 f["near_cols"], int(f["noise_cluster"])))
    print("ALL PASS" if ok else "SOME FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
