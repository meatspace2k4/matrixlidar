#!/usr/bin/env python3
"""Pipeline regression: clip round-trip, corrections, split, robustness.

Builds synthetic clips in a temp dir (no hardware) and exercises the data path
the model training will use. Run from the repo root:

  python3 tests/test_pipeline.py
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bathroom import dataset, config                       # noqa: E402
from bathroom.classify import median_window, DEFAULT_BASELINE  # noqa: E402
from bathroom.zones import N, NO_TARGET, ORIENTATION, from_label, State  # noqa: E402

FAILS = []


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


def _clip_doc(clip_id, label, n=20, corrections=None):
    frames = []
    for i in range(n):
        mm = [1100] * (N * N)
        mm[0] = NO_TARGET            # one dropout cell, for the mask test
        frames.append({"t": round(i / 15.0, 3), "mm": mm,
                       "door_open_pct": 100.0, "state_auto": "EMPTY",
                       "state_engine": "EMPTY"})
    doc = {"schema": "bathroom-clip/1", "clip_id": clip_id, "label": label,
           "n": n, "hz": 15.0, "orientation": ORIENTATION,
           "baseline": DEFAULT_BASELINE, "started": 1.0, "ended": 1.0 + n / 15.0,
           "frames": frames}
    if corrections:
        doc["corrections"] = corrections
    return doc


def main():
    with tempfile.TemporaryDirectory() as d:
        # two clips, one with a mid-clip correction
        a = _clip_doc("empty_1", "empty", n=20)
        b = _clip_doc("sitting_2", "sitting", n=30,
                      corrections=[{"start": 0, "end": 10, "label": "standing"}])
        for doc in (a, b):
            with open(os.path.join(d, doc["clip_id"] + ".json"), "w") as f:
                json.dump(doc, f)

        clips = list(dataset.iter_clips(d))
        check(len(clips) == 2, "iter_clips loads both clips")

        ca = dataset.load_clip(os.path.join(d, "empty_1.json"))
        check(ca["labels"] == ["empty"] * 20, "no-correction clip -> uniform per-frame label")

        cb = dataset.load_clip(os.path.join(d, "sitting_2.json"))
        check(cb["labels"][:10] == ["standing"] * 10 and cb["labels"][10:] == ["sitting"] * 20,
              "correction range overrides per-frame labels")

        try:
            import numpy  # noqa: F401
            tr = dataset.to_training(cb)
            check(tr["x"].shape == (30, 8, 8), "to_training x is (T,8,8)")
            check(int(tr["mask"].sum()) == 30, "mask flags the per-frame dropout cell")
            check(tr["baseline"].shape == (8, 8), "baseline returned as 8x8")
        except ImportError:
            print("SKIP numpy not present; to_training array checks skipped")

        tr_ids, va_ids = dataset.train_val_split(0.5, d)
        tr2, va2 = dataset.train_val_split(0.5, d)
        check((tr_ids, va_ids) == (tr2, va2), "train_val_split is deterministic")
        check(sorted(tr_ids + va_ids) == ["empty_1", "sitting_2"], "split covers all clips")

        s = dataset.summary(d)
        check(s["n_clips"] == 2 and s["total_frames"] == 50, "summary counts clips+frames")

        # corrupt clip is skipped, not fatal
        with open(os.path.join(d, "broken.json"), "w") as f:
            f.write("{not json")
        check(len(list(dataset.iter_clips(d))) == 2, "corrupt clip skipped, others survive")

    # robustness: bad baseline file -> built-in default, no crash
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        f.write("{garbage")
        bad = f.name
    check(config.load_baseline(bad) == DEFAULT_BASELINE, "corrupt baseline falls back to default")
    os.unlink(bad)

    # empty-window guard
    try:
        median_window([])
        check(False, "median_window([]) should raise")
    except ValueError:
        check(True, "median_window([]) raises ValueError")

    # State.EMPTY (value 0) must round-trip through from_label without truthiness bugs
    check(from_label("empty") is State.EMPTY, "from_label('empty') -> EMPTY")

    print("ALL PASS" if not FAILS else "FAILED: %d" % len(FAILS))
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
