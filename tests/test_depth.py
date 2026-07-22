#!/usr/bin/env python3
"""Depth-aware classification regression, against REAL harvested frames.

Covers the misdetections observed in live sessions (fixtures/misdetections/):
- session 6, 13:41 window: mirror use right of the sensor was committed as
  SITTING_AT_TOILET five times. With depth (body-plane headroom + seat-depth
  + fixture position) none of those frames may classify as SITTING.
- session 5 tail: a near-lens body smeared to the door COLUMN at ~450mm and
  its vanish fired a false EXIT. The classifier must expose body_mm well
  inside the door plane for those frames (the occupancy depth gate's input).
- the labeled seeds must keep their states (regression half).

  python3 tests/test_depth.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import test_classify as tc                                   # noqa: E402
from bathroom import config, stream                          # noqa: E402
from bathroom.classify import classify                       # noqa: E402
from bathroom.occupancy import Params as OccParams           # noqa: E402
from bathroom.zones import N, State                          # noqa: E402

BASELINE = config.load_baseline()
FIXDIR = os.path.join(ROOT, "fixtures", "misdetections")

FAILS = []


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


def oriented(frame_csv):
    flat = [int(x) for x in frame_csv.split(",")]
    return stream.orient([flat[i * N:(i + 1) * N] for i in range(N)])


def session(sid):
    with open(os.path.join(FIXDIR, "session_%03d.json" % sid)) as f:
        return json.load(f)


def main():
    # regression: the four labeled seeds keep their states
    for name, frame, expected in tc.CASES:
        got, _f = classify(frame, BASELINE)
        check(got == expected, "seed %-14s still %s" % (name, expected.name))

    # mirror-use frames (session 6, committed ->SITTING at 13:41) must never
    # classify as SITTING again; STANDING or UNKNOWN are both acceptable.
    s6 = session(6)
    mirror = [tr for tr in s6["transitions"]
              if tr["mark"]["kind"] == "posture"
              and tr["mark"]["to"] == "SITTING_AT_TOILET"]
    check(len(mirror) >= 4, "fixture holds the mirror SITTING commits (%d)"
          % len(mirror))
    still_sitting = 0
    standing = 0
    for tr in mirror:
        st, f = classify(oriented(tr["frame"]), BASELINE)
        if st == State.SITTING_AT_TOILET:
            still_sitting += 1
        if st == State.STANDING_AT_TOILET:
            standing += 1
    check(still_sitting == 0,
          "no mirror frame classifies as SITTING (was 5/5, now %d)" % still_sitting)
    check(standing >= mirror.__len__() // 2,
          "most mirror frames read STANDING (%d/%d)" % (standing, len(mirror)))

    # session 5 tail: the frames of the near-lens crossing that caused the
    # false EXIT must expose a body depth far inside the door plane, so the
    # occupancy depth gate rejects the door-side vanish.
    s5 = session(5)
    tail = [tr for tr in s5["transitions"][-4:]
            if tr["mark"]["kind"] == "posture" and tr["mark"]["to"] == "WALKING"]
    check(len(tail) >= 1, "fixture holds the false-exit crossing frame")
    for tr in tail:
        st, f = classify(oriented(tr["frame"]), BASELINE)
        check(f["body_mm"] is not None
              and f["body_mm"] < OccParams.DOOR_PLANE_MM,
              "crossing frame body_mm=%s is inside the door plane (<%d)"
              % (f["body_mm"], OccParams.DOOR_PLANE_MM))

    # depth features sanity on the seeds: seated body reads at seat depth,
    # the standing seed does not satisfy the seat condition
    _st, f_sit = classify(tc.SITTING, BASELINE)
    check(f_sit["body_at_seat"], "SITTING seed body is at seat depth")
    _st, f_stand = classify(tc.STANDING, BASELINE)
    check(not f_stand["body_at_seat"] or f_stand["deep_upper_supported"] > 0,
          "STANDING seed cannot be mistaken for a seated body")

    print("ALL PASS" if not FAILS else "SOME FAILED")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
