#!/usr/bin/env python3
"""Door reading regression: the panel column is only the door when what is in
it is where a door can be AND shaped like one.

Every column of numbers below is real: the operator's two calibration
captures, and profiles lifted from attested corpus frames. Nothing here is
constructed to sit on a threshold.

  python3 tests/test_door.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bathroom import config, detect                            # noqa: E402
from bathroom.zones import N                                    # noqa: E402

FAILS = []


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = config.load_baseline()

# the jamb from the live scan, so these columns are read against the same
# invariant the daemon uses
JAMB_FROM_SCAN = [BASE[r][detect.JAMB_COL] for r in range(N)]


def frame(colH, colG=None):
    """A matrix carrying just the two columns the door reading looks at."""
    m = [[0] * N for _ in range(N)]
    for r in range(N):
        m[r][detect.PANEL_COL] = colH[r]
        m[r][detect.JAMB_COL] = (colG or JAMB_FROM_SCAN)[r]
    return m


def calibration(name):
    with open(os.path.join(ROOT, "data", name)) as f:
        return json.load(f)


def main():
    n = 0

    # --- the operator's own calibrations ------------------------------------
    shut = calibration("door_closed100.json")
    openc = calibration("door_open100.json")
    check(detect.door_open_pct(frame(shut["colH"], shut["colG"]), BASE) == 0.0,
          "the 37-frame 'closed 100%' capture reads shut")
    check(detect.door_open_pct(frame(openc["colH"], openc["colG"]), BASE) == 100.0,
          "the 37-frame '100% open' capture reads open")
    n += 2

    # --- a body in the doorway is not the door ------------------------------
    # Verbatim from attested frames. The old code fitted these and reported
    # 72-77% open, because collinear points fit any plane with zero residual:
    # this exact column measured rms 0.3mm, better than the real door's 5.9.
    NT = 4000
    bodies = [
        ([673, 686, 628, 676, 672, 683, NT, NT], "WALKING, body at ~650mm"),
        ([626, 609, 615, 653, 638, 617, NT, NT], "WALKING, body at ~620mm"),
        ([710, 737, 688, 767, 737, 739, 782, NT], "body at ~730mm"),
        ([NT, 765, NT, 834, 785, 819, 855, NT], "body at ~800mm, gappy"),
    ]
    for colH, why in bodies:
        got = detect.door_open_pct(frame(colH), BASE)
        check(got is None, "%s -> unreadable, not a door position (got %s)"
              % (why, got))
        n += 1

    # --- a blob at the leaf's distance is not the door either ---------------
    # This one IS in the leaf's band, so position alone lets it through. It
    # jitters 100mm ray to ray, which no flat leaf does. The old code called
    # it rms 0.02 and reported 76.7% open on an empty bathroom.
    got = detect.door_open_pct(frame([NT, 1136, 1037, 1078, 1037, 1077, NT, NT]), BASE)
    check(got is None, "a 100mm-jittering blob at leaf distance -> unreadable "
                       "(got %s)" % got)
    n += 1

    # --- the real door still reads, including through stray cells -----------
    # Attested shut-door columns from standing/sitting frames.
    for colH, why in (
            ([1114, 1117, 1126, 1135, 1143, 1126, 1119, 1112], "attested shut (standing)"),
            ([1132, 1125, 1135, 1117, 1142, 1129, 1127, 1124], "attested shut (sitting)"),
            ([1103, 1112, 1129, 1125, 1141, 1120, 1136, 1117], "attested shut, noisier")):
        check(detect.door_open_pct(frame(colH), BASE) == 0.0, "%s -> shut" % why)
        n += 1

    # a stray near cell among a shut door's returns is an outlier, not a
    # corrupted reading: it falls outside the leaf band and is simply not the
    # leaf. This is the case that must NOT be handled by a return-count gate.
    strayed = [1114, 1117, 300, 1135, 1143, 1126, 1119, 1112]
    check(detect.door_open_pct(frame(strayed), BASE) == 0.0,
          "one stray near cell among a shut door's rays does not move the reading")
    n += 1

    # and a door with most of its rays dropped, which the sensor does often,
    # still reads shut off the ones that came back
    sparse = [NT, NT, 1126, 1135, 1143, NT, NT, NT]
    check(detect.door_open_pct(frame(sparse), BASE) == 0.0,
          "3 surviving rays at leaf distance still read shut")
    n += 1

    # --- nothing there is open, not shut ------------------------------------
    check(detect.door_open_pct(frame([NT] * 8), BASE) == 100.0,
          "no returns at all -> open")
    # the hallway seen through the opening is not a leaf
    check(detect.door_open_pct(frame([3200, 3150, 3300, 3250, 3180, NT, NT, NT]),
                               BASE) == 100.0,
          "the far wall seen through the opening -> open")
    n += 2

    # --- no intermediate values are emitted ---------------------------------
    # The swing is a rotation about a vertical hinge and one column is one
    # azimuth, so there is no angle in this data to report. Anything between
    # the two ends would be fitted, not measured.
    from bathroom import ingest
    seen = set()
    for s in ingest.corpus():
        for m, st in zip(s.frames, s.states):
            pct = detect.door_open_pct(m, s.baseline)
            if pct is not None:
                seen.add(pct)
    check(seen <= {0.0, 100.0},
          "over the whole corpus the reading is only shut/open/unreadable "
          "(saw %s)" % sorted(seen))
    n += 1

    print("\n%d checks, %d failures" % (n, len(FAILS)))
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
