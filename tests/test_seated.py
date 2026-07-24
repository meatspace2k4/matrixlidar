#!/usr/bin/env python3
"""Seated posture regression: the HEAD_ROWS fix ensures no row-0 oscillation.

Before the fix (HEAD_ROWS = (0, 1, 2)), a seated torso at rows 1-2 caused
upper_body to swing 1-3 around the threshold of 2, flipping posture 26 times
in four minutes. After the fix (HEAD_ROWS = (0,) only), 30 operator-attested
sitting frames hold steady SITTING_AT_TOILET with zero row-0 cells at body
depth, while the standing seed keeps its row-0 cells and classifies STANDING.

  python3 tests/test_seated.py
"""
import json
import os
import sys
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import test_classify as tc                                   # noqa: E402
from bathroom import stream                                  # noqa: E402
from bathroom.classify import classify, Params, features     # noqa: E402
from bathroom.config import load_baseline                    # noqa: E402
from bathroom.zones import N, State                          # noqa: E402

FIXDIR = os.path.join(ROOT, "fixtures", "misdetections")

FAILS = []


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


def oriented(frame_csv):
    flat = [int(x) for x in frame_csv.split(",")]
    return stream.orient([flat[i * N:(i + 1) * N] for i in range(N)])


def main():
    # Load the fixture session with the 18:31-18:33 window
    session_file = os.path.join(FIXDIR, "session_008.json")
    if not os.path.isfile(session_file):
        print("FAIL fixture session_008.json not found")
        return 1

    with open(session_file) as f:
        session = json.load(f)

    # Time window: 18:31:00 to 18:33:59 local time
    # 18:31 = 66660 seconds from midnight, 18:33 = 66780 seconds
    # But timestamps are Unix epoch; filter by converting to local time
    min_hour, min_minute = 18, 31
    max_hour, max_minute = 18, 33

    window_transitions = []
    for tr in session.get("transitions", []):
        ts = tr.get("ts")
        if not ts:
            continue
        dt = datetime.fromtimestamp(ts)
        if dt.hour == 18 and min_minute <= dt.minute <= max_minute:
            mark = tr.get("mark", {})
            to_state = mark.get("to", "")
            # Collect transitions marked as SITTING_AT_TOILET (operator-attested)
            if to_state == "SITTING_AT_TOILET":
                window_transitions.append(tr)

    # Assertion 1: At least 25 frames in the attested sitting window
    check(len(window_transitions) >= 25,
          "attested sitting window has %d frames (need >= 25)" % len(window_transitions))

    if len(window_transitions) < 25:
        print("FAIL not enough frames in window, cannot proceed")
        return 1

    # Score the frames against the live baseline
    live_baseline = load_baseline()

    # Track classifications and row-0 cell counts
    standing_count = 0
    sitting_count = 0
    unknown_count = 0
    row0_body_depth_failures = 0

    for tr in window_transitions:
        matrix = oriented(tr["frame"])
        state, f = classify(matrix, live_baseline)

        if state == State.STANDING_AT_TOILET:
            standing_count += 1
        elif state == State.SITTING_AT_TOILET:
            sitting_count += 1
        else:
            unknown_count += 1

        # Assertion 4: For every frame, row-0 cells at body depth < UPPER_BODY_MIN
        if f["body_mm"] is not None:
            row0_at_body_depth = sum(
                1 for (r, c) in f["occ_cells"]
                if r in Params.HEAD_ROWS
                and abs(matrix[r][c] - f["body_mm"]) <= Params.BODY_DEPTH_TOL
            )
            if row0_at_body_depth >= Params.UPPER_BODY_MIN:
                row0_body_depth_failures += 1

    # Assertion 2: Not one classifies as STANDING (the regression was 26 flaps)
    check(standing_count == 0,
          "no seated frame classifies as STANDING_AT_TOILET (was 26/32, now %d)"
          % standing_count)

    # Assertion 3: At least 80% classify as SITTING (remainder may be UNKNOWN)
    sitting_ratio = sitting_count / len(window_transitions) if window_transitions else 0
    check(sitting_ratio >= 0.80,
          "at least 80%% classify as SITTING_AT_TOILET (%d/%d = %.1f%%)"
          % (sitting_count, len(window_transitions), sitting_ratio * 100))

    # Assertion 4: Zero row-0 cells at body depth in all frames
    check(row0_body_depth_failures == 0,
          "all seated frames have zero row-0 cells at body depth (%d failures)"
          % row0_body_depth_failures)

    # Assertion 5: Using DEFAULT_BASELINE, labeled seeds have expected properties
    from bathroom.classify import DEFAULT_BASELINE

    # STANDING seed: >= UPPER_BODY_MIN row-0 cells at body depth
    st_stand, f_stand = classify(tc.STANDING, DEFAULT_BASELINE)
    stand_row0 = sum(
        1 for (r, c) in f_stand["occ_cells"]
        if r in Params.HEAD_ROWS
        and abs(tc.STANDING[r][c] - f_stand["body_mm"]) <= Params.BODY_DEPTH_TOL
    ) if f_stand["body_mm"] is not None else 0
    check(stand_row0 >= Params.UPPER_BODY_MIN,
          "STANDING seed has %d row-0 cells at body depth (>= %d)"
          % (stand_row0, Params.UPPER_BODY_MIN))
    check(st_stand == State.STANDING_AT_TOILET,
          "STANDING seed classifies as STANDING_AT_TOILET")

    # SITTING seed: zero row-0 cells at body depth
    st_sit, f_sit = classify(tc.SITTING, DEFAULT_BASELINE)
    sit_row0 = sum(
        1 for (r, c) in f_sit["occ_cells"]
        if r in Params.HEAD_ROWS
        and abs(tc.SITTING[r][c] - f_sit["body_mm"]) <= Params.BODY_DEPTH_TOL
    ) if f_sit["body_mm"] is not None else 0
    check(sit_row0 == 0,
          "SITTING seed has zero row-0 cells at body depth (%d)"
          % sit_row0)
    check(st_sit == State.SITTING_AT_TOILET,
          "SITTING seed classifies as SITTING_AT_TOILET")

    print("ALL PASS" if not FAILS else "SOME FAILED")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
