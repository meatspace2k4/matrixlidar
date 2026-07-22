#!/usr/bin/env python3
"""Occupancy engine confirmation tests: entry confirmation paths,
anti-latch decay, and edge cases for the T_CONFIRM and door-crossing logic.

  python3 tests/test_confirmation.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bathroom.occupancy import OccupancyEngine, RoomState, Params  # noqa: E402
from bathroom.zones import State                                   # noqa: E402

HZ = 10.0


class FakeUpdate:
    """Just enough of engine.Update for OccupancyEngine (.instant, .features)."""

    def __init__(self, col, mm=None):
        if col is None:
            self.instant = State.EMPTY
            self.features = {"occ_cells": [], "body_mm": None}
        else:
            self.instant = State.WALKING
            self.features = {"occ_cells": [(2, col), (3, col), (4, col)],
                             "body_mm": mm}


def _val(x, i, n):
    """Scalar or (start, end) ramp -> value at step i of n."""
    if isinstance(x, tuple):
        return x[0] + (x[1] - x[0]) * i / max(1, n - 1)
    return x


def drive(eng, t, steps):
    """steps: (seconds, col, mm[, door]) tuples; col/mm may be (start, end)
    ramps, col None = empty field. Returns (t, events seen)."""
    events = []
    for step in steps:
        seconds, col, mm = step[0], step[1], step[2]
        door = step[3] if len(step) > 3 else None
        n = max(1, int(seconds * HZ))
        for i in range(n):
            c = col if col is None else int(round(_val(col, i, n)))
            m = None if mm is None else int(round(_val(mm, i, n)))
            ru = eng.update(FakeUpdate(c, m), t, door_pct=door)
            if ru.event:
                events.append(ru.event)
            t += 1.0 / HZ
    return t, events


FAILS = []


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


def main():
    # Test 1: From UNKNOWN, ~7 second mid-field evidence (centroid col 2-3,
    # depth ~500-1000mm) followed by absence must NOT stay OCCUPIED. It reports
    # OCCUPIED during the episode, then decays back to UNKNOWN with
    # ENTRY_ABORTED. Must never emit ENTRY_CONFIRMED.
    e = OccupancyEngine()
    t, ev = drive(e, 0.0, [(7.0, 2.5, 700)])
    check(e.room == RoomState.OCCUPIED and "ENTRY_CONFIRMED" not in ev,
          "1a: mid-field 7s reports OCCUPIED but not confirmed")
    t, ev = drive(e, t, [(2.0, None, None)])
    check(e.room == RoomState.UNKNOWN and "ENTRY_ABORTED" in ev,
          "1b: after absence, mid-field episode aborts back to UNKNOWN")

    # Test 2: From EMPTY, the same short episode decays back to EMPTY with
    # ENTRY_ABORTED (not to UNKNOWN).
    e = OccupancyEngine()
    e.room = RoomState.EMPTY
    t, ev = drive(e, 0.0, [(7.0, 2.5, 700)])
    check(e.room == RoomState.OCCUPIED and "ENTRY_CONFIRMED" not in ev,
          "2a: mid-field 7s from EMPTY reports OCCUPIED but not confirmed")
    t, ev = drive(e, t, [(2.0, None, None)])
    check(e.room == RoomState.EMPTY and "ENTRY_ABORTED" in ev,
          "2b: after absence, mid-field episode from EMPTY aborts to EMPTY")

    # Test 3: From UNKNOWN, a door crossing (appear at door-side col >= DOOR_COL,
    # then penetrate to field col <= FIELD_COL) confirms OCCUPIED instantly with
    # ENTRY_CONFIRMED.
    e = OccupancyEngine()
    t, ev = drive(e, 0.0, [(0.4, 6, 1050), (0.8, (6, 3), (1050, 700))])
    check(e.room == RoomState.OCCUPIED and "ENTRY_CONFIRMED" in ev
          and e.occupants == 1,
          "3: door entry from UNKNOWN penetrates field -> OCCUPIED confirmed")

    # Test 4: From EMPTY, door crossing confirms OCCUPIED instantly (unchanged
    # behavior).
    e = OccupancyEngine()
    e.room = RoomState.EMPTY
    t, ev = drive(e, 0.0, [(0.4, 6, 1050), (0.8, (6, 3), (1050, 700))])
    check(e.room == RoomState.OCCUPIED and "ENTRY_CONFIRMED" in ev
          and e.occupants == 1,
          "4: door entry from EMPTY penetrates field -> OCCUPIED confirmed")

    # Test 5: From UNKNOWN, continuous mid-field evidence lasting longer than
    # Params.T_CONFIRM confirms OCCUPIED with ENTRY_CONFIRMED (boot-with-person
    # path: person present at startup, never crossed door, but continuous
    # evidence > T_CONFIRM confirms occupancy).
    e = OccupancyEngine()
    duration = Params.T_CONFIRM + 1.0
    t, ev = drive(e, 0.0, [(duration, 2, 700)])
    check(e.room == RoomState.OCCUPIED and "ENTRY_CONFIRMED" in ev
          and e.occupants == 1,
          "5: continuous mid-field > T_CONFIRM confirms OCCUPIED")

    # Test 6: Evidence shorter than Params.T_CONFIRM in a single continuous
    # episode does not confirm during the presence. After vanish and decay, it
    # aborts.
    e = OccupancyEngine()
    e.room = RoomState.EMPTY
    duration = Params.T_CONFIRM - 5.0
    t, ev = drive(e, 0.0, [(duration, 2, 700)])
    check("ENTRY_CONFIRMED" not in ev,
          "6a: mid-field < T_CONFIRM does not confirm during presence")
    t, ev = drive(e, t, [(2.0, None, None)])
    check("ENTRY_ABORTED" in ev,
          "6b: short mid-field episode aborts after vanish+decay")

    # Test 7: A confirmed OCCUPIED followed by a door-side vanish (trailing
    # centroid at/right of DOOR_COL with depth >= DOOR_PLANE_MM) exits to EMPTY.
    e = OccupancyEngine()
    e.room = RoomState.EMPTY
    t, _ = drive(e, 0.0, [(0.4, 6, 1050), (0.8, (6, 3), (1050, 700))])
    check(e.room == RoomState.OCCUPIED,
          "7a: door entry establishes OCCUPIED")
    # trajectory must actually reach the door band (smoothed centroid >=
    # DOOR_COL) at door-plane depth; a ramp topping out below FIELD_COL is a
    # mid-field vanish and correctly does NOT exit
    t, ev = drive(e, t, [(0.8, (3, 6), (700, 1050)), (1.5, None, None)])
    check(e.room == RoomState.EMPTY and "EXIT" in ev,
          "7b: confirmed OCCUPIED + door-side vanish (depth >= 900) -> EXIT")

    # Test 8: Gap over Params.T_GAP still resets to UNKNOWN with GAP_RESET
    # event, regardless of prior state.
    e = OccupancyEngine()
    e.room = RoomState.EMPTY
    t, _ = drive(e, 0.0, [(0.4, 6, 1050), (0.8, (6, 3), (1050, 700))])
    check(e.room == RoomState.OCCUPIED,
          "8a: door entry confirms OCCUPIED")
    # probe the gap with an EMPTY frame: evidence on the gap frame would
    # (by design) immediately re-report maybe-OCCUPIED from UNKNOWN
    ru = e.update(FakeUpdate(None), t + Params.T_GAP + 1.0)
    check(ru.room == RoomState.UNKNOWN and ru.event == "GAP_RESET",
          "8b: gap > T_GAP resets to UNKNOWN with GAP_RESET")

    print("ALL PASS" if not FAILS else "SOME FAILED")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
