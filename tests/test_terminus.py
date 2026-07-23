#!/usr/bin/env python3
"""Occupancy engine terminus classification: the six exit-path scenarios with
corroboration and noise guards, driven synthetically at 10 Hz.

  python3 tests/test_terminus.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bathroom.occupancy import OccupancyEngine, Params, RoomState  # noqa: E402
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


def _occupied_engine():
    """An engine in confirmed-OCCUPIED via a clean door entry."""
    e = OccupancyEngine()
    e.room = RoomState.EMPTY
    t, _ = drive(e, 0.0, [(0.4, 6, 1050), (0.8, (6, 3), (1050, 700))])
    assert e.room == RoomState.OCCUPIED
    return e, t


def main():
    # 1. DEEP terminus: receded to door plane at door column, vanish with NO
    # door signal -> EXIT, EMPTY, occupants 0. Door is not consulted at all.
    e, t = _occupied_engine()
    # depth smoothing lags like column smoothing: the ramp must exceed the
    # door plane enough that the TRAILING average clears DOOR_PLANE_MM
    t, ev = drive(e, t, [(0.6, (4, 6), (900, 1200)),  # doorward ramp past door plane
                         (Params.T_ABSENT + 0.1, None, None)])  # vanish, no door
    check(e.room == RoomState.EMPTY and e.occupants == 0 and "EXIT" in ev,
          "DEEP terminus (door plane depth + door column) -> EXIT")

    # 2. NEAR terminus without corroboration: close pass (500mm) at door column,
    # vanish, then long emptiness with door constant -> no EXIT, stays OCCUPIED.
    e, t = _occupied_engine()
    t, ev = drive(e, t, [(0.6, (4, 6), 500),           # doorward to 500mm (near)
                         (Params.T_ABSENT + 0.1, None, None, 30.0),  # vanish, door 30%
                         (3.0, None, None, 30.0)])     # long quiet, door unchanged
    check(e.room == RoomState.OCCUPIED and "EXIT" not in ev,
          "NEAR terminus no door change -> no EXIT, stays OCCUPIED")

    # 3. NEAR terminus with corroboration: close pass, vanish with door state,
    # then door changes by >= DOOR_MOVE_PCT, held for DOOR_MOVE_N frames and
    # T_DOOR_STILL -> EXIT, EMPTY, occupants 0.
    e, t = _occupied_engine()
    door_start = 30.0
    door_end = door_start + Params.DOOR_MOVE_PCT + 5.0  # ensure >DOOR_MOVE_PCT change
    door_hold = max(Params.DOOR_MOVE_N / HZ + 0.1, 0.4)  # hold for >N frames
    t, ev = drive(e, t, [(0.6, (4, 6), 500),           # doorward to 500mm (near)
                         (Params.T_ABSENT + 0.1, None, None, door_start),  # vanish
                         (door_hold, None, None, door_end),  # door moves & holds
                         (Params.T_DOOR_STILL + 0.1, None, None, door_end)])  # stillness
    check(e.room == RoomState.EMPTY and e.occupants == 0 and "EXIT" in ev,
          "NEAR terminus + door corroboration -> EXIT")

    # 4. OFF terminus + door change: vanish leftward (off-side), then large
    # held door change -> no EXIT, stays OCCUPIED (hallway protection).
    e, t = _occupied_engine()
    t, ev = drive(e, t, [(0.5, (3, 0), 700),           # vanish left (off-side)
                         (Params.T_ABSENT + 0.1, None, None, 30.0),  # vanish
                         (0.5, None, None, 90.0),      # door operated
                         (2.0, None, None, 90.0)])     # held change
    check(e.room == RoomState.OCCUPIED and "EXIT" not in ev,
          "OFF terminus + door change is ignored (hallway protection)")

    # 5. Reappearance clears pending: NEAR vanish, body returns before door
    # change, then vanishes left, then door change -> no EXIT, stays OCCUPIED.
    e, t = _occupied_engine()
    t, ev = drive(e, t, [(0.6, (4, 6), 500),           # doorward to 500mm (near)
                         (Params.T_ABSENT + 0.1, None, None, 30.0),  # vanish
                         (0.2, 3, 700)])               # body reappears before door
    check(e.room == RoomState.OCCUPIED and "ENTRY_CONFIRMED" not in ev,
          "reappearance during NEAR pending clears door pending")
    t, ev = drive(e, t, [(0.5, (3, 0), 700),           # vanish left
                         (Params.T_ABSENT + 0.1, None, None, 30.0),  # vanish off-side
                         (0.5, None, None, 90.0),      # door change
                         (2.0, None, None, 90.0)])
    check(e.room == RoomState.OCCUPIED and "EXIT" not in ev,
          "reappearance clears NEAR pending, later off-side vanish + door ignored")

    # 6. Noise guard: NEAR terminus vanish, door flaps between two values but
    # never holds changed value for DOOR_MOVE_N consecutive frames -> no EXIT.
    e, t = _occupied_engine()
    t, ev = drive(e, t, [(0.6, (4, 6), 500),           # doorward to 500mm (near)
                         (Params.T_ABSENT + 0.1, None, None, 30.0)])  # vanish at 30%
    # Flap door: change by >DOOR_MOVE_PCT but don't hold it for N consecutive frames
    frame_dur = 1.0 / HZ
    for _ in range(2):
        # Move door, hold for N-1 frames (not enough)
        t, ev_part = drive(e, t, [((Params.DOOR_MOVE_N - 1) * frame_dur - 0.01,
                                    None, None, 60.0)])
        ev.extend(ev_part)
        # Revert door before completing the hold
        t, ev_part = drive(e, t, [(frame_dur + 0.01, None, None, 30.0)])
        ev.extend(ev_part)
    check(e.room == RoomState.OCCUPIED and "EXIT" not in ev,
          "NEAR terminus with door flapping (no sustained hold) -> no EXIT")

    print("ALL PASS" if not FAILS else "SOME FAILED")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
