#!/usr/bin/env python3
"""Occupancy engine regression: the scenarios from docs/occupancy-design.md
section 9, driven synthetically at 10 Hz.

  python3 tests/test_occupancy.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bathroom.occupancy import OccupancyEngine, RoomState   # noqa: E402
from bathroom.zones import State                            # noqa: E402

HZ = 10.0


class FakeUpdate:
    """Just enough of engine.Update for OccupancyEngine (.instant, .features)."""

    def __init__(self, col):
        if col is None:
            self.instant = State.EMPTY
            self.features = {"occ_cells": []}
        else:
            self.instant = State.WALKING
            self.features = {"occ_cells": [(2, col), (3, col), (4, col)]}


def drive(eng, t, steps):
    """steps: (seconds, col_or_None) pairs; col may be a (start, end) ramp.
    Returns (t, events seen)."""
    events = []
    for seconds, col in steps:
        n = max(1, int(seconds * HZ))
        for i in range(n):
            if isinstance(col, tuple):
                c = int(round(col[0] + (col[1] - col[0]) * i / max(1, n - 1)))
            else:
                c = col
            ru = eng.update(FakeUpdate(c), t)
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
    # boot into a quiet room: UNKNOWN, never EMPTY by timeout
    e = OccupancyEngine()
    _, ev = drive(e, 0.0, [(4.0, None)])
    check(e.room == RoomState.UNKNOWN and not ev, "quiet boot stays UNKNOWN")

    # boot with a person mid-room: OCCUPIED after T_CONFIRM; door exit empties
    e = OccupancyEngine()
    t, ev = drive(e, 0.0, [(3.0, 3)])
    check(e.room == RoomState.OCCUPIED and e.occupants == 1
          and "ENTRY_CONFIRMED" in ev, "boot with person -> OCCUPIED")
    t, ev = drive(e, t, [(1.0, (3, 6)), (1.5, None)])
    check(e.room == RoomState.EMPTY and e.occupants == 0 and "EXIT" in ev,
          "walk to door + vanish -> EMPTY")

    # normal cycle: enter at door, penetrate, shower (vanish left), return, exit
    e = OccupancyEngine()
    e.room = RoomState.EMPTY
    t, ev = drive(e, 0.0, [(0.4, 6), (0.8, (6, 3))])
    check(e.room == RoomState.OCCUPIED and "ENTRY_CONFIRMED" in ev,
          "door entry penetrates -> OCCUPIED confirmed")
    t, ev = drive(e, t, [(0.5, (3, 0)), (3.0, None)])
    check(e.room == RoomState.OCCUPIED and not ev,
          "vanish left (shower) keeps OCCUPIED")
    t, ev = drive(e, t, [(0.5, 2), (1.0, (2, 6)), (1.5, None)])
    check(e.room == RoomState.EMPTY and "EXIT" in ev,
          "reappear + door vanish -> EMPTY")

    # step-in step-out: reported OCCUPIED while present, exits to EMPTY
    e = OccupancyEngine()
    e.room = RoomState.EMPTY
    t, ev = drive(e, 0.0, [(0.4, 6)])
    check(e.room == RoomState.OCCUPIED, "evidence reports OCCUPIED instantly")
    t, ev = drive(e, t, [(1.5, None)])
    check(e.room == RoomState.EMPTY and "EXIT" in ev,
          "step-in-step-out -> EMPTY via exit")

    # anti-latch: brief mid-field blob that vanishes off-door aborts to EMPTY
    e = OccupancyEngine()
    e.room = RoomState.EMPTY
    t, ev = drive(e, 0.0, [(0.5, 3), (2.0, None)])
    check(e.room == RoomState.EMPTY and "ENTRY_ABORTED" in ev,
          "unconfirmed mid vanish decays -> EMPTY (anti-latch)")

    # second entry while occupied: +1; any exit still empties the room
    e = OccupancyEngine()
    e.room = RoomState.EMPTY
    t, _ = drive(e, 0.0, [(0.4, 6), (0.8, (6, 3)), (0.5, (3, 0)), (2.0, None)])
    t, ev = drive(e, t, [(0.4, 6), (0.8, (6, 3))])
    check(e.room == RoomState.OCCUPIED and e.occupants == 2,
          "second door entry -> occupants 2")
    t, ev = drive(e, t, [(1.0, (3, 6)), (1.5, None)])
    check(e.room == RoomState.EMPTY and e.occupants == 0 and "EXIT" in ev,
          "any exit -> EMPTY (delta rule)")

    # stream gap: forces UNKNOWN
    e = OccupancyEngine()
    e.room = RoomState.EMPTY
    t, _ = drive(e, 0.0, [(0.4, 6), (0.8, (6, 3))])
    ru = e.update(FakeUpdate(3), t + 10.0)
    check(ru.room == RoomState.UNKNOWN and ru.event == "GAP_RESET",
          "stream gap -> UNKNOWN")

    print("ALL PASS" if not FAILS else "SOME FAILED")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
