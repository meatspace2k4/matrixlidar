#!/usr/bin/env python3
"""Occupancy engine regression: the design-doc scenarios plus the
depth/door-motion failure modes observed in real sessions, driven
synthetically at 10 Hz.

  python3 tests/test_occupancy.py
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
    # boot into a quiet room: UNKNOWN, never EMPTY by timeout
    e = OccupancyEngine()
    _, ev = drive(e, 0.0, [(4.0, None, None)])
    check(e.room == RoomState.UNKNOWN and not ev, "quiet boot stays UNKNOWN")

    # boot with a person mid-room: OCCUPIED only after sustained cumulative
    # evidence (T_CONFIRM; no door crossing available); door exit empties
    e = OccupancyEngine()
    t, ev = drive(e, 0.0, [(Params.T_CONFIRM + 1.0, 3, 1200)])
    check(e.room == RoomState.OCCUPIED and e.occupants == 1
          and "ENTRY_CONFIRMED" in ev, "boot with person -> OCCUPIED (sustained)")
    t, ev = drive(e, t, [(1.0, (3, 6), (1200, 1050)), (1.5, None, None)])
    check(e.room == RoomState.EMPTY and e.occupants == 0 and "EXIT" in ev,
          "walk to door (receding depth) + vanish -> EMPTY")

    # normal cycle: enter at door, penetrate, shower (vanish left), return, exit
    e = OccupancyEngine()
    e.room = RoomState.EMPTY
    t, ev = drive(e, 0.0, [(0.4, 6, 1050), (0.8, (6, 3), (1050, 700))])
    check(e.room == RoomState.OCCUPIED and "ENTRY_CONFIRMED" in ev,
          "door entry penetrates -> OCCUPIED confirmed")
    t, ev = drive(e, t, [(0.5, (3, 0), 700), (3.0, None, None)])
    check(e.room == RoomState.OCCUPIED and not ev,
          "vanish left (shower) keeps OCCUPIED")
    t, ev = drive(e, t, [(0.5, 2, 800), (1.0, (2, 6), (800, 1100)),
                         (1.5, None, None)])
    check(e.room == RoomState.EMPTY and "EXIT" in ev,
          "reappear + door vanish -> EMPTY")

    # step-in step-out: reported OCCUPIED while present, exits to EMPTY
    e = OccupancyEngine()
    e.room = RoomState.EMPTY
    t, ev = drive(e, 0.0, [(0.4, 6, 1050)])
    check(e.room == RoomState.OCCUPIED, "evidence reports OCCUPIED instantly")
    t, ev = drive(e, t, [(1.5, None, None)])
    check(e.room == RoomState.EMPTY and "EXIT" in ev,
          "step-in-step-out -> EMPTY via exit")

    # anti-latch: brief mid-field blob that vanishes off-door aborts to EMPTY
    e = OccupancyEngine()
    e.room = RoomState.EMPTY
    t, ev = drive(e, 0.0, [(0.5, 3, 700), (2.0, None, None)])
    check(e.room == RoomState.EMPTY and "ENTRY_ABORTED" in ev,
          "unconfirmed mid vanish decays -> EMPTY (anti-latch)")

    # second entry while occupied: +1; any exit still empties the room
    e = OccupancyEngine()
    e.room = RoomState.EMPTY
    t, _ = drive(e, 0.0, [(0.4, 6, 1050), (0.8, (6, 3), (1050, 700)),
                          (0.5, (3, 0), 700), (2.0, None, None)])
    t, ev = drive(e, t, [(0.4, 6, 1050), (0.8, (6, 3), (1050, 700))])
    check(e.room == RoomState.OCCUPIED and e.occupants == 2,
          "second door entry -> occupants 2")
    t, ev = drive(e, t, [(1.0, (3, 6), (700, 1050)), (1.5, None, None)])
    check(e.room == RoomState.EMPTY and e.occupants == 0 and "EXIT" in ev,
          "any exit -> EMPTY (delta rule)")

    # stream gap: forces UNKNOWN
    e = OccupancyEngine()
    e.room = RoomState.EMPTY
    t, _ = drive(e, 0.0, [(0.4, 6, 1050), (0.8, (6, 3), (1050, 700))])
    ru = e.update(FakeUpdate(3, 700), t + 10.0)
    check(ru.room == RoomState.UNKNOWN and ru.event == "GAP_RESET",
          "stream gap -> UNKNOWN")

    # DEPTH: near-lens crossing smears to the door COLUMN at ~450mm and
    # vanishes; column-only logic called this an exit (real failure,
    # session 5). Depth must reject it: room stays OCCUPIED, no EXIT.
    e, t = _occupied_engine()
    t, ev = drive(e, t, [(0.5, (1, 4), 450), (0.3, (4, 6), 450),
                         (1.5, None, None)])
    check(e.room == RoomState.OCCUPIED and "EXIT" not in ev,
          "near-lens flip to door column is NOT an exit (depth)")

    # DEPTH: same trajectory at door-plane depth IS an exit
    e, t = _occupied_engine()
    t, ev = drive(e, t, [(0.5, (1, 4), (700, 900)), (0.3, (4, 6), (900, 1100)),
                         (1.5, None, None)])
    check(e.room == RoomState.EMPTY and "EXIT" in ev,
          "same trajectory receded to door plane IS an exit")

    # DOOR MOTION: body vanishes mid-field (missed exit), then the door
    # open% changes with nobody visible and holds -> EXIT after stillness
    # (real failures: exit behind closing door / door left ajar).
    e, t = _occupied_engine()
    t, ev = drive(e, t, [(0.5, (3, 4), 500),          # drifts, still visible
                         (1.0, None, None, 0.0),      # vanish mid; door 0%
                         (0.5, None, None, 60.0),     # door moved to 60% ajar
                         (2.5, None, None, 60.0)])    # stillness
    check(e.room == RoomState.EMPTY and "EXIT" in ev,
          "door moved while empty + stillness -> EXIT (door-motion rule)")

    # DOOR MOTION guard: shower vanish with the door NEVER moving stays
    # OCCUPIED indefinitely (no timeout exit).
    e, t = _occupied_engine()
    t, ev = drive(e, t, [(0.5, (3, 0), 700),
                         (1.0, None, None, 30.0),
                         (6.0, None, None, 30.0)])
    check(e.room == RoomState.OCCUPIED and "EXIT" not in ev,
          "shower vanish, door untouched -> stays OCCUPIED")

    print("ALL PASS" if not FAILS else "SOME FAILED")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
