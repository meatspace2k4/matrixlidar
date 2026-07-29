#!/usr/bin/env python3
"""Entry, exit and occupancy, gated on the recorded captures.

This is where entry and exit are pinned, because it is the only data that
contains a crossing: continuous frame sequences with the operator's own note on
each. It replaces the entry/exit assertions that used to live in
test_scene_engine.py, test_occlusion.py and test_replay_sessions.py, which
pinned the model that inferred a crossing from the DOOR (doorway-band cells,
then interior cells, then ENTERED) and fed it data with no crossing in it:
hand-edited baseline cells in two of them, and archived transition snapshots
held constant in the third.

  python3 tests/test_crossing.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from bathroom import config, detect                            # noqa: E402
from bathroom.occupancy import OccupancyEngine, RoomState       # noqa: E402
from bathroom.zones import N                                    # noqa: E402

# (clip, operator note, expected crossings, expected final room)
CASES = [
    ("walking_1784952866", "in through the door, out again, twice",
     ["ENTRY_CONFIRMED", "EXIT", "ENTRY_CONFIRMED", "EXIT"], RoomState.EMPTY),
    ("sitting_1785074559", "walk in, sit 20s, leave",
     ["ENTRY_CONFIRMED"], RoomState.OCCUPIED),      # the capture ends seated
    # These two open with the operator already at the toilet, so no crossing
    # is ever observed and the room stays UNKNOWN. That is the model: the
    # crossing is the only occupancy evidence, in both directions. The field
    # briefly latched these OCCUPIED (the PRESENCE event), and the same rule
    # latched an EMPTY room off a 1.0s posture flicker with no recovery path
    # (07-29 11:29, held for hours). The body here still gets its audio: the
    # washlet sit trigger runs on posture, not room state.
    ("sitting_1784950347", "upright, back against tank",
     [], RoomState.UNKNOWN),
    ("standing_1784950390", "standing at the toilet, not the mirror",
     [], RoomState.UNKNOWN),
]

FAILS = []


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


def main():
    base = config.load_baseline()
    for clip, note, want_events, want_room in CASES:
        path = os.path.join(ROOT, "data", "clips", clip + ".json")
        if not os.path.isfile(path):
            check(False, "%s: capture missing" % clip)
            continue
        with open(path) as f:
            doc = json.load(f)
        frames = [[fr["mm"][i * N:(i + 1) * N] for i in range(N)]
                  for fr in doc["frames"]]
        times = [fr["t"] for fr in doc["frames"]]

        eng = OccupancyEngine(baseline=base)
        door = detect.DoorReading(base)
        got = []
        for m, t in zip(frames, times):
            ru = eng.update(m, t, door_pct=door.update(m))
            if ru.event in ("ENTRY_CONFIRMED", "EXIT"):
                got.append(ru.event)

        check(got == want_events,
              "%-22s %-42r  crossings %s" % (clip, note, got))
        room = eng.room if isinstance(eng.room, str) else eng.room.name
        check(room == str(want_room),
              "%-22s %-42s  ends %s" % (clip, "", room))

    print("ALL PASS" if not FAILS else "SOME FAILED")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
