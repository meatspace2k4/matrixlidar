#!/usr/bin/env python3
"""SitLatch regression: one play per visit, the latch cleared by the NEXT
ENTRY. No clock exists. Driven synthetically at 10 Hz.

  python3 tests/test_washlet.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bathroom.occupancy import RoomState                          # noqa: E402
from bathroom.washlet import (PLAY, QUEUE, REARM, STOP, UNQUEUE,  # noqa: E402
                              SitLatch)
from bathroom.zones import State                                  # noqa: E402

HZ = 10.0
SIT = State.SITTING_AT_TOILET
STAND = State.STANDING_AT_TOILET
WALK = State.WALKING
EMPTY = State.EMPTY
OCC = RoomState.OCCUPIED
VACANT = RoomState.EMPTY
UNK = RoomState.UNKNOWN

FAILS = []


def drive(latch, t, steps):
    actions = []
    for seconds, committed, room in steps:
        for _ in range(max(1, int(seconds * HZ))):
            a = latch.update(committed, room, t)
            if a is not None:
                actions.append((t, a))
            t += 1.0 / HZ
    return t, actions


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


def names(actions):
    return [a for _t, a in actions]


VISIT = [(1.0, WALK, OCC), (4.0, SIT, OCC), (1.0, WALK, OCC),
         (2.0, EMPTY, VACANT)]


def main():
    n = 0

    # 1. entry with no sit: entry sound on the edge, stopped on the exit
    l = SitLatch(hold=2.0)
    _, ev = drive(l, 0.0, [(2.0, WALK, OCC), (1.0, EMPTY, VACANT)])
    check(names(ev) == [QUEUE, UNQUEUE], "sit-less visit: QUEUE then UNQUEUE")
    n += 1

    # 2. sub-hold sit never reaches the main track (debounce)
    l = SitLatch(hold=2.0)
    _, ev = drive(l, 0.0, [(1.0, WALK, OCC), (1.5, SIT, OCC),
                           (1.0, WALK, OCC), (1.0, EMPTY, VACANT)])
    check(names(ev) == [QUEUE, UNQUEUE], "sit shorter than hold never plays")
    n += 1

    # 3. the happy path: QUEUE, PLAY at the hold boundary, STOP on vacancy
    l = SitLatch(hold=2.0)
    t, ev = drive(l, 0.0, VISIT)
    check(names(ev) == [QUEUE, PLAY, STOP],
          "entry queues; sit plays; vacancy stops the audio")
    check(l.latched, "vacancy does NOT clear the latch")
    n += 2

    # 4. stand/sit cycles inside one visit never replay
    l = SitLatch(hold=2.0)
    _, ev = drive(l, 0.0, [(1.0, WALK, OCC), (3.0, SIT, OCC),
                           (2.0, STAND, OCC), (5.0, SIT, OCC)])
    check(names(ev) == [QUEUE, PLAY], "one play per visit, cycles blocked")
    n += 1

    # 5. THE NEXT ENTRY REARMS: full visit, vacancy, then a new entry plays
    l = SitLatch(hold=2.0)
    t, ev = drive(l, 0.0, VISIT)
    t, ev2 = drive(l, t, [(60.0, EMPTY, VACANT)])
    check(names(ev2) == [], "the empty room between visits emits nothing")
    t, ev3 = drive(l, t, VISIT)
    check(names(ev3) == [REARM, PLAY, STOP],
          "next entry rearms (consuming its QUEUE edge) and plays again")
    n += 2

    # 6. the departing body cannot rearm its own visit: exit then re-enter
    #    immediately still needs a real OCCUPIED edge
    l = SitLatch(hold=2.0)
    t, ev = drive(l, 0.0, [(1.0, WALK, OCC), (3.0, SIT, OCC),
                           (1.0, EMPTY, OCC), (2.0, EMPTY, VACANT),
                           (1.0, WALK, OCC), (3.0, SIT, OCC)])
    check(names(ev) == [QUEUE, PLAY, STOP, REARM, PLAY],
          "exit, then a genuine new entry: rearm and play")
    n += 1

    # 7. room EMPTY misread under a still-seated body: no stop, no rearm
    l = SitLatch(hold=2.0)
    _, ev = drive(l, 0.0, [(1.0, WALK, OCC), (3.0, SIT, OCC),
                           (30.0, SIT, VACANT)])
    check(names(ev) == [QUEUE, PLAY],
          "room-EMPTY misread while seated never cuts the track")
    n += 1

    # 8. off-FOV vanish (posture EMPTY, room OCCUPIED) is not a vacancy
    l = SitLatch(hold=2.0)
    _, ev = drive(l, 0.0, [(1.0, WALK, OCC), (3.0, SIT, OCC),
                           (10.0, EMPTY, OCC), (3.0, SIT, OCC)])
    check(names(ev) == [QUEUE, PLAY], "off-FOV vanish holds everything")
    n += 1

    # 9. a gap (UNKNOWN room) while latched neither stops nor replays;
    #    the eventual real vacancy stops
    l = SitLatch(hold=2.0)
    _, ev = drive(l, 0.0, [(1.0, WALK, OCC), (3.0, SIT, OCC),
                           (6.0, EMPTY, UNK), (3.0, SIT, OCC),
                           (1.0, EMPTY, VACANT)])
    check(names(ev) == [QUEUE, PLAY, STOP],
          "a gap holds; the real vacancy stops the audio")
    n += 1

    # 10. daemon starting into an occupied room queues and plays
    l = SitLatch(hold=2.0)
    _, ev = drive(l, 0.0, [(1.0, STAND, OCC), (3.0, SIT, OCC)])
    check(names(ev) == [QUEUE, PLAY], "present-at-start body queues then plays")
    n += 1

    # 11. a restart (a fresh SitLatch) comes up rearmed
    l2 = SitLatch(hold=2.0)
    _, ev = drive(l2, 0.0, VISIT)
    check(names(ev) == [QUEUE, PLAY, STOP], "a fresh latch plays immediately")
    n += 1

    # 12. idle empty room emits nothing
    l = SitLatch(hold=2.0)
    _, ev = drive(l, 0.0, [(10.0, EMPTY, VACANT)])
    check(names(ev) == [], "idle empty room is silent")
    n += 1

    print("\n%d checks, %d failures" % (n, len(FAILS)))
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
