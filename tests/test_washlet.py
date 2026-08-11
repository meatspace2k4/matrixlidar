#!/usr/bin/env python3
"""SitLatch regression: one play per visit, fired by the ACT OF SITTING DOWN
(committed STANDING_AT_TOILET -> SITTING_AT_TOILET), latch cleared by the
NEXT ENTRY. No clock and no dwell. Driven synthetically at 10 Hz.

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
UNKNOWN = State.UNKNOWN
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


# a whole visit: walk in, stand at the toilet, sit, get up, leave
VISIT = [(1.0, WALK, OCC), (1.0, STAND, OCC), (4.0, SIT, OCC),
         (1.0, WALK, OCC), (2.0, EMPTY, VACANT)]


def main():
    n = 0

    # 1. entry with no sit: entry sound on the edge, stopped on the exit
    l = SitLatch()
    _, ev = drive(l, 0.0, [(2.0, WALK, OCC), (1.0, EMPTY, VACANT)])
    check(names(ev) == [QUEUE, UNQUEUE], "sit-less visit: QUEUE then UNQUEUE")
    n += 1

    # 2. THE TRIGGER IS THE TRANSITION, and it fires immediately
    l = SitLatch()
    t, ev = drive(l, 0.0, [(1.0, WALK, OCC), (1.0, STAND, OCC), (4.0, SIT, OCC)])
    check(names(ev) == [QUEUE, PLAY], "standing then sitting plays")
    check(abs(ev[1][0] - 2.0) < 0.2,
          "and it plays AS the body sits, with no dwell")
    n += 2

    # 3. sitting WITHOUT having stood at the toilet is not a sit-down: a
    #    sitting-shaped reading that never came from standing never fires
    l = SitLatch()
    _, ev = drive(l, 0.0, [(1.0, WALK, OCC), (60.0, SIT, OCC)])
    check(names(ev) == [QUEUE], "sitting-shaped with no stand before it: no play")
    n += 1

    # 4. walking straight to sitting is likewise not it: the body was never
    #    committed as standing AT THE TOILET
    l = SitLatch()
    _, ev = drive(l, 0.0, [(1.0, WALK, OCC), (30.0, SIT, OCC)])
    check(names(ev) == [QUEUE], "walking -> sitting alone does not play")
    n += 1

    # 5. UNKNOWN between the two is the same event: the classifier is least
    #    sure exactly while the body is mid-motion
    l = SitLatch()
    _, ev = drive(l, 0.0, [(1.0, WALK, OCC), (1.0, STAND, OCC),
                           (0.6, UNKNOWN, OCC), (3.0, SIT, OCC)])
    check(names(ev) == [QUEUE, PLAY],
          "STANDING -> UNKNOWN -> SITTING still plays")
    n += 1

    # 6. one play per visit: stand/sit cycles inside a visit never replay
    l = SitLatch()
    _, ev = drive(l, 0.0, [(1.0, WALK, OCC), (1.0, STAND, OCC), (3.0, SIT, OCC),
                           (2.0, STAND, OCC), (5.0, SIT, OCC),
                           (2.0, STAND, OCC), (5.0, SIT, OCC)])
    check(names(ev) == [QUEUE, PLAY], "later sit-downs in the same visit are silent")
    n += 1

    # 7. the full visit: play, then vacancy stops the audio, latch held
    l = SitLatch()
    t, ev = drive(l, 0.0, VISIT)
    check(names(ev) == [QUEUE, PLAY, STOP],
          "visit: queue, play on sitting, stop on vacancy")
    check(l.latched, "vacancy does NOT clear the latch")
    n += 2

    # 8. the next entry rearms, and the next sit-down plays again
    t, ev2 = drive(l, t, [(60.0, EMPTY, VACANT)])
    check(names(ev2) == [], "the empty room between visits emits nothing")
    _, ev3 = drive(l, t, VISIT)
    check(names(ev3) == [REARM, PLAY, STOP],
          "next entry rearms (consuming its QUEUE edge) and plays again")
    n += 2

    # 9. room EMPTY misread under a still-seated body: no stop
    l = SitLatch()
    _, ev = drive(l, 0.0, [(1.0, WALK, OCC), (1.0, STAND, OCC), (3.0, SIT, OCC),
                           (30.0, SIT, VACANT)])
    check(names(ev) == [QUEUE, PLAY],
          "room-EMPTY misread while seated never cuts the track")
    n += 1

    # 10. off-FOV vanish (posture EMPTY, room OCCUPIED) is not a vacancy
    l = SitLatch()
    _, ev = drive(l, 0.0, [(1.0, WALK, OCC), (1.0, STAND, OCC), (3.0, SIT, OCC),
                           (10.0, EMPTY, OCC), (3.0, SIT, OCC)])
    check(names(ev) == [QUEUE, PLAY], "off-FOV vanish holds everything")
    n += 1

    # 11. a stream gap (UNKNOWN room) neither stops nor rearms mid-visit
    l = SitLatch()
    _, ev = drive(l, 0.0, [(1.0, WALK, OCC), (1.0, STAND, OCC), (3.0, SIT, OCC),
                           (6.0, EMPTY, UNK), (3.0, SIT, OCC),
                           (1.0, EMPTY, VACANT)])
    check(names(ev) == [QUEUE, PLAY, STOP],
          "a gap holds; the real vacancy stops the audio")
    n += 1

    # 12. daemon starting into an ALREADY SEATED body: no sit-down was
    #     observed, so no play. Stated cost of the transition rule.
    l = SitLatch()
    _, ev = drive(l, 0.0, [(60.0, SIT, OCC)])
    check(names(ev) == [QUEUE],
          "a body already seated at startup queues but does not play")
    n += 1

    # 13. ... and it plays the moment they actually stand and sit again
    _, ev = drive(l, 60.0, [(2.0, STAND, OCC), (3.0, SIT, OCC)])
    check(names(ev) == [PLAY], "the next real sit-down plays")
    n += 1

    # 14. a restart (a fresh SitLatch) comes up rearmed
    l2 = SitLatch()
    _, ev = drive(l2, 0.0, VISIT)
    check(names(ev) == [QUEUE, PLAY, STOP], "a fresh latch plays immediately")
    n += 1

    # 15. idle empty room emits nothing
    l = SitLatch()
    _, ev = drive(l, 0.0, [(10.0, EMPTY, VACANT)])
    check(names(ev) == [], "idle empty room is silent")
    n += 1

    print("\n%d checks, %d failures" % (n, len(FAILS)))
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
