#!/usr/bin/env python3
"""SitLatch regression: the sit trigger the HomePod is driven from, now latched
for the whole DAY rather than the visit. Driven synthetically at 10 Hz, with the
wall clock advanced in lockstep so the 05:00 rollover is a real crossing.

  python3 tests/test_washlet.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bathroom.occupancy import RoomState                          # noqa: E402
from bathroom.washlet import (DAY_START_H, PLAY, QUEUE, REARM,    # noqa: E402
                              STOP, UNQUEUE, SitLatch,
                              _service_day)
from bathroom.zones import State                                  # noqa: E402

HZ = 10.0

SIT = State.SITTING_AT_TOILET
STAND = State.STANDING_AT_TOILET
WALK = State.WALKING
EMPTY = State.EMPTY

OCC = RoomState.OCCUPIED
VACANT = RoomState.EMPTY
UNK = RoomState.UNKNOWN


def local(y, mo, d, h, mi=0, s=0):
    """Epoch seconds for a LOCAL wall time, whatever this machine's zone is.

    The rollover is defined in local time, so the fixtures have to be too;
    hard-coded epochs would pass in Denver and fail in UTC.
    """
    return time.mktime((y, mo, d, h, mi, s, 0, 0, -1))


DAY1_9AM = local(2026, 7, 27, 9, 0, 0)


def drive(latch, t, wall, steps):
    """steps: (seconds, committed_state, room_state) tuples, played at 10 Hz.

    Returns (t, wall, [(t, action), ...]) for every non-None action.
    """
    actions = []
    for seconds, committed, room in steps:
        for _ in range(max(1, int(seconds * HZ))):
            a = latch.update(committed, room, t, wall=wall)
            if a is not None:
                actions.append((t, a))
            t += 1.0 / HZ
            wall += 1.0 / HZ
    return t, wall, actions


FAILS = []


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

    # --- the day window itself ---------------------------------------------
    # 04:59 belongs to the PREVIOUS day; 05:00 starts the new one.
    check(_service_day(local(2026, 7, 27, 4, 59, 59))
          == _service_day(local(2026, 7, 26, 12, 0, 0)),
          "04:59 still belongs to the previous day's window")
    check(_service_day(local(2026, 7, 27, 5, 0, 1))
          != _service_day(local(2026, 7, 27, 4, 59, 59)),
          "05:00 opens a new window")
    check(_service_day(local(2026, 7, 27, 5, 0, 1))
          == _service_day(local(2026, 7, 28, 4, 0, 0)),
          "the window runs 05:00 to 05:00, not midnight to midnight")
    n += 3

    # --- unchanged within a single visit ------------------------------------
    # 1. entry with no sit: entry sound on the edge, stopped when it evaporates
    l = SitLatch(hold=2.0)
    _, _, ev = drive(l, 0.0, DAY1_9AM,
                     [(2.0, WALK, OCC), (1.0, EMPTY, VACANT)])
    check(names(ev) == [QUEUE, UNQUEUE], "sit-less visit: QUEUE then UNQUEUE")
    n += 1

    # 2. sub-hold sit never reaches the main track (debounce)
    l = SitLatch(hold=2.0)
    _, _, ev = drive(l, 0.0, DAY1_9AM,
                     [(1.0, WALK, OCC), (1.5, SIT, OCC),
                      (1.0, WALK, OCC), (1.0, EMPTY, VACANT)])
    check(names(ev) == [QUEUE, UNQUEUE], "sit shorter than hold never plays")
    check(not l.latched, "a sit that never played leaves the day unspent")
    n += 2

    # 3. the happy path: entry sound, then the main track at the hold boundary
    l = SitLatch(hold=2.0)
    _, _, ev = drive(l, 0.0, DAY1_9AM, [(1.0, WALK, OCC), (60.0, SIT, OCC)])
    check(names(ev) == [QUEUE, PLAY], "entry queues; sustained sit plays once")
    check(abs(ev[1][0] - 3.0) < 0.2, "play lands at the hold boundary (~2s in)")
    n += 2

    # 4. the dwell DRAINS through jitter rather than resetting: a real sit that
    #    flickers out of SITTING still gets there, just later
    l = SitLatch(hold=2.0)
    _, _, ev = drive(l, 0.0, DAY1_9AM,
                     [(1.0, WALK, OCC), (1.5, SIT, OCC),
                      (0.5, STAND, OCC), (1.5, SIT, OCC)])
    check(names(ev) == [QUEUE, PLAY], "jitter drains the dwell, it does not "
                                      "reset it: the real sit still plays")
    n += 1

    # --- once per DAY, not once per visit -----------------------------------
    # 5. the played visit ends: audio stops, but the latch does NOT clear
    l = SitLatch(hold=2.0)
    t, w, ev = drive(l, 0.0, DAY1_9AM, VISIT)
    check(names(ev) == [QUEUE, PLAY, STOP],
          "the played visit ends with STOP (audio off), not REARM")
    check(l.latched, "leaving the bathroom does NOT clear the latch")
    n += 2

    # 6. ... so every later visit that same day is silent, sit or no sit
    _, w2, ev = drive(l, t, w, [(600.0, EMPTY, VACANT)] + VISIT + VISIT)
    check(names(ev) == [], "later visits the same day are silent: no QUEUE, "
                           "no second PLAY, no STOP")
    n += 1

    # 7. 05:00 the next morning clears it, and the first sit after that plays
    w3 = local(2026, 7, 28, 4, 59, 55)     # just before the boundary
    t2, w3, ev = drive(l, t + 1.0, w3, [(3.0, EMPTY, VACANT)])
    check(names(ev) == [], "still latched at 04:59")
    t2, w3, ev = drive(l, t2, w3, [(10.0, EMPTY, VACANT)])
    check(names(ev) == [REARM], "05:00 fires exactly one REARM")
    check(not l.latched, "the latch is clear after the rollover")
    _, _, ev = drive(l, t2, w3, VISIT)
    check(names(ev) == [QUEUE, PLAY, STOP],
          "the first sit of the new day plays again")
    n += 4

    # 8. a rollover landing MID-SIT must not cut the track; it rearms, and the
    #    body already past the hold plays on the very next frame
    l = SitLatch(hold=2.0)
    t, w, _ = drive(l, 0.0, local(2026, 7, 27, 9, 0, 0), VISIT)
    ev = drive(l, t, local(2026, 7, 28, 4, 59, 50),
               [(20.0, SIT, OCC)])[2]
    check(names(ev) == [REARM, PLAY],
          "rollover mid-sit rearms without a STOP, then plays")
    check(ev[1][0] - ev[0][0] < 0.2, "the waiting sit plays on the next frame")
    n += 2

    # 9. STATELESS BY DESIGN: a restart (a fresh SitLatch, which is all a
    #    restart is) comes up clear and plays again the same day
    l2 = SitLatch(hold=2.0)
    _, _, ev = drive(l2, 0.0, DAY1_9AM + 3600, VISIT)
    check(names(ev) == [QUEUE, PLAY, STOP],
          "a restart rearms: nothing is persisted, on purpose")
    n += 1

    # --- the exit guards, unchanged -----------------------------------------
    # 10. room EMPTY misread under a still-seated body: no STOP mid-track
    l = SitLatch(hold=2.0)
    _, _, ev = drive(l, 0.0, DAY1_9AM,
                     [(1.0, WALK, OCC), (3.0, SIT, OCC), (30.0, SIT, VACANT)])
    check(names(ev) == [QUEUE, PLAY],
          "room-EMPTY misread while seated never cuts the track")
    n += 1

    # 11. off-FOV vanish (posture EMPTY, room OCCUPIED) is not an exit
    l = SitLatch(hold=2.0)
    _, _, ev = drive(l, 0.0, DAY1_9AM,
                     [(1.0, WALK, OCC), (3.0, SIT, OCC),
                      (10.0, EMPTY, OCC), (3.0, SIT, OCC)])
    check(names(ev) == [QUEUE, PLAY],
          "off-FOV vanish (room still OCCUPIED) is not an exit")
    n += 1

    # 12. a stream gap while the track plays neither stops it nor replays it
    l = SitLatch(hold=2.0)
    _, _, ev = drive(l, 0.0, DAY1_9AM,
                     [(1.0, WALK, OCC), (3.0, SIT, OCC),
                      (6.0, EMPTY, UNK), (3.0, SIT, OCC),
                      (1.0, EMPTY, VACANT)])
    check(names(ev) == [QUEUE, PLAY, STOP],
          "a gap holds; the real exit stops the audio")
    n += 1

    # 13. daemon starting into an occupied room queues immediately
    l = SitLatch(hold=2.0)
    _, _, ev = drive(l, 0.0, DAY1_9AM, [(1.0, STAND, OCC), (3.0, SIT, OCC)])
    check(names(ev) == [QUEUE, PLAY], "present-at-start body queues then plays")
    n += 1

    # 14. idle empty room emits nothing, latched or not
    l = SitLatch(hold=2.0)
    _, _, ev = drive(l, 0.0, DAY1_9AM, [(10.0, EMPTY, VACANT)])
    check(names(ev) == [], "idle empty room is silent")
    n += 1

    check(DAY_START_H == 5, "the shipped day starts at 05:00 local")
    n += 1

    print("\n%d checks, %d failures" % (n, len(FAILS)))
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
