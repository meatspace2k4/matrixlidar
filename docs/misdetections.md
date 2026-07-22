# Misdetection root causes (depth refactor)

Observed failures from real sessions (raw-frame snapshots preserved in
`fixtures/misdetections/`, harvested from the Pi's RAM ring), the root cause
of each, and the change that fixes it. No thresholds were calibrated against
these sessions; the fixes read the depth the sensor was already reporting.

## 1. False EXIT on a walking direction flip (session 5, 13:15:29)

**Observed:** a single flip from walking-left to walking-right fired EXIT and
closed the session.

**Root cause:** the vanish-side test was column-only. The final crossing ran
at 393-533mm from the lens; that close, a body smears across grid columns by
angular projection and reached column G, so the trailing centroid looked
door-side. Its depth said mid-room: the door plane at the ~45° sensor angle
is ~1.0-1.4m.

**Fix:** `occupancy._vanish_door_side` requires the trailing track depth to
have receded to `DOOR_PLANE_MM` (900mm) as well as the column condition.
Pinned by `test_occupancy` ("near-lens flip ... is NOT an exit") and
`test_depth` (real crossing frame `body_mm=491`).

## 2. Missed EXITs / stuck OCCUPIED (sessions 6 and 8; 19 min, then hours)

**Observed:** real exits were never recognized; the room stayed OCCUPIED
until (and past) later walk-ins. Session 8's final transition shows the door
open% jumping the moment the field emptied: the exit happened behind the
moving door.

**Root cause:** two-dimensional vanish classification again: exits whose last
visible track didn't satisfy the column heuristic were filed as mid/left
vanishes (which correctly keep OCCUPIED, for the shower case). The door
column is excluded from presence, so a body behind/at the door is invisible
at the exact moment that matters. Meanwhile the one signal that was
unambiguous, the door being operated with nobody visible, was ignored.

**Fix:** door-motion rule in `occupancy.update`: while OCCUPIED with an empty
field, a door open% change vs. the vanish moment that holds for
`DOOR_MOVE_N` frames, followed by `T_DOOR_STILL` of stillness, is the missed
exit. The shower case is unaffected: the door does not move. Pinned by
`test_occupancy` ("door moved while empty ..." and "shower vanish, door
untouched ...").

## 3. False SITTING during mirror use (session 6, 13:41, five commits)

**Observed:** standing at the mirror right of the sensor repeatedly read
`SITTING_AT_TOILET`.

**Root cause:** sitting was "body with clear headroom": the deep-occlusion
headroom test failed for a body ~0.8-1.0m out (its absolute depression over a
1.2-1.7m baseline is below `DEEP_DELTA`), so a standing body fell through to
SITTING. Position and depth were never consulted.

**Fix (classify.py):**
- Body-plane headroom: upper-region cells occupied at the body's own median
  depth (`upper_body`) mean the body reaches the top of the FOV → STANDING,
  regardless of absolute depression.
- SITTING is positional and depth-consistent: the lower cluster must overlap
  the toilet fixture (`zones.TOTO`, now the 6-cell static fixture B-D x rows
  7-8) AND the body must read at the seat's own baseline depth
  (`body_at_seat`, tolerance `SEAT_TOL`); a floor-stander beside the fixture
  is systematically closer than the seat.

Result on the five bad frames: 4 STANDING, 1 UNKNOWN (safe fallback), 0
SITTING; the labeled seated seed still reads SITTING. Pinned by `test_depth`.

## Not addressed here

- Left-of-FOV coverage (sink/tub/shower): stated non-issue.
- Toilet-seat up/down detection: future work; the fixture cells make the seat
  itself measurable when unobscured.
- EMPTY/UNKNOWN posture flapping filling the transition archive: noise in the
  marker stream, not a state error; candidate for a later debounce.
