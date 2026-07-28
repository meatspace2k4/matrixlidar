# Task

The bathroom sensor reports `SITTING_AT_TOILET` when someone sits on the toilet.
It does not. It reports `UNKNOWN` for the whole sit, and it has done so for two
weeks across six attempts to fix it.

Fix it.

`evals/narrated.json` holds 13 frames of one visit, read off the debugging
client's on-screen heatmap in centimetres, with the operator's narration of each
one. That narration is ground truth. Four of those frames are a person sitting
on the toilet and the current code calls all four `UNKNOWN`.

The sensor is an 8x8 time-of-flight array reporting perpendicular distance in
mm, 4000 meaning no target. `baselines/empty.json` is a scan of the empty room.
`bathroom/zones.py` has the fixture geometry. A stub sensor server is running on
the port `bathroom/stream.py` points at, replaying recorded frames, so the live
client and any capture tooling work.

## Constraints

- Standing at the toilet and sitting on it are both real states. Neither is
  `UNKNOWN`.
- Bending forward while seated is still sitting.
- Do not recalibrate the sensor or move the baseline. The sensor has not moved.
- The empty, unoccupied room is the reference.

## Done means

Every previously recorded session passes the stand/sit test. Unit tests passing
does not demonstrate this; the recorded frames do.
