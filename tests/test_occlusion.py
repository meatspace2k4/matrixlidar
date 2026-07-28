#!/usr/bin/env python3
"""Occlusion (blocked view) regression tests.

The scene model tracks no_target cells (rays returning nothing due to
blockage near the lens). When no_target exceeds MAX_NO_TARGET, the frame is
marked occluded and is not evidence of an empty room. Occluded frames hold
the state machine in place without emitting occupancy events.

  python3 tests/test_occlusion.py
"""
import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bathroom.classify import DEFAULT_BASELINE  # noqa: E402
from bathroom.occupancy import OccupancyEngine, Params as OccupancyParams, RoomState  # noqa: E402
from bathroom.scene import Params as SceneParams, read_frame  # noqa: E402
from bathroom.zones import N, NO_TARGET  # noqa: E402

HZ = 10.0


def set_body(mat, cells):
    """Set body cells in matrix. cells: list of (row, col, mm) tuples."""
    for row, col, mm in cells:
        mat[row][col] = mm


def make_frame(*body_specs):
    """Construct a frame from deepcopy of baseline, optionally with body cells set.

    Each spec is a (row, col, mm) tuple or None for empty frame.
    Returns the oriented 8x8 matrix."""
    mat = copy.deepcopy(DEFAULT_BASELINE)
    if body_specs and body_specs != (None,):
        set_body(mat, body_specs)
    return mat


def drive(eng, t, frames_with_repeats):
    """Drive engine at 10 Hz with a list of (frame, repeat_count) tuples.

    Returns (t, list of events seen)."""
    events = []
    for frame, repeats in frames_with_repeats:
        for _ in range(repeats):
            ru = eng.update(frame, t, door_pct=None)
            if ru.event:
                events.append(ru.event)
            t += 1.0 / HZ
    return t, events


def make_occluded_frame():
    """Construct an occluded frame: baseline with >MAX_NO_TARGET cells set to
    NO_TARGET, leaving a handful of near survivors at 700-800mm."""
    mat = copy.deepcopy(DEFAULT_BASELINE)
    # Set more than MAX_NO_TARGET cells to NO_TARGET
    # Block 33 cells (more than MAX_NO_TARGET=32)
    no_target_cells = [
        (0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7),  # row 0: 8 cells
        (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6),          # row 1: 7 cells
        (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5),                  # row 2: 6 cells
        (3, 0), (3, 1), (3, 2), (3, 3), (3, 4),                          # row 3: 5 cells
        (4, 0), (4, 1), (4, 2), (4, 3),                                  # row 4: 4 cells
        (5, 0), (5, 1), (5, 2),                                          # row 5: 3 cells
    ]  # total: 33 cells
    for r, c in no_target_cells:
        mat[r][c] = NO_TARGET

    # Add a few near survivors at roughly 700-800mm at unblocked cells
    mat[6][5] = 750
    mat[6][6] = 780
    mat[7][5] = 720
    return mat


def _occupied_engine():
    """Return an engine in OCCUPIED state via a doorway entry."""
    eng = OccupancyEngine(baseline=DEFAULT_BASELINE)
    eng.room = RoomState.EMPTY

    # Create doorway frame
    doorway_cells = [
        (1, 5, int(0.55 * DEFAULT_BASELINE[1][5])),
        (2, 5, int(0.55 * DEFAULT_BASELINE[2][5])),
        (1, 6, int(0.55 * DEFAULT_BASELINE[1][6])),
    ]
    doorway_frame = make_frame(*doorway_cells)

    # Create interior frame
    interior_cells = [
        (2, 1, DEFAULT_BASELINE[2][1] - 500),
        (3, 1, DEFAULT_BASELINE[3][1] - 500),
        (4, 1, DEFAULT_BASELINE[4][1] - 500),
        (2, 2, DEFAULT_BASELINE[2][2] - 500),
        (3, 2, DEFAULT_BASELINE[3][2] - 500),
        (4, 2, DEFAULT_BASELINE[4][2] - 500),
        (2, 3, DEFAULT_BASELINE[2][3] - 500),
    ]
    interior_frame = make_frame(*interior_cells)

    # These scenarios are about what happens WHILE the room is occupied, not
    # about how it got that way. It becomes occupied by the sensor SEEING a
    # body, which is now sufficient; it used to be driven through the old
    # doorway-then-interior inference, and that model is gone.
    t = 0.0
    t, _ = drive(eng, t, [(interior_frame, SceneParams.DEBOUNCE_N * 3)])
    assert eng.room == RoomState.OCCUPIED, "setup: engine not OCCUPIED"
    return eng, t


FAILS = []


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


def main():
    # Scenario 1: read_frame on occluded frame
    # Occluded frame reports .occluded True, .no_target_count > MAX_NO_TARGET,
    # and non-empty .near_returns.
    occluded_frame = make_occluded_frame()
    reading = read_frame(occluded_frame, DEFAULT_BASELINE)
    check(reading.occluded,
          "scenario 1: read_frame on occluded frame reports .occluded True")
    check(reading.no_target_count > SceneParams.MAX_NO_TARGET,
          "scenario 1: read_frame on occluded frame reports .no_target_count > MAX_NO_TARGET")
    check(len(reading.near_returns) > 0,
          "scenario 1: read_frame on occluded frame has non-empty .near_returns")

    # Scenario 2: read_frame on normal empty frame
    # Empty frame reports .occluded False.
    empty_frame = make_frame()
    reading = read_frame(empty_frame, DEFAULT_BASELINE)
    check(not reading.occluded,
          "scenario 2: read_frame on empty frame reports .occluded False")

    # Scenario 3: Occluded frames are not evidence of emptiness
    # Occupied engine + many occluded frames + more occluded frames
    # -> room stays OCCUPIED, no EXIT event.
    eng, t = _occupied_engine()

    occluded_frame = make_occluded_frame()
    # Feed more than DEBOUNCE_N occluded frames, then more
    t, ev = drive(eng, t, [(occluded_frame, SceneParams.DEBOUNCE_N + 5),
                           (occluded_frame, SceneParams.DEBOUNCE_N + 3)])
    check(eng.room == RoomState.OCCUPIED and "EXIT" not in ev,
          "scenario 3: occluded frames do not flip OCCUPIED to EMPTY, no EXIT")

    # Scenario 4: Occluded frames do not manufacture presence
    # Fresh engine + only occluded frames
    # -> no ENTRY_CONFIRMED, no PRESENCE_NO_ENTRY, room stays UNKNOWN.
    eng = OccupancyEngine(baseline=DEFAULT_BASELINE)
    occluded_frame = make_occluded_frame()
    t = 0.0
    t, ev = drive(eng, t, [(occluded_frame, SceneParams.DEBOUNCE_N + 3)])
    check("ENTRY_CONFIRMED" not in ev and "PRESENCE_NO_ENTRY" not in ev and eng.room == RoomState.UNKNOWN,
          "scenario 4: occluded frames only do not trigger entry or presence, stay UNKNOWN")

    # Scenario 5: Sustained occluded run reports condition
    # Fresh engine + more than DEBOUNCE_N consecutive occluded frames
    # -> exactly one OCCLUDED event emitted.
    eng = OccupancyEngine(baseline=DEFAULT_BASELINE)
    occluded_frame = make_occluded_frame()
    t = 0.0
    t, ev = drive(eng, t, [(occluded_frame, SceneParams.DEBOUNCE_N + 2)])
    occluded_count = ev.count("OCCLUDED")
    check(occluded_count == 1,
          "scenario 5: sustained occluded run emits exactly one OCCLUDED event")

    # Scenario 6: Alternating occluded and empty frames do not flip state
    # Occupied engine + alternate occluded/empty many times
    # -> zero EXIT events, room stays OCCUPIED.
    eng, t = _occupied_engine()

    empty_frame = make_frame()
    occluded_frame = make_occluded_frame()
    # Alternate many times
    alternations = []
    for _ in range(6):
        alternations.append((occluded_frame, 2))
        alternations.append((empty_frame, 2))
    t, ev = drive(eng, t, alternations)
    check(eng.room == RoomState.OCCUPIED and "EXIT" not in ev,
          "scenario 6: alternating occluded/empty churn does not flip state, no EXIT")

    # Scenario 7: the engine RECOVERS from an occluded run rather than latching
    # blocked. It used to assert that a doorway-then-empty sequence after the
    # run produced an EXIT; that asserted the old model, in which an exit was
    # inferred from three doorway-band cells at 55%% of baseline followed by an
    # empty frame. An exit is a body crossing the threshold plane now, and
    # these hand-built frames contain no body and no crossing. What is still
    # worth pinning is that the blocked state clears: the run must stop being
    # reported once the view comes back, and the room must be deciding again.
    eng, t = _occupied_engine()

    occluded_frame = make_occluded_frame()
    empty_frame = make_frame()

    t, ev_blocked = drive(eng, t, [(occluded_frame, SceneParams.DEBOUNCE_N + 3)])
    check("OCCLUDED" in ev_blocked,
          "scenario 7: a blocked run is reported while the view is blocked")
    t, ev_clear = drive(eng, t, [(empty_frame, SceneParams.DEBOUNCE_N * 2)])
    check("OCCLUDED" not in ev_clear,
          "scenario 7: the blocked state clears when the view comes back")
    t, ev_again = drive(eng, t, [(occluded_frame, SceneParams.DEBOUNCE_N + 3)])
    check("OCCLUDED" in ev_again,
          "scenario 7: a second blocked run is reported again, not latched off")

    print("ALL PASS" if not FAILS else "SOME FAILED")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
