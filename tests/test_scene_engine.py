#!/usr/bin/env python3
"""Scene automaton (doorway/interior detection) regression tests.

Synthetic frame construction validates the scene model's frame reduction
(read_frame, SceneAutomaton) against its design scenarios.

  python3 tests/test_scene_engine.py
"""
import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bathroom.classify import DEFAULT_BASELINE  # noqa: E402
from bathroom.occupancy import OccupancyEngine, Params as OccupancyParams, RoomState  # noqa: E402
from bathroom.scene import Params as SceneParams  # noqa: E402
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


FAILS = []


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


def _occupied_engine():
    """Return an engine in OCCUPIED state via a doorway entry."""
    eng = OccupancyEngine(baseline=DEFAULT_BASELINE)
    eng.room = RoomState.EMPTY

    # Create doorway frame: >= MIN_DOORWAY_CELLS cells on doorway columns
    # at 0.55 * baseline, inside the 0.40-0.95 band, no sub-floor cells.
    doorway_cells = [
        (1, 5, int(0.55 * DEFAULT_BASELINE[1][5])),
        (2, 5, int(0.55 * DEFAULT_BASELINE[2][5])),
        (1, 6, int(0.55 * DEFAULT_BASELINE[1][6])),
    ]
    doorway_frame = make_frame(*doorway_cells)

    # Create interior frame: >= MIN_INTERIOR_CELLS cells on columns 0-3
    # at baseline minus 500 mm (representing a body).
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

    # Drive: doorway, then interior (enough debounce frames to lock in)
    t = 0.0
    t, _ = drive(eng, t, [(doorway_frame, SceneParams.DEBOUNCE_N),
                           (interior_frame, SceneParams.DEBOUNCE_N)])
    assert eng.room == RoomState.OCCUPIED, "setup: engine not OCCUPIED after entry"
    return eng, t


def main():
    # Scenario 1: Entry from UNKNOWN
    # Doorway frames followed by interior frames -> ENTRY_CONFIRMED, OCCUPIED.
    eng = OccupancyEngine(baseline=DEFAULT_BASELINE)

    doorway_cells = [
        (1, 5, int(0.55 * DEFAULT_BASELINE[1][5])),
        (2, 5, int(0.55 * DEFAULT_BASELINE[2][5])),
        (1, 6, int(0.55 * DEFAULT_BASELINE[1][6])),
    ]
    doorway_frame = make_frame(*doorway_cells)

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
    empty_frame = make_frame()

    t = 0.0
    t, ev = drive(eng, t, [(doorway_frame, SceneParams.DEBOUNCE_N),
                           (interior_frame, SceneParams.DEBOUNCE_N)])
    check(eng.room == RoomState.OCCUPIED and "ENTRY_CONFIRMED" in ev and eng.occupants == 1,
          "entry from UNKNOWN: doorway then interior -> ENTRY_CONFIRMED, OCCUPIED, occupants=1")

    # Scenario 2: Exit
    # Continue from occupied engine, doorway frames then empty -> EXIT, EMPTY.
    t, ev = drive(eng, t, [(doorway_frame, SceneParams.DEBOUNCE_N),
                           (empty_frame, SceneParams.DEBOUNCE_N * 2)])
    check(eng.room == RoomState.EMPTY and "EXIT" in ev and eng.occupants == 0,
          "exit: occupied then doorway then empty -> EXIT, EMPTY, occupants=0")

    # Scenario 3: Artifact never latches
    # Fresh engine, interior frames from boot, then empty -> PRESENCE_NO_ENTRY, never OCCUPIED.
    eng = OccupancyEngine(baseline=DEFAULT_BASELINE)
    t = 0.0
    t, ev = drive(eng, t, [(interior_frame, SceneParams.DEBOUNCE_N * 2),
                           (empty_frame, SceneParams.DEBOUNCE_N * 2)])
    check("PRESENCE_NO_ENTRY" in ev and eng.room == RoomState.UNKNOWN and eng.occupants == 0,
          "artifact never latches: interior from boot stays UNKNOWN, PRESENCE_NO_ENTRY seen")

    # Scenario 4: Smear veto
    # Occupied engine, near-lens smear (sub-floor cells) then empty -> no EXIT, stays OCCUPIED.
    eng, t = _occupied_engine()

    # Near-lens smear: >= 5 cells on columns 5-7 at 0.20 * baseline (sub-floor)
    # plus 2 cells at 0.55 * baseline on column 6.
    smear_cells = [
        (1, 5, int(0.20 * DEFAULT_BASELINE[1][5])),
        (2, 5, int(0.20 * DEFAULT_BASELINE[2][5])),
        (3, 5, int(0.20 * DEFAULT_BASELINE[3][5])),
        (1, 6, int(0.20 * DEFAULT_BASELINE[1][6])),
        (2, 6, int(0.20 * DEFAULT_BASELINE[2][6])),
        (3, 6, int(0.55 * DEFAULT_BASELINE[3][6])),
        (4, 6, int(0.55 * DEFAULT_BASELINE[4][6])),
        (1, 7, int(0.20 * DEFAULT_BASELINE[1][7])),
    ]
    smear_frame = make_frame(*smear_cells)

    t, ev = drive(eng, t, [(smear_frame, SceneParams.DEBOUNCE_N),
                           (empty_frame, SceneParams.DEBOUNCE_N * 2)])
    check(eng.room == RoomState.OCCUPIED and "EXIT" not in ev,
          "smear veto: sub-floor cells veto doorway, no EXIT, stays OCCUPIED")

    # Scenario 5: Out-of-FOV
    # Occupied engine, interior frames then sustained empty (no doorway) -> no EXIT, stays OCCUPIED.
    eng, t = _occupied_engine()

    t, ev = drive(eng, t, [(interior_frame, SceneParams.DEBOUNCE_N),
                           (empty_frame, SceneParams.DEBOUNCE_N * 3)])
    check(eng.room == RoomState.OCCUPIED and "EXIT" not in ev,
          "out-of-FOV: interior vanishes but no doorway exit observed, stays OCCUPIED")

    # Scenario 6: Gap
    # Occupied engine, skip to t + T_GAP + 1 with empty frame -> GAP_RESET, UNKNOWN.
    eng, t = _occupied_engine()

    t_gap = t + OccupancyParams.T_GAP + 1.0
    empty_frame = make_frame()
    ru = eng.update(empty_frame, t_gap, door_pct=None)
    check(ru.event == "GAP_RESET" and eng.room == RoomState.UNKNOWN and eng.occupants == 0,
          "gap: unobserved stretch >= T_GAP resets to UNKNOWN")

    # Scenario 7: Doorway re-sighting while occupied
    # Occupied engine, doorway then interior -> NO ENTRY_CONFIRMED event, stays OCCUPIED.
    eng, t = _occupied_engine()

    t, ev = drive(eng, t, [(doorway_frame, SceneParams.DEBOUNCE_N),
                           (interior_frame, SceneParams.DEBOUNCE_N)])
    check(eng.room == RoomState.OCCUPIED and "ENTRY_CONFIRMED" not in ev and eng.occupants == 1,
          "doorway re-sighting while occupied: no ENTRY_CONFIRMED, occupants stays 1")

    # Scenario 8: door_pct irrelevance
    # Repeat scenario 5 (out-of-FOV) passing door_pct 0 and 100 alternately -> same result.
    eng, t = _occupied_engine()

    events = []
    door_pct_values = [0, 100, 0, 100]
    for door_pct in door_pct_values:
        ru = eng.update(interior_frame, t, door_pct=door_pct)
        if ru.event:
            events.append(ru.event)
        t += 1.0 / HZ

    t, ev = drive(eng, t, [(empty_frame, SceneParams.DEBOUNCE_N * 3)])
    check(eng.room == RoomState.OCCUPIED and "EXIT" not in ev,
          "door_pct irrelevance: swinging door_pct 0/100 has no effect on occupancy")

    print("ALL PASS" if not FAILS else "SOME FAILED")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
