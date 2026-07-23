"""Per-frame doorway scene model.

Promoted from lab/scene_model.py after validation over all captured
sessions (see that commit); lab/validate_scene.py remains the replay
validator.

Reads each oriented 8x8 frame directly against the static scene
(classify.DEFAULT_BASELINE, byte-identical to baselines/empty.json) and
reduces it to a SceneReading:

    interior_body : bool + cells   a body inside the room
    doorway_body  : bool + cells   a body inside the doorway passage
    door_panel    : str            informational only, never consulted for
                                   occupancy (occupancy-design.md I5)

No temporal smoothing, no centroids, no velocity. The only temporal
construct is an N-frame debounce (N=2) on the two booleans, plus a tiny
event automaton over the debounced values.

Geometry (display orientation, row 0 = top, col 0 = column A)
-------------------------------------------------------------
The sensor is mounted at ~45 degrees to the door; the door swings OUTWARD.
The doorway passage occupies the door-adjacent rays, columns F,G (5,6) and
H (7, the door column), across a RANGE of depths: from the near edge of the
frame to the panel/hallway plane -- the "gradient left of the jamb".

Passage band, derived per-cell from the baseline
------------------------------------------------
For a doorway-ray cell with (capped) baseline depth B, the passage band is

    [BAND_LO * B, BAND_HI * B]  =  [0.40 * B, 0.95 * B]

which on these rays (B ~ 1045-1170 mm) is ~420-1100 mm absolute. A
doorway_body cell is a body cell (depression >= MARGIN_MM below B) whose
absolute depth lies inside the band. Because a body cell already satisfies
v <= B - 300, the binding ceiling is ~0.71-0.74*B; 0.95 is the formal scene
boundary (empty-scene doorway readings across all five fixture sessions sit
at 0.93-1.26*B and never enter the band).

The 0.40 floor is the near edge of the passage and is the empirically
observed boundary in the fixture corpus (sessions 001/005/006/008/010, no
per-session tuning):
  - every doorway-ray cell during the attested real passage transits
    (001 t=0.00 entry and t=8.16 exit; 006 t=0.00 and t=879.55 entries,
    t=76.57 at-door; 008 t=5901.98 entry, t=27.34 and t=5944.14 at-door)
    reads >= 0.402*B (minima 0.402/0.406/0.411), and NONE reads below 0.40;
  - the near-lens smear frames (005 t=264.94 -- the old false EXIT, at
    353-538 mm; 001 t=0.68; 006 t=949.87, t=1033.51; 008 t=7800.89) paint
    9-19 doorway-ray cells at 0.16-0.39*B, below the floor.
This matches docs/misdetections.md #1: the false-exit crossing ran at
393-533 mm from the lens, and "that close, a body smears across grid
columns by angular projection"; the door plane at the ~45 degree angle is
~1.0-1.4 m, so nothing inside the passage can return that near.

Smear veto: a sub-floor return on a doorway ray proves a body NEARER than
the passage is occluding those rays, so any same-frame in-band doorway
cells are angular smear, not a passage body: the frame's doorway reading is
False. The sub-floor cells themselves are counted as interior evidence
(geometrically they are a body inside the room). In the corpus the veto is
clean: real transits have zero sub-floor cells, smears have >= 9.

Interior body
-------------
A body cell on columns A-E, plus any sub-floor doorway-ray cell (see
above). interior_body requires MIN_INTERIOR_CELLS = 7 cells: the scene has
drifted since the baseline capture at B4/C4/C5/D6/D7/D8 (up to ~500 mm of
persistent depression, present in every session), and those drift cells
never exceed 6 simultaneously anywhere in the corpus, while every attested
body reads >= 9 interior cells. 7 splits the observed gap; it is one global
constant, not per-session tuning.

Debounce and gap semantics
--------------------------
Each boolean flips only after DEBOUNCE_N = 2 consecutive contrary readings
(dropout robustness at 10 Hz). The debounce operates on contiguous evidence
only: across an unobserved gap the model cannot attest continuity, so the
reading history is reset and the next frame is adopted as-is ("fresh").
Because the prior state cannot be attested across a gap, a fresh True
reading is ALWAYS treated as a rising edge (a new sighting), even when the
boolean was True before the gap; a fresh False is a falling edge only
against a previously-known True. Whether a gap is contiguous is the
caller's call (validate_scene bridges gaps <= 1.0 s at 10 Hz by repeating
the earlier frame, and treats anything longer as a gap).

Event automaton (SceneAutomaton)
--------------------------------
  - doorway_body rising edge          -> DOORWAY_EVENT (opens a pending
                                         transit)
  - pending + interior_body True      -> ENTERED, occupied := True
  - pending + both debounced False    -> EXITED,  occupied := False
  - interior rising, no pending,
      not occupied                    -> PRESENCE_NO_ENTRY (reported,
                                         never latches occupancy)
  - interior falls with no pending    -> no event: still occupied
                                         (out-of-FOV is not an exit)
"""

from .classify import DEFAULT_BASELINE, _eff_base
from .zones import N, NO_TARGET, label


class Params:
    MARGIN_MM = 300          # body cell: this many mm closer than baseline
                             # (classify.DELTA_MM, unchanged)
    DOORWAY_COLS = (5, 6, 7)  # F, G (door-adjacent) and H (door column)
    BAND_LO = 0.40           # passage floor, fraction of the ray's baseline
    BAND_HI = 0.95           # passage ceiling (formal; margin binds first)
    MIN_DOORWAY_CELLS = 2    # in-band cells needed (single-cell noise guard)
    MIN_INTERIOR_CELLS = 7   # above the 6-cell worst-case baseline drift,
                             # below the 9-cell minimum attested body
    DEBOUNCE_N = 2           # consecutive contrary readings to flip a bool
    # door panel classification (informational only):
    PANEL_OPEN = 1.35        # H reading >= this * baseline -> hallway seen
    PANEL_CLOSED_HI = 1.12   # H median within [0.95, this] * baseline


def passage_band(baseline=None):
    """{(r, c): (lo_mm, hi_mm)} for every doorway-ray cell."""
    base = baseline or DEFAULT_BASELINE
    band = {}
    for r in range(N):
        for c in Params.DOORWAY_COLS:
            b = _eff_base(base[r][c])
            band[(r, c)] = (Params.BAND_LO * b, Params.BAND_HI * b)
    return band


def door_panel_state(matrix, baseline=None):
    """Coarse per-frame panel state from column H. Informational only."""
    base = baseline or DEFAULT_BASELINE
    ratios = []
    occluded = 0
    for r in range(N):
        v = matrix[r][7]
        b = _eff_base(base[r][7])
        if v >= NO_TARGET:
            ratios.append(Params.PANEL_OPEN + 1)  # no target = hallway
            continue
        ratios.append(v / b)
        if b - v >= Params.MARGIN_MM:
            occluded += 1
    open_cells = sum(1 for x in ratios if x >= Params.PANEL_OPEN)
    if occluded >= 2:
        return "OCCLUDED(body at door)"
    if open_cells >= N // 2:
        return "OPEN"
    if open_cells >= 1:
        return "MOVING/PARTIAL"
    med = sorted(ratios)[N // 2]
    if med <= Params.PANEL_CLOSED_HI:
        return "CLOSED"
    return "AJAR"


class SceneReading:
    """Direct per-frame reduction. Cell lists carry (label, mm) pairs;
    doorway cells carry (label, mm, band_lo, band_hi)."""

    __slots__ = ("interior_body", "interior_cells", "doorway_body",
                 "doorway_cells", "subfloor_cells", "vetoed", "door_panel")

    def __init__(self, interior_body, interior_cells, doorway_body,
                 doorway_cells, subfloor_cells, vetoed, door_panel):
        self.interior_body = interior_body
        self.interior_cells = interior_cells
        self.doorway_body = doorway_body
        self.doorway_cells = doorway_cells
        self.subfloor_cells = subfloor_cells
        self.vetoed = vetoed
        self.door_panel = door_panel


def read_frame(matrix, baseline=None):
    """Reduce one oriented frame to a SceneReading. Pure, per-frame."""
    base = baseline or DEFAULT_BASELINE
    interior = []   # (label, mm) body cells, columns A-E
    in_band = []    # (label, mm, lo, hi) doorway-ray body cells in the band
    sub_floor = []  # (label, mm) doorway-ray body cells below the floor
    for r in range(N):
        for c in range(N):
            v = matrix[r][c]
            if v >= NO_TARGET:
                continue
            b = _eff_base(base[r][c])
            if b - v < Params.MARGIN_MM:
                continue
            if c in Params.DOORWAY_COLS:
                lo, hi = Params.BAND_LO * b, Params.BAND_HI * b
                if v < lo:
                    sub_floor.append((label(r, c), v))
                elif v <= hi:
                    in_band.append((label(r, c), v, int(lo), int(hi)))
                # v > hi cannot happen for a body cell (margin binds first)
            else:
                interior.append((label(r, c), v))

    vetoed = bool(sub_floor) and bool(in_band)
    doorway_body = (len(in_band) >= Params.MIN_DOORWAY_CELLS
                    and not sub_floor)
    # sub-floor doorway-ray returns are a body inside the room
    interior_evidence = interior + sub_floor
    interior_body = len(interior_evidence) >= Params.MIN_INTERIOR_CELLS
    return SceneReading(interior_body, interior_evidence, doorway_body,
                        in_band, sub_floor, vetoed,
                        door_panel_state(matrix, base))


class _Debounced:
    """N-frame debounce of one boolean, with explicit gap resets."""

    def __init__(self, n):
        self.n = n
        self.value = False
        self._contrary = 0

    def step(self, reading):
        """Contiguous-evidence update. Returns (rose, fell)."""
        if reading == self.value:
            self._contrary = 0
            return False, False
        self._contrary += 1
        if self._contrary >= self.n:
            self.value = reading
            self._contrary = 0
            return (reading, not reading)
        return False, False

    def reset(self, reading):
        """Adopt a fresh reading after an unobserved gap. Continuity cannot
        be attested across the gap, so a fresh True is always a rising edge
        (a new sighting); a fresh False falls only against a known True."""
        prev = self.value
        self.value = reading
        self._contrary = 0
        return (bool(reading), prev and not reading)


class SceneAutomaton:
    """Tiny event automaton over the debounced per-frame readings."""

    def __init__(self):
        self.door = _Debounced(Params.DEBOUNCE_N)
        self.interior = _Debounced(Params.DEBOUNCE_N)
        self.pending = None      # ts of the unresolved DOORWAY_EVENT
        self.occupied = False
        self.first_interior_t = None
        self.first_doorway_t = None

    def step(self, t, reading, fresh=False):
        """Feed one reading; returns a list of event dicts."""
        events = []
        if fresh:
            door_rose, _ = self.door.reset(reading.doorway_body)
            int_rose, _ = self.interior.reset(reading.interior_body)
        else:
            door_rose, _ = self.door.step(reading.doorway_body)
            int_rose, _ = self.interior.step(reading.interior_body)

        if self.interior.value and self.first_interior_t is None:
            self.first_interior_t = t

        if door_rose:
            if self.first_doorway_t is None:
                self.first_doorway_t = t
            self.pending = t
            events.append({"t": t, "kind": "DOORWAY_EVENT",
                           "cells": list(reading.doorway_cells),
                           "panel": reading.door_panel})

        if self.pending is not None:
            if self.interior.value:
                events.append({"t": t, "kind": "ENTERED",
                               "already_occupied": self.occupied,
                               "since_event": t - self.pending})
                self.occupied = True
                self.pending = None
            elif not self.door.value:
                events.append({"t": t, "kind": "EXITED",
                               "was_occupied": self.occupied,
                               "since_event": t - self.pending,
                               "panel": reading.door_panel})
                self.occupied = False
                self.pending = None
        elif int_rose and not self.occupied:
            events.append({"t": t, "kind": "PRESENCE_NO_ENTRY",
                           "cells": list(reading.interior_cells),
                           "panel": reading.door_panel})
        return events
