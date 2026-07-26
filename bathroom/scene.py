"""Per-frame doorway scene model.

Promoted from lab/scene_model.py after validation over all captured
sessions (see that commit); lab/validate_scene.py remains the replay
validator.

Reads each oriented 8x8 frame directly against the static scene
(the empty-room scan; classify.DEFAULT_BASELINE is a pre-move scan and is
not interchangeable with it) and
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
    # scene validity: the empty scene's walls and floor return on essentially
    # every ray (the baseline itself carries exactly one no-target cell), so a
    # frame reporting nothing almost everywhere is not an empty room, it is a
    # blocked view. A body close to the lens does not read "close": it
    # INVALIDATES rays and they come back no-target. Corpus separation is
    # wide and unambiguous: normal frames carry 0-25 no-target cells (262
    # historical frames across 11 sessions, plus a 148-frame live sample, all
    # below 40), while the churn-window frames carry 40-59 with only 3-17
    # survivors sitting at 673-824mm. 32 sits in that gap, at half the field.
    MAX_NO_TARGET = 32       # more no-target cells than this => view blocked
    # door panel classification (informational only):
    PANEL_OPEN = 1.35        # H reading >= this * baseline -> hallway seen
    PANEL_CLOSED_HI = 1.12   # H median within [0.95, this] * baseline


def passage_band(baseline):
    """{(r, c): (lo_mm, hi_mm)} for every doorway ray that HAS a baseline surface.

    The band is a fraction of the distance to the surface the scan found on that
    ray, so a ray the scan found nothing on gets no band and is skipped. It used
    to call _eff_base() here, which substitutes 1500mm for no-target and turns
    the band into arithmetic on an invented number: a scan taken with the door
    open has all of column H at no-target, so every one of those rays got the
    band (600, 1425), and a CLOSED door at 1085mm then sat permanently inside it
    and read as a body crossing the threshold. That fired phantom EXITs in an
    empty room. Columns F and G (interior-adjacent and the jamb) always have a
    surface, and MIN_DOORWAY_CELLS is 2, so the test still has what it needs.
    """
    band = {}
    for r in range(N):
        for c in Params.DOORWAY_COLS:
            b = baseline[r][c]
            if b >= NO_TARGET:
                continue
            band[(r, c)] = (Params.BAND_LO * b, Params.BAND_HI * b)
    return band


def door_panel_state(matrix, baseline):
    """Coarse per-frame panel state from column H. Informational only."""
    base = baseline
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
                 "doorway_cells", "subfloor_cells", "vetoed", "door_panel",
                 "occluded", "no_target_count", "near_returns")

    def __init__(self, interior_body, interior_cells, doorway_body,
                 doorway_cells, subfloor_cells, vetoed, door_panel,
                 occluded=False, no_target_count=0, near_returns=()):
        self.interior_body = interior_body
        self.interior_cells = interior_cells
        self.doorway_body = doorway_body
        self.doorway_cells = doorway_cells
        self.subfloor_cells = subfloor_cells
        self.vetoed = vetoed
        self.door_panel = door_panel
        self.occluded = occluded            # the view is blocked: this frame
                                            # is NOT evidence of an empty room
        self.no_target_count = no_target_count
        self.near_returns = near_returns    # surviving near-field cells: the
                                            # positive evidence of a body


def read_frame(matrix, baseline):
    """Reduce one oriented frame to a SceneReading. Pure, per-frame."""
    base = baseline
    interior = []   # (label, mm) body cells, columns A-E
    in_band = []    # (label, mm, lo, hi) doorway-ray body cells in the band
    sub_floor = []  # (label, mm) doorway-ray body cells below the floor
    no_target = 0   # rays returning nothing: the scene-validity signal
    near = []       # (label, mm) surviving near-field returns
    for r in range(N):
        for c in range(N):
            v = matrix[r][c]
            b = base[r][c]
            if v >= NO_TARGET:
                no_target += 1
                # The beam went out where the scan found a surface and nothing
                # came back: mass in the way. A SEATED body is mostly this, so
                # counting only cells that read 300mm nearer made interior_body
                # go False the moment the operator sat down, and the room then
                # resolved an open transit as an EXIT with someone on the
                # toilet (measured: EXIT 11.6s into an attested sit). The door
                # column is excluded, as everywhere else: it swings on its own.
                if b < NO_TARGET and c not in Params.DOORWAY_COLS:
                    interior.append((label(r, c), v))
                continue
            if b >= NO_TARGET:
                # the scan found no surface on this ray, so "300mm closer than
                # the empty room" and "40-95% of the way to it" are both
                # arithmetic on a number that does not exist. This used to call
                # _eff_base(), which substitutes 1500mm: a scan taken with the
                # door open has all of column H at no-target, every one of
                # those rays got the band (600, 1425), and a CLOSED door at
                # 1085mm sat permanently inside it. Two such cells is all
                # doorway_body needs, so an empty room with a shut door read as
                # a body crossing the threshold and fired phantom EXITs.
                continue
            # a surviving return that is body-depth (the same criterion a
            # body cell uses). On a blocked view these are the positive
            # evidence that something is there: the real churn frames carry
            # ~4 of them at 673-824mm while the rest of the field is dead.
            if v > 0 and b - v >= Params.MARGIN_MM:
                near.append((label(r, c), v))
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
    # the walls do not disappear: this many dead rays means the view is
    # blocked, so the frame says nothing about whether the room is empty
    occluded = no_target > Params.MAX_NO_TARGET
    return SceneReading(interior_body, interior_evidence, doorway_body,
                        in_band, sub_floor, vetoed,
                        door_panel_state(matrix, base),
                        occluded=occluded, no_target_count=no_target,
                        near_returns=near)


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
        self._interior_at_open = False
        self._occluded_run = 0   # consecutive blocked frames
        self._occluded_reported = False

    def step(self, t, reading, fresh=False):
        """Feed one reading; returns a list of event dicts."""
        events = []

        if reading.occluded:
            # The view is blocked: absence of returns is not absence of a
            # body, so this frame is NOT evidence. Hold every debounced
            # value, feed no edges, and never resolve a pending transit off
            # it (that is what manufactured the EXIT/PRESENCE churn).
            self._occluded_run += 1
            if (self._occluded_run > Params.DEBOUNCE_N
                    and not self._occluded_reported):
                self._occluded_reported = True
                events.append({"t": t, "kind": "OCCLUDED",
                               "no_target": reading.no_target_count,
                               "near": list(reading.near_returns),
                               "panel": reading.door_panel})
            return events
        self._occluded_run = 0
        self._occluded_reported = False

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
            # which side the body was on when the transit opened
            self._interior_at_open = self.interior.value
            events.append({"t": t, "kind": "DOORWAY_EVENT",
                           "cells": list(reading.doorway_cells),
                           "panel": reading.door_panel})

        if self.pending is not None:
            # ENTRY and EXIT differ only in ORDER: entering is doorway THEN
            # interior, leaving is interior THEN doorway. Reading one frame's
            # edge cannot tell them apart, because on the way OUT the interior
            # is still True at the instant the body reaches the doorway (you
            # are inside the room until you are in the doorway). So the transit
            # is resolved against what the interior was WHEN IT OPENED:
            #
            #   opened with the interior clear -> the body came from the
            #     doorway. It is an entry once the interior fills.
            #   opened with the interior set   -> the body came from inside.
            #     It is an exit once BOTH clear.
            #
            # EXITED must test the interior itself, not rely on reaching an
            # elif. An earlier attempt at this ordering left the exit branch as
            # `elif not self.door.value`, which then fired whenever the doorway
            # blipped during a sit and the interior was still plainly occupied:
            # EXIT 11.3s into an attested sit, 9.5s into an attested stand.
            if self.interior.value and not self._interior_at_open:
                events.append({"t": t, "kind": "ENTERED",
                               "already_occupied": self.occupied,
                               "since_event": t - self.pending})
                self.occupied = True
                self.pending = None
            elif not self.interior.value and not self.door.value:
                events.append({"t": t, "kind": "EXITED",
                               "was_occupied": self.occupied,
                               "since_event": t - self.pending,
                               "panel": reading.door_panel})
                self.occupied = False
                self.pending = None
        # PRESENCE_NO_ENTRY is reported on ANY interior rise while the room is
        # not known-occupied, including while a transit is open. It used to be
        # an `elif` on the pending block, so a transit that opened with the
        # interior already set (a capture that starts mid-posture, or a doorway
        # blip) swallowed it and the room reported nothing at all while a body
        # stood in plain view.
        if int_rose and not self.occupied:
            events.append({"t": t, "kind": "PRESENCE_NO_ENTRY",
                           "cells": list(reading.interior_cells),
                           "panel": reading.door_panel})
        return events
