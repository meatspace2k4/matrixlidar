"""Posture from one frame, the way the heatmap reads to a person.

The body is a SHAPE in the picture, made of two kinds of cell:

  nearer than the empty room   the beam came back early: mass in the way
  nothing came back at all     the beam went out where the scan says a wall is
                               and nothing returned: mass in the way. A seated
                               torso is mostly this. Dropping these cells as
                               "nothing there" is what made an occupied toilet
                               read EMPTY and UNKNOWN.

Take the largest connected run of those cells (one body, not scattered cells),
then answer the three questions you answer by looking at the grid:

  is there a shape at all           no -> EMPTY
  does it reach the top row         no -> the body is low. Then ask the SEAT,
                                          against its own baseline: if the
                                          fixture's cells have come forward, the
                                          body is on it. SITTING.
  it reaches the top, how near      at the lens -> crossing in front of it,
                                          WALKING. Further -> STANDING at the
                                          toilet.

Measured on 597 operator-attested frames, the top row is the whole sitting/
standing split: it is reached by 96% of standing frames, 99% of walking frames,
and 2% of sitting frames.

Single frames flicker at the silhouette edge, where one 7.5 degree zone
straddles the body's boundary and alternates between the body and the wall
behind it. That is not smoothed here. engine.StateEngine debounces the label
(commit_after consecutive frames before the committed state moves), which is
the layer that owns it.
"""
import statistics

from .zones import N, NO_TARGET, DOOR_COL, State, ZONE_CELLS, Zone

NEAR_MM = 250       # nearer than the empty room by this much: mass in the way
BODY_MIN = 5        # a connected shape smaller than this is not a body
LENS_MM = 680       # an upright shape this near is crossing in front of the
                    # lens, not standing at the toilet. Attested: walking reads
                    # 290-776mm (median 612), standing 680-903 (median 760).
TOP_ROW = 0
# THE SEAT, MEASURED AGAINST ITSELF.
#
# The seat cells are the only cells whose baseline is a NEAR FIXTURE rather than
# a far wall, so a flat depression threshold cannot read them. NEAR_MM is 250 on
# a ~1500mm wall, which is 17%. Seventeen percent of the seat's own ~1000mm
# baseline is ~200mm, and ~200mm is exactly the standoff a body sitting on the
# seat has. That is why "a person-shaped thing 300mm in front of the seat" was
# undetectable: the number was scaled to the wall and applied to the fixture.
#
# Proportional to each cell's own baseline instead. Measured over 759 attested
# frames, cells of 6 advanced past 20% of their own baseline:
#
#     sitting    1-5, median 2      >= 1 in ALL 323 frames
#     standing   0-2, median 1
#     walking    0-6, median 1
#     empty      0 in ALL 161 frames
#
# So >= 1 is a necessary condition for sitting with no false negatives and no
# empty-room noise. It replaces three separate tests (toilet-column overlap,
# body distance, and a margin behind the seat) that together scored the same.
SEAT_ADVANCE_FRAC = 0.20
SEAT_ADVANCE_MIN = 1
# A STANDING BODY WHOSE TOP ROW BLINKS IS STILL STANDING.
#
# The sit/stand split was the top row alone, and one row is not enough to carry
# it. Inside 27 seconds of attested standing, four consecutive frames lost their
# top-row cell while everything else held: 26 cells, rows 2-8, seat reading
# unchanged. The classifier called it SITTING, the daemon committed it, PLAY
# fired, and the one-shot latch was spent 45 seconds before the operator
# actually sat down.
#
# What separates the two is size, and only for frames that have lost the top
# row. Standing bodies that blink are 25-28 cells; 98.5% of attested sitting is
# under 24.
#
#     attested sitting       7-29 cells, median 13
#     attested standing     15-33 cells, median 22
#     standing, top row absent (9 frames)     25-28 cells
#
# So a body that has lost its top row and is still standing-sized has not sat
# down, and it reads STANDING, not UNKNOWN. A body the sensor can plainly see is
# never UNKNOWN: that was the whole complaint this file exists to answer.
UPRIGHT_CELLS = 24
# THE TOILET DOES NOT MOVE, so neither does where a seated body can be. A sit
# is AT the seat: over 288 attested sitting frames the body's median return is
# never more than 338mm in front of the fixture's own scanned distance (p95 is
# 201mm, the torso leaning forward). The sitting branch never asked. A body
# crossing 8 inches from the lens read 703-731mm in front of the seat, kept its
# top row out of frame, occluded the seat's rays (which is what seat_advance
# measures: rays answered early, whether by a body on the seat or a body in
# front of it), came in under UPRIGHT_CELLS, and committed SITTING; the play
# latch was spent on someone leaning at the sensor. 450mm keeps every attested
# sit with 112mm to spare and excludes that frame by 250mm. A low small shape
# further out than this is a body in front of the lens, and falls through to
# the upright branches, where LENS_MM already reads it as WALKING.
SIT_MAX_FRONT_MM = 450
BLOCKED_FRACTION = 0.5  # more than half the rays THE SCAN ANSWERS coming back
                        # dead is not a body, it is something over the sensor.
                        # The largest attested body leaves 12 rays dead; a
                        # blocked view leaves 24 and up.

# the toilet's own columns, from the zone geometry rather than typed in
TOILET_COLS = frozenset(c for (_r, c) in ZONE_CELLS[Zone.TOTO])
TOILET_ROW_FROM = min(r for (r, _c) in ZONE_CELLS[Zone.TOTO]) - 2


def cells(matrix, baseline):
    """(near cells -> distance, cells that stopped returning).

    The DOOR column is excluded from both: it swings on its own and says
    nothing about a body.
    """
    near, gone = {}, set()
    for r in range(N):
        for c in range(N):
            if c == DOOR_COL:
                continue
            b = baseline[r][c]
            if b >= NO_TARGET:          # the scan saw nothing here either
                continue
            v = matrix[r][c]
            if v >= NO_TARGET:
                gone.add((r, c))
            elif b - v >= NEAR_MM:
                near[(r, c)] = v
    return near, gone


def largest_shape(cs):
    """The biggest connected run (8-neighbour). One body, not scattered cells."""
    todo = set(cs)
    best = set()
    while todo:
        comp, stack = set(), [todo.pop()]
        while stack:
            cell = stack.pop()
            comp.add(cell)
            r, c = cell
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    n = (r + dr, c + dc)
                    if n in todo:
                        todo.discard(n)
                        stack.append(n)
        if len(comp) > len(best):
            best = comp
    return best


def seat_mm(baseline):
    """How far the toilet is, from the scan. Not a typed-in distance."""
    v = [baseline[r][c] for (r, c) in ZONE_CELLS[Zone.TOTO]
         if baseline[r][c] < NO_TARGET]
    return int(statistics.median(v)) if v else None


def seat_advance(matrix, baseline):
    """Toilet cells whose return has come forward of the FIXTURE'S OWN baseline
    by a fixture-scale amount. This is the seat measured against itself."""
    n = 0
    for (r, c) in ZONE_CELLS[Zone.TOTO]:
        b = baseline[r][c]
        if b >= NO_TARGET or matrix[r][c] >= NO_TARGET:
            continue
        if b - matrix[r][c] >= SEAT_ADVANCE_FRAC * b:
            n += 1
    return n


def present_cells(matrix, baseline, floor):
    """Cells that have come nearer than the empty room EVER did, plus dead rays.

    This answers "is anything there", which is a different question from "what
    shape is it" and needs a different cut. NEAR_MM is a flat 250mm and the empty
    room's own spread exceeds that on 42 of 64 cells, so presence measured with
    it fires on nothing: 40 consecutive frames of an empty bathroom committed
    STANDING_AT_TOILET. The floor is per cell and measured, so it cannot.
    """
    n = 0
    for r in range(N):
        for c in range(N):
            if c == DOOR_COL:
                continue
            b = baseline[r][c]
            if b >= NO_TARGET:
                continue
            v = matrix[r][c]
            if v >= NO_TARGET:
                n += 1
                continue
            f = floor[r][c] if floor else None
            if f is None:
                if b - v >= NEAR_MM:
                    n += 1
            elif v < f:
                n += 1
    return n


def features(matrix, baseline, floor=None):
    """What the shape is and where it sits. Plain dict, logged next to the state."""
    near, gone = cells(matrix, baseline)
    shape = largest_shape(set(near) | gone)
    near = {rc: v for rc, v in near.items() if rc in shape}
    dists = sorted(near.values())
    answerable = sum(1 for r in range(N) for c in range(N)
                     if c != DOOR_COL and baseline[r][c] < NO_TARGET)
    dead = sum(1 for r in range(N) for c in range(N)
               if c != DOOR_COL and matrix[r][c] >= NO_TARGET
               and baseline[r][c] < NO_TARGET)
    return {
        "present": present_cells(matrix, baseline, floor),
        "blocked": bool(answerable) and dead > BLOCKED_FRACTION * answerable,
        "dead_rays": dead,
        "seat_mm": seat_mm(baseline),
        "seat_advance": seat_advance(matrix, baseline),
        "body_cells": len(shape),
        "returned": len(near),
        "no_return": len(shape) - len(near),
        "touches_top": any(r == TOP_ROW for (r, _c) in shape),
        "over_toilet": any(r >= TOILET_ROW_FROM and c in TOILET_COLS
                           for (r, c) in shape),
        "body_mm": dists[len(dists) // 2] if dists else None,
        "nearest_mm": dists[0] if dists else None,
        "rows": sorted({r for (r, _c) in shape}),
        "cols": sorted({c for (_r, c) in shape}),
        "shape": sorted(shape),
    }


def preflight(baseline):
    """Startup line for the washlet daemon, which logs this before its frame
    loop so a bad scan surfaces at START rather than on frame 1. There is no
    scene to reconstruct here, so what it reports is the fixture this
    classifier keys off, read out of the scan."""
    return ("toilet at %smm, columns %s; %d rays answered by the scan"
            % (seat_mm(baseline), sorted(TOILET_COLS),
               sum(1 for r in range(N) for c in range(N)
                   if c != DOOR_COL and baseline[r][c] < NO_TARGET)))


_FLOOR_CACHE = {}


def _floor_for(baseline):
    """The measured noise floor that goes with this scan, or None."""
    key = id(baseline)
    if key not in _FLOOR_CACHE:
        try:
            from . import config
            fl = config.load_noise_floor()
            live = config.load_baseline()
            _FLOOR_CACHE[key] = fl if live == baseline else None
        except Exception:
            _FLOOR_CACHE[key] = None
    return _FLOOR_CACHE[key]


def classify(matrix, baseline):
    """One oriented frame -> (State, features)."""
    f = features(matrix, baseline, floor=_floor_for(baseline))

    if f["blocked"]:
        # Something is over the sensor. The empty room answers on essentially
        # every ray, so a frame that answers almost nowhere is not a posture and
        # not an empty room: it is a blocked view, and the honest report is that
        # nothing can be seen. This is the one place UNKNOWN means something.
        return State.UNKNOWN, f

    # PRESENCE against the measured floor, not the flat cut. The shape below is
    # still measured with NEAR_MM, because the two questions are different: one
    # asks whether anything is there, the other asks what shape it is.
    if f["present"] < BODY_MIN:
        return State.EMPTY, f

    # Low AND small AND on the seat: sitting. Small matters because a standing
    # body that merely blinks its top row is still standing-sized (see
    # UPRIGHT_CELLS), and it must not read as a sit. AT the seat matters
    # because seat_advance cannot tell a body on the fixture from a body
    # between the lens and the fixture: both answer the seat's rays early
    # (see SIT_MAX_FRONT_MM). No distance evidence (body_mm None: a torso of
    # pure no-returns) does not veto the sit.
    if (not f["touches_top"] and f["body_cells"] < UPRIGHT_CELLS
            and f["seat_advance"] >= SEAT_ADVANCE_MIN
            and (f["body_mm"] is None or f["seat_mm"] is None
                 or f["seat_mm"] - f["body_mm"] <= SIT_MAX_FRONT_MM)):
        return State.SITTING_AT_TOILET, f

    # Everything else the sensor can see is an upright body: at the lens, or at
    # the toilet. There is no UNKNOWN here. A body in plain view always gets a
    # posture; only a blocked view above declines to answer.
    if f["body_mm"] is not None and f["body_mm"] < LENS_MM:
        return State.WALKING, f
    return State.STANDING_AT_TOILET, f


def explain(f):
    if f["body_cells"] < BODY_MIN:
        return "no body"
    if f.get("blocked"):
        return "view blocked (%d dead rays)" % f["dead_rays"]
    return ("%d cells (%d returned, %d no-return) rows %d-%d top=%s seat=%d/6 "
            "body=%smm" % (f["body_cells"], f["returned"], f["no_return"],
                           f["rows"][0] + 1, f["rows"][-1] + 1,
                           f["touches_top"], f["seat_advance"], f["body_mm"]))
