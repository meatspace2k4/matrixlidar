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
  does it reach the top row         no -> the body is low. Over the toilet
                                          columns, that is SITTING.
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
BEHIND_SEAT_MM = 250  # a body ON the seat cannot read further away than the
                      # seat plus a body's depth. The scan says where the seat
                      # is (median of the toilet's own cells), so this is a
                      # margin, not a distance. Attested sitting runs 691-1179mm
                      # against a seat at 1029mm; the frames this rejects sat at
                      # 1436-1454mm, i.e. 400mm BEHIND the toilet, and were
                      # far-field dropouts that happened to fall over its
                      # columns.
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


def features(matrix, baseline):
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
        "blocked": bool(answerable) and dead > BLOCKED_FRACTION * answerable,
        "dead_rays": dead,
        "seat_mm": seat_mm(baseline),
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


def classify(matrix, baseline):
    """One oriented frame -> (State, features)."""
    f = features(matrix, baseline)

    if f["blocked"]:
        # Something is over the sensor. The empty room answers on essentially
        # every ray, so a frame that answers almost nowhere is not a posture and
        # not an empty room: it is a blocked view, and the honest report is that
        # nothing can be seen. This is the one place UNKNOWN means something.
        return State.UNKNOWN, f

    if f["body_cells"] < BODY_MIN:
        return State.EMPTY, f

    if not f["touches_top"]:
        # the shape does not reach the top of the frame: the body is low
        if (f["over_toilet"] and f["body_mm"] is not None
                and f["seat_mm"] is not None
                and f["body_mm"] <= f["seat_mm"] + BEHIND_SEAT_MM):
            return State.SITTING_AT_TOILET, f
        return State.UNKNOWN, f

    # upright body: at the lens, or at the toilet
    if f["body_mm"] is not None and f["body_mm"] < LENS_MM:
        return State.WALKING, f
    return State.STANDING_AT_TOILET, f


def explain(f):
    if f["body_cells"] < BODY_MIN:
        return "no body"
    if f.get("blocked"):
        return "view blocked (%d dead rays)" % f["dead_rays"]
    return ("%d cells (%d returned, %d no-return) rows %d-%d top=%s toilet=%s "
            "body=%smm" % (f["body_cells"], f["returned"], f["no_return"],
                           f["rows"][0] + 1, f["rows"][-1] + 1,
                           f["touches_top"], f["over_toilet"], f["body_mm"]))
