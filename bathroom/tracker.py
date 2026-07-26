"""Track the closest thing in the room.

The scene is the 8x8 of ranges. Rebuild it, take the closest return, and that is
the object. Where it is tells you what it is doing.

  nothing closer than the empty room   ->  EMPTY
  the closest thing is at the toilet   ->  at the toilet, sitting or standing by
                                           its height off the floor
  the closest thing is at the lens     ->  WALKING past

No blobs, no clusters, no bounding boxes, no thresholds per row.
"""
import math

from .zones import N, NO_TARGET, DOOR_COL, State, ZONE_CELLS, Zone

ZONE_DEG = 7.5          # 60 deg across 8 zones
HALF = 3.5

# a return this much nearer than the empty room is an object, not the room
DELTA_MM = 250


def _angles(r, c):
    """(elevation, azimuth) of a cell, radians."""
    return (math.radians((HALF - r) * ZONE_DEG),
            math.radians((c - HALF) * ZONE_DEG))


def closest(matrix, baseline):
    """The nearest object in the room: (cell, range_mm, height_mm, plan_mm).

    height is mm above the sensor, plan is distance out from the lens along the
    floor. Both come straight out of the ray's own angle. A ray that died where
    the room used to answer is the object at the room's own range: something
    swallowed it.
    """
    best = None
    for r in range(N):
        for c in range(N):
            if c == DOOR_COL:
                continue
            b = baseline[r][c]
            if b >= NO_TARGET:
                continue
            v = matrix[r][c]
            if v >= NO_TARGET:
                d = b                       # swallowed: at most this far
            elif b - v >= DELTA_MM:
                d = v
            else:
                continue
            if best is None or d < best[1]:
                el, az = _angles(r, c)
                best = ((r, c), d, d * math.sin(el), d * math.cos(el))
    return best


def fixture(baseline):
    """Where the toilet is: its range and height, off the scan."""
    rs = []
    for (r, c) in ZONE_CELLS[Zone.TOTO]:
        b = baseline[r][c]
        if b < NO_TARGET:
            el, _az = _angles(r, c)
            rs.append((b, b * math.sin(el)))
    if not rs:
        return None
    rs.sort()
    return rs[len(rs) // 2]
