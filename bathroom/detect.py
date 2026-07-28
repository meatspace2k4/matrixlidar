"""Deterministic zone-derived values (door opening, per-zone summaries).

These are explicit, requested mappings of what the sensor measured, not
presence heuristics; presence/state lives in classify.py and posture.py.

EVERY CELL IS A RAY WITH A MEASURED DISTANCE ALONG IT
----------------------------------------------------
This is a time-of-flight sensor. A cell is not a pixel and a column is not a
strip of an image: it is 8 rays at 8 known elevations, each reporting the
perpendicular distance to whatever it hit. A flat wall seen by one column is
therefore 8 points on one PLANE, and the thing to measure is the plane.

`median(column)` is not a distance to anything. It mixes 8 different ray
geometries into one number, and so does the difference of two column medians,
which is how the door opening used to be computed: `median(colH)` interpolated
between the literals 1090 and 1400. Those literals were also unattached to
anything in the frame, so a nudged sensor would shift every reading with
nothing to notice it.

Mirrors the Pi's bathroom/detect.py in intent, not line for line: the Pi's copy
still carries the column-median version.
"""
import math
import statistics

from . import geometry
from .zones import N, NO_TARGET, ZONE_CELLS

JAMB_COL = N - 2        # column G: the wall the doorway is cut into
PANEL_COL = N - 1       # column H: the door leaf itself

MIN_RAYS = 3            # fewer returns than a plane needs
SWING_DEG = 90.0        # a door's travel, by construction. See door_open_pct.
PLANE_RMS_MAX = 25.0    # a coherent flat surface. The jamb fits at 4.9-5.7mm
                        # rms and the shut panel at 5.9mm across three
                        # independent captures, so this is ~4x the observed
                        # scatter of a real surface. Past it the column is not
                        # looking at one plane (a body in the doorway splits
                        # it: some rays hit the person, some the panel) and the
                        # honest answer is None, not a number.


def _plane(matrix, col):
    """Fit one column's returns as a plane in metric 3D.

    Returns (unit_normal, offset_mm, rms_mm, centroid), normal pointing away
    from the lens, or None if the column has too few returns to define a plane.
    Too few returns is itself a reading: on the panel column it means the
    sensor is looking through the opening.
    """
    pts = [geometry.unproject(r, col, matrix[r][col])
           for r in range(N) if 0 < matrix[r][col] < NO_TARGET]
    if len(pts) < MIN_RAYS:
        return None
    n, off, rms = geometry.fit_plane(pts)
    if off < 0:
        n, off = tuple(-x for x in n), -off
    cen = tuple(sum(p[i] for p in pts) / len(pts) for i in range(3))
    return n, off, rms, cen


def jamb_plane(baseline):
    """The doorway's own wall, from the empty-room scan.

    THE JAMB IS THE REFERENCE. It is the one part of the doorway that cannot
    move: the door swings, the panel leaves the field entirely, a body walks
    through, and the wall the opening is cut into stays exactly where it is.
    Across three independent captures (the current scan, and the operator's
    37-frame closed and open calibrations) its plane offset varies by 16mm and
    its normal by 0.8 degrees, while the panel goes from a plane at +83mm
    outboard to no return at all.

    Taken from the baseline rather than the live frame deliberately: a body
    standing in the doorway occludes column G, and the scan is the invariant.
    """
    return _plane(baseline, JAMB_COL)


def door_open_angle(matrix, baseline):
    """How far the door has swung, in degrees off shut, or None if unreadable.

    A door ROTATES about a hinge, so the measurable is the angle between the
    panel's plane and the jamb's plane. Both come from rays, so the reading is
    invariant to where the unit sits: nudge it and both planes move together
    while the angle between them does not change.

    Three cases, two of them attested by the operator's calibrations:
      no panel returns  swung clear of the field, sensor seeing into the
                        hallway. data/door_open100.json: 0 of 8 rays return,
                        labeled "100% open".
      panel parallel    shut. data/door_closed100.json: the panel fits a plane
                        0.8 deg off the jamb's and +83mm outboard of it,
                        labeled "closed 100%".
      panel rotated     partly open, by that angle.
    """
    jamb = jamb_plane(baseline)
    if jamb is None:
        return None                 # no jamb in the scan: nothing to measure against
    panel = _plane(matrix, PANEL_COL)
    if panel is None:
        return SWING_DEG            # seeing through the opening
    if panel[2] > PLANE_RMS_MAX or jamb[2] > PLANE_RMS_MAX:
        return None                 # not one surface; caller holds its last reading
    dot = sum(jamb[0][i] * panel[0][i] for i in range(3))
    return math.degrees(math.acos(max(-1.0, min(1.0, abs(dot)))))


def door_open_pct(matrix, baseline=None):
    """`door_open_angle` as a percentage of a door's travel, or None.

    `baseline` defaults to the empty-room scan, because the washlet daemon on
    the Pi calls this with the matrix alone. The angle is measured against the
    jamb, which the scan locates, so there is nothing to substitute if it is
    missing: config.load_baseline raises rather than inventing one.

    SWING_DEG is 90 by construction of a door, not by measurement: the two
    calibration captures pin 0% (panel parallel to the jamb) and 100% (no panel
    return at all), and the corpus holds no partly-open door, so the ramp
    between them is geometry rather than something fitted. One named constant
    in place of the two unattached literals it replaces.
    """
    if baseline is None:
        from . import config
        baseline = config.load_baseline()
    deg = door_open_angle(matrix, baseline)
    if deg is None:
        return None
    return round(min(100.0, max(0.0, deg / SWING_DEG * 100.0)), 1)


def zone_values(matrix, zone):
    """Per-zone summary of RAY distances over the zone's cells.

    median/min/max are summaries of a set of ray measurements, which is what the
    on-screen panel wants and all it is good for. None of them is the distance
    to a surface: for that, fit the surface (see `_plane`).

    NO_TARGET rays are excluded and counted separately. Including them made the
    summary arithmetic on a sentinel: an open doorway with nothing in range
    reported "DOOR 400 cm" on the operator's panel, which reads as a surface
    four metres away rather than as no surface at all. `median` is None when
    nothing returned.
    """
    vals = [matrix[r][c] for (r, c) in ZONE_CELLS[zone]]
    seen = [v for v in vals if 0 < v < NO_TARGET]
    return {
        "median": int(statistics.median(seen)) if seen else None,
        "min": min(seen) if seen else None,
        "max": max(seen) if seen else None,
        "rays": len(vals),
        "returns": len(seen),
    }


class DoorReading:
    """Holds the last coherent door reading across unreadable frames.

    A body standing in the doorway splits the panel column, so no single plane
    describes it and `door_open_pct` returns None. The door itself has not
    changed in that moment; only our view of it has, so the right answer is the
    last one we could actually measure, not 0 and not a guess. `coherent` says
    which you are looking at, so a caller can show that it is stale.
    """

    def __init__(self, baseline):
        self.baseline = baseline
        self.pct = None
        self.angle = None
        self.coherent = False

    def update(self, matrix):
        deg = door_open_angle(matrix, self.baseline)
        self.coherent = deg is not None
        if self.coherent:
            self.angle = deg
            self.pct = round(min(100.0, max(0.0, deg / SWING_DEG * 100.0)), 1)
        return self.pct
