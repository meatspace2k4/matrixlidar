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

# A DOOR IS WHERE A DOOR CAN BE. RMS CANNOT TELL YOU THAT.
#
# `PLANE_RMS_MAX` was the only test standing between "column H returned
# something" and "that something is the door", and it does not work, for a
# reason that is geometric rather than a matter of tuning. One column is one
# vertical fan of rays, so a flat surface puts every return on a LINE, and
# collinear points fit infinitely many planes with zero residual. The fitted
# normal is then arbitrary and so is its angle to the jamb. Measured: a BODY
# crossing the doorway at ~650mm fits at rms 0.3-1.2mm, better than the real
# shut door's 5.9, and the code reported it as the door 72-77% open.
#
# What actually separates them is position, against the jamb plane the module
# already treats as the invariant. A leaf hangs IN the opening; it cannot be
# half a metre into the room:
#
#   the closed-door calibration      +80 .. +108mm outboard of the jamb
#   every attested shut-door frame   +105 .. +142mm  (n=198 standing, n=173 sitting)
#   a body in the doorway            -378 .. -688mm  IN FRONT of the jamb
#   the hallway through the opening  +1895 .. +2703mm
#
# So a return is the leaf only inside this band, and the band is per-ray:
# stray cells (which this sensor produces constantly) fall outside it and are
# ignored, rather than dragging a whole-column fit the way a least-squares
# plane does. Nothing is gated on how many rays came back.
LEAF_FRONT_MM = 100     # a leaf never reads further forward of the jamb than this
LEAF_BEHIND_MM = 400    # further out than this is seen THROUGH the opening

# A DOOR IS A STRAIGHT LINE. A PERSON IS BLOBBY.
#
# The straightness test has to be a LINE in the column's own vertical section
# (height against depth), not the plane fit, and again for a geometric reason
# rather than a tuning one. Every return in one column lies in that column's
# azimuth fan, so the plane fit's zero-residual answer is the fan itself: it
# reports rms 0.02 for `1136 1037 1078 1037 1077`, which is a 100mm-jittering
# blob, and then hands back the fan's normal as if it were a surface. That is
# where the fabricated "76.7% open" readings on an empty bathroom came from.
#
# In the section a flat leaf is a line and scatter about it is real. Measured:
#
#   closed-door calibration        5.7mm
#   attested shut door             3.3 .. 16.3mm  (p95 14.8, n=396)
#   the blobs above               25 .. 250mm
#
# 25mm is the same coherence figure PLANE_RMS_MAX was reaching for, applied to
# a fit that can actually fail.
LEAF_LINE_RMS_MAX = 25.0

# THE SWING ANGLE IS NOT MEASURABLE FROM ONE COLUMN, AND WAS NEVER MEASURED.
#
# A door rotates about a VERTICAL hinge, so swinging it changes how distance
# varies with AZIMUTH. One column is one azimuth. It carries no information
# about that rotation, which is why fitting an angle to it produced numbers
# uncorrelated with any door: 1.4-13.8% on a door the operator attested as
# fully shut, 76.7% on an empty bathroom.
#
# What column H does carry is whether the leaf is THERE. Both calibrations are
# exactly that: "closed 100%" is 8 returns at leaf distance, "100% open" is 0
# returns. The corpus holds no partly-open door, so there is nothing to fit a
# ramp to and nothing to validate one against. The reading is therefore what
# the sensor can support, and no more.
SHUT_PCT = 0.0
OPEN_PCT = 100.0


def _section_line_rms(matrix, rows, col=PANEL_COL):
    """Scatter of these returns about a straight line in the column's section.

    The section is (height, depth) down one azimuth, which is where a flat
    surface is a line and a body is not. Fewer than 3 points define a line
    exactly and prove nothing, so they get None rather than a flattering 0.
    """
    pts = []
    for r in rows:
        v = matrix[r][col]
        if 0 < v < NO_TARGET:
            _x, y, z = geometry.unproject(r, col, v)
            pts.append((y, z))
    if len(pts) < 3:
        return None
    my = sum(p[0] for p in pts) / len(pts)
    mz = sum(p[1] for p in pts) / len(pts)
    syy = sum((y - my) ** 2 for y, _z in pts)
    if syy == 0:
        return None
    slope = sum((y - my) * (z - mz) for y, z in pts) / syy
    res = [z - (mz + slope * (y - my)) for y, z in pts]
    return math.sqrt(sum(r * r for r in res) / len(res))


def _leaf_rays(matrix, jamb, col=PANEL_COL):
    """Split the panel column's returns by where they are, against the jamb.

    (leaf, in_front, behind): rows whose return lies in the leaf's band, rows
    hundreds of mm nearer than the jamb (a body standing in the opening), and
    rows well outboard of it (the hallway, seen through the opening).

    Per ray, so one stray cell is one outlier rather than a corrupted fit.
    """
    n, off = jamb[0], jamb[1]
    leaf, in_front, behind = [], [], []
    for r in range(N):
        v = matrix[r][col]
        if not 0 < v < NO_TARGET:
            continue
        p = geometry.unproject(r, col, v)
        d = sum(n[i] * p[i] for i in range(3)) - off   # + is outboard
        if d < -LEAF_FRONT_MM:
            in_front.append(r)
        elif d > LEAF_BEHIND_MM:
            behind.append(r)
        else:
            leaf.append(r)
    return leaf, in_front, behind


def _plane(matrix, col, rows=None):
    """Fit one column's returns as a plane in metric 3D.

    Returns (unit_normal, offset_mm, rms_mm, centroid), normal pointing away
    from the lens, or None if the column has too few returns to define a plane.
    Too few returns is itself a reading: on the panel column it means the
    sensor is looking through the opening.
    """
    rows = range(N) if rows is None else rows
    pts = [geometry.unproject(r, col, matrix[r][col])
           for r in rows if 0 < matrix[r][col] < NO_TARGET]
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

    The rays are sorted by WHERE they are first (see LEAF_FRONT_MM), because
    a column full of returns is not a door just because it fits something:

      nothing in the leaf band, something in front
                        a body is standing in the opening and the leaf is
                        behind it. Unreadable: None, and the caller holds its
                        last reading. This used to fit the body itself and
                        report a door angle off it.
      nothing at all, or only returns seen through the opening
                        swung clear of the field. data/door_open100.json: 0 of
                        8 rays return, labeled "100% open".
      leaf band          the panel. data/door_closed100.json fits a plane
                        0.8 deg off the jamb's and +83mm outboard of it,
                        labeled "closed 100%"; the angle between the two
                        planes is how far it has swung.

    Returns SHUT_PCT, OPEN_PCT or None. See SHUT_PCT: the swing angle is not
    recoverable from one column, and every intermediate number this used to
    emit was an artefact of fitting one anyway.
    """
    jamb = jamb_plane(baseline)
    if jamb is None:
        return None                 # no jamb in the scan: nothing to measure against
    if jamb[2] > PLANE_RMS_MAX:
        return None
    leaf, in_front, _behind = _leaf_rays(matrix, jamb)
    if not leaf:
        if in_front:
            return None             # a body in the opening hides the leaf
        return OPEN_PCT             # nothing there, or the hallway beyond it
    rms = _section_line_rms(matrix, leaf)
    if rms is None:
        # one or two returns at leaf distance. Something is there and it is
        # where the leaf lives, but two points are a line by definition and
        # cannot show it is flat. Not a reading.
        return None
    if rms > LEAF_LINE_RMS_MAX:
        return None                 # blobby: in the leaf's band, not the leaf
    return SHUT_PCT


def door_open_pct(matrix, baseline=None):
    """How open the door is: 0.0, 100.0, or None when it cannot be read.

    `baseline` defaults to the empty-room scan, because the washlet daemon on
    the Pi calls this with the matrix alone. The reading is made against the
    jamb, which the scan locates, so there is nothing to substitute if it is
    missing: config.load_baseline raises rather than inventing one.

    Still a percentage rather than an enum because it is displayed as one and
    every caller treats it as telemetry (OccupancyEngine takes `door_pct` and
    ignores it). What changed is that the values in between are gone: they were
    never measured, only fitted, and they were wrong (see SHUT_PCT). If a
    partly-open door is ever captured and labeled, the ramp can come back
    attached to it.
    """
    if baseline is None:
        from . import config
        baseline = config.load_baseline()
    return door_open_angle(matrix, baseline)


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
