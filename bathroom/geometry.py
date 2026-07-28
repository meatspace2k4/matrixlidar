"""Metric 3D geometry for the matrix lidar: rays, unprojection, room planes.

Everything above this file has treated the 8x8 frame as an IMAGE: row indices
as height, column indices as width, thresholds per row band. That is wrong in
a specific and costly way. The frame is a depth map, so a row index is an
ANGLE, not a height: the physical height of a return is `range * tan(angle)`.
A body at 700mm and a body at 1400mm subtend completely different row spans,
which is why every row-band constant (HEAD_ROWS, UPPER_ROWS, LOWER_ROWS,
TORSO_ROWS) had to be retuned per situation and still flapped. This module
converts a frame into metric 3D points once, so nothing downstream needs a
row constant.

Sensor and mounting (all of it verified against the data, see tests/test_geometry.py)
------------------------------------------------------------------------------
SEN0628: 8x8 zones, 60 deg x 60 deg, 90 deg diagonal (vendor spec). A ~90 deg
diagonal is only consistent with an EQUAL-ANGLE zone grid: 7.5 deg per zone
puts the corner ray at sqrt(2)*30 = 42.4 deg off axis (84.8 deg diagonal),
where an equal-tangent/pinhole grid would give 78.4 deg. So zone centres are
laid out in equal angle, not equal tangent.

DEPTH SEMANTICS: the sensor reports PERPENDICULAR Z, not radial range. This
was not documented anywhere; it was recovered by segmenting the empty room
into planes under both models. Perpendicular explains the room with 4 planes
at 14.4mm weighted rms where radial needs 5 at 17.5mm, and, decisively, under
perpendicular the wall patches come out VERTICAL (|n_y| <= 0.032) while under
radial the same walls come out tilted 9-11 deg off vertical, which no wall is.

MOUNTING: the unit is portrait, aimed about 45 deg across the room in plan.
`stream.orient` resolves to orient(m)[r][c] = m[c][7-r], which transposes and
mirrors the raw grid, so in DISPLAY coordinates row is elevation and column is
azimuth. A square 60x60 array keeps 7.5 deg/zone under a portrait rotation.
The ~45 deg is yaw only: three independent wall patches all return normals
perpendicular to the sensor's Y axis, so Y is gravity to within ~1 deg and
there is no appreciable pitch or roll to correct.

CONSEQUENCE WORTH KNOWING: the vertical half-FOV is 26.25 deg and the unit
sits at roughly waist height (~330mm above the toilet seat plane). At z=1m it
therefore sees only +-490mm about its own height. A standing head and a seated
head are BOTH outside or on the edge of that band, so no elevation-derived
feature can separate the postures; `standing_seed`'s topmost point lands
exactly on the FOV ceiling. Plan-view position is the axis that survives.

Frame convention
----------------
X right (+column), Y up (-row), Z forward out of the lens, millimetres,
origin at the sensor. Y is gravity-aligned, so Y is literally height.
"""
import math

from .zones import N, NO_TARGET, DOOR_COL

ZONE_DEG = 7.5          # 60 deg / 8 zones, equal-angle grid
HALF_FOV_DEG = 26.25    # centre of the outermost zone, (3.5 * 7.5)
BASELINE_CAP = 1500     # no-target baseline cells capped here, as in classify

# per-cell ray tangents, precomputed: p = (d*TAN_AZ[c], d*TAN_EL[r], d)
TAN_AZ = [math.tan(math.radians((c - 3.5) * ZONE_DEG)) for c in range(N)]
TAN_EL = [math.tan(math.radians((3.5 - r) * ZONE_DEG)) for r in range(N)]


def unproject(r, c, d):
    """One cell -> a metric 3D point in the sensor frame.

    `d` is the PERPENDICULAR distance the sensor reports, so it is already the
    Z coordinate and the lateral offsets are plain tangents. No trig per call.
    """
    return (d * TAN_AZ[c], d * TAN_EL[r], float(d))


def max_visible_height(z):
    """Highest point the sensor can see at distance `z` (mm above the sensor).

    The reason a head is not a usable feature: this is only ~490mm at z=1m.
    """
    return z * math.tan(math.radians(HALF_FOV_DEG))


def cloud(matrix, skip_door=True):
    """Whole frame -> {(r, c): (x, y, z)}, dropping no-target cells."""
    pts = {}
    for r in range(N):
        for c in range(N):
            if skip_door and c == DOOR_COL:
                continue
            d = matrix[r][c]
            if d >= NO_TARGET or d <= 0:
                continue
            pts[(r, c)] = unproject(r, c, d)
    return pts


def foreground(matrix, baseline, delta_mm=300, skip_door=True):
    """Metric points for cells meaningfully closer than the empty room.

    Same occupancy rule as classify.features (a depression below a captured
    baseline), but the result is 3D points instead of grid coordinates.
    """
    pts = {}
    for r in range(N):
        for c in range(N):
            if skip_door and c == DOOR_COL:
                continue
            d = matrix[r][c]
            if d >= NO_TARGET or d <= 0:
                continue
            b = baseline[r][c]
            b = b if b < NO_TARGET else BASELINE_CAP
            if b - d < delta_mm:
                continue
            pts[(r, c)] = unproject(r, c, d)
    return pts


def fit_plane(points):
    """Least-squares plane through >=3 metric points.

    Returns (normal, offset, rms) with the normal unit-length; a point p is on
    the plane when dot(normal, p) == offset. Smallest-eigenvector of the
    covariance, found by power iteration on (trace*I - cov) so there is no
    numpy dependency on the Pi.
    """
    pts = list(points)
    m = len(pts)
    if m < 3:
        raise ValueError("fit_plane needs at least 3 points")
    cen = [sum(p[i] for p in pts) / m for i in range(3)]
    cov = [[0.0] * 3 for _ in range(3)]
    for p in pts:
        v = [p[i] - cen[i] for i in range(3)]
        for i in range(3):
            for j in range(3):
                cov[i][j] += v[i] * v[j]
    tr = sum(cov[i][i] for i in range(3))
    a = [[tr * (i == j) - cov[i][j] for j in range(3)] for i in range(3)]
    n = [0.5773, 0.5774, 0.5775]
    for _ in range(300):
        w = [sum(a[i][j] * n[j] for j in range(3)) for i in range(3)]
        mag = math.sqrt(sum(x * x for x in w))
        if mag < 1e-12:
            break
        n = [x / mag for x in w]
    off = sum(n[i] * cen[i] for i in range(3))
    res = [sum(n[i] * p[i] for i in range(3)) - off for p in pts]
    return tuple(n), off, math.sqrt(sum(x * x for x in res) / m)


def _neighbours(rc):
    r, c = rc
    return ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1))


def segment_planes(pts, tol=40.0, min_cells=4):
    """Region-grow 4-connected planar patches out of a cloud, largest first.

    Connectivity matters: a free RANSAC over 64 noisy points happily returns
    spatially incoherent cell sets that fit a plane by accident. Growing only
    into grid neighbours keeps every patch a real surface.

    Returns [{"cells", "normal", "offset", "rms"}], plus the leftover cells.
    """
    todo = dict(pts)
    out = []
    while len(todo) >= min_cells:
        best = None
        for s in list(todo):
            patch = [s] + [k for k in _neighbours(s) if k in todo]
            if len(patch) < 3:
                continue
            n, off, _ = fit_plane([todo[k] for k in patch])
            grown = set(patch)
            changed = True
            while changed:
                changed = False
                for k in list(grown):
                    for nb in _neighbours(k):
                        if nb in todo and nb not in grown:
                            if abs(sum(n[i] * todo[nb][i]
                                       for i in range(3)) - off) <= tol:
                                grown.add(nb)
                                changed = True
                if len(grown) >= 3:
                    n, off, _ = fit_plane([todo[k] for k in sorted(grown)])
            if best is None or len(grown) > len(best):
                best = grown
        if best is None or len(best) < min_cells:
            break
        cells = sorted(best)
        n, off, rms = fit_plane([todo[k] for k in cells])
        out.append({"cells": cells, "normal": n, "offset": off, "rms": rms})
        for k in cells:
            todo.pop(k, None)
    return out, sorted(todo)


def gravity_check(planes, min_cells=6):
    """How far the sensor's Y axis is from true vertical, in degrees.

    Walls are vertical, so a wall's normal is perpendicular to gravity. Taking
    the largest near-vertical patches, the worst |n_y| bounds the mount's
    combined pitch/roll error. Returns None if no patch is big enough to judge.
    """
    cands = [p for p in planes if len(p["cells"]) >= min_cells
             and abs(p["normal"][1]) < 0.35]
    if not cands:
        return None
    # The LARGEST patch, not the worst. A small patch can have swallowed part
    # of a fixture and tilted: in one scan an 8-cell "wall" holding two toilet
    # cells read 3.6 deg off vertical beside a clean 29-cell wall at 0.7.
    big = max(cands, key=lambda p: len(p["cells"]))
    return math.degrees(math.asin(abs(big["normal"][1])))


def fixture_points(baseline, cells, cluster_tol=250.0):
    """Metric 3D points of a static fixture, read out of the empty baseline.

    The toilet is not a grid region, it is an OBJECT AT A PLACE, and the two
    are not the same thing. ZONE_CELLS[Zone.TOTO] is a hand-drawn 3x2 rectangle
    of cells, but its column-D cells return ~450mm further out than the rest:
    they miss the fixture and read the wall and the floor behind it. Averaging
    the rectangle therefore puts the "toilet" half a metre from the toilet.

    So keep only the near depth cluster: the cells that actually come back off
    the object. Everything past `cluster_tol` behind the nearest return is
    background that the rectangle happens to overlap.
    """
    pts = [unproject(r, c, baseline[r][c]) for (r, c) in cells
           if baseline[r][c] < NO_TARGET]
    if not pts:
        return []
    near = min(p[2] for p in pts)
    return [p for p in pts if p[2] - near <= cluster_tol]


def fixture_frame(baseline, cells):
    """Summarise a fixture as (centre_x, centre_z, seat_y).

    `seat_y` is the highest return off the fixture, i.e. the seat plane; the
    centre is the plan-view position a body's occupancy is measured against.

    Sanity anchor for the current room: this returns a seat plane 328mm below
    the sensor, and the operator's AR measurement puts the sensor 700mm above
    the floor, so the seat lands at 372mm. That is a real toilet seat height,
    and it is arrived at without ever being told one.
    """
    pts = fixture_points(baseline, cells)
    if not pts:
        return None
    cx = sum(p[0] for p in pts) / len(pts)
    cz = sum(p[2] for p in pts) / len(pts)
    return cx, cz, max(p[1] for p in pts)


def plan_offset(points, cx, cz):
    """Plan-view (x, z) distance from a body cloud's centroid to a fixture.

    Plan position is the one axis a waist-height mount does NOT clip, which is
    why it separates postures when height and surface-orientation features
    cannot.
    """
    pts = list(points)
    if not pts:
        return None
    bx = sum(p[0] for p in pts) / len(pts)
    bz = sum(p[2] for p in pts) / len(pts)
    return math.hypot(bx - cx, bz - cz)


def plan_axis(cx, cz):
    """Unit vector from the sensor toward a fixture, in plan view."""
    L = math.hypot(cx, cz)
    if L < 1e-6:
        return 0.0, 1.0
    return cx / L, cz / L


def footprint_depth(points, cx, cz):
    """Front-to-back extent of a body cloud along the sensor->fixture axis.

    The posture feature that survives this mount. A seated body is deep in
    plan: the knees stand ~0.5m proud of the hips while the back stays at the
    seat. A standing body is shallow, roughly chest-to-back. Crucially this is
    invariant to LEAN, which killed the two obvious alternatives: leaning
    forward flips the sign of depth-vs-height slope and swings the body
    surface normal, but it does not move the knees, so the footprint stays
    deep either way.
    """
    pts = list(points)
    if len(pts) < 5:
        return None
    ax, az = plan_axis(cx, cz)
    along = [p[0] * ax + p[2] * az for p in pts]
    return max(along) - min(along)


def largest_blob(cloud):
    """The biggest 4-connected group of cells in a foreground {(r,c): point}.

    A person is ONE connected object. Loose cells scattered across the grid are
    baseline drift, not a body, and averaging them into a centroid drags it
    somewhere nobody is standing. In the walk-through capture the room after
    the last exit holds a steady 3-4 foreground cells in two disjoint pairs
    (B4/C4/C5 and D7/D8, the latter 400-550mm off baseline, i.e. the scene
    really has changed there); by count that looks like a body, by connectivity
    it plainly is not.

    Returns a {(r, c): point} sub-dict, empty if the cloud is.
    """
    seen = set()
    best = []
    for start in cloud:
        if start in seen:
            continue
        stack, group = [start], []
        seen.add(start)
        while stack:
            k = stack.pop()
            group.append(k)
            for nb in _neighbours(k):
                if nb in cloud and nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        if len(group) > len(best):
            best = group
    return {k: cloud[k] for k in best}


def surface_rake(points):
    """|dz/dy| of a body's visible surface: how far it leans out of vertical.

    A STANDING body is a column, so its visible surface is vertical and its
    depth does not change with height: the rake is ~0 (measured 0.00-0.18 over
    198 attested frames of standing and of bending to wipe). A SEATED body is
    raked either way it is sitting: upright puts the knees near and the torso
    back (+0.48 median), leaning forward brings the head near and down (-0.29
    median). The MAGNITUDE is the feature; the sign only says which way they
    are sitting, which is why the signed slope failed to separate anything.

    Unlike the ratio of the top return to the FOV ceiling, this does not
    algebraically reduce to a row index: both z and y are real metric
    quantities here and neither cancels.
    """
    pts = list(points)
    if len(pts) < 5:
        return None
    ys = [p[1] for p in pts]
    zs = [p[2] for p in pts]
    my = sum(ys) / len(ys)
    mz = sum(zs) / len(zs)
    syy = sum((y - my) ** 2 for y in ys)
    if syy < 1e-6:
        return None
    return abs(sum((ys[i] - my) * (zs[i] - mz) for i in range(len(ys))) / syy)


def height_above_floor(point, floor_y):
    """Absolute height of a metric point, given the floor's y in sensor coords."""
    return point[1] - floor_y


def seat_cells(baseline, cells, tol=40.0):
    """The subset of a fixture's cells that return its TOP surface.

    For the toilet those are the two cells that come back off the seat plane
    (they agree on its height to 2mm); the rest see the bowl front below it.
    """
    pts = [(rc, unproject(rc[0], rc[1], baseline[rc[0]][rc[1]]))
           for rc in cells if baseline[rc[0]][rc[1]] < NO_TARGET]
    if not pts:
        return []
    top = max(p[1] for _rc, p in pts)
    return [rc for rc, p in pts if top - p[1] <= tol]


def surface_height(matrix, cells, floor_y):
    """Mean height above the floor of whatever those cells return right now.

    Pointed at a fixture's seat cells this is the whole posture measurement:
    the toilet is a static object of known height, so the seat cells read the
    bare seat when it is clear and read whatever is ON it when it is not. A
    seated body puts its thighs there, roughly one thigh-thickness above the
    seat plane, and that offset is a body dimension rather than a tuned
    threshold, which is why it is so repeatable.
    """
    hs = []
    for (r, c) in cells:
        d = matrix[r][c]
        if d >= NO_TARGET or d <= 0:
            continue
        hs.append(unproject(r, c, d)[1] - floor_y)
    return sum(hs) / len(hs) if hs else None


def over_fixture(points, cx, cz, seat_y, half_extent=300.0, drop=50.0):
    """Fraction of a body cloud sitting in the volume above a fixture's seat.

    A true volumetric occupancy test in millimetres: inside a box of
    +-half_extent about the fixture in plan, and no lower than `drop` below
    the seat plane.
    """
    pts = list(points)
    if not pts:
        return 0.0
    hit = sum(1 for p in pts
              if abs(p[0] - cx) < half_extent and abs(p[2] - cz) < half_extent
              and p[1] > seat_y - drop)
    return hit / len(pts)
