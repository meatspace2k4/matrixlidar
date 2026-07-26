"""Metric posture classifier: is the body ON the toilet, or somewhere else?

Replaces the row-band decision in classify.py. Two things had to change, and
only the first one is about thresholds.

1. A NO-RETURN CELL IS EVIDENCE OF A BODY, NOT OF AN EMPTY ROOM
---------------------------------------------------------------
Every earlier version of this decision started with

    if d >= NO_TARGET: continue        # "nothing there"

A cell that reads 4000 where the empty-room scan had a solid return at
1300-1700mm did not see nothing. Something absorbed the beam short of a
surface we know is there. That is an occluder, and in this room the only
occluder is a person.

It matters specifically for SITTING, because a seated torso is what stops
returning. In the operator's 09:00 capture, frame 122 of the attested sit:

    1110 1191 1275 1388 1388 1226 1116 4000
    1183 1308 1327 4000 1477 1216 1139 4000     nine cells that read
    1318 1439 4000 4000 1367 1215 1130 4000     1300-1700mm in the empty
    1354 4000 4000 4000 1393 1223 1099 4000     room now read 4000. The
    1357 4000 4000 4000 1361 1247 1103 4000     torso is exactly there.
    1350 1058 4000 1112 1405 1224 1089 4000
    1363 1071  918  717 1034 1014 1114 4000     verdict, dropping them: EMPTY
    1244 1180 1009  751  860 1157 1115 4000     (4 foreground cells left)

Counting returns only, a seated body scores 2-22 foreground cells (median 6)
and drops under any presence floor in 19 of 119 attested seated frames.
Counting returns plus occluded rays, the same frames never score below 10,
while an empty room scores 0-6. Presence stops being a coin flip.

An occluded cell has no measured depth, but its RAY is known and so is the
distance at which the scan says a surface should have answered. The constraint
it carries is "an occluder lies on this ray, nearer than the baseline", which
is enough to ask whether it passes through the space above the seat.

2. WHAT SITTING IS: THE BODY'S EVIDENCE IS CONCENTRATED OVER THE SEAT
---------------------------------------------------------------------
Sitting on the toilet means your mass is in the air directly above the seat,
and the seat is a fixed object the scan measures. So: reconstruct the seat,
then ask what FRACTION of everything the sensor knows about the body (returns
plus occluded rays) lies in the column above it. Sitting concentrates the
evidence there; standing in front of the toilet does not.

Measured over the whole labeled corpus, per frame, against per-frame ground
truth (323 attested sitting frames across three days and three poses, 201
frames of "standing at the toilet, not the mirror", 73 walking, 161 empty):

                        seat fraction          foreground returns
    sitting             0.04-0.85 (med 0.56)    2-22 (med  7)
    standing            0.00-0.31 (med 0.08)   13-27 (med 20)
    walking             0.00-0.33 (med 0.03)    3-46 (med 22)

    SITTING iff seat_frac >= 0.30

                        row bands       metric 3D
    sitting recall      0.502           0.929
    standing recall     0.731           0.960
    false sitting       0/436           4/436
    overall             0.682           0.924

The return count separates as well (sitting 2-22, standing 13-27) and an
earlier revision of this file used it as a second gate. It is not here, because
it costs the LEANING-FORWARD sit: 30 attested frames of it return 13-19 cells,
overlapping standing completely, and only the seat fraction (0.36-0.68 there)
separates them. A tighter seat corridor does the same job without that cost;
see COL_Z_MM.

DEAD HYPOTHESES, recorded so they are not retried
--------------------------------------------------
- ROW BANDS (HEAD_ROWS / UPPER_ROWS / LOWER_ROWS / TORSO_ROWS in classify.py).
  A row is an ANGLE, not a height: a return's height is range*tan(angle), so
  the row a shoulder lands in moves with distance. Retuning a band only moves
  which distance it is right at.
- SEAT OCCUPANCY BY DEPRESSION ("a person-shaped thing on top of the seat").
  Geometrically impossible: a body sitting on the seat is within a body
  thickness of the seat surface, so it cannot read 300mm nearer than the seat.
  In the four narrated sitting frames the seat cells move 0-299mm and the
  positional gate scored zero every time. What used to make it fire was a
  forward LEAN putting a torso over the seat cells, i.e. lean, not sitting.
- SEAT DEPTH (|body_mm - seat_mm| <= 120). Not a discriminator: attested
  sitting sits 17-187mm off the seat plane and the mirror-use frames it was
  installed to reject sit 124-164mm off it. It was holding by 4mm.
- SURFACE RAKE |dz/dy|. Looked decisive (94%) against a stale baseline that
  put the toilet where it no longer was, so the fixture's own cells entered
  every foreground. Against a scan that describes the room: 75-78%.
- TOPMOST RETURN / FOV CEILING RATIO. Algebraically identical to the top-row
  index: top/ceiling = z*tan(el)/z*tan(26.25) cancels z. Any feature that
  cancels depth is a row band however it is written.

KNOWN LIMITS
------------
- One person, one room. Nothing here has seen a second body.
- A body BENT OVER the seat occupies the same space as a body on it. Those two
  are not separated, and no feature tried separates them; both read at the
  toilet. The operator's requirement is standing vs sitting, both of which are
  reported, so this is a documented limit rather than an open bug.
- 23 of 323 attested seated frames still miss per frame, all of them to
  STANDING_AT_TOILET, none to EMPTY or UNKNOWN. The engine's commit_after
  hysteresis absorbs isolated ones (the 221-frame 08:02 capture commits
  EMPTY -> WALKING -> STANDING -> SITTING and holds SITTING for 137 unbroken
  frames), but the figures above are the per-frame ones, not the committed ones.
- Mid-rise off the seat reads SITTING for a frame or two: standing up puts your
  mass over the seat on the way. Same ambiguity as bending over it.
"""
from . import geometry, roomscan
from .zones import N, NO_TARGET, DOOR_COL, State, ZONE_CELLS, Zone

# --- metric thresholds. Millimetres of real space, no grid indices. ---------
SEAT_TOL_MM = 60.0       # baseline fixture cells within this of the topmost
                         # one ARE the seat surface
COL_X_MM = 90.0          # half-width of the column above the seat, in plan
COL_Z_MM = 130.0         # half-depth of the same column. NOT a free parameter:
                         # the seat's own returns span z=973-1078mm and a body
                         # standing against the front of the bowl reads
                         # 810-870mm, i.e. 110-160mm proud of the front edge.
                         # 130 is where "in the air above the seat" stops
                         # including "standing in front of it". At 170 it
                         # admitted 15 of 198 attested standing frames on 11mm
                         # of margin; at 130 it admits 1.
COL_RISE_MM = 80.0       # a return must clear the seat by this to be a body
                         # ON it rather than the seat itself
SEAT_FRAC = 0.30         # fraction of the body's evidence (returns + occluded
                         # rays) that must lie in the column above the seat
MIN_BODY = 6             # returns + occluded rays below this is not a body
NEAR_PASS_MM = 600.0     # body nearer than this along the toilet axis is
                         # crossing in front of the lens
AT_FIXTURE_MM = 450.0    # plan distance within which a body is "at" the toilet

_SCENES = {}


def scene_for(baseline):
    """Scene model for a baseline, built once and cached."""
    key = tuple(tuple(row) for row in baseline)
    sc = _SCENES.get(key)
    if sc is None:
        sc = _SCENES[key] = roomscan.Scene(baseline)
    return sc


def preflight(baseline):
    """Build the scene up front so a daemon fails at STARTUP, not on frame 1.

    roomscan.Scene raises SceneError rather than falling back to defaults,
    which is right: a silent fallback makes every measurement downstream
    quietly wrong. features() builds lazily, so without this a bad baseline
    would surface mid-loop. Returns the scene description, for the startup log.
    """
    return scene_for(baseline).describe()


def seat_surface(baseline):
    """The toilet's seat, as metric points read out of the empty-room scan."""
    pts = geometry.fixture_points(baseline, ZONE_CELLS[Zone.TOTO])
    if not pts:
        return []
    top = max(p[1] for p in pts)
    return [p for p in pts if top - p[1] <= SEAT_TOL_MM]


def occluded_rays(matrix, baseline):
    """Cells that answered NO_TARGET where the scan says a surface is.

    Not "nothing there": something absorbed the beam short of a known surface.
    Returns [(r, c, z_limit)], z_limit being the depth the scan expected, i.e.
    the far end of the stretch the occluder must lie in.
    """
    out = []
    for r in range(N):
        for c in range(N):
            if c == DOOR_COL:          # the door swings on its own
                continue
            if matrix[r][c] < NO_TARGET:
                continue
            b = baseline[r][c]
            if b >= NO_TARGET:         # the scan saw nothing here either
                continue
            out.append((r, c, float(b)))
    return out


def _over_seat_point(p, seat_y, seat_x, seat_z):
    return (abs(p[0] - seat_x) < COL_X_MM and abs(p[2] - seat_z) < COL_Z_MM
            and p[1] >= seat_y + COL_RISE_MM)


def seat_column(points, seat):
    """Returns standing in the air directly above the seat: a body ON it."""
    if not seat:
        return 0
    seat_y = max(s[1] for s in seat)
    return sum(1 for p in points
               if any(_over_seat_point(p, seat_y, s[0], s[2]) for s in seat))


def seat_column_rays(rays, seat):
    """Occluded rays whose corridor passes through the column above the seat.

    The occluder's depth is unknown, so the ray is tested over the whole
    stretch it could occupy (0 .. the depth the scan expected). x and y are
    linear in z along a ray, so the seat's own depth window is the only part
    of the ray that can matter.
    """
    if not seat:
        return 0
    seat_y = max(s[1] for s in seat)
    n = 0
    for (r, c, z_limit) in rays:
        tx, ty = geometry.TAN_AZ[c], geometry.TAN_EL[r]
        hit = False
        for s in seat:
            lo = max(s[2] - COL_Z_MM, 0.0)
            hi = min(s[2] + COL_Z_MM, z_limit)
            if lo >= hi:
                continue
            for z in (lo, (lo + hi) / 2.0, hi):
                if (abs(z * tx - s[0]) < COL_X_MM
                        and z * ty >= seat_y + COL_RISE_MM):
                    hit = True
                    break
            if hit:
                break
        n += hit
    return n


def features(matrix, baseline):
    """Metric description of whatever is in the room. All units millimetres."""
    sc = scene_for(baseline)
    seat = seat_surface(baseline)
    cx, cz = _fixture_plan(baseline, sc)
    fg = geometry.foreground(matrix, baseline)
    pts = list(fg.values())
    rays = occluded_rays(matrix, baseline)
    body = len(pts) + len(rays)
    ev = seat_column(pts, seat) + seat_column_rays(rays, seat)
    out = {
        "returns": len(pts),
        "occluded": len(rays),
        "body_cells": body,
        "seat_evidence": ev,
        "seat_frac": (ev / body) if body else 0.0,
        "fixture_x": cx, "fixture_z": cz,
        "sensor_height": sc.sensor_height,
        "seat_height": sc.fixture_top,
        "footprint_depth": None,
        "plan_offset": None,
        "nearest": None,
        "top_height": None,
    }
    if not pts:
        return out
    ax, az = geometry.plan_axis(cx, cz)
    along = [p[0] * ax + p[2] * az for p in pts]
    out["footprint_depth"] = max(along) - min(along)
    out["nearest"] = min(along)
    out["plan_offset"] = geometry.plan_offset(pts, cx, cz)
    out["top_height"] = max(sc.height(p) for p in pts)
    return out


def _fixture_plan(baseline, sc):
    """Plan-view centre of the toilet.

    roomscan.Scene finds the nearest solid by depth clustering, which is right
    in principle but lands on whichever few cells happen to be closest. The
    toilet is a wide object, so take every cell of its zone that returns it
    (still depth-clustered, so cells that miss it are dropped).
    """
    pts = geometry.fixture_points(baseline, ZONE_CELLS[Zone.TOTO])
    if len(pts) < len(sc.fixture_cells):
        return sc.fixture_x, sc.fixture_z
    return (sum(p[0] for p in pts) / len(pts),
            sum(p[2] for p in pts) / len(pts))


def classify(matrix, baseline):
    """Map one oriented frame to a (State, features) pair, metrically."""
    f = features(matrix, baseline)

    # presence counts occluded rays: a body that stops returning is still a
    # body. This is the test that used to call a seated person an empty room.
    if f["body_cells"] < MIN_BODY:
        return State.EMPTY, f

    # SITTING: the body's evidence is concentrated in the air above the seat.
    # Decided before the pass-by test, because a seated body is also NEAR
    # (knees stand half a metre proud of hips), so "near => walking" first
    # would swallow every sitting frame.
    if f["seat_frac"] >= SEAT_FRAC:
        return State.SITTING_AT_TOILET, f

    # AT the toilet, upright. Tested before the pass-by, not after: standing at
    # the toilet and walking past it overlap almost completely on how near the
    # nearest point is (standing 544-824mm, walking 261-669mm) and not at all
    # on plan distance from the fixture. Asking "near?" first cost 48 of 201
    # attested standing frames to WALKING; asking "at the toilet?" first costs
    # 3, and no sitting frame reads WALKING any more.
    if f["plan_offset"] is not None and f["plan_offset"] <= AT_FIXTURE_MM:
        return State.STANDING_AT_TOILET, f

    if f["nearest"] is not None and f["nearest"] < NEAR_PASS_MM:
        return State.WALKING, f

    # a body is there (possibly only as occluded rays), but it is neither at
    # the toilet nor crossing the lens: make no claim
    return State.UNKNOWN, f


def explain(f):
    """One-line audit string, for logging next to a frame."""
    if not f or f["body_cells"] < MIN_BODY:
        return "no body"
    return ("body=%d (%d returned, %d occluded) seat=%d/%.2f depth=%.0fmm "
            "offset=%.0fmm" % (f["body_cells"], f["returns"], f["occluded"],
                               f["seat_evidence"], f["seat_frac"],
                               f["footprint_depth"] or -1,
                               f["plan_offset"] or -1))
