"""The empty room, derived from the background instead of typed in.

Metric reconstruction of the scene from an empty-room scan. (Named roomscan
rather than scene because scene.py already owns the door/passage reading that
the occupancy engine consumes; these are two different models of the room and
both are live.)

Everything here comes out of the baseline depth map. Nothing is a grid index
and nothing is a measured constant from a tape or an AR app. The two things
this replaces:

  FIXTURE_CELLS = [(6,1),(6,2),(7,1),(7,2)]   hand-picked grid coordinates
  FLOOR_DROP_MM = 700.0                       a scalar typed in from an AR app

Both were the same mistake in different clothes: describing the room to the
code, rather than letting the code read the room off the scan it already has.
A hand-picked cell list silently re-anchors to whatever those cells happen to
hit if the sensor shifts; a typed-in floor height cannot notice it is wrong.

What gets derived, and from what
-------------------------------
GRAVITY. Walls are vertical, so a wall's normal is perpendicular to gravity.
Segmenting the empty room yields several large planar patches; gravity is the
one direction perpendicular to all of their normals at once (the smallest
eigenvector of sum(n n^T)). Three independent walls agreeing pins it, and the
residual is reported so a bad mount cannot pass silently.

FLOOR. Once gravity is known, every baseline point has a height along it. The
floor is the bottom of that distribution: the lowest return that has support
from at least one other return nearby, so a single dropout cannot define it.
No tape measure involved; the operator's "about 70cm" becomes a cross-check
rather than an input.

FIXTURE. The toilet is not a rectangle of cells, it is the nearest solid in
the empty room. Sorting the baseline by depth, there is a wide gap between the
returns that come off the fixture and everything behind it; the cluster below
that gap is the object. The gap is found in the data, so there is no distance
threshold to tune either.
"""
import math

from . import geometry
from .zones import N, NO_TARGET


class SceneError(Exception):
    """The baseline does not describe a room this code can model. Raised loudly
    rather than falling back to defaults, because a silent fallback here means
    every downstream measurement is quietly wrong."""


def _smallest_eigenvector(m):
    """Direction minimising sum (n.v)^2, by power iteration on trace*I - M."""
    tr = sum(m[i][i] for i in range(3))
    a = [[tr * (i == j) - m[i][j] for j in range(3)] for i in range(3)]
    v = [0.5773, 0.5774, 0.5775]
    for _ in range(300):
        w = [sum(a[i][j] * v[j] for j in range(3)) for i in range(3)]
        mag = math.sqrt(sum(x * x for x in w))
        if mag < 1e-12:
            break
        v = [x / mag for x in w]
    return tuple(v)


class Scene:
    """Metric model of the empty room, built once from a baseline."""

    def __init__(self, baseline, min_wall_cells=6):
        self.baseline = baseline
        cloud = geometry.cloud(baseline)
        if len(cloud) < 20:
            raise SceneError("baseline has only %d usable cells" % len(cloud))
        self.planes, _rest = geometry.segment_planes(cloud)

        # --- gravity, from the walls -------------------------------------
        walls = [p for p in self.planes
                 if len(p["cells"]) >= min_wall_cells and abs(p["normal"][1]) < 0.35]
        if len(walls) < 2:
            raise SceneError("need >=2 wall patches to fix gravity, found %d"
                             % len(walls))
        # The walls in THIS room are all nearly parallel: their normals sit
        # within a few degrees of each other, so they span one direction, not
        # two. Perpendicularity to them therefore leaves gravity free in a
        # whole plane, and taking the smallest eigenvector picks an arbitrary
        # member of that family (it landed 4.2 deg off, dragging the floor
        # 87mm low). Degenerate input must be detected, not averaged over.
        # Weight by nothing: take the LARGEST patch. When the walls are
        # parallel (detected below) the others add no independent constraint,
        # and averaging actively hurts, because a patch can be contaminated.
        # In one scan an 8-cell "wall" had swallowed two toilet cells and sat
        # 3.6 deg off vertical; averaging it with a clean 29-cell wall at
        # 0.7 deg tilted gravity to 2.16 deg and dragged the floor 100mm.
        biggest = max(walls, key=lambda p: len(p["cells"]))
        w = list(biggest["normal"])
        wm = math.sqrt(sum(x * x for x in w))
        w = [x / wm for x in w]
        spread = max(math.degrees(math.acos(max(-1.0, min(1.0, abs(
                     sum(w[i] * p["normal"][i] for i in range(3)))))))
                     for p in walls)
        self.wall_spread_deg = spread
        self.walls_degenerate = spread < 15.0
        if self.walls_degenerate:
            # One wall direction only. It constrains gravity in that direction
            # and says nothing about the perpendicular one, so constrain what
            # it can and take the mount's own axis for the rest: project the
            # sensor Y axis onto the plane perpendicular to the wall normal.
            up = [-w[0] * w[1], 1.0 - w[1] * w[1], -w[2] * w[1]]
        else:
            m = [[0.0] * 3 for _ in range(3)]
            for p in walls:
                n = p["normal"]
                for i in range(3):
                    for j in range(3):
                        m[i][j] += n[i] * n[j]
            up = list(_smallest_eigenvector(m))
        mag = math.sqrt(sum(x * x for x in up))
        g = tuple(x / mag for x in up)
        if g[1] < 0:
            g = tuple(-x for x in g)
        self.gravity = g
        self.n_walls = len(walls)
        # worst angle between gravity and a wall plane; 0 would be perfect
        self.gravity_residual_deg = max(
            abs(90.0 - math.degrees(math.acos(
                max(-1.0, min(1.0, sum(g[i] * p["normal"][i] for i in range(3)))))))
            for p in walls)
        if self.gravity_residual_deg > 5.0:
            raise SceneError("walls disagree about vertical by %.1f deg"
                             % self.gravity_residual_deg)

        # --- floor, from the bottom of the height distribution ------------
        # The lowest return, provided it is not a lone outlier. The tolerance
        # is generous because the floor is seen at a steep grazing angle: the
        # bottom row's rays only reach it out at ~1.4m, so few cells land on it
        # and they land far apart. A tight tolerance rejected the one true
        # floor return (E8, z=1379mm, exactly where a -26.25 deg ray meets the
        # floor) because its nearest companion was 50mm up, and fell through to
        # a cell that was actually looking at the toilet.
        hs = sorted((self.height_raw(p), rc) for rc, p in cloud.items())
        floor_h = None
        for i, (h, _rc) in enumerate(hs):
            support = [x for x, _ in hs[i + 1:i + 4] if x - h <= 100.0]
            if support:
                floor_h = h
                break
        if floor_h is None:
            raise SceneError("no supported floor return in the baseline")
        self.floor_offset = floor_h
        self.sensor_height = -floor_h

        # --- fixture, as the nearest solid ---------------------------------
        by_z = sorted(cloud.items(), key=lambda kv: kv[1][2])
        gap_at, gap = 0, 0.0
        # only look for the break in the near third; the far tail is wall
        for i in range(1, max(2, len(by_z) // 3)):
            d = by_z[i][1][2] - by_z[i - 1][1][2]
            if d > gap:
                gap, gap_at = d, i
        if gap_at < 2:
            raise SceneError("no near cluster stands out of the baseline")
        self.fixture_cells = sorted(rc for rc, _p in by_z[:gap_at])
        self.fixture_gap = gap
        pts = [cloud[rc] for rc in self.fixture_cells]
        self.fixture_x = sum(p[0] for p in pts) / len(pts)
        self.fixture_z = sum(p[2] for p in pts) / len(pts)
        # the fixture's top surface: its highest return, i.e. the seat
        self.fixture_top = max(self.height(p) for p in pts)
        if self.fixture_top > self.sensor_height:
            raise SceneError("nearest solid is above the sensor; not a fixture")

        # --- the doorway, as a place rather than a panel -------------------
        # occupancy.py infers entry and exit from the door's STATE: open%,
        # a door-motion rule, a DOOR_PLANE_MM depth gate. That encodes a
        # swing-IN door. This one opens OUT and the sensor sees straight
        # through the frame into the hallway, so the panel is not a proxy for
        # anything: when it matters most (someone in the doorway) the panel is
        # not even in the way.
        #
        # The threshold is a fixed plane in the room. Derived here from the
        # door column of a closed-door baseline, it comes out 7.6mm rms flat
        # and 0.0 deg off the vertical this class derived independently, at
        # 1174mm from the lens against the operator's 1200mm AR measurement.
        # A body is then inside or outside by the sign of its distance to that
        # plane, which does not care which way the door swings or where it is.
        # The panel IS the doorway, so use it when the door is shut. But a
        # baseline taken with the door OPEN has that whole column at no-target
        # and cannot fix the plane at all, so fall back to the jamb: the wall
        # the doorway is cut into, always present, same plane to ~30mm. The
        # jamb sits inboard of the opening, so it puts the threshold ~110mm
        # nearer the toilet; the panel is the better estimate when available.
        self.door_plane = None
        self.door_source = None
        for col, tag in ((N - 1, "panel"), (N - 2, "jamb")):
            pts_d = [geometry.unproject(r, col, baseline[r][col])
                     for r in range(N) if baseline[r][col] < NO_TARGET]
            if len(pts_d) < 4:
                continue
            zs = sorted(p[2] for p in pts_d)
            if zs[-1] - zs[0] >= 300.0:
                continue
            n, off, rms = geometry.fit_plane(pts_d)
            if rms >= 40.0:
                continue
            if off < 0:
                n, off = tuple(-x for x in n), -off
            self.door_plane = (n, off, rms)
            self.door_distance = off
            self.door_x = sum(p[0] for p in pts_d) / len(pts_d)
            self.door_z = sum(p[2] for p in pts_d) / len(pts_d)
            self.door_source = tag
            break

    def outside_door(self, point):
        """Signed mm past the threshold plane. Positive = at or beyond it.

        Returns None when the baseline could not fix the doorway.
        """
        if self.door_plane is None:
            return None
        n, off, _rms = self.door_plane
        return sum(n[i] * point[i] for i in range(3)) - off

    # --- metric queries -------------------------------------------------
    def height_raw(self, point):
        """Height along gravity, in sensor coordinates (floor not subtracted)."""
        return sum(point[i] * self.gravity[i] for i in range(3))

    def height(self, point):
        """Metric height of a point above the derived floor plane."""
        return self.height_raw(point) - self.floor_offset

    def plan_axis(self):
        return geometry.plan_axis(self.fixture_x, self.fixture_z)

    def describe(self):
        return ("scene: %d planes, %d walls, gravity residual %.1f deg, "
                "sensor %.0fmm above floor, fixture %d cells at "
                "(x=%+.0f z=%.0f) top %.0fmm above floor, near-gap %.0fmm"
                % (len(self.planes), self.n_walls, self.gravity_residual_deg,
                   self.sensor_height, len(self.fixture_cells),
                   self.fixture_x, self.fixture_z, self.fixture_top,
                   self.fixture_gap)
                + (", threshold from %s at %.0fmm" % (self.door_source,
                                                      self.door_distance)
                   if self.door_plane else ", NO threshold"))
