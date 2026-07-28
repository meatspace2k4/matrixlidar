"""Entry and exit as a body crossing a plane, not as a door's state.

A parallel client. occupancy.py is untouched.

What was wrong before
---------------------
occupancy.py decides entry and exit from the DOOR: `door_open_pct`, a
door-motion rule, a `DOOR_PLANE_MM` depth gate. That encodes a swing-in door.
This one opens outward and the sensor sees straight through the frame into the
hallway, so the panel is not a proxy for a person. Worse, presence detection
excludes the door column entirely (`zones.DOOR_COL`), on the reasoning that
the door swings on its own and says nothing about a body. The consequence is
that at the exact moment that matters, someone standing in the doorway, the
body is in the one column nothing looks at.

That is visible in the harvested sessions: at every ENTRY_CONFIRMED the body
is already 555-871mm inside the room, and the single archived EXIT frame has
no body in view at all. The old logic never saw a crossing. It inferred one
afterwards from the door.

What this does instead
----------------------
roomscan.Scene derives the threshold as a fixed PLANE from the closed-door
baseline (7.6mm rms, 0.0 deg off vertical, 1174mm from the lens against a
1200mm AR measurement). Then the door column is read as what it actually is: a
depth probe pointed at the doorway. Unprojecting its nearest return gives a
signed distance to that plane, and a body walking through produces a clean
sign change:

    entry   +137  ->   -65  ->  -111  ->  -222  ->  -352   (body now inside)
    exit    -248  ->   -24  ->   +32  ->   +78  ->  +448   (body now outside)

Direction of the sign change is the event. Nothing here cares which way the
door swings, where the panel currently is, or whether the sensor can see past
it.

Rejecting what is not a body:
- no-target returns are an open doorway looking down the hallway, not a body
- returns far past the threshold are hallway floor and wall (seen at +1412,
  +1658, +1780mm), so only a window either side of the plane is considered
- a single stray frame is not a crossing: the track must show CONSECUTIVE
  frames on both sides, close in time, which rejects the isolated -625mm
  blip in the capture where nobody was in the doorway
- and the door column is consulted ONLY when the main field cannot see the
  body, because that column covers room space as well as the doorway (see
  body_position)
- a SHUT door lies on the threshold plane itself and is rejected by MOTION,
  not by shape. Flatness does not work: a person crossing the doorway fills
  the column just as flat as a panel does (8 rows, 39-147mm spread), because
  a body's front is also a surface. What separates them is that the door
  sits still at a fixed offset while a person sweeps through: +140 to -520mm
  in 1.5s on one pass, +171 to -461mm in 1.0s on the other
"""
from . import geometry
from .zones import N, NO_TARGET, DOOR_COL

WINDOW_MM = 900.0     # beyond this past the plane it is hallway, not a body
CONFIRM_N = 2         # consecutive frames needed on each side of the plane
MAX_GAP_S = 2.5       # a crossing happens quickly; older evidence is stale
MIN_BODY = 5          # main-field returns that count as SEEING the body
BLIND_MAX = 1         # ... and this few means the field is genuinely empty
DOORWAY_N = 3            # a body in a doorway occludes at least this many rows
MIN_SWEEP_MM = 150.0     # a crossing SWEEPS through the plane; a door sits still
MIN_PROBE_MOVE_MM = 60.0 # ... and a doorway probe that has not moved this much
                         # since the last probe frame IS the door, not a body

ENTRY = "ENTRY"
EXIT = "EXIT"


def door_probe(matrix, scene):
    """Median signed mm past the threshold of what is filling the doorway.

    None when the doorway is clear (no-target: looking down the hallway), when
    too few rows are occluded to be a body standing in it, or when nothing is
    within a body's reach of the plane.
    """
    signed = []
    for r in range(N):
        d = matrix[r][DOOR_COL]
        if d >= NO_TARGET or d <= 0:
            continue
        sg = scene.outside_door(geometry.unproject(r, DOOR_COL, d))
        if sg is not None and abs(sg) <= WINDOW_MM:
            signed.append(sg)
    # A person in a doorway occludes a RUN of its rows. One lone return is a
    # reflection or a dropout: after the last exit in the capture a single cell
    # read -625mm with the room empty, and that alone manufactured an ENTRY.
    if len(signed) < DOORWAY_N:
        return None
    signed.sort()
    return signed[len(signed) // 2]


def body_position(matrix, scene):
    """Signed mm past the threshold of the body, from whichever view has it.

    The main field is authoritative whenever it can see the body. The door
    column is ONLY a fill for the blind spot that excluding that column
    creates, i.e. the moment the body is standing in the doorway.

    Using the door column unconditionally does not work, and the capture shows
    why: column H is a 60-degree slice of the room, not a peephole at the
    door, so a body walking about INSIDE also intersects it. Read as a doorway
    probe it produced a phantom EXIT at t=6.4s with the body 827mm inside the
    room, and three phantom events in a clip where nobody went near the door.

    Between those two there is a band that is neither: a couple of scattered
    returns that could be a sparse seated body or could be baseline drift. It
    ABSTAINS there rather than guessing, because guessing is what produced the
    worst artefacts. Mixing the two sources fabricates sweeps out of nothing:
    the main field reporting a body at -535mm and the probe reporting a shut
    door at +85mm alternate into a 620mm "crossing" while the subject sits
    perfectly still.
    """
    pts = list(geometry.foreground(matrix, scene.baseline).values())
    if len(pts) >= MIN_BODY:
        c = tuple(sum(p[i] for p in pts) / len(pts) for i in range(3))
        return scene.outside_door(c), "field"
    if len(pts) <= BLIND_MAX:
        p = door_probe(matrix, scene)
        return (p, "probe") if p is not None else (None, None)
    return None, None


class Crossing:
    """Emits ENTRY / EXIT when the body SWEEPS through the threshold plane.

    Pure and frame-driven, like SitLatch: update() takes one frame and a time
    and returns an event or None.

    A sign change alone is not enough, because a shut door parks on the plane
    and its sign dithers on noise. A crossing is a sign change plus travel:
    the reading must have moved MIN_SWEEP_MM across the plane within
    MAX_GAP_S. A stationary door never travels, so it never fires.
    """

    __slots__ = ("_hist", "_last_probe")

    def __init__(self):
        self._hist = []          # [(t, signed_mm, source)], recent only
        self._last_probe = None  # last doorway-probe reading, to spot a door

    def update(self, matrix, scene, t):
        s, src = body_position(matrix, scene)
        if s is None:
            return None
        if src == "probe":
            # A SHUT DOOR fills the doorway exactly as a body does, and it lies
            # ON the plane, so the probe reads it at ~0. What separates them is
            # that the door SITS STILL while a person sweeps through. A static
            # probe reading is therefore the door, and must not be handed to
            # the crossing test: the field measures a body and the probe
            # measures whatever is in the doorway, so a body simply walking out
            # of the FIELD (to the sink, off the left edge) would hand over
            # from "body at -400" to "door at 0", and that handover alone reads
            # as a 400mm sweep across the plane. Phantom EXIT with the operator
            # still in the room. The module docstring names this ("mixing the
            # two sources fabricates sweeps out of nothing"); nothing enforced
            # it. No previous probe reading means this one cannot be SHOWN to
            # be moving, so it abstains rather than being trusted: that costs
            # one frame at the handover and it is the difference between
            # reading a shut door as a departure and not.
            still = (self._last_probe is None
                     or abs(s - self._last_probe) < MIN_PROBE_MOVE_MM)
            self._last_probe = s
            if still:
                return None
        else:
            self._last_probe = None
        self._hist = [(ts, v, sr) for ts, v, sr in self._hist
                      if t - ts <= MAX_GAP_S]
        event = None
        for _ts, v, sr in self._hist:
            if (v > 0) != (s > 0) and abs(s - v) >= MIN_SWEEP_MM:
                event = ENTRY if v > 0 else EXIT
                break
        self._hist.append((t, s, src))
        if event:
            self._hist = [(t, s, src)]
        return event
