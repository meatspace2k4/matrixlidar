"""Wave-to-stop: a hand oscillating NEAR the sensor, tracked in 4D.

The decision is made on a trajectory through 3D space over time, never on
the flat 8x8 array. Each frame's returns are LIFTED to metric 3D points
(cell ray direction x measured distance: the same construction waveview
draws), the points inside the near-field are clustered to a centroid, and a
wave is that centroid's LATERAL oscillation: direction reversals too fast
and too many to be anything but a hand shaken at the lens.

What can NOT fire it, by construction rather than by tuning:

  - anything past NEAR_Z_MM: sitting, standing, standing up, walking at
    normal range never produce near-field points at all
  - a body crossing close to the lens (LENS_MM territory): translation is
    one direction of travel, at most one reversal per pass; a wave needs
    REVERSALS_MIN inside WINDOW_S, which is a >=2Hz shake
  - a stationary hand or a held object: reversals require the swing to
    cover AMP_MM each way; jitter does not travel that far

There is no 2D fallback anywhere. A malformed frame raises. If this module
cannot lift and track, the caller crashes, loudly, by design.
"""
from . import geometry
from .zones import N, NO_TARGET
NEAR_Z_MM = 600       # the near-field: a wave happens AT the sensor. The
                      # nearest attested walking pass reads 290mm and normal
                      # standing 680+, so a hand at arm's length from the
                      # unit is unmistakably inside this and a body is not
                      # reliably inside it long enough to oscillate.
MIN_PTS = 2           # a hand subtends few cells; one lone cell is noise
AMP_MM = 70           # each swing must travel this far laterally, each way
REVERSALS_MIN = 2     # direction changes inside the window. The stream runs
                      # ~7.3Hz and a real shake aliases hard: the attested
                      # wave capture (wave_1786304046, 61-67s) shows swings
                      # of +117mm and -163mm inside a second, i.e. ~2
                      # qualifying reversals per window, not the 3 a faster
                      # sensor would see. Two 70mm+ reversals inside 2s is
                      # still there-back-there: a body passing the lens
                      # reverses at most once per pass.
WINDOW_S = 2.0
REFRACTORY_S = 2.0    # one WAVE per gesture, not one per reversal
GONE_S = 0.45         # the hand must be unseen this long before the gesture
                      # is called over. At ~7Hz a shaking hand blinks out for
                      # a frame or two mid-swing (it thins edge-on and slips
                      # between rays): the live wave of 08-09 13:57 arrived
                      # as sub-second near-field bursts split by exactly such
                      # blinks, and a tracker that reset on the first empty
                      # frame threw away its reversals every time. Three
                      # missing frames (~0.43s) is a blink; longer is the
                      # hand actually withdrawn.

WAVE = "WAVE"


def lift(r, c, mm):
    """One return -> a metric 3D point, via the ONE spatial model this repo
    has (geometry.unproject, the same math the crossing runs on). Raises on
    a non-return: a NO_TARGET has no position, and asking for one is a
    caller bug, not a value."""
    if not 0 < mm < NO_TARGET:
        raise ValueError("lift of a non-return: %r" % (mm,))
    return geometry.unproject(r, c, mm)


def near_centroid(matrix, near_z=NEAR_Z_MM):
    """Centroid of the near-field 3D points, or None if fewer than MIN_PTS.

    None is a stated observation ("no near-field cluster this frame"), not a
    fallback: the tracker consumes it as evidence the hand is gone.
    """
    if len(matrix) != N or any(len(row) != N for row in matrix):
        raise ValueError("frame is not %dx%d" % (N, N))
    pts = []
    for r in range(N):
        for c in range(N):
            v = matrix[r][c]
            if 0 < v < near_z:
                pts.append(lift(r, c, v))
    if len(pts) < MIN_PTS:
        return None
    n = float(len(pts))
    return (sum(p[0] for p in pts) / n,
            sum(p[1] for p in pts) / n,
            sum(p[2] for p in pts) / n)


class WaveTracker:
    """Feed it (matrix, t) per frame; it says WAVE when a hand shakes at
    the lens. Pure state machine over the 3D centroid track, no I/O."""

    __slots__ = ("_last_fire", "_axes", "_last_seen")

    def __init__(self):
        # One swing tracker per axis. A wave is oscillation of the near
        # cluster in ANY direction: the attested capture shakes hardest
        # TOWARD the sensor (z, 19 direction flips) and vertically (y, 7),
        # with side-to-side (x) weakest at 5, and an x-only rule missed the
        # live waves that shook the other way.
        self._axes = [{"extreme": None, "dir": 0, "revs": []}
                      for _ in range(3)]
        self._last_fire = None
        self._last_seen = None  # t of the last frame with a near-field hand

    def _reset(self):
        for a in self._axes:
            a["extreme"] = None
            a["dir"] = 0
            a["revs"] = []
        self._last_seen = None

    def update(self, matrix, t):
        c = near_centroid(matrix)
        if c is None:
            # unseen for a blink: hold the gesture. Unseen for GONE_S: over.
            if self._last_seen is not None and t - self._last_seen > GONE_S:
                self._reset()
            return None
        self._last_seen = t
        fired = False
        for i, a in enumerate(self._axes):
            v = c[i]
            if a["extreme"] is None:
                a["extreme"] = v
                continue
            swing = v - a["extreme"]
            d = 1 if swing > 0 else -1
            if a["dir"] == 0:
                if abs(swing) >= AMP_MM:
                    a["dir"] = d
                    a["extreme"] = v
            elif d != a["dir"]:
                if abs(swing) >= AMP_MM:
                    # travelled AMP_MM against the run: a reversal
                    a["revs"].append(t)
                    a["dir"] = d
                    a["extreme"] = v
            else:
                if (d > 0 and v > a["extreme"]) or (d < 0 and v < a["extreme"]):
                    a["extreme"] = v
            a["revs"] = [rt for rt in a["revs"] if t - rt <= WINDOW_S]
            if len(a["revs"]) >= REVERSALS_MIN:
                fired = True
        if fired:
            if self._last_fire is None or t - self._last_fire > REFRACTORY_S:
                self._last_fire = t
                for a in self._axes:
                    a["revs"] = []
                return WAVE
        return None
