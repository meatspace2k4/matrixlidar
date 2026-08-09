"""Room occupancy: is anyone in the bathroom.

One input: bathroom/threshold.py, did a body cross the door plane, and which
way. That is the whole model.

  a body crossed the plane inward       -> OCCUPIED.
  a body crossed the plane outward      -> EMPTY. The door is the only way out.
  the view is BLOCKED                   -> HOLD, and say so. Blocked means
                                           the ENTIRE field is dead: a towel
                                           over the unit kills every ray the
                                           scan can answer. A frame with any
                                           ray answering is a scene, never an
                                           occlusion, however close the body.
  anything else                         -> HOLD.

The room has ONE door and the sensor points at it. A body cannot be in the
room without having crossed the plane inward in view of the sensor, and it
cannot leave except back out through the same plane. So the crossing is the
only occupancy evidence there is, in both directions, and the field holds no
veto: a body the field stops seeing is at the sink or in the tub, not gone,
and a posture flicker in a room with no observed ENTRY is a misread, not a
person.

THE FIELD USED TO LATCH OCCUPANCY (the PRESENCE event), on the argument that
seeing someone is direct evidence. It also latched off seeing no one: one
committed 1.0s STANDING flicker in an empty room latched OCCUPIED, and since
clearing required an EXIT crossing that no one was there to make, it held for
hours (07-29 11:29, live). Debouncing the latch shrinks that window; it
cannot close it, and a false OCCUPIED has no recovery path by design. The
latch is gone instead. Detection errors are detection bugs, to be fixed in
the classifier, not absorbed as room state.

The cost: a body already inside when the daemon starts (or seen only after a
stream gap) is not OCCUPIED until its exit is seen, because its entry was
never observed. That body still gets its audio: the washlet's sit trigger
runs on posture, not on room state.

ENTRY AND EXIT were once inferred from the DOOR PANEL. This door opens
outward, the sensor looks through the frame into the hallway, and the panel
can be operated from either side by someone who never comes in, so the panel
is not a proxy for a person. The crossing is a body crossing a fixed plane:
the sign of its distance to a plane derived from the scan, confirmed by
travel across it. door_pct stays telemetry, never evidence.

UNKNOWN survives for one thing only: before anything has been observed, and
after a stream gap long enough that nothing about the room can be attested.
"""
from . import roomscan, threshold
from .scene import Params as SceneParams
from .zones import N, NO_TARGET


class RoomState:
    UNKNOWN = "UNKNOWN"
    EMPTY = "EMPTY"
    OCCUPIED = "OCCUPIED"


BLOCKED_FRACTION = 1.0   # OCCLUSION MEANS THE ENTIRE SENSOR IS COVERED:
                         # every ray the scan can answer coming back dead is
                         # a towel over the unit. Anything less is a scene
                         # with a body in it, and this engine does not decide
                         # which cells are body and which are blockage. The
                         # old half-coverage rule did exactly that, and it
                         # switched the crossing off at the worst possible
                         # moment: a body passing the sensor on its way OUT
                         # drops most rays under minimum range, the frame
                         # read "blocked", and the exit swept past unseen
                         # (08-09 13:58; the room stayed OCCUPIED with
                         # nobody in it).


class Params:
    T_GAP = 5.0  # stream silence that forces UNKNOWN (and a fresh reading)


class RoomUpdate:
    __slots__ = ("room", "occupants", "event", "since", "blocked")

    def __init__(self, room, occupants, event, since, blocked=False):
        self.room = room            # externally reported RoomState.*
        self.occupants = occupants  # 0 unless OCCUPIED
        self.event = event          # ENTRY_CONFIRMED/EXIT/GAP_RESET/OCCLUDED
        self.since = since          # t when the reported room state last changed
        self.blocked = blocked      # the view is occluded NOW (level, not edge)


class OccupancyEngine:
    """Feed it oriented frames; it says whether the room is occupied."""

    def __init__(self, baseline=None):
        # The washlet daemon constructs this with no argument. That is fine: it
        # loads the same scan every other entrypoint loads. config.load_baseline
        # RAISES on a missing or corrupt scan rather than substituting one, so
        # this is not the old silent fallback to DEFAULT_BASELINE that had the
        # room engine measuring a pre-move pose while posture used the real one.
        if baseline is None:
            from . import config
            baseline = config.load_baseline()
        # required, never defaulted. It used to default to None, which
        # scene.read_frame resolved to classify.DEFAULT_BASELINE, and BOTH live
        # front-ends constructed this with no argument: the room engine ran
        # against a pre-move scan (wrong by up to 514mm on the fixture's own
        # cells) while the posture path used the real one. Nothing reported it,
        # because a wrong baseline yields plausible states.
        self.baseline = baseline
        self._scene = roomscan.Scene(baseline)
        self._crossing = threshold.Crossing()
        self._blocked_run = 0
        self._blocked_said = False
        # how many rays the SCAN itself answered. A ray it could not answer
        # (the whole door column, with the door open when the scan was taken)
        # says nothing about the view being blocked either, so the blocked test
        # is a fraction of these rather than an absolute count out of 64.
        self._answerable = sum(1 for r in range(N) for c in range(N)
                               if baseline[r][c] < NO_TARGET)
        self.room = RoomState.UNKNOWN
        self.occupants = 0
        self.since = 0.0
        self._last_t = None

    def _set_room(self, room, occupants, t):
        if room != self.room:
            self.since = t
        self.room = room
        self.occupants = occupants

    def update(self, frame, t, door_pct=None, crossed_out=False):
        """One frame. Accepts either an oriented 8x8 matrix or an engine.Update
        (the washlet daemon passes the latter), so the daemon's call site is
        unchanged. `door_pct` is accepted and ignored: the panel is telemetry,
        operated from both sides and obscured by whoever operates it, and it
        never decides occupancy. `crossed_out` is likewise accepted and ignored:
        the door-plane crossing is computed here now, from the frame, instead of
        being computed twice and passed back in."""
        matrix = getattr(frame, "matrix", frame)
        event = None
        if self._last_t is not None and t - self._last_t > Params.T_GAP:
            # unobserved stretch: nothing about the room can be attested
            self._set_room(RoomState.UNKNOWN, 0, t)
            # the crossing history is stale: a body committed before the gap
            # is not evidence of a body after it
            self._crossing = threshold.Crossing()
            event = "GAP_RESET"
        self._last_t = t

        # A BLOCKED VIEW IS NOT AN OBSERVATION. The empty room answers on
        # essentially every ray, so a frame that answers almost nowhere is not
        # an empty room, it is something over the sensor. Decide nothing from
        # it: no crossing, no posture, no state change. Say it once per run.
        dead = sum(1 for r in range(N) for c in range(N)
                   if matrix[r][c] >= NO_TARGET
                   and self.baseline[r][c] < NO_TARGET)
        if self._answerable and dead >= BLOCKED_FRACTION * self._answerable:
            self._blocked_run += 1
            debounced = self._blocked_run > SceneParams.DEBOUNCE_N
            if debounced and not self._blocked_said:
                self._blocked_said = True
                event = event or "OCCLUDED"
            return RoomUpdate(self.room, self.occupants, event, self.since,
                              blocked=debounced)
        self._blocked_run = 0
        self._blocked_said = False

        crossed = self._crossing.update(matrix, self._scene, t)
        if crossed == threshold.EXIT:
            # exits are deltas and cannot be attributed, so the count zeroes
            self._set_room(RoomState.EMPTY, 0, t)
            return RoomUpdate(self.room, self.occupants, "EXIT", self.since)
        if crossed == threshold.ENTRY:
            n = self.occupants + 1 if self.room == RoomState.OCCUPIED else 1
            self._set_room(RoomState.OCCUPIED, n, t)
            return RoomUpdate(self.room, self.occupants, "ENTRY_CONFIRMED",
                              self.since)

        # No crossing this frame. Hold. The field never decides the room:
        # OCCUPIED is entered by an ENTRY crossing and left by an EXIT
        # crossing, nothing else. The room has one door and the sensor points
        # at it, so a body cannot be in here without having crossed inward in
        # view, and a body the field stops seeing is at the sink or in the
        # tub, not gone. A posture flicker in an empty room is a misread, and
        # under the old PRESENCE rule one 1.0s flicker latched OCCUPIED with
        # no way back (07-29 11:29, held for hours).
        return RoomUpdate(self.room, self.occupants, event, self.since)


def fmt_duration(seconds):
    """92 -> '1m32s', 3672 -> '1h01m'."""
    s = max(0, int(seconds))
    if s < 60:
        return "%ds" % s
    if s < 3600:
        return "%dm%02ds" % (s // 60, s % 60)
    return "%dh%02dm" % (s // 3600, (s % 3600) // 60)
