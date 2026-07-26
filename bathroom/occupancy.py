"""Room occupancy: is anyone in the bathroom.

Two inputs, and both of them measure a BODY:

  bathroom/posture3d.py    is there a body in the field
  bathroom/threshold.py  did a body cross the door plane, and which way

That is the whole model.

  a body is in the field                -> OCCUPIED. Seeing someone is direct
                                           evidence that someone is there.
  a body crossed the plane outward      -> EMPTY. The door is the only way out.
  a body crossed the plane inward       -> OCCUPIED.
  the view is BLOCKED                   -> HOLD, and say so. Something over the
                                           sensor (a towel, a coat on the hook,
                                           a hand) kills the returns across the
                                           whole field. That frame is not
                                           evidence of anything, least of all
                                           of an empty room, so nothing is
                                           decided off it.
  the field is empty, nothing crossed   -> HOLD. THE FIELD IS NOT THE ROOM. It
                                           does not cover the sink or the tub,
                                           so a body that walks out of the
                                           FIELD, off to the left, has not left
                                           the ROOM. Only a crossing empties it.

WHAT THIS REPLACES, AND WHY IT WAS WRONG

The previous engine reduced each frame to {interior_body, doorway_body} with
scene.read_frame and ran a doorway automaton over that. Three things were
broken at once, and each one hid the others:

- INTERIOR PRESENCE COULD NOT LATCH OCCUPANCY. A body seen in the room with no
  observed entry was reported as PRESENCE_NO_ENTRY and the room stayed UNKNOWN.
  So the operator sat on the toilet in full view of the sensor and the room
  line read `UNKNOWN(0) 8m31s`. UNKNOWN is not the cautious answer when a
  person is visibly sitting there, it is a wrong one that declines to commit.

- interior_body COULD NOT SEE A SEATED BODY. It wanted 7 cells reading 300mm
  nearer than the empty room, and a seated body is mostly a HOLE: the beam goes
  out where the scan found a surface and nothing comes back. posture.py counts
  those, so this does too, by asking posture.py.

- ENTRY AND EXIT WERE INFERRED FROM THE DOOR. Doorway-band cells followed by
  interior cells meant ENTERED. This door opens outward, the sensor looks
  straight through the frame into the hallway, and it can be opened and shut
  from the hallway by someone who never comes in, so the panel is not a proxy
  for a person. It is a body crossing a fixed plane now: the sign of its
  distance to a plane derived from the scan, confirmed by travel across it.

UNKNOWN survives for one thing only: before anything has been observed, and
after a stream gap long enough that nothing about the room can be attested. It
is never used to avoid committing to what the sensor can plainly see.
"""
from . import roomscan, threshold
from .engine import StateEngine
from .scene import Params as SceneParams
from .zones import N, NO_TARGET, State


class RoomState:
    UNKNOWN = "UNKNOWN"
    EMPTY = "EMPTY"
    OCCUPIED = "OCCUPIED"


BLOCKED_FRACTION = 0.5   # more than half the rays the scan answers coming back
                         # dead is not an empty room, it is a blocked view


class Params:
    T_GAP = 5.0  # stream silence that forces UNKNOWN (and a fresh reading)


class RoomUpdate:
    __slots__ = ("room", "occupants", "event", "since", "blocked")

    def __init__(self, room, occupants, event, since, blocked=False):
        self.room = room            # externally reported RoomState.*
        self.occupants = occupants  # 0 unless OCCUPIED
        self.event = event          # ENTRY_CONFIRMED/EXIT/PRESENCE/GAP_RESET
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
        # the DEBOUNCED posture, not the raw per-frame verdict. A handful of
        # frames of an empty room read as a body (5 of 111 in the walk-through
        # capture), and latching occupancy off one of those put the room back
        # to OCCUPIED a second after the operator had walked out. The temporal
        # engine already exists to absorb exactly that; occupancy had been
        # re-deciding per frame beside it.
        self._posture = StateEngine(baseline=baseline)
        self._blocked_run = 0
        self._blocked_said = False
        # when the last outward crossing happened. A body walking OUT through
        # the plane is still a body in the field on those same frames, so
        # PRESENCE re-latched OCCUPIED one frame after the EXIT and the room
        # said occupied with the bathroom empty. Observed live at 13:38:12:
        #   room -> EMPTY     (EXIT)
        #   room -> OCCUPIED  (PRESENCE)
        # After a crossing out, presence has to see the field go clear before it
        # may latch again.
        self._exited_at = None
        self._clear_since_exit = False
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
            # the crossing history AND the posture debounce are both stale: a
            # body committed before the gap is not evidence of a body after it,
            # and holding that history reported the pre-gap occupant on the
            # first frame back.
            self._crossing = threshold.Crossing()
            self._posture = StateEngine(baseline=self.baseline)
            event = "GAP_RESET"
        self._last_t = t

        # A BLOCKED VIEW IS NOT AN OBSERVATION. The empty room answers on
        # essentially every ray, so a frame that answers almost nowhere is not
        # an empty room, it is something over the sensor. Decide nothing from
        # it: no crossing, no posture, no state change. Say it once per run.
        dead = sum(1 for r in range(N) for c in range(N)
                   if matrix[r][c] >= NO_TARGET
                   and self.baseline[r][c] < NO_TARGET)
        if self._answerable and dead > BLOCKED_FRACTION * self._answerable:
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
            self._exited_at = t
            self._clear_since_exit = False
            return RoomUpdate(self.room, self.occupants, "EXIT", self.since)
        if crossed == threshold.ENTRY:
            n = self.occupants + 1 if self.room == RoomState.OCCUPIED else 1
            self._set_room(RoomState.OCCUPIED, n, t)
            return RoomUpdate(self.room, self.occupants, "ENTRY_CONFIRMED",
                              self.since)

        state = self._posture.update(matrix, t).committed
        if state == State.EMPTY:
            self._clear_since_exit = True
        if state != State.EMPTY:
            # a body is in the field. That is the whole argument, EXCEPT
            # immediately after an outward crossing: the departing body is still
            # in the field on those frames, and re-latching off it is how the
            # room ended up occupied with nobody in it.
            if self._exited_at is not None and not self._clear_since_exit:
                return RoomUpdate(self.room, self.occupants, event, self.since)
            if self.room != RoomState.OCCUPIED:
                self._set_room(RoomState.OCCUPIED, 1, t)
                event = "PRESENCE"
            return RoomUpdate(self.room, self.occupants, event, self.since)

        # The field is empty, which is NOT an empty room. Hold what it was.
        return RoomUpdate(self.room, self.occupants, event, self.since)


def fmt_duration(seconds):
    """92 -> '1m32s', 3672 -> '1h01m'."""
    s = max(0, int(seconds))
    if s < 60:
        return "%ds" % s
    if s < 3600:
        return "%dm%02ds" % (s // 60, s % 60)
    return "%dh%02dm" % (s // 3600, (s % 3600) // 60)
