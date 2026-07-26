"""Room occupancy over the per-frame scene model (bathroom/scene.py).

The engine consumes the ORIENTED MATRIX directly: the scan already contains
the scene. No centroids, no smoothing, no velocity, no derived track. The
scene model reduces each frame to {interior_body, doorway_body} against the
static baseline; a tiny automaton turns doorway transits into entries and
exits. The door PANEL is telemetry only and is never consulted for
occupancy: it is operated from both sides, obscured by whoever operates it,
and its column is noisy.

Room semantics (operator rules):
- The door is the only way in or out. ENTERED (doorway transit resolving to
  interior presence) -> OCCUPIED; while already occupied it increments the
  occupant count. EXITED (doorway transit resolving to emptiness) -> EMPTY
  with the count zeroed: exits are deltas and cannot be attributed.
- Interior presence with NO doorway transit (mid-field artifact, presence
  already there at boot) NEVER latches occupancy: it is surfaced as a
  PRESENCE_NO_ENTRY event and the room state is unchanged.
- Boot and stream gaps are UNKNOWN, never assumed EMPTY. Only an observed
  transit moves the state out of UNKNOWN.
"""
from .scene import SceneAutomaton, read_frame


class RoomState:
    UNKNOWN = "UNKNOWN"
    EMPTY = "EMPTY"
    OCCUPIED = "OCCUPIED"


class Params:
    T_GAP = 5.0  # stream silence that forces UNKNOWN (and a fresh reading)


class RoomUpdate:
    __slots__ = ("room", "occupants", "event", "since")

    def __init__(self, room, occupants, event, since):
        self.room = room            # externally reported RoomState.*
        self.occupants = occupants  # 0 unless OCCUPIED
        self.event = event          # ENTRY_CONFIRMED/EXIT/PRESENCE_NO_ENTRY/
                                    # OCCLUDED/GAP_RESET/None
        self.since = since          # t when the reported room state last changed


class OccupancyEngine:
    def __init__(self, baseline):
        # required. It defaulted to None, which scene.read_frame then silently
        # resolved to classify.DEFAULT_BASELINE: BOTH live front-ends
        # constructed this with no argument, so the room engine has been
        # measuring the door and the passage against a pre-move scan (wrong by
        # up to 514mm on the fixture) while the posture path used the real one.
        # Nothing reported it, because a wrong baseline yields plausible states.
        self.baseline = baseline
        self.room = RoomState.UNKNOWN
        self.occupants = 0
        self.since = 0.0
        self._last_t = None
        self._auto = SceneAutomaton()

    def _set_room(self, room, occupants, t):
        if room != self.room:
            self.since = t
        self.room = room
        self.occupants = occupants

    def update(self, matrix, t, door_pct=None):
        """Feed one oriented 8x8 frame. door_pct is accepted for interface
        compatibility and ignored: the panel never drives occupancy."""
        event = None
        fresh = False
        if self._last_t is not None and t - self._last_t > Params.T_GAP:
            # unobserved stretch: nothing about the room can be attested
            self._set_room(RoomState.UNKNOWN, 0, t)
            self._auto.pending = None
            self._auto.occupied = False
            event = "GAP_RESET"
            fresh = True
        self._last_t = t

        reading = read_frame(matrix, self.baseline)
        for ev in self._auto.step(t, reading, fresh=fresh):
            if ev["kind"] == "ENTERED":
                if ev.get("already_occupied"):
                    # a doorway re-sighting by someone already inside is
                    # indistinguishable from a second entry in the scan;
                    # never inflate the count or re-mark on it
                    continue
                self._set_room(RoomState.OCCUPIED, 1, t)
                event = "ENTRY_CONFIRMED"
            elif ev["kind"] == "EXITED":
                # ANY observed exit empties the room (delta-only rule)
                self._set_room(RoomState.EMPTY, 0, t)
                event = "EXIT"
            elif ev["kind"] == "PRESENCE_NO_ENTRY" and event is None:
                # surfaced, never latched
                event = "PRESENCE_NO_ENTRY"
            elif ev["kind"] == "OCCLUDED" and event is None:
                # the view is blocked; room state is deliberately unchanged
                event = "OCCLUDED"
        return RoomUpdate(self.room, self.occupants, event, self.since)


def fmt_duration(seconds):
    """92 -> '1m32s', 3672 -> '1h01m'."""
    s = max(0, int(seconds))
    if s < 60:
        return "%ds" % s
    if s < 3600:
        return "%dm%02ds" % (s // 60, s % 60)
    return "%dh%02dm" % (s // 3600, (s % 3600) // 60)
