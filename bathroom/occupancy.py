"""Room occupancy engine: UNKNOWN / EMPTY / OCCUPIED on top of the posture
stack (docs/occupancy-design.md). Consumes engine.Update per frame; never
touches sensing, orientation, zones, or the posture classifier.

Rules (operator semantics):
- The door is the only exit. A body vanishing off the left FOV edge
  (sink/tub/shower) or mid-field stays OCCUPIED.
- Exits are deltas: bodies leaving cannot be counted, so ANY door-side vanish
  marks the room EMPTY, from any state.
- Entries are observable: a door-side appear that penetrates into the field
  while already OCCUPIED increments the occupant count.
- Boot and stream gaps are UNKNOWN, never EMPTY; only an observed exit (or a
  confirmed entry after quiet) leaves UNKNOWN.
- Anti-latch: presence that never amounts to a real entry (hallway shimmer,
  a vanish before T_CONFIRM without door penetration) decays back to EMPTY so
  the EMPTY->OCCUPIED edge re-arms.

Internally there is a MAYBE phase (body evidence not yet confirmed); it is
reported externally as OCCUPIED so consumers react instantly, but it can
decay (ENTRY_ABORTED). Events: ENTRY_CONFIRMED, ENTRY_ABORTED, EXIT,
GAP_RESET.
"""
from collections import deque

from .zones import State


class RoomState:
    UNKNOWN = "UNKNOWN"
    EMPTY = "EMPTY"
    OCCUPIED = "OCCUPIED"


class Params:
    T_CONFIRM = 2.0      # presence needed to confirm occupancy without penetration
    T_DECAY = 0.7        # quiet time before an unconfirmed MAYBE aborts
    T_ABSENT = 0.5       # sustained absence that ends a presence run (a vanish)
    T_GAP = 5.0          # stream gap that forces UNKNOWN
    K = 3                # centroid smoothing window (frames)
    L = 4                # trailing track points used for the vanish side
    DOOR_COL = 5.0       # smoothed centroid at/right of this is door-side
    FIELD_COL = 4.0      # centroid at/left of this is real field penetration
    LEFT_COL = 1.0       # centroid at/left of this is the left FOV edge
    D_DRIFT = 0.5        # cols of trailing drift that biases the vanish side


class RoomUpdate:
    __slots__ = ("room", "occupants", "event", "since")

    def __init__(self, room, occupants, event, since):
        self.room = room            # externally reported RoomState.*
        self.occupants = occupants  # 0 unless OCCUPIED
        self.event = event          # ENTRY_CONFIRMED/ENTRY_ABORTED/EXIT/GAP_RESET/None
        self.since = since          # t when the reported room state last changed


class OccupancyEngine:
    def __init__(self):
        self.room = RoomState.UNKNOWN
        self.occupants = 0
        self.since = 0.0
        self._last_t = None
        self._maybe = False          # OCCUPIED reported but entry not confirmed
        self._cent = deque(maxlen=Params.K)   # raw centroid cols, current run
        self._track = []             # (t, smoothed_col) for the current run
        self._run_start = None
        self._run_presence = 0.0     # accumulated presence in the MAYBE episode
        self._appear_door = False
        self._counted = False        # this run already incremented occupants
        self._absent_since = None
        self._abort_at = None

    # ---- helpers ---------------------------------------------------------

    def _set_room(self, room, occupants, t):
        if room != self.room:
            self.since = t
        self.room = room
        self.occupants = occupants

    def _end_run(self):
        self._cent.clear()
        self._track = []
        self._run_start = None
        self._appear_door = False
        self._counted = False
        self._absent_since = None

    def _vanish_door_side(self):
        """Door-side vanish: trailing centroid at the door band, or one column
        short of it while drifting doorward (a fast exit between frames)."""
        if not self._track:
            return False
        tail = self._track[-Params.L:]
        end = sum(c for _t, c in tail) / len(tail)
        drift = tail[-1][1] - tail[0][1]
        if end >= Params.DOOR_COL:
            return True
        return end >= Params.DOOR_COL - 1.0 and drift >= Params.D_DRIFT

    # ---- one step --------------------------------------------------------

    def update(self, u, t):
        """u: engine.Update (only .instant and .features are read)."""
        event = None

        if self._last_t is not None and t - self._last_t > Params.T_GAP:
            self._set_room(RoomState.UNKNOWN, 0, t)
            self._maybe = False
            self._abort_at = None
            self._end_run()
            event = "GAP_RESET"
        self._last_t = t

        cells = u.features["occ_cells"]
        present = u.instant != State.EMPTY and bool(cells)

        if present:
            self._abort_at = None
            self._cent.append(sum(c for _r, c in cells) / float(len(cells)))
            smoothed = sum(self._cent) / len(self._cent)
            if self._run_start is None:
                self._run_start = t
                self._appear_door = smoothed >= Params.DOOR_COL
            self._absent_since = None
            self._track.append((t, smoothed))
            run_len = t - self._run_start

            if self.room == RoomState.EMPTY:
                # evidence: report OCCUPIED immediately, confirm below
                self._maybe = True
                self._run_presence = 0.0
                self._set_room(RoomState.OCCUPIED, 1, t)

            penetrated = self._appear_door and smoothed <= Params.FIELD_COL
            if self._maybe and (penetrated
                                or self._run_presence + run_len >= Params.T_CONFIRM):
                self._maybe = False
                self._counted = True
                event = "ENTRY_CONFIRMED"
            elif self.room == RoomState.UNKNOWN and run_len >= Params.T_CONFIRM:
                # present since boot: they never "entered", but they are here
                self._set_room(RoomState.OCCUPIED, 1, t)
                self._counted = True
                event = "ENTRY_CONFIRMED"
            elif (self.room == RoomState.OCCUPIED and not self._maybe
                    and not self._counted and penetrated):
                # a new door-side appear that came into the field = +1 entry
                self._counted = True
                self.occupants += 1
                event = "ENTRY_CONFIRMED"
        else:
            if self._run_start is not None:
                if self._absent_since is None:
                    self._absent_since = t
                elif t - self._absent_since >= Params.T_ABSENT:
                    # the run is over: a vanish
                    door_side = self._vanish_door_side()
                    run_len = self._absent_since - self._run_start
                    self._end_run()
                    if door_side:
                        # ANY observed exit empties the room (delta-only rule)
                        self._maybe = False
                        self._set_room(RoomState.EMPTY, 0, t)
                        event = "EXIT"
                    elif self._maybe:
                        # unconfirmed body gone off-side: arm the anti-latch
                        self._run_presence += run_len
                        self._abort_at = t + Params.T_DECAY
                    # confirmed OCCUPIED off-side vanish: stay OCCUPIED
            if (self._abort_at is not None and t >= self._abort_at
                    and self._maybe):
                self._maybe = False
                self._abort_at = None
                self._set_room(RoomState.EMPTY, 0, t)
                event = "ENTRY_ABORTED"

        return RoomUpdate(self.room, self.occupants, event, self.since)


def fmt_duration(seconds):
    """92 -> '1m32s', 3672 -> '1h01m'."""
    s = max(0, int(seconds))
    if s < 60:
        return "%ds" % s
    if s < 3600:
        return "%dm%02ds" % (s // 60, s % 60)
    return "%dh%02dm" % (s // 3600, (s % 3600) // 60)
