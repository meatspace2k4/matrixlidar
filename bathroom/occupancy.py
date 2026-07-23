"""Room occupancy engine: UNKNOWN / EMPTY / OCCUPIED on top of the posture
stack (docs/occupancy-design.md). Consumes engine.Update per frame; never
touches sensing, orientation, zones, or the posture classifier.

Rules (operator semantics):
- The door is the only exit. A body vanishing off the left FOV edge
  (sink/tub/shower) or mid-field stays OCCUPIED.
- Exits derive from the BODY TRACK's terminus, judged in column AND depth
  (this is a ToF depth map). A terminus in the door band at door-plane depth
  is an exit outright. A terminus in the door band at close depth (at the
  45-degree mount an exit passes close by the lens) is an exit only when the
  door is then operated and stays changed: track first, door corroborates.
- The door alone can never change room state: it is operated from both
  sides, it is obscured by whoever operates it, and its column is noisy. A
  door change after an off-side vanish (shower, mid-field occlusion,
  artifact) is ignored.
- Entries are observable: a door-side appear that penetrates into the field
  while already OCCUPIED increments the occupant count.
- Boot and stream gaps are UNKNOWN, never EMPTY; only an observed exit (or a
  confirmed entry after quiet) leaves UNKNOWN.
- Anti-latch: presence that never amounts to a real entry (hallway shimmer,
  a mid-field artifact, a vanish without door penetration and below the
  cumulative T_CONFIRM) decays back to the state the episode started from
  (EMPTY stays EMPTY, UNKNOWN stays UNKNOWN) so the entry edge re-arms and
  an unverified room is never promoted to a verified one.

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
    T_CONFIRM = 30.0     # CUMULATIVE presence needed to confirm occupancy
                         # WITHOUT an observed door crossing. A door crossing
                         # (CROSS_IN penetration) confirms instantly and does
                         # not use this. 2.0s let a 7s mid-field artifact
                         # latch OCCUPIED permanently (session 10); a real
                         # body that never crossed the door (present at boot)
                         # keeps producing evidence and still confirms.
    T_DECAY = 0.7        # quiet time before an unconfirmed MAYBE aborts
    T_ABSENT = 0.5       # sustained absence that ends a presence run (a vanish)
    T_GAP = 5.0          # stream gap that forces UNKNOWN
    K = 3                # centroid smoothing window (frames)
    L = 4                # trailing track points used for the vanish side
    DOOR_COL = 5.0       # smoothed centroid at/right of this is door-side
    FIELD_COL = 4.0      # centroid at/left of this is real field penetration
    LEFT_COL = 1.0       # centroid at/left of this is the left FOV edge
    D_DRIFT = 0.5        # cols of trailing drift that biases the vanish side
    # depth (this is a ToF depth map; the sensor is ~45 deg to the door, so
    # the approach to the door is a depth gradient, not just a column):
    DOOR_PLANE_MM = 900  # a body must have receded at least this far for a
                         # vanish to be a door exit. A body 400-500mm from the
                         # lens smears across columns (angular projection) and
                         # can reach the door COLUMN while standing mid-room;
                         # its depth exposes that. Unknown depth = permissive.
    DOOR_MOVE_PCT = 20.0 # door open% change (vs when the body vanished) that
                         # counts as "someone operated the door"
    DOOR_MOVE_N = 3      # consecutive empty-field frames the change must hold
    T_DOOR_STILL = 2.0   # stillness after the door moved before EXIT fires


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
        self._prior = RoomState.UNKNOWN  # room state before the current maybe
                                     # episode; unconfirmed evidence decays
                                     # BACK to this (UNKNOWN stays UNKNOWN:
                                     # honest boot; EMPTY stays EMPTY)
        self._cent = deque(maxlen=Params.K)   # raw centroid cols, current run
        self._depth = deque(maxlen=Params.K)  # raw body depths (mm), current run
        self._track = []             # (t, smoothed_col, smoothed_mm) per frame
        self._run_start = None
        self._run_presence = 0.0     # accumulated presence in the MAYBE episode
        self._appear_door = False
        self._counted = False        # this run already incremented occupants
        self._absent_since = None
        self._abort_at = None
        # door-motion exit tracking (door operated while nobody is visible)
        self._vanish_pct = None      # door open% when the field went empty
        self._door_moved_n = 0       # consecutive frames the pct change held
        self._door_moved_at = None   # t when the door movement was confirmed

    # ---- helpers ---------------------------------------------------------

    def _set_room(self, room, occupants, t):
        if room != self.room:
            self.since = t
        self.room = room
        self.occupants = occupants

    def _end_run(self):
        self._cent.clear()
        self._depth.clear()
        self._track = []
        self._run_start = None
        self._appear_door = False
        self._counted = False
        self._absent_since = None

    def _vanish_terminus(self):
        """Classify where the track TERMINATED. The body is the primary
        object; the door is never consulted here.

          "deep"  trailing centroid in the door band, receded to the door
                  plane (>= DOOR_PLANE_MM): an unambiguous walk-out.
          "near"  trailing centroid in the door band at close depth: at the
                  45-degree mount an exit passes close by the lens at the
                  door column, indistinguishable by track alone from a close
                  pass that stays inside. Needs corroboration.
          "off"   anywhere else: not an exit terminus, and no later door
                  signal may reinterpret it (doors are operated from both
                  sides; a hallway-side operation says nothing about a body
                  that vanished mid-field or left).
        """
        if not self._track:
            return "off"
        tail = self._track[-Params.L:]
        end = sum(p[1] for p in tail) / len(tail)
        drift = tail[-1][1] - tail[0][1]
        col_ok = (end >= Params.DOOR_COL
                  or (end >= Params.DOOR_COL - 1.0 and drift >= Params.D_DRIFT))
        if not col_ok:
            return "off"
        depths = [p[2] for p in tail if p[2] is not None]
        if not depths:
            return "deep"  # no depth info: legacy column-only behavior
        end_mm = sum(depths) / len(depths)
        return "deep" if end_mm >= Params.DOOR_PLANE_MM else "near"

    # ---- one step --------------------------------------------------------

    def update(self, u, t, door_pct=None):
        """u: engine.Update (only .instant and .features are read).
        door_pct: current door open% (detect.door_open_pct), used for the
        door-motion exit rule; None disables that rule for this step."""
        event = None

        if self._last_t is not None and t - self._last_t > Params.T_GAP:
            self._set_room(RoomState.UNKNOWN, 0, t)
            self._maybe = False
            self._abort_at = None
            self._vanish_pct = None
            self._door_moved_at = None
            self._door_moved_n = 0
            self._end_run()
            event = "GAP_RESET"
        self._last_t = t

        cells = u.features["occ_cells"]
        present = u.instant != State.EMPTY and bool(cells)

        if present:
            self._abort_at = None
            self._vanish_pct = None       # body visible again: door-motion moot
            self._door_moved_at = None
            self._door_moved_n = 0
            self._cent.append(sum(c for _r, c in cells) / float(len(cells)))
            smoothed = sum(self._cent) / len(self._cent)
            body_mm = u.features.get("body_mm")
            if body_mm is not None:
                self._depth.append(body_mm)
            smoothed_mm = (sum(self._depth) / len(self._depth)
                           if self._depth else None)
            if self._run_start is None:
                self._run_start = t
                self._appear_door = smoothed >= Params.DOOR_COL
            self._absent_since = None
            self._track.append((t, smoothed, smoothed_mm))
            run_len = t - self._run_start

            if not self._maybe and self.room in (RoomState.EMPTY,
                                                 RoomState.UNKNOWN):
                # evidence: report OCCUPIED immediately, confirm below.
                # UNKNOWN takes the same maybe path as EMPTY: it must not
                # hard-latch on unconfirmed evidence (session 10), and if the
                # episode decays it returns to UNKNOWN, not EMPTY.
                self._prior = self.room
                self._maybe = True
                self._run_presence = 0.0
                self._set_room(RoomState.OCCUPIED, 1, t)

            penetrated = self._appear_door and smoothed <= Params.FIELD_COL
            if self._maybe and (penetrated
                                or self._run_presence + run_len >= Params.T_CONFIRM):
                self._maybe = False
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
                    terminus = self._vanish_terminus()
                    run_len = self._absent_since - self._run_start
                    self._end_run()
                    if terminus == "deep":
                        # ANY observed exit empties the room (delta-only rule)
                        self._maybe = False
                        self._set_room(RoomState.EMPTY, 0, t)
                        event = "EXIT"
                    elif terminus == "near":
                        # door-side close-pass terminus: an exit iff the door
                        # is then operated; remember the door state at the
                        # vanish for that corroboration
                        if door_pct is not None:
                            self._vanish_pct = door_pct
                        if self._maybe:
                            self._run_presence += run_len
                            self._abort_at = t + Params.T_DECAY
                    else:
                        # off-side terminus: never an exit candidate; door
                        # signals must not reinterpret it (hallway-side
                        # operation, noise, obscured column)
                        self._vanish_pct = None
                        if self._maybe:
                            # unconfirmed body gone off-side: arm the anti-latch
                            self._run_presence += run_len
                            self._abort_at = t + Params.T_DECAY
            if (self._abort_at is not None and t >= self._abort_at
                    and self._maybe):
                self._maybe = False
                self._abort_at = None
                # decay to the state the episode started from, never assume
                # EMPTY out of an UNKNOWN boot
                self._set_room(self._prior, 0, t)
                event = "ENTRY_ABORTED"
            # door corroboration for a NEAR terminus only (_vanish_pct is set
            # exclusively by a door-side close-pass vanish): the track said
            # the body ended at the door; the door then being operated and
            # staying changed, with stillness, completes the exit. Off-side
            # vanishes never set _vanish_pct, so a hallway-side door
            # operation or a noisy/obscured door column can never empty the
            # room on its own.
            if (event is None and self.room == RoomState.OCCUPIED
                    and door_pct is not None and self._vanish_pct is not None):
                if self._door_moved_at is None:
                    if abs(door_pct - self._vanish_pct) >= Params.DOOR_MOVE_PCT:
                        self._door_moved_n += 1
                        if self._door_moved_n >= Params.DOOR_MOVE_N:
                            self._door_moved_at = t
                    else:
                        self._door_moved_n = 0
                elif t - self._door_moved_at >= Params.T_DOOR_STILL:
                    self._maybe = False
                    self._vanish_pct = None
                    self._door_moved_at = None
                    self._door_moved_n = 0
                    self._set_room(RoomState.EMPTY, 0, t)
                    event = "EXIT"

        return RoomUpdate(self.room, self.occupants, event, self.since)


def fmt_duration(seconds):
    """92 -> '1m32s', 3672 -> '1h01m'."""
    s = max(0, int(seconds))
    if s < 60:
        return "%ds" % s
    if s < 3600:
        return "%dm%02ds" % (s // 60, s % 60)
    return "%dh%02dm" % (s // 3600, (s % 3600) // 60)
