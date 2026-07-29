"""Report committed transitions to the Pi's in-memory transition archive.

ONLY THE DAEMON MARKS. This used to be constructed by heatmap_client.py and
bathroom/live.py as well, so every viewer left running wrote its own committed
transitions into the same 255-entry ring the washlet daemon writes to, with
nothing in the mark saying who sent it. The marks interleave, and the only way
to tell them apart afterwards is that each client's `t` is its own uptime: a
viewer running since Thursday stamped t=223111 next to the daemon's t=97, in
the same session, describing the same 25 seconds of one person using the
bathroom. The archive then disagreed with the daemon's own log and the
disagreement looked like a detection bug.

A viewer observes. It does not write to the system's event history.

The engines already track every transition; this only forwards them. Each
committed posture transition and each room-state change is sent as one
"MARK <json>" line back over the live stream socket (stream.send). The server
snapshots its latest RAW frame with the mark, in RAM, bounded at 255
transitions; nothing is detected or stored client-side.

Session boundaries follow the room engine: a session opens when the room
becomes OCCUPIED and closes when it stops being OCCUPIED (exit, abort, gap).
Retrieval/replay lives in replay.py at the repo root.
"""
import json

from . import stream
from .occupancy import RoomState


class Marker:
    def __init__(self):
        self._room = None

    def _send(self, obj):
        stream.send("MARK " + json.dumps(obj))

    def update(self, u, ru, t):
        """Forward this step's committed transitions, if any.

        u: engine.Update (posture), ru: occupancy.RoomUpdate, t: engine time.
        """
        prev = self._room
        self._room = ru.room
        if prev is not None and ru.room != prev:
            boundary = "open" if ru.room == RoomState.OCCUPIED else "close"
            self._send({"t": round(t, 3), "kind": "room", "from": prev,
                        "to": ru.room, "event": ru.event, "boundary": boundary})
        elif ru.event is not None:
            # room event without a state change (e.g. +1 entry while occupied)
            self._send({"t": round(t, 3), "kind": "room", "from": ru.room,
                        "to": ru.room, "event": ru.event, "boundary": None})
        if u.changed and u.event is not None:
            self._send({"t": round(t, 3), "kind": "posture",
                        "from": u.event.state.name, "to": u.committed.name,
                        "event": None, "boundary": None})

    def ir(self, ev, t):
        """Mark one IR fingerprint into the session archive.

        The server's persisted log keeps only {ts,t,kind,from,to,event,
        boundary} per mark (lidar_server._event_only), so the one field that
        must survive a power cycle, the FINGERPRINT, is packed into `from`;
        `event` carries the repeat tag. `to` is the bit count, which is
        redundant with the fingerprint's prefix but keeps the printed line
        readable without parsing.

        The server also dedupes on (kind, from, to, event, boundary) within
        2s: for IR that means a held button's identical repeat frames collapse
        to about one mark per 2s, while the press itself always lands because
        its `event` differs from its repeats' ("repeat" vs None). The clip
        recorder keeps every frame; the session archive records that the
        button was down, not the ~30 Hz re-send.
        """
        self._send({
            "t": round(t, 3), "kind": "ir",
            "from": ev.get("fp", "?"),
            "to": "%db" % ev.get("bits", 0),
            "event": "repeat" if ev.get("repeat") else None,
            "boundary": None,
            # ring-only detail (stripped from the persisted log)
            "repeat": ev.get("repeat"), "ir_ts": ev.get("ts"),
        })
