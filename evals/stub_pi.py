#!/usr/bin/env python3
"""Stub Pi: speaks the lidar_server wire protocol from recorded frames.

Not an emulator and not a Pi image. One TCP server that does what
~/lidar_server.py does, so a client cannot tell the difference:

  - accepts connections, then writes one line per frame, 64 comma-separated
    raw row-major mm values, at the recorded rate
  - accepts "MARK <json>" from the client and keeps it in a RAM ring with a
    snapshot of the latest raw frame, grouped into sessions
  - answers "DUMP" with `ARCHIVE <json>` of the ring
  - answers "LOG [n]" with `ARCHIVE <json>` of persisted sessions

The frames come from a recorded clip, in RAW sensor order. The clips store
ORIENTED frames, so this un-orients them: the client's stream.orient is
`orient(m)[r][c] = m[c][7-r]`, and the inverse is applied here so the client
sees exactly what the sensor sent.

  python3 evals/stub_pi.py --clip data/clips/sitting_1785074559.json --port 8628
"""
import argparse
import json
import os
import socket
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from bathroom.zones import N                                     # noqa: E402

RING_MAX = 255


def unorient(m):
    """Inverse of stream.orient, so the wire carries raw sensor order.

    orient(raw)[r][c] == raw[c][N-1-r], so raw[a][b] == oriented[N-1-b][a].
    """
    return [[m[N - 1 - b][a] for b in range(N)] for a in range(N)]


def load_frames(paths):
    """[(dt_seconds, raw 8x8)] from one or more recorded clips, in order."""
    out = []
    for p in paths:
        with open(p) as f:
            doc = json.load(f)
        prev = None
        for fr in doc["frames"]:
            oriented = [fr["mm"][i * N:(i + 1) * N] for i in range(N)]
            dt = 0.0 if prev is None else max(0.0, fr["t"] - prev)
            prev = fr["t"]
            out.append((dt, unorient(oriented)))
    return out


class Stub:
    def __init__(self, frames, loop=True, empty_gap=0.0):
        self.frames = frames
        self.loop = loop
        self.empty_gap = empty_gap
        self.latest_raw = frames[0][1] if frames else [[4000] * N for _ in range(N)]
        self.ring = []          # [{session, ts, mark, frame}]
        self.sessions_closed = []
        self._session = 1
        self._lock = threading.Lock()

    # --- the wire ---------------------------------------------------------
    def serve(self, port, host="127.0.0.1"):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((host, port))
        srv.listen(8)
        sys.stderr.write("stub pi on %s:%d, %d frames, loop=%s\n"
                         % (host, port, len(self.frames), self.loop))
        sys.stderr.flush()
        while True:
            conn, _addr = srv.accept()
            threading.Thread(target=self._client, args=(conn,), daemon=True).start()

    def _client(self, conn):
        reader = threading.Thread(target=self._read, args=(conn,), daemon=True)
        reader.start()
        try:
            while True:
                for dt, raw in self.frames:
                    if dt:
                        time.sleep(dt)
                    with self._lock:
                        self.latest_raw = raw
                    line = ",".join(str(v) for row in raw for v in row) + "\n"
                    conn.sendall(line.encode())
                if not self.loop:
                    break
                if self.empty_gap:
                    time.sleep(self.empty_gap)
        except OSError:
            pass
        finally:
            try:
                conn.close()
            except OSError:
                pass

    def _read(self, conn):
        """MARK / DUMP / LOG from the client, same as the real server."""
        try:
            with conn.makefile("r") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("MARK "):
                        self._mark(json.loads(line[5:]))
                    elif line == "DUMP":
                        conn.sendall(("ARCHIVE " + json.dumps(self._dump())
                                      + "\n").encode())
                    elif line.startswith("LOG"):
                        conn.sendall(("ARCHIVE " + json.dumps(
                            {"sessions": self.sessions_closed}) + "\n").encode())
        except (OSError, ValueError):
            pass

    def _mark(self, mark):
        with self._lock:
            frame = ",".join(str(v) for row in self.latest_raw for v in row)
            if mark.get("kind") == "room" and mark.get("boundary") == "open":
                self._session += 1
            self.ring.append({"session": self._session, "ts": time.time(),
                              "mark": mark, "frame": frame})
            del self.ring[:-RING_MAX]
            if mark.get("kind") == "room" and mark.get("boundary") == "close":
                trs = [e for e in self.ring if e["session"] == self._session]
                self.sessions_closed.append(
                    {"id": self._session, "closed": time.time(),
                     "events": [e["mark"] for e in trs]})

    def _dump(self):
        sessions = []
        for e in self.ring:
            if not sessions or sessions[-1]["id"] != e["session"]:
                sessions.append({"id": e["session"], "transitions": []})
            sessions[-1]["transitions"].append(e)
        return {"sessions": sessions}


def main():
    ap = argparse.ArgumentParser(description="Stub Pi lidar server")
    ap.add_argument("--clip", action="append", required=True,
                    help="recorded clip to replay (repeatable, in order)")
    ap.add_argument("--port", type=int, default=8628)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--once", action="store_true", help="do not loop the clips")
    ap.add_argument("--gap", type=float, default=0.0,
                    help="seconds of pause between loops")
    args = ap.parse_args()
    frames = load_frames(args.clip)
    if not frames:
        sys.stderr.write("no frames loaded\n")
        return 1
    Stub(frames, loop=not args.once, empty_gap=args.gap).serve(args.port, args.host)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
