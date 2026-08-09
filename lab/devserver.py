#!/usr/bin/env python3
"""Dev server: the Pi's server->client pipeline, and nothing else of the Pi.

Reproduces exactly what a client sees on port 8628 (one CSV line of 64 mm
values per frame, STATE/IR broadcast lines, WHAT/DUMP/LOG answered) with
none of the on-device business logic: no audio, no latch, no engines. Two
sources, chosen at start:

  --clip a.json [--clip b.json]   replay recorded clips at their own rate,
                                  looped, with their IR logs on the wire
  --live [host]                   relay the REAL server. ONE connection
                                  upstream no matter how many clients hang
                                  off this process: the fan-out lives here,
                                  so the production pipeline never grows a
                                  second consumer per viewer.
  --follow [host]                 live relay that turns into a replay of
                                  each visit as it ends. The relay already
                                  carries every dense frame and every STATE
                                  line, so follow keeps a rolling in-memory
                                  buffer of the raw wire and, when the
                                  daemon's own EXIT event rides past, loops
                                  the buffered visit VERBATIM: same frames,
                                  same cadence, same state feed the live
                                  view had. Nothing is watched, nothing
                                  polls, nothing is fetched twice, and the
                                  replay cannot look different from live
                                  because it is a recording of it.

Everything is in memory. Nothing is written to disk, ever: this exists so
the SD card and the production daemon are both left alone while the wave
work iterates against real frames.

The browser cannot open raw TCP, so the same event stream is also served
over HTTP as Server-Sent Events, plus the waveview/ page itself:

  http://localhost:8899/          the 3D TUI (waveview/index.html)
  http://localhost:8899/feed      SSE: every wire line, verbatim
  tcp  localhost:8629             the wire protocol, for python clients

  python3 lab/devserver.py --clip data/clips/sitting_1785074559.json
  python3 lab/devserver.py --live bath-pi.local
"""
import argparse
import http.server
import json
import os
import socket
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from bathroom.zones import N                                    # noqa: E402

TCP_PORT = 8629
HTTP_PORT = 8899

# fan-out state: every consumer (tcp client, sse response) owns a queue of
# lines. One producer appends to all. Plain dict guarded by one lock.
_subs = {}          # id -> {"lines": [..], "cv": Condition}
_subs_lock = threading.Lock()
_last_state = None  # latest STATE line, replayed to a client that asks WHAT
_next_id = [0]


def _subscribe():
    with _subs_lock:
        _next_id[0] += 1
        sid = _next_id[0]
        _subs[sid] = {"lines": [], "cv": threading.Condition()}
        return sid


def _unsubscribe(sid):
    with _subs_lock:
        _subs.pop(sid, None)


def _publish(line):
    global _last_state
    if line.startswith("STATE "):
        _last_state = line
    with _subs_lock:
        subs = list(_subs.values())
    for s in subs:
        with s["cv"]:
            s["lines"].append(line)
            if len(s["lines"]) > 512:      # a stalled consumer drops oldest,
                del s["lines"][:-512]      # it never stalls the producer
            s["cv"].notify()


def _next_lines(sid, timeout=5.0):
    with _subs_lock:
        s = _subs.get(sid)
    if s is None:
        return None
    with s["cv"]:
        if not s["lines"]:
            s["cv"].wait(timeout)
        out, s["lines"][:] = list(s["lines"]), []
        return out


# --- sources ---------------------------------------------------------------

def _unorient(m):
    """Inverse of stream.orient: the wire carries raw sensor order."""
    return [[m[N - 1 - b][a] for b in range(N)] for a in range(N)]


def _clip_source(paths):
    """Loop the clips forever at their recorded rate, IR included."""
    plan = []           # (dt, line) across all clips, in order
    for p in paths:
        with open(p) as f:
            doc = json.load(f)
        events = []
        for fr in doc["frames"]:
            oriented = [fr["mm"][i * N:(i + 1) * N] for i in range(N)]
            raw = _unorient(oriented)
            events.append((fr["t"],
                           ",".join(str(v) for row in raw for v in row)))
            if fr.get("state_engine"):
                events.append((fr["t"], "STATE " + json.dumps(
                    {"posture": fr["state_engine"], "room": "REPLAY",
                     "occupants": 0, "t": fr["t"], "event": None,
                     "door": fr.get("door_open_pct"), "occ": 0, "dir": "",
                     "since": 0, "blocked": False, "sit_s": 0})))
        for ev in doc.get("ir") or []:
            events.append((ev.get("t", 0.0), "IR " + json.dumps(ev)))
        events.sort(key=lambda e: e[0])
        prev = None
        for t, line in events:
            plan.append((0.0 if prev is None else max(0.0, t - prev), line))
            prev = t
    sys.stderr.write("clip source: %d wire events from %d clip(s)\n"
                     % (len(plan), len(paths)))
    while True:
        for dt, line in plan:
            if dt:
                time.sleep(dt)
            _publish(line)


BUFFER_MAX = 20000     # wire lines kept (frames ~7-10/s + STATE ~1/s:
                       # comfortably over 20 minutes of visit)
PRE_ROLL_S = 3.0       # replay starts this far before the entry


def _live_source(host, port, follow=False):
    """ONE upstream connection, relaying every line verbatim. Reconnects.

    With follow=True the raw wire is also buffered in memory, and the
    daemon's own EXIT event freezes the buffered visit (from PRE_ROLL_S
    before its entry) and loops it verbatim: the replay is a recording of
    the live wire, so its frame rate and its state feed are exactly what
    live showed. The next EXIT swaps the loop. No files, no polling, no
    second fetch of anything."""
    from collections import deque
    buf = deque(maxlen=BUFFER_MAX)      # (wall_t, line)
    entry_wall = [None]
    replaying = {"stop": None}
    while True:
        try:
            sock = socket.create_connection((host, port), timeout=5)
            sock.settimeout(10)
            sys.stderr.write("live source: connected to %s:%d (the one "
                             "upstream consumer)\n" % (host, port))
            sock.sendall(b"WHAT\n")
            with sock.makefile("r") as f:
                for line in f:
                    line = line.rstrip("\n")
                    now = time.time()
                    if follow:
                        buf.append((now, line))
                        if line.startswith("STATE "):
                            if '"event": "ENTRY_CONFIRMED"' in line:
                                entry_wall[0] = now
                            elif ('"event": "EXIT"' in line
                                    and entry_wall[0] is not None):
                                cut = entry_wall[0] - PRE_ROLL_S
                                visit = [e for e in buf if e[0] >= cut]
                                entry_wall[0] = None
                                if replaying["stop"] is not None:
                                    replaying["stop"].set()
                                stop = threading.Event()
                                replaying["stop"] = stop
                                threading.Thread(target=_visit_loop,
                                                 args=(visit, stop),
                                                 daemon=True).start()
                                continue
                    if replaying["stop"] is None:
                        _publish(line)
        except OSError as e:
            sys.stderr.write("live source: %s; reconnecting\n" % e)
            time.sleep(1)


def _visit_loop(visit, stop):
    """Loop one buffered visit verbatim: same lines, same spacing."""
    sys.stderr.write("follow: replaying the last visit (%d wire lines, "
                     "%.0fs)\n" % (len(visit), visit[-1][0] - visit[0][0]))
    while not stop.is_set():
        prev = None
        for wall, line in visit:
            if stop.is_set():
                return
            dt = 0.0 if prev is None else max(0.0, wall - prev)
            prev = wall
            while dt > 0 and not stop.is_set():
                step = min(dt, 0.25)
                time.sleep(step)
                dt -= step
            _publish(line)
        time.sleep(1.0)


def _tcp_client(conn):
    sid = _subscribe()

    def reader():
        try:
            with conn.makefile("r") as f:
                for line in f:
                    line = line.strip()
                    if line == "WHAT" and _last_state:
                        conn.sendall((_last_state + "\n").encode())
                    elif line == "DUMP" or line.startswith("LOG"):
                        conn.sendall(b'ARCHIVE {"sessions": []}\n')
                    # MARK accepted and dropped: no archive here, on purpose
        except OSError:
            pass

    threading.Thread(target=reader, daemon=True).start()
    try:
        while True:
            lines = _next_lines(sid)
            if lines is None:
                break
            for line in lines:
                conn.sendall((line + "\n").encode())
    except OSError:
        pass
    finally:
        _unsubscribe(sid)
        try:
            conn.close()
        except OSError:
            pass


def _tcp_serve(port):
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", port))
    srv.listen(8)
    while True:
        conn, _ = srv.accept()
        threading.Thread(target=_tcp_client, args=(conn,), daemon=True).start()


# --- the HTTP side, for the browser ----------------------------------------

class _Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == "/feed":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            sid = _subscribe()
            try:
                while True:
                    lines = _next_lines(sid)
                    if lines is None:
                        break
                    for line in lines:
                        self.wfile.write(("data: " + line + "\n\n").encode())
                    if not lines:
                        self.wfile.write(b": ping\n\n")   # keepalive
                    self.wfile.flush()
            except OSError:
                pass
            finally:
                _unsubscribe(sid)
            return
        name = {"/": "index.html", "/scene.cfg": "scene.cfg"}.get(self.path)
        if name is None:
            self.send_error(404)
            return
        path = os.path.join(ROOT, "waveview", name)
        try:
            with open(path, "rb") as f:
                body = f.read()
        except OSError:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type",
                         "text/html" if name.endswith(".html") else "text/plain")
        self.end_headers()
        self.wfile.write(body)


def _tui_source(port):
    """The REAL heatmap client, run as a subprocess against our own TCP port,
    its ANSI stdout captured and republished as SCREEN lines. Each repaint
    starts with ESC[H; the startup ESC[2J / cursor-hide prefix is part of the
    first chunk and stripped by the split. The browser parses these bytes with
    a terminal emulator, so the page shows what the terminal shows because it
    IS what the terminal was sent."""
    import subprocess
    proc = subprocess.Popen(
        [sys.executable, os.path.join(ROOT, "lab", "tui_capture.py"), str(port)],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, cwd=ROOT)
    buf = b""
    while True:
        chunk = proc.stdout.read1(65536)
        if not chunk:
            sys.stderr.write("tui source: client exited rc=%s\n" % proc.poll())
            return
        buf += chunk
        parts = buf.split(b"\x1b[H")
        # everything before the LAST unterminated repaint is publishable
        for part in parts[1:-1]:
            _publish("SCREEN " + json.dumps(("\x1b[H" + part.decode(
                "utf-8", "replace"))))
        buf = b"\x1b[H" + parts[-1] if len(parts) > 1 else parts[0]


def main():
    ap = argparse.ArgumentParser(description="Dev pipeline server (no business logic)")
    ap.add_argument("--clip", action="append", help="replay this clip (repeatable)")
    ap.add_argument("--live", nargs="?", const="bath-pi.local", default=None,
                    help="relay the real server (one upstream connection)")
    ap.add_argument("--follow", nargs="?", const="bath-pi.local", default=None,
                    help="live until a visit ends, then loop that visit; "
                         "each EXIT on the stream swaps the loop")
    ap.add_argument("--port", type=int, default=TCP_PORT)
    ap.add_argument("--http", type=int, default=HTTP_PORT)
    a = ap.parse_args()
    if len([x for x in (a.clip, a.live, a.follow) if x]) != 1:
        ap.error("exactly one of --clip / --live / --follow")

    if a.clip:
        src = threading.Thread(target=_clip_source, args=(a.clip,), daemon=True)
    else:
        src = threading.Thread(target=_live_source,
                               args=(a.live or a.follow, 8628, bool(a.follow)),
                               daemon=True)
    src.start()
    threading.Thread(target=_tcp_serve, args=(a.port,), daemon=True).start()
    threading.Thread(target=_tui_source, args=(a.port,), daemon=True).start()
    sys.stderr.write("tcp :%d   http :%d\n" % (a.port, a.http))
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", a.http), _Handler)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
