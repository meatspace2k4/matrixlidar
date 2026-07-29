"""Warm-stream sensor access for the client side.

The Pi runs an always-on server (~/lidar_server.py) that holds /dev/ttyACM0
and broadcasts each frame as 64 comma-separated raw row-major mm values, one
line per frame, on PORT. Reading from it means no serial cold start and no
contention: capture and the live heatmap can share the stream, and the serial
port is never stolen from the server (unlike the direct-serial snapshot tool).

`frames()` yields oriented matrices (display orientation, row 0 = top, col 0 =
column A) and transparently reconnects if the Pi reboots or the link drops.

Two other line types share the stream and are NOT frames: "STATE <json>" (the
detection daemon's published state, a level) and "IR <json>" (a decoded IR
signal, an event). Both are consumed here and handed on through `state()` and
`drain_ir()` rather than reaching the frame consumer.
"""
import json
import socket
import time

from collections import deque

from .zones import N

HOST = "bath-pi.local"
PORT = 8628
CONNECT_TIMEOUT = 2.0
READ_TIMEOUT = 5.0  # frames arrive ~10/s; silence this long means a stalled
                    # link (e.g. wake-from-sleep) -> drop and reconnect
RETRY_DELAY = 1.0
MAX_ATTEMPTS = 60  # ~ covers a full Pi reboot

_cached_ip = None  # last good peer IP; skips slow .local mDNS on reconnects
_sock = None       # the socket frames() is currently reading (for send())

# The detection daemon's published state, rebroadcast by the server as
# "STATE <json>" lines on the same stream. Viewers DISPLAY this; they do not
# run their own engines (the whole v3.00 rule). Updated by frames().
last_state = None     # dict from the daemon's most recent STATE line
last_state_ts = None  # time.time() when it arrived
STATE_STALE_S = 5.0   # daemon heartbeats at 1s; older than this means no feed

# Decoded IR signals arriving as "IR <json>" lines on the same stream. A
# QUEUE, not a `last_ir`: STATE is a level (the newest one is the truth and
# missing one costs nothing) and an IR signal is an EVENT (each one is a
# distinct press, and dropping one loses it). Bounded, so a consumer that
# never drains cannot grow it without limit -- it just keeps the newest.
IR_QUEUE_MAX = 256
_ir = deque(maxlen=IR_QUEUE_MAX)


def drain_ir():
    """Every IR signal received since the last call, oldest first."""
    out = []
    while True:
        try:
            out.append(_ir.popleft())
        except IndexError:
            return out


def state():
    """The daemon's published state, or None if absent/stale."""
    if last_state is None or time.time() - last_state_ts > STATE_STALE_S:
        return None
    return last_state


def send(line):
    """Best-effort send of one line back to the server on the live stream
    socket (the server understands MARK/DUMP). Never raises; drops the line
    if not currently connected."""
    s = _sock
    if s is None:
        return False
    try:
        s.sendall((line.rstrip("\n") + "\n").encode())
        return True
    except OSError:
        return False


def rotate_cw(m):
    return [[m[N - 1 - c][r] for c in range(N)] for r in range(N)]


def flip_x(m):
    return [row[:] for row in m[::-1]]


def flip_y(m):
    return [row[::-1] for row in m]


def orient(m):
    """Same calibrated chain as the Pi heatmap: cw, flip top/bottom, flip l/r."""
    return flip_y(flip_x(rotate_cw(m)))


def _to_matrix(flat):
    return [list(flat[i * N:(i + 1) * N]) for i in range(N)]


def connect_with_retry(host=HOST, port=PORT, on_wait=None):
    """Return a connected socket, or None after MAX_ATTEMPTS. Never raises.

    The socket carries READ_TIMEOUT so a connection that goes silent (stalled
    server, wake-from-sleep) raises instead of hanging the render loop."""
    global _cached_ip
    for attempt in range(1, MAX_ATTEMPTS + 1):
        for target in ([_cached_ip] if _cached_ip else []) + [host]:
            try:
                s = socket.create_connection((target, port), timeout=CONNECT_TIMEOUT)
                _cached_ip = s.getpeername()[0]
                s.settimeout(READ_TIMEOUT)
                return s
            except OSError:
                continue
        if on_wait:
            on_wait(attempt, MAX_ATTEMPTS)
        time.sleep(RETRY_DELAY)
    return None


def frames(host=HOST, port=PORT, on_wait=None):
    """Yield oriented 8x8 matrices forever, reconnecting as needed.

    Raises ConnectionError only if the Pi can't be reached after MAX_ATTEMPTS.
    """
    global _sock
    while True:
        sock = connect_with_retry(host, port, on_wait)
        if sock is None:
            raise ConnectionError("could not reach %s:%d after %d attempts"
                                  % (host, port, MAX_ATTEMPTS))
        _sock = sock
        global last_state, last_state_ts
        try:
            send("WHAT")  # prime: current state now, not at the next heartbeat
            with sock.makefile("r") as f:
                for line in f:
                    if line.startswith("STATE "):
                        try:
                            last_state = json.loads(line[6:])
                            last_state_ts = time.time()
                        except ValueError:
                            pass
                        continue
                    if line.startswith("IR "):
                        try:
                            _ir.append(json.loads(line[3:]))
                        except ValueError:
                            pass
                        continue
                    parts = line.strip().split(",")
                    if len(parts) != N * N:
                        continue
                    try:
                        flat = [int(x) for x in parts]
                    except ValueError:
                        continue
                    yield orient(_to_matrix(flat))
        except OSError:
            pass  # dropped; loop back and reconnect
        finally:
            # makefile() wraps the socket but does NOT close it; close it here.
            # This is a generator, so on an early `break` the finally runs when
            # the generator is closed (GC or explicit .close()), releasing the fd.
            if _sock is sock:
                _sock = None
            try:
                sock.close()
            except OSError:
                pass
