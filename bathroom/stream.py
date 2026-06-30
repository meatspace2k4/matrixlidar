"""Warm-stream sensor access for the client side.

The Pi runs an always-on server (~/lidar_server.py) that holds /dev/ttyACM0
and broadcasts each frame as 64 comma-separated raw row-major mm values, one
line per frame, on PORT. Reading from it means no serial cold start and no
contention: capture and the live heatmap can share the stream, and the serial
port is never stolen from the server (unlike the direct-serial snapshot tool).

`frames()` yields oriented matrices (display orientation, row 0 = top, col 0 =
column A) and transparently reconnects if the Pi reboots or the link drops.
"""
import socket
import time

from .zones import N

HOST = "bath-pi.local"
PORT = 8628
CONNECT_TIMEOUT = 2.0
RETRY_DELAY = 1.0
MAX_ATTEMPTS = 60  # ~ covers a full Pi reboot


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
    """Return a connected socket, or None after MAX_ATTEMPTS. Never raises."""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return socket.create_connection((host, port), timeout=CONNECT_TIMEOUT)
        except OSError:
            if on_wait:
                on_wait(attempt, MAX_ATTEMPTS)
            time.sleep(RETRY_DELAY)
    return None


def frames(host=HOST, port=PORT, on_wait=None):
    """Yield oriented 8x8 matrices forever, reconnecting as needed.

    Raises ConnectionError only if the Pi can't be reached after MAX_ATTEMPTS.
    """
    while True:
        sock = connect_with_retry(host, port, on_wait)
        if sock is None:
            raise ConnectionError("could not reach %s:%d after %d attempts"
                                  % (host, port, MAX_ATTEMPTS))
        try:
            with sock.makefile("r") as f:
                for line in f:
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
            try:
                sock.close()
            except OSError:
                pass
