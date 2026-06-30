#!/usr/bin/env python3
"""Local heatmap client. Connects to the Pi's always-on warm stream
(bath-pi.local:8628), renders the 8x8 heatmap + zone panel LOCALLY, and now
also shows the live ground-truth STATE from the classifier so you can see
what a recording is being labeled as while you capture it.

Only the tiny raw frames cross the network; the per-frame ANSI repaint happens
on this machine. Run from the project root so `import bathroom` resolves:

  python3 heatmap_client.py
"""
import sys
from collections import deque

from bathroom import config, detect, stream
from bathroom.classify import classify_window
from bathroom.zones import N, NO_TARGET, Zone

NEAR_MM = 100
FAR_MM = 2000
CELL_W = 6
WINDOW = 5
STOPS = [
    (0.00, (12, 12, 40)), (0.20, (30, 60, 180)), (0.40, (0, 170, 170)),
    (0.60, (40, 190, 60)), (0.78, (240, 220, 40)), (0.90, (240, 120, 20)),
    (1.00, (240, 40, 40)),
]


def lerp(a, b, f):
    return tuple(int(a[i] + (b[i] - a[i]) * f) for i in range(3))


def color_for(mm):
    if mm >= NO_TARGET:
        return STOPS[0][1]
    if mm <= NEAR_MM:
        t = 1.0
    elif mm >= FAR_MM:
        t = 0.0
    else:
        t = 1.0 - (mm - NEAR_MM) / float(FAR_MM - NEAR_MM)
    for i in range(len(STOPS) - 1):
        t0, c0 = STOPS[i]
        t1, c1 = STOPS[i + 1]
        if t0 <= t <= t1:
            f = 0 if t1 == t0 else (t - t0) / (t1 - t0)
            return lerp(c0, c1, f)
    return STOPS[-1][1]


def render(m, state):
    out = ["\x1b[H", "\x1b[1m DFRobot Matrix Lidar  8x8 heatmap (mm)\x1b[0m\x1b[K\n\n"]
    for r in range(N):
        line = []
        for c in range(N):
            v = m[r][c]
            br, bg, bb = color_for(v)
            fg = "30" if v < FAR_MM and v < NEAR_MM + 900 else "97"
            txt = "" if v >= NO_TARGET else str(v)
            line.append("\x1b[48;2;%d;%d;%dm\x1b[%sm%s\x1b[0m"
                        % (br, bg, bb, fg, txt.center(CELL_W)))
        out.append("".join(line) + "\x1b[K\n")
    out.append("\x1b[K\n near ")
    for mm in (100, 300, 600, 1000, 1500, 2000, 4000):
        br, bg, bb = color_for(mm)
        out.append("\x1b[48;2;%d;%d;%dm   \x1b[0m" % (br, bg, bb))
    out.append(" far/none\x1b[K\n\x1b[K\n")
    for z in (Zone.TOTO, Zone.DOOR, Zone.LEFT_EDGE):
        v = detect.zone_values(m, z)
        extra = "  open=%.0f%%" % detect.door_open_pct(m) if z == Zone.DOOR else ""
        out.append("  %d %-9s %4d mm%s\x1b[K\n" % (int(z), z.name, v["median"], extra))
    out.append("\x1b[K\n \x1b[1mSTATE: %-20s\x1b[0m\x1b[K\n" % state.name)
    sys.stdout.write("".join(out))
    sys.stdout.flush()


def _waiting(attempt, total):
    sys.stderr.write("\rwaiting for %s:%d ... %d/%d  "
                     % (stream.HOST, stream.PORT, attempt, total))
    sys.stderr.flush()


def main():
    baseline = config.load_baseline()
    win = deque(maxlen=WINDOW)
    started = False
    try:
        for m in stream.frames(on_wait=_waiting):
            if not started:
                sys.stderr.write("\r\x1b[K")
                sys.stdout.write("\x1b[2J\x1b[?25l")  # clear, hide cursor
                started = True
            win.append(m)
            state, _ = classify_window(list(win), baseline)
            render(m, state)
    except (KeyboardInterrupt, ConnectionError):
        pass
    finally:
        if started:
            sys.stdout.write("\x1b[?25h\n")  # restore cursor
            sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
