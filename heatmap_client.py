#!/usr/bin/env python3
"""Local heatmap client. Connects to the Pi's always-on warm stream
(bath-pi.local:8628) and renders locally: the 8x8 heatmap (cell values in cm,
brief dropouts held for display), the zone panel with door open%, the debounced
STATE (with a <- / -> direction arrow while WALKING), and the occupancy
sparkline + state strip.

Only the tiny raw frames cross the network; the per-frame ANSI repaint happens
on this machine. Run from the project root so `import bathroom` resolves:

  python3 heatmap_client.py
"""
import sys
import time
from collections import deque

from bathroom import archive, config, detect, occupancy, sparkline, stream
from bathroom.engine import StateEngine
from bathroom.live import DRIFT_FRAMES, _direction
from bathroom.zones import N, NO_TARGET, State, Zone

NEAR_MM = 100
FAR_MM = 2000
CELL_W = 6
HIST = 40       # columns of rolling history under the panel
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


HOLD_FRAMES = 6  # bridge dropouts this long; beyond it the cell is truly empty
_hold = [[NO_TARGET] * N for _ in range(N)]       # last real value (display only)
_stale = [[HOLD_FRAMES + 1] * N for _ in range(N)]  # frames since that value


def render(m, state_name, room_txt, occ_hist, state_hist):
    out = ["\x1b[H", "\x1b[1m DFRobot Matrix Lidar  8x8 heatmap (cm)\x1b[0m\x1b[K\n\n"]
    for r in range(N):
        line = []
        for c in range(N):
            raw = m[r][c]
            if 0 < raw < NO_TARGET:     # 0 = dust/noise, never a real reading
                _hold[r][c], _stale[r][c] = raw, 0
            else:
                _stale[r][c] += 1
            # brief dropout: show the held value; persistently empty (open
            # doorway, hallway out of range): show truly blank
            v = _hold[r][c] if _stale[r][c] <= HOLD_FRAMES else NO_TARGET
            br, bg, bb = color_for(v)
            fg = "30" if v < FAR_MM and v < NEAR_MM + 900 else "97"
            txt = "" if v >= NO_TARGET else str((v + 5) // 10)  # shown in cm
            line.append("\x1b[48;2;%d;%d;%dm\x1b[%sm%s\x1b[0m"
                        % (br, bg, bb, fg, txt.center(CELL_W)))
        out.append("".join(line) + "\x1b[K\n")
    out.append("\x1b[K\n near ")
    for mm in (100, 300, 600, 1000, 1500, 2000, 4000):
        br, bg, bb = color_for(mm)
        out.append("\x1b[48;2;%d;%d;%dm   \x1b[0m" % (br, bg, bb))
    out.append(" far/none\x1b[K\n\x1b[K\n")
    for z in (Zone.TOTO, Zone.DOOR, Zone.FRAME):
        v = detect.zone_values(m, z)
        extra = "  open=%.0f%%" % detect.door_open_pct(m) if z == Zone.DOOR else ""
        out.append("  %d %-9s %4d cm%s\x1b[K\n"
                   % (int(z), z.name, (v["median"] + 5) // 10, extra))
    out.append("\x1b[K\n \x1b[1mSTATE: %-22s\x1b[0m\x1b[K\n" % state_name)
    out.append(" \x1b[1mROOM:  %-26s\x1b[0m\x1b[K\n" % room_txt)
    hi = max(8, max(occ_hist) if occ_hist else 8)
    out.append(" occ  |%s|\x1b[K\n" % sparkline.sparkline(occ_hist, lo=0, hi=hi))
    out.append(" time  %s\x1b[K\n" % sparkline.state_strip(state_hist))
    sys.stdout.write("".join(out))
    sys.stdout.flush()


def _waiting(attempt, total):
    sys.stderr.write("\rwaiting for %s:%d ... %d/%d  "
                     % (stream.HOST, stream.PORT, attempt, total))
    sys.stderr.flush()


def main():
    baseline = config.load_baseline()
    eng = StateEngine(baseline=baseline)
    room_eng = occupancy.OccupancyEngine()
    marker = archive.Marker()
    occ_hist = deque(maxlen=HIST)
    state_hist = deque(maxlen=HIST)
    col_hist = deque(maxlen=DRIFT_FRAMES)
    started = False
    t0 = None
    try:
        for m in stream.frames(on_wait=_waiting):
            if not started:
                sys.stderr.write("\r\x1b[K")
                sys.stdout.write("\x1b[2J\x1b[?25l")  # clear, hide cursor
                started = True
            now = time.time()
            t0 = t0 if t0 is not None else now
            u = eng.update(m, now - t0)
            ru = room_eng.update(m, now - t0,
                                 door_pct=detect.door_open_pct(m))
            marker.update(u, ru, now - t0)
            occ_hist.append(u.features["occ_count"])
            state_hist.append(u.committed)
            cells = u.features["occ_cells"]
            if cells:
                col_hist.append(sum(c for _r, c in cells) / float(len(cells)))
            else:
                col_hist.clear()
            shown = u.committed.name
            if u.committed == State.WALKING:
                shown += _direction(col_hist)   # '<-' left / '->' right
            room_txt = "%s(%d) %s" % (ru.room, ru.occupants,
                                      occupancy.fmt_duration((now - t0) - ru.since))
            render(m, shown, room_txt, occ_hist, state_hist)
    except (KeyboardInterrupt, ConnectionError):
        pass
    finally:
        if started:
            sys.stdout.write("\x1b[?25h\n")  # restore cursor
            sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
