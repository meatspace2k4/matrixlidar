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

from bathroom import detect, occupancy, sparkline, stream
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
_zone_hold = {}  # last real zone median / door pct (display only, starred when held)


def render(m, state_name, room_txt, occ_hist, state_hist, door_pct):
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
        # hold the last real reading instead of flapping to --: a blank frame
        # changes the line width and shoves the panel around
        if v["median"] is not None:
            _zone_hold[z] = v["median"]
        mm = v["median"] if v["median"] is not None else _zone_hold.get(z)
        cm = " -- " if mm is None else "%4d" % ((mm + 5) // 10)
        extra = ""
        if z == Zone.DOOR:
            # the service's published door reading, not anything derived here
            if door_pct is not None:
                _zone_hold["door_pct"] = door_pct
            pct = _zone_hold.get("door_pct")
            extra = ("  open=  --" if pct is None else "  open=%3.0f%%" % pct)
        out.append("  %d %-9s %s cm  %d/%d rays%s\x1b[K\n"
                   % (int(z), z.name, cm, v["returns"], v["rays"], extra))
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
    # THE SERVICE DERIVES, THE VIEWER DISPLAYS. Everything on screen except
    # the raw cells comes from the published STATE feed: posture, room,
    # duration, door%, occlusion, occ count, walking direction. This process
    # runs no engine and reads no baseline.
    occ_hist = deque(maxlen=HIST)
    state_hist = deque(maxlen=HIST)
    started = False
    seen_ts = None      # last STATE arrival consumed into the histories
    try:
        for m in stream.frames(on_wait=_waiting):
            if not started:
                sys.stderr.write("\r\x1b[K")
                sys.stdout.write("\x1b[2J\x1b[?25l")  # clear, hide cursor
                started = True
            st = stream.state()
            if st is None:
                shown = "NO STATE FEED"
                room_txt = "(service not publishing)"
                door_pct = None
            else:
                if stream.last_state_ts != seen_ts:   # one column per publish
                    seen_ts = stream.last_state_ts
                    occ_hist.append(st.get("occ", 0))
                    posture = st.get("posture", "UNKNOWN")
                    state_hist.append(State[posture] if posture in State.__members__
                                      else State.UNKNOWN)
                shown = st.get("posture", "UNKNOWN")
                if st.get("blocked"):
                    shown = "VIEW BLOCKED"
                elif st.get("dir"):
                    shown += " " + st["dir"]        # '<-' left / '->' right
                room_txt = "%s(%d) %s" % (
                    st.get("room"), st.get("occupants", 0),
                    occupancy.fmt_duration(st.get("t", 0) - st.get("since", 0)))
                door_pct = st.get("door")
            render(m, shown, room_txt, occ_hist, state_hist, door_pct)
    except (KeyboardInterrupt, ConnectionError):
        pass
    finally:
        if started:
            sys.stdout.write("\x1b[?25h\n")  # restore cursor
            sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
