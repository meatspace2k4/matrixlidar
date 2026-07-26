"""Live end-to-end state readout: stream -> engine -> committed state, with a
rolling occupancy sparkline and a colored state strip.

  python3 -m bathroom.live              # one rewriting status line + a sparkline

State transitions are logged to scrollback as they commit; the bottom line
rewrites in place with the current state (plus a <- / -> direction arrow while
WALKING), door%, the occupancy sparkline, and the recent state strip. On exit
it prints the session timeline.
"""
import argparse
import shutil
import sys
import time
from collections import deque

from . import occupancy, stream, sparkline
from .zones import State

HIST = 32  # columns of rolling history in the sparkline + state strip
DRIFT_FRAMES = 6    # centroid-column history for walking direction
DRIFT_MIN = 0.8     # columns of net drift before a direction is claimed


def _direction(col_hist):
    """'<-' (leftward), '->' (rightward), or '' from centroid-column drift."""
    if len(col_hist) < DRIFT_FRAMES:
        return ""
    drift = col_hist[-1] - col_hist[0]
    if drift <= -DRIFT_MIN:
        return " <-"
    if drift >= DRIFT_MIN:
        return " ->"
    return ""


def main():
    ap = argparse.ArgumentParser(description="Live e2e state readout with sparkline")
    ap.parse_args()

    # THE SERVICE DERIVES, THE VIEWER DISPLAYS. Everything on this line comes
    # from the published STATE feed; this process runs no engine and reads no
    # baseline.
    occ_hist = deque(maxlen=HIST)
    state_hist = deque(maxlen=HIST)
    last_shown = None   # (posture, room) last logged to scrollback
    door_held = None
    seen_ts = None      # last STATE arrival consumed into the histories
    try:
        for m in stream.frames(on_wait=_waiting):
            st = stream.state()
            if st is None:
                shown = "NO STATE FEED"
                room_txt = "(service not publishing)"
            else:
                posture = st.get("posture", "UNKNOWN")
                room = st.get("room")
                if stream.last_state_ts != seen_ts:   # one column per publish
                    seen_ts = stream.last_state_ts
                    occ_hist.append(st.get("occ", 0))
                    state_hist.append(State[posture] if posture in State.__members__
                                      else State.UNKNOWN)
                if (posture, room) != last_shown:
                    if last_shown is not None:
                        sys.stdout.write("\r\x1b[K%s  %-18s room=%-9s%s\n"
                                         % (time.strftime("%H:%M:%S"), posture,
                                            room, (" (%s)" % st["event"])
                                            if st.get("event") else ""))
                    last_shown = (posture, room)
                shown = posture
                if st.get("blocked"):
                    shown = "VIEW BLOCKED"
                elif st.get("dir"):
                    shown += " " + st["dir"]        # '<-' left / '->' right
                room_txt = "%s(%d) %s" % (
                    room, st.get("occupants", 0),
                    occupancy.fmt_duration(st.get("t", 0) - st.get("since", 0)))
                if st.get("door") is not None:
                    door_held = st["door"]
            door_txt = "  --" if door_held is None else "%3.0f%%" % door_held
            lead = "%-20s %-21s door=%-5s " % (room_txt, shown, door_txt)
            # fit the line to the terminal: a wrapped line cannot be
            # rewritten with \r and scrolls a new row per frame instead
            cols = shutil.get_terminal_size(fallback=(120, 24)).columns
            if len(lead) > cols - 1:
                lead = lead[:max(0, cols - 1)]  # plain text, safe to slice
            avail = cols - len(lead) - 1
            spark = strip = ""
            if avail >= 12 and occ_hist:
                spark_w = min(HIST, max(8, (avail - 3) // 2))
                strip_w = min(HIST, avail - spark_w - 3)
                spark = "|%s|" % sparkline.sparkline(
                    list(occ_hist)[-spark_w:], lo=0, hi=max(8, max(occ_hist)))
                if strip_w >= 8:
                    spark += " " + sparkline.state_strip(
                        list(state_hist)[-strip_w:])
            sys.stdout.write("\r\x1b[K%s%s" % (lead, spark))
            sys.stdout.flush()
    except (KeyboardInterrupt, ConnectionError) as e:
        sys.stdout.write("\n")
        if isinstance(e, ConnectionError):
            sys.stdout.write("%s\n" % e)
    return 0


def _waiting(attempt, total):
    sys.stdout.write("\r\x1b[Kwaiting for %s:%d ... %d/%d"
                     % (stream.HOST, stream.PORT, attempt, total))
    sys.stdout.flush()


if __name__ == "__main__":
    raise SystemExit(main())
