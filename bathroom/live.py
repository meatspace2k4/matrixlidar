"""Live end-to-end state readout: stream -> engine -> committed state, with a
rolling occupancy sparkline and a colored state strip.

  python3 -m bathroom.live              # one rewriting status line + a sparkline
  python3 -m bathroom.live --verbose    # also log each committed transition

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

from . import config, detect, occupancy, stream, sparkline
from .classify import features
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
    ap.add_argument("--verbose", action="store_true",
                    help="log the feature vector on each published transition")
    args = ap.parse_args()

    try:
        baseline = config.load_baseline()
    except config.BaselineError as e:
        sys.stderr.write("%s\n" % e)
        return 1
    # THE DAEMON DERIVES, THE VIEWER DISPLAYS. Same rule as heatmap_client:
    # this readout used to run its own StateEngine + OccupancyEngine over the
    # stream and diverge from the daemon. STATE/ROOM now come from the
    # daemon's published feed; the per-frame features below are display
    # telemetry (sparkline, direction arrow), never state.
    door = detect.DoorReading(baseline)
    occ_hist = deque(maxlen=HIST)
    state_hist = deque(maxlen=HIST)
    col_hist = deque(maxlen=DRIFT_FRAMES)
    last_shown = None   # (posture, room) last logged to scrollback
    room_shown = None
    room_since = None
    try:
        for m in stream.frames(on_wait=_waiting):
            now = time.time()
            f = features(m, baseline)
            door.update(m)
            occ_hist.append(f["occ_count"])
            cells = f["occ_cells"]
            if cells:
                col_hist.append(sum(c for _r, c in cells) / float(len(cells)))
            else:
                col_hist.clear()

            st = stream.state()
            if st is None:
                posture, room, occupants = "UNKNOWN", None, 0
                shown = "NO STATE FEED"
                room_txt = "(daemon not publishing)"
                state_hist.append(State.UNKNOWN)
            else:
                posture = st.get("posture", "UNKNOWN")
                room, occupants = st.get("room"), st.get("occupants", 0)
                state_hist.append(State[posture] if posture in State.__members__
                                  else State.UNKNOWN)
                if room != room_shown:
                    room_shown, room_since = room, now
                if (posture, room) != last_shown:
                    if last_shown is not None:
                        sys.stdout.write("\r\x1b[K%s  %-18s room=%-9s%s\n"
                                         % (time.strftime("%H:%M:%S"), posture,
                                            room, (" (%s)" % st["event"])
                                            if st.get("event") else ""))
                        if args.verbose:
                            sys.stdout.write("           occ=%d vnear=%d up=%d low=%d toto=%d min=%s\n"
                                             % (f["occ_count"], f["vnear_count"], f["upper_count"],
                                                f["lower_count"], int(f["toto_occluded"]), f["min_mm"]))
                    last_shown = (posture, room)
                shown = posture
                if st.get("event") == "OCCLUDED":
                    shown = "VIEW BLOCKED"
                elif posture == State.WALKING.name:
                    shown += _direction(col_hist)   # '<-' left / '->' right
                room_txt = "%s(%d) %s" % (room, occupants,
                                          occupancy.fmt_duration(now - room_since))
            door_txt = ("  --" if door.pct is None
                        else "%3.0f%%%s" % (door.pct, "" if door.coherent else "*"))
            lead = "%-20s %-21s door=%-5s " % (room_txt, shown, door_txt)
            # fit the line to the terminal: a wrapped line cannot be
            # rewritten with \r and scrolls a new row per frame instead
            cols = shutil.get_terminal_size(fallback=(120, 24)).columns
            if len(lead) > cols - 1:
                lead = lead[:max(0, cols - 1)]  # plain text, safe to slice
            avail = cols - len(lead) - 1
            spark = strip = ""
            if avail >= 12:
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
