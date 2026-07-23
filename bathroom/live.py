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

from . import archive, config, detect, occupancy, stream, sparkline
from .engine import StateEngine
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
                    help="log the feature vector on each committed transition")
    ap.add_argument("--commit-after", type=int, default=4,
                    help="frames a candidate must hold before the state flips")
    args = ap.parse_args()

    baseline = config.load_baseline()
    eng = StateEngine(baseline=baseline, commit_after=args.commit_after)
    room_eng = occupancy.OccupancyEngine()
    marker = archive.Marker()
    occ_hist = deque(maxlen=HIST)
    state_hist = deque(maxlen=HIST)
    col_hist = deque(maxlen=DRIFT_FRAMES)
    t0 = None
    try:
        for m in stream.frames(on_wait=_waiting):
            now = time.time()
            t0 = t0 if t0 is not None else now
            u = eng.update(m, now - t0)
            door = detect.door_open_pct(m)
            ru = room_eng.update(m, now - t0, door_pct=door)
            marker.update(u, ru, now - t0)
            occ_hist.append(u.features["occ_count"])
            state_hist.append(u.committed)
            cells = u.features["occ_cells"]
            if cells:
                col_hist.append(sum(c for _r, c in cells) / float(len(cells)))
            else:
                col_hist.clear()

            if ru.event is not None:
                sys.stdout.write("\r\x1b[K%s  room -> %-9s (%s)\n"
                                 % (time.strftime("%H:%M:%S"), ru.room, ru.event))
            if u.changed and u.event is not None:
                sys.stdout.write("\r\x1b[K%s  %-18s -> %-18s (%.1fs)\n"
                                 % (time.strftime("%H:%M:%S"), u.event.state.name,
                                    u.committed.name, u.event.duration))
                if args.verbose:
                    f = u.features
                    sys.stdout.write("           occ=%d vnear=%d up=%d low=%d toto=%d min=%s\n"
                                     % (f["occ_count"], f["vnear_count"], f["upper_count"],
                                        f["lower_count"], int(f["toto_occluded"]), f["min_mm"]))

            shown = u.committed.name
            if u.committed == State.WALKING:
                shown += _direction(col_hist)   # '<-' left / '->' right
            room_txt = "%s(%d) %s" % (ru.room, ru.occupants,
                                      occupancy.fmt_duration((now - t0) - ru.since))
            lead = "%-20s %-21s inst=%-18s door=%3.0f%% " % (
                room_txt, shown, u.instant.name, door)
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
        eng.close()
        sys.stdout.write("\n\nsession timeline:\n")
        for s in eng.segments():
            sys.stdout.write("  %-18s %6.1fs  %4d frames\n"
                             % (s["state"], s["duration"], s["n_frames"]))
        if isinstance(e, ConnectionError):
            sys.stdout.write("%s\n" % e)
    return 0


def _waiting(attempt, total):
    sys.stdout.write("\r\x1b[Kwaiting for %s:%d ... %d/%d"
                     % (stream.HOST, stream.PORT, attempt, total))
    sys.stdout.flush()


if __name__ == "__main__":
    raise SystemExit(main())
