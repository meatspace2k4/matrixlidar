"""Live end-to-end state readout: stream -> engine -> committed state, with a
rolling occupancy sparkline and a colored state strip.

  python3 -m bathroom.live              # one rewriting status line + a sparkline
  python3 -m bathroom.live --verbose    # also log each committed transition

State transitions are logged to scrollback as they commit; the bottom line
rewrites in place with the current state, door%, the occupancy sparkline, and
the recent state strip. On exit it prints the session timeline.
"""
import argparse
import sys
import time
from collections import deque

from . import config, detect, stream, sparkline
from .engine import StateEngine

HIST = 32  # columns of rolling history in the sparkline + state strip


def main():
    ap = argparse.ArgumentParser(description="Live e2e state readout with sparkline")
    ap.add_argument("--verbose", action="store_true",
                    help="log the feature vector on each committed transition")
    ap.add_argument("--commit-after", type=int, default=4,
                    help="frames a candidate must hold before the state flips")
    args = ap.parse_args()

    baseline = config.load_baseline()
    eng = StateEngine(baseline=baseline, commit_after=args.commit_after)
    occ_hist = deque(maxlen=HIST)
    state_hist = deque(maxlen=HIST)
    t0 = None
    try:
        for m in stream.frames(on_wait=_waiting):
            now = time.time()
            t0 = t0 if t0 is not None else now
            u = eng.update(m, now - t0)
            occ_hist.append(u.features["occ_count"])
            state_hist.append(u.committed)
            door = detect.door_open_pct(m)

            if u.changed and u.event is not None:
                sys.stdout.write("\r\x1b[K%s  %-18s -> %-18s (%.1fs)\n"
                                 % (time.strftime("%H:%M:%S"), u.event.state.name,
                                    u.committed.name, u.event.duration))
                if args.verbose:
                    f = u.features
                    sys.stdout.write("           occ=%d vnear=%d up=%d low=%d toto=%d min=%s\n"
                                     % (f["occ_count"], f["vnear_count"], f["upper_count"],
                                        f["lower_count"], int(f["toto_occluded"]), f["min_mm"]))

            spark = sparkline.sparkline(occ_hist, lo=0, hi=max(8, max(occ_hist)))
            strip = sparkline.state_strip(state_hist)
            sys.stdout.write("\r\x1b[K%-18s inst=%-18s door=%3.0f%% |%s| %s"
                             % (u.committed.name, u.instant.name, door, spark, strip))
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
