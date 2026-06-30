"""Live ground-truth state readout (text only).

Connects to the warm stream and prints the current classified state, door
open%, and the features behind the decision. Bounded, single rewriting line
by default so it can run forever in a corner; --verbose adds the feature dump
each time the state changes (useful while tuning).

  python3 -m bathroom.live
  python3 -m bathroom.live --verbose
"""
import argparse
import sys
import time
from collections import deque

from . import config
from . import detect
from . import stream
from .classify import classify_window
from .zones import State

WINDOW = 5


def main():
    ap = argparse.ArgumentParser(description="Live ground-truth state readout")
    ap.add_argument("--verbose", action="store_true",
                    help="print the feature vector each time the state changes")
    args = ap.parse_args()

    baseline = config.load_baseline()
    win = deque(maxlen=WINDOW)
    last = None
    try:
        for m in stream.frames(on_wait=_waiting):
            win.append(m)
            st, f = classify_window(list(win), baseline)
            door = detect.door_open_pct(m)
            sys.stdout.write("\r\x1b[K%-18s  door=%3.0f%%  occ=%-2d vnear=%-2d "
                             "up=%d low=%d toto=%d"
                             % (st.name, door, f["occ_count"], f["vnear_count"],
                                f["upper_count"], f["lower_count"],
                                int(f["toto_occluded"])))
            sys.stdout.flush()
            if args.verbose and st != last:
                sys.stdout.write("\n%s -> %s  min=%s top=%s\n"
                                 % (time.strftime("%H:%M:%S"), st.name,
                                    f["min_mm"], f["top_row"]))
            last = st
    except (KeyboardInterrupt, ConnectionError) as e:
        sys.stdout.write("\n%s\n" % (e or "stopped"))
    return 0


def _waiting(attempt, total):
    sys.stdout.write("\r\x1b[Kwaiting for %s:%d ... %d/%d"
                     % (stream.HOST, stream.PORT, attempt, total))
    sys.stdout.flush()


if __name__ == "__main__":
    raise SystemExit(main())
