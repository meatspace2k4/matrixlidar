"""Baseline staleness + drift checks.

The classifier is baseline-relative, so a stale empty-room reference (lighting,
temperature, furniture moved) silently degrades ground truth. This reports the
baseline's age and compares it cell-by-cell against a fresh reference, either a
recorded EMPTY clip or a short live capture.

  python3 -m bathroom.baselines --age
  python3 -m bathroom.baselines --check-drift          # vs newest empty clip
  python3 -m bathroom.baselines --check-drift --live 6 # vs a 6s live capture
"""
import argparse
import sys
import time

from . import config, ingest
from .classify import median_window
from .zones import N, NO_TARGET, from_label, State

DRIFT_PCT = 8.0  # per-cell drift beyond this (of baseline mm) is flagged


def _age_line():
    meta = config.baseline_meta()
    cap = meta.get("captured")
    if not cap:
        print("baseline: no capture timestamp (using built-in default or legacy file)")
        return
    age_days = (time.time() - cap) / 86400.0
    print("baseline captured %.1f days ago (%d frames)" % (age_days, meta.get("frames", 0)))
    if age_days > 7:
        print("  -> older than a week; consider `python3 -m bathroom.record --baseline`")


def _newest_empty_clip():
    best = None
    for s in ingest.clips():
        if from_label(s.label) is State.EMPTY:
            best = s  # ingest.clips() is sorted; last EMPTY wins
    return best


def _drift(baseline, ref):
    """Per-cell |ref-baseline| as % of baseline; ignores no-target cells."""
    flagged = []
    worst = 0.0
    for r in range(N):
        for c in range(N):
            b, v = baseline[r][c], ref[r][c]
            if b >= NO_TARGET or v >= NO_TARGET or b == 0:
                continue
            pct = abs(v - b) / float(b) * 100.0
            worst = max(worst, pct)
            if pct > DRIFT_PCT:
                flagged.append((r, c, b, v, pct))
    return flagged, worst


def main():
    ap = argparse.ArgumentParser(description="Baseline age + drift checks")
    ap.add_argument("--age", action="store_true")
    ap.add_argument("--check-drift", action="store_true")
    ap.add_argument("--live", type=int, default=0, metavar="SECONDS",
                    help="capture a fresh empty reference live for N seconds")
    args = ap.parse_args()

    if args.age or not (args.check_drift):
        _age_line()
    if not args.check_drift:
        return 0

    baseline = config.load_baseline()
    if args.live:
        from . import stream
        frames = []
        sys.stderr.write("leave the room; capturing %ds...\n" % args.live)
        start = time.time()
        for m in stream.frames():
            frames.append(m)
            if time.time() - start >= args.live:
                break
        if not frames:
            print("no frames captured")
            return 1
        ref = median_window(frames)
        src = "%d-frame live capture" % len(frames)
    else:
        s = _newest_empty_clip()
        if s is None:
            print("no EMPTY-labeled clip found; record one or use --live")
            return 1
        ref = median_window(s.frames)
        src = "empty clip %s" % s.id

    flagged, worst = _drift(baseline, ref)
    print("drift vs %s: worst cell %.1f%%, %d cell(s) over %.0f%%"
          % (src, worst, len(flagged), DRIFT_PCT))
    for (r, c, b, v, pct) in flagged[:12]:
        print("  r%d c%d  base=%4d now=%4d  %+.0f%%" % (r, c, b, v, pct))
    if worst > DRIFT_PCT:
        print("  -> baseline looks stale; recapture with --baseline")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
