#!/usr/bin/env python3
"""Replay a harvested fixture session through the occupancy engine of a given
code tree, at the session's REAL timestamps.

Purpose: old-vs-new comparison on real captured data. Point --code-root at a
git worktree of the old code and at the current tree, run both, compare the
event streams and final room state.

Faithfulness contract (stated, not hidden): ring fixtures are transition
snapshots, not full streams. --bridge repeats each snapshot's frame at --hz
until the next snapshot, which is only valid when the live record attests
continuous evidence between those snapshots (session 10: the live client
committed WALKING continuously 16:24:02-16:24:09, so bridging its present
snapshots reconstructs the recorded episode; without --bridge the engine
sees only the sparse instants and duration-based behavior cannot
reproduce). The aftermath is driven by repeating the session's final
captured EMPTY frame for --tail-seconds, encoding the operator-attested
fact that the room was empty afterward. Long multi-minute sessions are NOT
faithfully replayable this way; do not use this script to make claims about
them.

  python3 scripts/replay_fixture.py fixtures/misdetections/session_010.json
  python3 scripts/replay_fixture.py <fixture> --code-root /tmp/old_tree
"""
import argparse
import importlib
import json
import sys


def load_stack(root):
    """Import bathroom.* from a specific tree, isolated from this process's
    cached modules."""
    saved = {k: v for k, v in sys.modules.items() if k.startswith("bathroom")}
    for k in list(saved):
        del sys.modules[k]
    sys.path.insert(0, root)
    try:
        classify = importlib.import_module("bathroom.classify")
        occupancy = importlib.import_module("bathroom.occupancy")
        stream = importlib.import_module("bathroom.stream")
        detect = importlib.import_module("bathroom.detect")
        config = importlib.import_module("bathroom.config")
        baseline = config.load_baseline()
    finally:
        sys.path.pop(0)
        for k in list(sys.modules):
            if k.startswith("bathroom"):
                del sys.modules[k]
        sys.modules.update(saved)
    return classify, occupancy, stream, detect, baseline


class Shim:
    """Adapter: a classified frame -> the .instant/.features surface the
    occupancy engine consumes."""

    def __init__(self, state, features):
        self.instant = state
        self.features = features


def main():
    ap = argparse.ArgumentParser(description="Replay a fixture session")
    ap.add_argument("fixture")
    ap.add_argument("--code-root", default=".",
                    help="tree whose bathroom/ package to run (default: cwd)")
    ap.add_argument("--tail-seconds", type=float, default=60.0,
                    help="attested-empty aftermath to append (final frame)")
    ap.add_argument("--hz", type=float, default=10.0)
    ap.add_argument("--bridge", action="store_true",
                    help="repeat each snapshot's frame at --hz until the next "
                         "snapshot, but only across gaps <= 1.0s (longer "
                         "stretches are unobserved and must not be fabricated; "
                         "the engine's own gap handling covers them)")
    a = ap.parse_args()

    classify_m, occupancy_m, stream_m, detect_m, baseline = load_stack(a.code_root)
    with open(a.fixture) as f:
        s = json.load(f)
    trs = sorted(s["transitions"], key=lambda tr: tr["ts"])
    if not trs:
        print("fixture has no transitions")
        return 1

    def to_update(frame_csv):
        flat = [int(x) for x in frame_csv.split(",")]
        mat = stream_m.orient([flat[i * 8:(i + 1) * 8] for i in range(8)])
        return mat, detect_m.door_open_pct(mat)

    eng = occupancy_m.OccupancyEngine()
    t0 = trs[0]["ts"]
    events = []

    def step(pair, t):
        mat, door_pct = pair
        ru = eng.update(mat, t, door_pct=door_pct)
        if ru.event:
            events.append((t, ru.event, ru.room))

    # real snapshots at real relative timestamps
    if a.bridge:
        for i, tr in enumerate(trs):
            t_here = tr["ts"] - t0
            step(to_update(tr["frame"]), t_here)
            if i + 1 < len(trs):
                t_next = trs[i + 1]["ts"] - t0
                if t_next - t_here <= 1.0:  # only attestable continuity
                    t_b = t_here
                    while t_b + 1.0 / a.hz < t_next:
                        t_b += 1.0 / a.hz
                        step(to_update(tr["frame"]), t_b)
    else:
        for tr in trs:
            step(to_update(tr["frame"]), tr["ts"] - t0)
    # attested-empty aftermath: the session's own final captured frame
    last = trs[-1]
    t = trs[-1]["ts"] - t0
    n = int(a.tail_seconds * a.hz)
    for i in range(n):
        t += 1.0 / a.hz
        step(to_update(last["frame"]), t)

    print("code-root: %s" % a.code_root)
    print("fixture:   %s  (%d snapshots over %.1fs + %.0fs attested-empty tail)"
          % (a.fixture, len(trs), trs[-1]["ts"] - t0, a.tail_seconds))
    for t, ev, room in events:
        print("  t=%7.2fs  %-15s -> room=%s" % (t, ev, room))
    print("final room state: %s (occupants=%d)" % (eng.room, eng.occupants))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
