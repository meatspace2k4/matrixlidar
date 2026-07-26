#!/usr/bin/env python3
"""Seated posture regression, against every attested sit in the corpus.

WHAT THIS TEST USED TO DO, AND WHY IT PASSED WHILE SITTING WAS BROKEN
--------------------------------------------------------------------
It sampled the 18:31-18:33 window of session_008 by scanning transitions for
`mark.kind == "posture" and "TOILET" in mark.to`, and reported 28/30 SITTING.
Those are TRANSITION frames: one frame is stored per committed state change.
A frame the classifier called UNKNOWN mid-sit produces no transition to a
toilet posture, so it can never appear in the sample. The test therefore
measured the classifier on the subset of frames the classifier had already got
right, and read 93% while the operator watched STATE: UNKNOWN for 146
consecutive seconds of an attested sit.

The 30-frame window is still here, now as a labeled fixture with that bias
written on it, because it is the hardest attested sit in the corpus (leaning
forward, return count fully overlapping standing). But the real regression
gate is the whole labeled corpus, per frame, against per-frame ground truth.

  python3 tests/test_seated.py
"""
import collections
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from bathroom import ingest                                    # noqa: E402
from bathroom.classify import classify                         # noqa: E402
from bathroom.zones import State                               # noqa: E402

# Per-frame floors, set below what is measured today so ordinary capture noise
# does not fail the build, and far above what the row-band classifier managed
# (0.457 recall: 151 of 293 attested sitting frames came out UNKNOWN).
MIN_SIT_RECALL = 0.90
MAX_FALSE_SIT = 0.03
MIN_STAND_RECALL = 0.90

FAILS = []


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


def main():
    samples = ingest.corpus()
    if not samples:
        print("FAIL no labeled corpus")
        return 1

    pairs = []
    per_sample = collections.defaultdict(collections.Counter)
    for s in samples:
        for want, frame in zip(s.states, s.frames):
            if want is None:
                continue
            got, _f = classify(frame, s.baseline)
            pairs.append((want, got))
            per_sample[s.id][(want, got)] += 1

    sit = [(w, g) for w, g in pairs if w == State.SITTING_AT_TOILET]
    stand = [(w, g) for w, g in pairs if w == State.STANDING_AT_TOILET]
    not_sit = [(w, g) for w, g in pairs if w != State.SITTING_AT_TOILET]
    check(len(sit) >= 250, "corpus holds %d attested sitting frames (need >= 250)"
          % len(sit))
    check(len(stand) >= 150, "corpus holds %d attested standing frames (need >= 150)"
          % len(stand))

    sit_rec = sum(1 for w, g in sit if g == w) / len(sit)
    check(sit_rec >= MIN_SIT_RECALL,
          "sitting recall %.3f >= %.2f (%d/%d frames; row bands managed 0.457)"
          % (sit_rec, MIN_SIT_RECALL, sum(1 for w, g in sit if g == w), len(sit)))

    false_sit = sum(1 for w, g in not_sit if g == State.SITTING_AT_TOILET)
    check(false_sit / len(not_sit) <= MAX_FALSE_SIT,
          "false sitting %d/%d = %.3f <= %.2f"
          % (false_sit, len(not_sit), false_sit / len(not_sit), MAX_FALSE_SIT))

    stand_rec = sum(1 for w, g in stand if g == w) / len(stand)
    check(stand_rec >= MIN_STAND_RECALL,
          "standing recall %.3f >= %.2f (%d/%d frames)"
          % (stand_rec, MIN_STAND_RECALL,
             sum(1 for w, g in stand if g == w), len(stand)))

    # no attested sit may be called an EMPTY room. This is the failure the
    # occlusion fix addresses: a seated torso that stops returning used to
    # leave 4 foreground cells and read EMPTY.
    sit_empty = sum(1 for w, g in sit if g == State.EMPTY)
    check(sit_empty == 0,
          "no attested sitting frame reads EMPTY (a body that stops returning "
          "is still a body): %d" % sit_empty)

    # and none may read UNKNOWN: the operator's complaint, stated as a gate
    sit_unknown = sum(1 for w, g in sit if g == State.UNKNOWN)
    check(sit_unknown == 0,
          "no attested sitting frame reads UNKNOWN: %d" % sit_unknown)

    print()
    for sid in sorted(per_sample):
        cm = per_sample[sid]
        n = sum(cm.values())
        hit = sum(v for (w, g), v in cm.items() if w == g)
        print("  %-38s %3d frames  %3d correct  %s"
              % (sid, n, hit,
                 " ".join("%s->%s=%d" % (w.name[:5], g.name[:5], v)
                          for (w, g), v in sorted(cm.items()) if w != g) or "clean"))

    print("\n" + ("ALL PASS" if not FAILS else "SOME FAILED"))
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
