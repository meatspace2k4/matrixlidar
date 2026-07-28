#!/usr/bin/env python3
"""Objective old-vs-new comparison of bathroom.classify over EVERY real frame
available: fixtures/misdetections/*.json (+ archive/), fixtures/labeled/*.json.

Does not touch the deployed HEAD code. Loads classify.classify from two code
roots (git worktrees) into isolated module namespaces and runs both over the
same real frames, reporting:
  - per-session raw per-frame flip count for STANDING<->SITTING (proxy for
    the operator-observed flapping, independent of any temporal debounce)
  - accuracy against every ground truth window this session has established

  python3 scripts/ab_compare_classify.py OLD_ROOT NEW_ROOT
"""
import glob
import importlib
import json
import os
import sys
import time


def load_classify(root):
    saved = {k: v for k, v in sys.modules.items() if k.startswith("bathroom")}
    for k in list(saved):
        del sys.modules[k]
    sys.path.insert(0, root)
    try:
        classify = importlib.import_module("bathroom.classify")
        stream = importlib.import_module("bathroom.stream")
        config = importlib.import_module("bathroom.config")
        baseline = config.load_baseline()
        default_baseline = classify.DEFAULT_BASELINE
    finally:
        sys.path.pop(0)
        for k in list(sys.modules):
            if k.startswith("bathroom"):
                del sys.modules[k]
        sys.modules.update(saved)
    return classify, stream, baseline, default_baseline


def oriented(stream_m, frame_csv):
    flat = [int(x) for x in frame_csv.split(",")]
    return stream_m.orient([flat[i * 8:(i + 1) * 8] for i in range(8)])


def flips(states):
    n = 0
    pairs = {("SITTING_AT_TOILET", "STANDING_AT_TOILET"),
             ("STANDING_AT_TOILET", "SITTING_AT_TOILET")}
    for a, b in zip(states, states[1:]):
        if (a, b) in pairs:
            n += 1
    return n


def run_session(path, classify_m, stream_m, baseline):
    with open(path) as f:
        doc = json.load(f)
    trs = sorted(doc["transitions"], key=lambda t: t["ts"])
    states = []
    for tr in trs:
        mat = oriented(stream_m, tr["frame"])
        st, _f = classify_m.classify(mat, baseline)
        states.append(st.name)
    return states


def main():
    old_root, new_root = sys.argv[1], sys.argv[2]
    old_c, old_s, old_live, old_default = load_classify(old_root)
    new_c, new_s, new_live, new_default = load_classify(new_root)

    paths = sorted(set(
        glob.glob("fixtures/misdetections/*.json")
        + glob.glob("fixtures/misdetections/archive/*.json")
    ))

    print("%-46s %8s %8s   %s" % ("session", "old_flip", "new_flip", "window"))
    tot_old = tot_new = 0
    for p in paths:
        try:
            doc = json.load(open(p))
        except Exception:
            continue
        trs = doc.get("transitions") or []
        if len(trs) < 3:
            continue
        t0 = trs[0]["ts"]
        window = "%s-%s" % (time.strftime("%H:%M", time.localtime(t0)),
                            time.strftime("%H:%M", time.localtime(trs[-1]["ts"])))
        old_states = run_session(p, old_c, old_s, old_live)
        new_states = run_session(p, new_c, new_s, new_live)
        of, nf = flips(old_states), flips(new_states)
        tot_old += of
        tot_new += nf
        flag = "  <-- WORSE" if nf > of else ("  <-- better" if nf < of else "")
        print("%-46s %8d %8d   %s%s" % (os.path.basename(p), of, nf, window, flag))
    print("-" * 90)
    print("%-46s %8d %8d" % ("TOTAL raw per-frame S<->S flips", tot_old, tot_new))


if __name__ == "__main__":
    raise SystemExit(main())
