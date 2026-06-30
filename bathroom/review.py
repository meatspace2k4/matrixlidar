"""Inspect and correct a recorded clip.

A clip carries one operator label for its whole span, but a real capture can
contain a transition (entered STANDING, then SITTING) or a mislabel. This tool
renders the clip (state strip + occupancy sparkline + timeline) so you can see
what was captured, and writes per-frame `corrections` (frame-index ranges) or a
new whole-clip label back into the clip JSON. dataset.effective_labels() then
applies those corrections as per-frame ground truth for training.

  python3 -m bathroom.review data/clips/sitting_123.json            # show
  python3 -m bathroom.review <clip> --set-label standing            # relabel all
  python3 -m bathroom.review <clip> --relabel 0:90=standing,90:240=sitting

Ranges are frame indices, end-exclusive. Use --show to see the index<->time map.
"""
import argparse
import json
import sys

from . import config, sparkline
from .engine import run_clip
from .zones import N, from_label


def _read(path):
    with open(path) as f:
        return json.load(f)


def _write(path, doc):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(doc, f)
    import os
    os.replace(tmp, path)


def _frames(doc):
    return [[fr["mm"][i * N:(i + 1) * N] for i in range(N)] for fr in doc["frames"]]


def show(doc):
    frames = _frames(doc)
    baseline = doc.get("baseline") or config.load_baseline()
    per_frame, segs, occ = run_clip(frames, baseline=baseline)
    print("clip %s  label=%r  n=%d  hz=%s"
          % (doc.get("clip_id"), doc.get("label"), doc.get("n"), doc.get("hz")))
    if doc.get("corrections"):
        print("corrections:")
        for c in doc["corrections"]:
            print("  frames [%d:%d) -> %s" % (c.get("start", 0), c.get("end", doc["n"]), c.get("label")))
    print(sparkline.render_clip(per_frame, occ))
    print()
    print("engine timeline (frame index <-> time):")
    idx = 0
    for s in segs:
        print("  [%4d ..]  %-18s %6.2fs  %4d frames" % (idx, s["state"], s["duration"], s["n_frames"]))
        idx += s["n_frames"]
    print()
    print(sparkline.legend(color=False))


def _parse_ranges(spec, n):
    """'0:90=standing,90:240=sitting' -> [{start,end,label}]; validates labels."""
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        rng, _, lab = part.partition("=")
        lab = lab.strip()
        if not lab:
            raise ValueError("range %r missing '=label'" % part)
        lo_s, _, hi_s = rng.partition(":")
        lo = int(lo_s) if lo_s.strip() else 0
        hi = int(hi_s) if hi_s.strip() else n
        if not (0 <= lo < hi <= n):
            raise ValueError("range %d:%d out of bounds for n=%d" % (lo, hi, n))
        if from_label(lab) is None:
            sys.stderr.write("note: label %r is not a known state; stored verbatim\n" % lab)
        out.append({"start": lo, "end": hi, "label": lab})
    return out


def main():
    ap = argparse.ArgumentParser(description="Inspect / correct a recorded clip")
    ap.add_argument("clip")
    ap.add_argument("--set-label", help="overwrite the whole-clip ground-truth label")
    ap.add_argument("--relabel", help="per-frame corrections, e.g. 0:90=standing,90:240=sitting")
    ap.add_argument("--clear-corrections", action="store_true")
    args = ap.parse_args()

    doc = _read(args.clip)
    changed = False
    if args.set_label:
        doc["label"] = args.set_label.strip()
        changed = True
        print("set label -> %r" % doc["label"])
    if args.clear_corrections:
        doc.pop("corrections", None)
        changed = True
        print("cleared corrections")
    if args.relabel:
        try:
            doc["corrections"] = _parse_ranges(args.relabel, doc["n"])
        except ValueError as e:
            ap.error("bad --relabel spec: %s" % e)
        changed = True
        print("set %d correction range(s)" % len(doc["corrections"]))
    if changed:
        _write(args.clip, doc)
    show(doc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
