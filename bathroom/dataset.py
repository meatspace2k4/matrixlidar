"""Load recorded clips for training and inspect the corpus.

A clip on disk is a self-describing JSON (schema bathroom-clip/1, or /2 with
an IR log alongside the frames; the frame fields are identical). This turns
clips into arrays a transformer can consume and provides the corpus tooling
the operator needs before training: class distribution, a deterministic
train/val split, and a vocabulary export.

  from bathroom import dataset
  for clip in dataset.iter_clips():          # corrupt files are skipped+warned
      tr = dataset.to_training(clip)         # x:(T,8,8), mask, per-frame labels, ...

CLI:
  python3 -m bathroom.dataset --summary      # class balance over the corpus
  python3 -m bathroom.dataset --vocab        # write data/vocab.json + print
  python3 -m bathroom.dataset --split 0.2    # deterministic train/val clip ids
  python3 -m bathroom.dataset --validate     # scan for corrupt/unreadable clips

`state_auto` is the per-frame classifier (advisory). `label` is the operator's
ground truth for the whole clip; per-frame `corrections` (written by
bathroom.review) override sub-ranges. Frames are oriented, row-major flattened.
"""
import argparse
import glob
import hashlib
import json
import os
import sys

from . import config
from .zones import N, NO_TARGET, ORIENTATION, State, from_label

EXPECTED_ORIENTATION = ORIENTATION
GAP_FACTOR = 3.0  # a frame gap > this * median dt is flagged as a stream gap


def iter_clip_files(clips_dir=None):
    clips_dir = clips_dir or config.CLIPS_DIR
    return sorted(glob.glob(os.path.join(clips_dir, "*.json")))


def _reshape(flat):
    return [list(flat[i * N:(i + 1) * N]) for i in range(N)]


def _dt_and_gaps(times):
    if len(times) < 2:
        return [], 0
    dt = [round(times[i] - times[i - 1], 4) for i in range(1, len(times))]
    ordered = sorted(dt)
    med = ordered[len(ordered) // 2] or 1e-9
    gaps = sum(1 for d in dt if d > GAP_FACTOR * med)
    return dt, gaps


def effective_labels(doc):
    """Per-frame ground-truth labels: the global label, with any per-frame
    `corrections` ranges (end-exclusive frame indices) overriding sub-spans."""
    n = doc["n"]
    labels = [doc.get("label")] * n
    for cor in doc.get("corrections", []) or []:
        lo = max(0, int(cor.get("start", 0)))
        hi = min(n, int(cor.get("end", n)))
        for i in range(lo, hi):
            labels[i] = cor.get("label")
    return labels


def load_clip(path, strict=False):
    """Flattened view of one clip, or None if unreadable (warns; strict=raise).

    Warns (does not fail) if the stored orientation differs from this client's,
    so clips from a differently-oriented setup are not silently mixed in."""
    try:
        with open(path) as f:
            doc = json.load(f)
        frames = doc["frames"]
        times = [fr["t"] for fr in frames]
        dt, gaps = _dt_and_gaps(times)
        orient = doc.get("orientation")
        if orient and orient != EXPECTED_ORIENTATION:
            sys.stderr.write("warning: %s orientation %r != expected %r\n"
                             % (os.path.basename(path), orient, EXPECTED_ORIENTATION))
        return {
            "clip_id": doc.get("clip_id", os.path.splitext(os.path.basename(path))[0]),
            "label": doc.get("label"),
            "hz": doc.get("hz"),
            "started": doc.get("started"),
            "ended": doc.get("ended"),
            "orientation": orient,
            "times": times,
            "dt": dt,
            "gaps": gaps,
            "frames": [fr["mm"] for fr in frames],          # T x 64
            "state_auto": [fr.get("state_auto") for fr in frames],
            "state_engine": [fr.get("state_engine") for fr in frames],
            "door_open_pct": [fr["door_open_pct"] for fr in frames],
            "labels": effective_labels(doc),                # per-frame ground truth
            "baseline": doc.get("baseline"),
            "timeline": doc.get("timeline", []),
            "n": doc.get("n", len(frames)),
            "path": path,
        }
    except (OSError, ValueError, KeyError) as e:
        if strict:
            raise
        sys.stderr.write("warning: skipping unreadable clip %s: %s\n" % (path, e))
        return None


def iter_clips(clips_dir=None):
    for p in iter_clip_files(clips_dir):
        clip = load_clip(p)
        if clip is not None:
            yield clip


def to_arrays(clip):
    """Compat view: (x:(T,8,8) int32, y:label str, state_auto:list). Requires
    numpy. For the full training view (mask, per-frame labels, timing) use
    to_training()."""
    import numpy as np
    x = np.asarray(clip["frames"], dtype=np.int32).reshape(-1, N, N)
    return x, clip["label"], clip["state_auto"]


def to_training(clip):
    """Full training view for a sequence model. numpy-backed where it helps:
      x        (T,8,8) int32 oriented distances
      mask     (T,8,8) bool, True where the cell is a no-target dropout (4000)
      y        operator label (whole clip)
      y_frames per-frame ground-truth labels (corrections applied)
      times/dt relative timestamps and inter-frame deltas (s); gaps flagged
      baseline (8,8) the empty reference the auto states were computed against
    """
    import numpy as np
    x = np.asarray(clip["frames"], dtype=np.int32).reshape(-1, N, N)
    mask = x >= NO_TARGET
    base = np.asarray(clip["baseline"], dtype=np.int32) if clip.get("baseline") else None
    return {
        "clip_id": clip["clip_id"],
        "x": x,
        "mask": mask,
        "y": clip["label"],
        "y_frames": clip["labels"],
        "times": clip["times"],
        "dt": clip["dt"],
        "gaps": clip["gaps"],
        "door_open_pct": clip["door_open_pct"],
        "state_auto": clip["state_auto"],
        "state_engine": clip["state_engine"],
        "baseline": base,
        "hz": clip["hz"],
    }


def labels(clips_dir=None):
    """Set of operator labels present in the corpus."""
    out = set()
    for clip in iter_clips(clips_dir):
        if clip["label"]:
            out.add(clip["label"])
    return sorted(out)


def summary(clips_dir=None):
    """Per-label clip + frame counts and gap totals over the corpus."""
    by_label = {}
    total_frames = 0
    total_gaps = 0
    n_clips = 0
    for clip in iter_clips(clips_dir):
        n_clips += 1
        lab = clip["label"] or "?"
        d = by_label.setdefault(lab, {"clips": 0, "frames": 0, "state": from_label(lab)})
        d["clips"] += 1
        d["frames"] += clip["n"]
        total_frames += clip["n"]
        total_gaps += clip["gaps"]
    return {"n_clips": n_clips, "total_frames": total_frames,
            "total_gaps": total_gaps, "by_label": by_label}


def train_val_split(val_fraction=0.2, clips_dir=None):
    """Deterministic split by clip_id hash (no RNG, stable across runs).
    Returns (train_ids, val_ids)."""
    train, val = [], []
    for p in iter_clip_files(clips_dir):
        cid = os.path.splitext(os.path.basename(p))[0]
        h = int(hashlib.sha1(cid.encode()).hexdigest(), 16) % 1000
        (val if h < val_fraction * 1000 else train).append(cid)
    return sorted(train), sorted(val)


def vocab(clips_dir=None, write=True):
    """Label vocabulary: the canonical State names plus any operator labels seen
    in the corpus. Optionally writes data/vocab.json as the single source of
    truth for a training script."""
    canonical = [s.name for s in State if s is not State.UNKNOWN]
    seen = labels(clips_dir)
    doc = {"states": canonical, "operator_labels": seen}
    if write:
        config.ensure_dirs()
        with open(config.VOCAB_FILE, "w") as f:
            json.dump(doc, f, indent=1)
    return doc


def main():
    ap = argparse.ArgumentParser(description="Inspect the recorded-clip corpus")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--vocab", action="store_true")
    ap.add_argument("--split", type=float, metavar="VAL_FRACTION", default=None)
    ap.add_argument("--validate", action="store_true", help="scan for corrupt clips")
    args = ap.parse_args()
    did = False
    if args.summary:
        did = True
        s = summary()
        print("corpus: %d clips, %d frames, %d gap(s)"
              % (s["n_clips"], s["total_frames"], s["total_gaps"]))
        for lab, d in sorted(s["by_label"].items()):
            st = d["state"].name if d["state"] is not None else "UNRECOGNIZED"
            print("  %-12s clips=%-3d frames=%-6d -> %s" % (lab, d["clips"], d["frames"], st))
        if s["n_clips"] and s["by_label"]:
            fewest = min(s["by_label"], key=lambda k: s["by_label"][k]["frames"])
            print("  lightest class: %s (record more)" % fewest)
    if args.vocab:
        did = True
        v = vocab()
        print("states: %s" % ", ".join(v["states"]))
        print("operator_labels: %s" % ", ".join(v["operator_labels"]) or "(none yet)")
        print("wrote %s" % config.VOCAB_FILE)
    if args.split is not None:
        did = True
        tr, va = train_val_split(args.split)
        print("train (%d): %s" % (len(tr), ", ".join(tr) or "(none)"))
        print("val   (%d): %s" % (len(va), ", ".join(va) or "(none)"))
    if args.validate:
        did = True
        bad = 0
        for p in iter_clip_files():
            if load_clip(p) is None:
                bad += 1
        print("scanned %d clips, %d unreadable" % (len(iter_clip_files()), bad))
        return 1 if bad else 0
    if not did:
        ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
