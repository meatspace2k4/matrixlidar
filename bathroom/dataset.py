"""Load recorded clips for training a sequence model.

A clip on disk is a self-describing JSON (schema bathroom-clip/1). This turns
it into arrays a transformer can consume. numpy is optional: `load_clip`
returns plain lists; `to_arrays` upgrades to ndarrays if numpy is present.

  from bathroom import dataset
  for clip in dataset.iter_clips():
      x = clip["frames"]          # list[T][64] oriented mm, row-major
      y = clip["label"]           # operator ground truth for the whole clip
      auto = clip["state_auto"]   # list[T] per-frame classifier state

Frames are oriented (display) and row-major flattened; reshape to (T, 8, 8)
with to_arrays(). Per-frame `state_auto` is advisory; `label` is ground truth.
"""
import glob
import json
import os

from . import config


def iter_clip_files(clips_dir=None):
    clips_dir = clips_dir or config.CLIPS_DIR
    return sorted(glob.glob(os.path.join(clips_dir, "*.json")))


def load_clip(path):
    """Return a flattened view of one clip: label, times, frames, auto states."""
    with open(path) as f:
        doc = json.load(f)
    frames = doc["frames"]
    return {
        "clip_id": doc["clip_id"],
        "label": doc["label"],
        "hz": doc.get("hz"),
        "times": [fr["t"] for fr in frames],
        "frames": [fr["mm"] for fr in frames],          # T x 64
        "state_auto": [fr["state_auto"] for fr in frames],
        "door_open_pct": [fr["door_open_pct"] for fr in frames],
        "n": doc["n"],
    }


def iter_clips(clips_dir=None):
    for p in iter_clip_files(clips_dir):
        yield load_clip(p)


def to_arrays(clip):
    """Upgrade a loaded clip to numpy: frames -> (T, 8, 8) int array. Requires
    numpy; raises ImportError otherwise so training code fails loudly."""
    import numpy as np
    x = np.asarray(clip["frames"], dtype=np.int32).reshape(-1, 8, 8)
    return x, clip["label"], clip["state_auto"]


def labels(clips_dir=None):
    """Set of operator labels present in the corpus (for building a vocab)."""
    return sorted({load_clip(p)["label"] for p in iter_clip_files(clips_dir)})
