"""Ingest real-world labeled snapshots into a uniform grounding corpus.

The model and the engine are both validated against the SAME labeled samples.
This module is the single reader that turns the operator's various labeled
artifacts into one shape:

  LabeledSample: id, label(str), state(State|None), frames(list[8x8]), source,
                 baseline(8x8)

Sources understood:
  fixtures/labeled/*.json   schema bathroom-labeled/1 (seed single frames)
  data/clips/*.json         schema bathroom-clip/1 (recorded time series)

Both store frames as flat row-major oriented `mm` arrays; we reshape to 8x8.
`state` is the normalized ground truth (None if the label is unrecognized, so
junk labels surface instead of silently scoring as a class).

Each sample also carries the baseline it was CAPTURED AGAINST: clips store
one, seeds predate the stored-baseline format and belong to the frozen
DEFAULT_BASELINE scene. Scoring a historical frame against today's live
baseline is wrong: the room drifts (a moved fixture, a shifted scene), and
that drift becomes phantom depression in cells the body never touched.
"""
import glob
import json
import os
import sys

from . import config
from .classify import DEFAULT_BASELINE
from .zones import N, from_label

FIXTURES_DIR = os.path.join(config.ROOT, "fixtures", "labeled")


class LabeledSample:
    __slots__ = ("id", "label", "state", "frames", "source", "path", "baseline")

    def __init__(self, id, label, frames, source, path, baseline=None):
        self.id = id
        self.label = label
        self.state = from_label(label)
        self.frames = frames
        self.source = source
        self.path = path
        # the scene this sample was captured against, not today's scene
        self.baseline = baseline or DEFAULT_BASELINE

    @property
    def n(self):
        return len(self.frames)


def _reshape(flat):
    return [list(flat[i * N:(i + 1) * N]) for i in range(N)]


def _load_file(path, strict=True):
    try:
        with open(path) as f:
            doc = json.load(f)
        frames = [_reshape(fr["mm"]) for fr in doc["frames"]]
    except (OSError, ValueError, KeyError) as e:
        if strict:
            raise
        sys.stderr.write("warning: skipping unreadable sample %s: %s\n" % (path, e))
        return None
    sid = os.path.splitext(os.path.basename(path))[0]
    return LabeledSample(sid, doc.get("label"), frames,
                         doc.get("source") or doc.get("schema", "?"), path,
                         baseline=doc.get("baseline"))


def _load_dir(d):
    out = [_load_file(p, strict=False)
           for p in sorted(glob.glob(os.path.join(d, "*.json")))]
    return [s for s in out if s is not None]


def fixtures(fixtures_dir=None):
    return _load_dir(fixtures_dir or FIXTURES_DIR)


def clips(clips_dir=None):
    return _load_dir(clips_dir or config.CLIPS_DIR)


def corpus(include_fixtures=True, include_clips=True):
    """All labeled samples available for grounding. Fixtures first, then clips."""
    out = []
    if include_fixtures:
        out += fixtures()
    if include_clips:
        out += clips()
    return out
