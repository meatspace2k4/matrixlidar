"""Project paths and baseline loading (the only file I/O for the baseline).

classify.py stays pure; this resolves the empty-room baseline and the
recorded-clip locations.

It does NOT degrade gracefully, deliberately. A missing or corrupt scan raises
BaselineError. See load_baseline for what the old fallback cost.
"""
import json
import os
import sys

from . import classify
from .zones import N, NO_TARGET

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINES_DIR = os.path.join(ROOT, "baselines")
DATA_DIR = os.path.join(ROOT, "data")
CLIPS_DIR = os.path.join(DATA_DIR, "clips")
MANIFEST = os.path.join(DATA_DIR, "manifest.jsonl")
BASELINE_FILE = os.path.join(BASELINES_DIR, "empty.json")
VOCAB_FILE = os.path.join(DATA_DIR, "vocab.json")


def _valid_matrix(m):
    return (isinstance(m, list) and len(m) == N
            and all(isinstance(row, list) and len(row) == N
                    and all(isinstance(v, int) for v in row) for row in m))


class BaselineError(Exception):
    """No usable empty-room scan. Everything downstream measures depth against
    this grid, so there is no such thing as degrading it: a substitute scan is
    not a worse answer to "what does this room look like", it is a confident
    answer about a DIFFERENT room, and every measurement taken against it is
    wrong in a way that reads as a plausible result rather than as an error."""


def load_baseline(path=None):
    """Return the empty-room 8x8 baseline matrix, or raise BaselineError.

    This used to fall back to classify.DEFAULT_BASELINE when the file was
    missing or corrupt. That fallback cost this project weeks. Three recorded
    clips took it at capture time and stored a scan of a room that no longer
    existed (wrong by up to 514mm on the toilet's own cells); measured against
    those, surface rake looked like a 94%-decisive posture feature and a whole
    classifier was built on it, while the feature that actually works looked
    marginal. Nothing announced itself: every run
    completed and printed plausible states.

    It was also PINNED BY A TEST ("corrupt baseline falls back to default",
    added one commit after the fallback itself), which turned the footgun into
    a specification that a red suite would defend.
    """
    path = path or BASELINE_FILE
    if not os.path.isfile(path):
        raise BaselineError(
            "no empty-room baseline at %s. Capture one (the room must be "
            "empty):\n  python3 -m bathroom.record --baseline" % path)
    try:
        with open(path) as f:
            doc = json.load(f)
        m = doc["matrix"] if isinstance(doc, dict) else doc
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise BaselineError("baseline %s is unreadable (%s); recapture it"
                            % (path, e))
    if not _valid_matrix(m):
        raise BaselineError("baseline %s is not an 8x8 int grid; recapture it"
                            % path)
    return m


def load_noise_floor(path=None):
    """The nearest each cell was ever observed to read in an EMPTY room, or None.

    Measured, not chosen. A flat depression cut sits inside the room's own spread
    on 42 of the 64 cells, which is how an empty bathroom built a body-sized
    shape and committed STANDING_AT_TOILET. Returns None per cell where the scan
    has no observation, and the caller falls back to the flat cut there.
    """
    doc = baseline_meta(path)
    return doc.get("noise_floor")


def baseline_meta(path=None):
    """Return the baseline file's metadata dict (captured ts, etc.) or {}."""
    path = path or BASELINE_FILE
    try:
        with open(path) as f:
            doc = json.load(f)
        return doc if isinstance(doc, dict) else {}
    except (OSError, ValueError):
        return {}


def ensure_dirs():
    for d in (BASELINES_DIR, DATA_DIR, CLIPS_DIR):
        if not os.path.isdir(d):
            os.makedirs(d)
