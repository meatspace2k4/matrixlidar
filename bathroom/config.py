"""Project paths and baseline loading (the only file I/O for the baseline).

classify.py stays pure; this resolves the empty-room baseline and the
recorded-clip locations, and degrades gracefully if the baseline file is
missing or corrupt instead of crashing every entrypoint.
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


def load_baseline(path=None):
    """Return an 8x8 baseline matrix. Falls back to the built-in default if the
    file is missing, unreadable, malformed, or not a valid 8x8 int grid, so a
    corrupt baseline never crashes recording/live/validation."""
    path = path or BASELINE_FILE
    if not os.path.isfile(path):
        return classify.DEFAULT_BASELINE
    try:
        with open(path) as f:
            doc = json.load(f)
        m = doc["matrix"] if isinstance(doc, dict) else doc
    except (OSError, ValueError, KeyError, TypeError) as e:
        sys.stderr.write("warning: bad baseline %s (%s); using built-in default\n"
                         % (path, e))
        return classify.DEFAULT_BASELINE
    if not _valid_matrix(m):
        sys.stderr.write("warning: baseline %s is not an 8x8 int grid; using default\n" % path)
        return classify.DEFAULT_BASELINE
    return m


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
