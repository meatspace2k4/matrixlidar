"""Project paths and baseline loading (the only file I/O in the package).

classify.py stays pure; this is where the empty-room baseline and the
recorded-clip locations are resolved.
"""
import json
import os

from . import classify

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINES_DIR = os.path.join(ROOT, "baselines")
DATA_DIR = os.path.join(ROOT, "data")
CLIPS_DIR = os.path.join(DATA_DIR, "clips")
MANIFEST = os.path.join(DATA_DIR, "manifest.jsonl")
BASELINE_FILE = os.path.join(BASELINES_DIR, "empty.json")


def load_baseline(path=None):
    """Return an 8x8 baseline matrix. Falls back to the built-in default if no
    captured baseline exists, so the classifier always has a reference."""
    path = path or BASELINE_FILE
    if os.path.isfile(path):
        with open(path) as f:
            doc = json.load(f)
        return doc["matrix"] if isinstance(doc, dict) else doc
    return classify.DEFAULT_BASELINE


def ensure_dirs():
    for d in (BASELINES_DIR, DATA_DIR, CLIPS_DIR):
        if not os.path.isdir(d):
            os.makedirs(d)
