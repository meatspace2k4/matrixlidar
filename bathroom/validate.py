"""Grounding + validation harness: score a predictor against labeled snapshots.

This is the tool that grounds the state engine today and the trained model
later. A PREDICTOR maps a sample's frames to a State; we compare against the
operator's ground-truth label and report accuracy, per-class precision/recall/
F1, and a confusion matrix. Same numbers for engine and model, so the model is
held to the bar the engine already clears.

  python3 -m bathroom.validate                       # engine vs labeled corpus
  python3 -m bathroom.validate --predictor frame      # per-frame classifier
  python3 -m bathroom.validate --frame-level          # score every frame
  python3 -m bathroom.validate --predictions m.jsonl  # score a MODEL's output
  python3 -m bathroom.validate --min-accuracy 0.9     # gate: nonzero exit if below

Model predictions file (JSONL): one object per sample,
  {"id": "<sample id>", "pred": "SITTING_AT_TOILET"}
keyed by the sample id from ingest (the fixture/clip filename stem).
"""
import argparse
import json
import sys

from . import config, ingest, metrics
from .classify import classify_window
from .engine import StateEngine
from .zones import State, from_label

CLASSES = [s.name for s in (State.EMPTY, State.WALKING,
                            State.STANDING_AT_TOILET, State.SITTING_AT_TOILET,
                            State.UNKNOWN)]


def _dominant(states):
    tally = {}
    for s in states:
        tally[s] = tally.get(s, 0) + 1
    return max(tally, key=tally.get) if tally else State.EMPTY


class FramePredictor:
    """Per-frame classifier; clip label = dominant per-frame state."""
    name = "frame"

    def __init__(self, baseline, window=5):
        self.baseline = baseline
        self.window = window

    def per_frame(self, frames, baseline=None):
        from collections import deque
        win = deque(maxlen=self.window)
        out = []
        for m in frames:
            win.append(m)
            st, _ = classify_window(list(win), baseline or self.baseline)
            out.append(st)
        return out

    def predict(self, frames, baseline=None):
        return _dominant(self.per_frame(frames, baseline))


class EnginePredictor:
    """Debounced temporal engine; clip label = dominant committed state."""
    name = "engine"

    def __init__(self, baseline, window=5, commit_after=4):
        self.baseline = baseline
        self.window = window
        self.commit_after = commit_after

    def per_frame(self, frames, baseline=None):
        eng = StateEngine(baseline=baseline or self.baseline, window=self.window,
                          commit_after=self.commit_after)
        out = []
        for i, m in enumerate(frames):
            out.append(eng.update(m, i / 15.0).committed)
        eng.close()
        return out

    def predict(self, frames, baseline=None):
        eng = StateEngine(baseline=baseline or self.baseline, window=self.window,
                          commit_after=self.commit_after)
        for i, m in enumerate(frames):
            eng.update(m, i / 15.0)
        eng.close()
        return eng.dominant_state()


class PredictionsFilePredictor:
    """Replays a MODEL's predictions from JSONL so the same harness scores it."""
    name = "model"

    def __init__(self, path):
        self.byid = {}
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                self.byid[d["id"]] = from_label(d["pred"])

    def predict_for(self, sample):
        if sample.id not in self.byid:
            raise KeyError("no model prediction for sample id %r" % sample.id)
        return self.byid[sample.id]


def run(predictor, samples, frame_level=False):
    """Return (clip_pairs, frame_pairs, rows) for reporting."""
    clip_pairs = []
    frame_pairs = []
    rows = []
    for s in samples:
        if s.state is None:
            rows.append((s.id, s.label, "?", "UNLABELED (label not recognized)"))
            continue
        truth = s.state.name
        if isinstance(predictor, PredictionsFilePredictor):
            pred = predictor.predict_for(s).name
            clip_pairs.append((truth, pred))
            rows.append((s.id, s.label, pred, "ok" if pred == truth else "MISS"))
            continue
        # score every sample against the baseline it was captured against,
        # never today's live scene (the room drifts; that drift would read as
        # depression in cells the body never touched)
        pred = predictor.predict(s.frames, s.baseline).name
        clip_pairs.append((truth, pred))
        if frame_level and hasattr(predictor, "per_frame"):
            for st in predictor.per_frame(s.frames, s.baseline):
                frame_pairs.append((truth, st.name))
        rows.append((s.id, s.label, pred, "ok" if pred == truth else "MISS"))
    return clip_pairs, frame_pairs, rows


def main():
    ap = argparse.ArgumentParser(description="Ground a predictor against labeled snapshots")
    ap.add_argument("--predictor", choices=["engine", "frame"], default="engine")
    ap.add_argument("--predictions", help="JSONL of model predictions to score instead")
    ap.add_argument("--frame-level", action="store_true",
                    help="also report per-frame metrics (engine/frame only)")
    ap.add_argument("--no-clips", action="store_true", help="fixtures only, skip recorded clips")
    ap.add_argument("--min-accuracy", type=float, default=None,
                    help="exit nonzero if clip accuracy is below this (regression gate)")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = ap.parse_args()

    samples = ingest.corpus(include_clips=not args.no_clips)
    if not samples:
        print("no labeled samples found (fixtures/labeled/ or data/clips/)")
        return 1

    baseline = config.load_baseline()
    if args.predictions:
        predictor = PredictionsFilePredictor(args.predictions)
    elif args.predictor == "frame":
        predictor = FramePredictor(baseline)
    else:
        predictor = EnginePredictor(baseline)

    clip_pairs, frame_pairs, rows = run(predictor, samples, args.frame_level)
    acc = metrics.accuracy(clip_pairs)

    if args.json:
        print(json.dumps({
            "predictor": predictor.name,
            "n_samples": len(samples),
            "clip_accuracy": acc,
            "clip_macro_f1": metrics.macro_f1(clip_pairs, CLASSES),
            "per_class": metrics.per_class(clip_pairs, CLASSES),
            "rows": [{"id": r[0], "label": r[1], "pred": r[2], "verdict": r[3]} for r in rows],
        }, indent=1))
    else:
        print("predictor: %s   samples: %d" % (predictor.name, len(samples)))
        print("-" * 64)
        for sid, label, pred, verdict in rows:
            print("  %-22s truth=%-10s pred=%-20s %s" % (sid, label, pred, verdict))
        print("-" * 64)
        print("CLIP-LEVEL")
        print(metrics.format_report(clip_pairs, CLASSES))
        if args.frame_level and frame_pairs:
            print("\nFRAME-LEVEL")
            print(metrics.format_report(frame_pairs, CLASSES))

    if args.min_accuracy is not None and acc < args.min_accuracy:
        sys.stderr.write("\nFAIL: clip accuracy %.3f < min %.3f\n" % (acc, args.min_accuracy))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
