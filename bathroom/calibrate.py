"""Threshold calibration sweep over the labeled corpus.

Mirrors the tickle FN-sweep idea: vary the classifier's load-bearing
thresholds and report accuracy across the grid so a tuning change is grounded
in the labeled data, not a guess. Reports the current setting and the best
setting found; never edits Params (read-only sweep).

  python3 -m bathroom.calibrate
"""
import argparse

from . import config, ingest, metrics
from . import classify
from .validate import EnginePredictor, FramePredictor, CLASSES

DELTA_GRID = [200, 250, 300, 350, 400]
VNEAR_MIN_GRID = [6, 8, 10, 12]


def _score(predictor_cls, baseline, samples):
    pred = predictor_cls(baseline)
    pairs = []
    for s in samples:
        if s.state is None:
            continue
        pairs.append((s.state.name, pred.predict(s.frames).name))
    return metrics.accuracy(pairs), metrics.macro_f1(pairs, CLASSES)


def main():
    ap = argparse.ArgumentParser(description="Calibrate classifier thresholds on labeled data")
    ap.add_argument("--predictor", choices=["engine", "frame"], default="frame")
    ap.add_argument("--no-clips", action="store_true")
    args = ap.parse_args()

    samples = ingest.corpus(include_clips=not args.no_clips)
    labeled = [s for s in samples if s.state is not None]
    if not labeled:
        print("no labeled samples to calibrate against")
        return 1
    baseline = config.load_baseline()
    predictor_cls = EnginePredictor if args.predictor == "engine" else FramePredictor

    cur_delta = classify.Params.DELTA_MM
    cur_vnm = classify.Params.VNEAR_MIN
    print("calibrating predictor=%s over %d labeled samples" % (args.predictor, len(labeled)))
    print("current: DELTA_MM=%d VNEAR_MIN=%d" % (cur_delta, cur_vnm))
    print()
    header = "DELTA\\VNEAR " + "".join("%8d" % v for v in VNEAR_MIN_GRID)
    print(header)
    best = (-1.0, None)
    grid = {}
    try:
        for d in DELTA_GRID:
            cells = []
            for v in VNEAR_MIN_GRID:
                classify.Params.DELTA_MM = d
                classify.Params.VNEAR_MIN = v
                acc, f1 = _score(predictor_cls, baseline, labeled)
                grid[(d, v)] = acc
                cells.append("%8.2f" % acc)
                if acc > best[0]:
                    best = (acc, (d, v))
            mark = " <-" if d == cur_delta else ""
            print("%-11d%s%s" % (d, "".join(cells), mark))
    finally:
        classify.Params.DELTA_MM = cur_delta
        classify.Params.VNEAR_MIN = cur_vnm

    print()
    print("current setting accuracy: %.3f" % grid.get((cur_delta, cur_vnm), float("nan")))
    print("best accuracy: %.3f at DELTA_MM=%d VNEAR_MIN=%d"
          % (best[0], best[1][0], best[1][1]))
    if grid.get((cur_delta, cur_vnm), 0) >= best[0]:
        print("=> current setting is on the best plateau; no change indicated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
