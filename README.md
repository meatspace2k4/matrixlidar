# matrixlidar — bathroom presence interface + grounding harness

Client-side interface to a DFRobot Matrix Lidar (SEN0628, 8x8 time-of-flight,
mm) watching a small bathroom. It reports **ground-truth presence states** from
the live sensor, runs an **end-to-end temporal state engine** (with a live
sparkline), **captures labeled time-series clips** for training a transformer,
and **grounds/validates** the engine — and later the model — against real
labeled snapshots.

The sensor and its always-on server run on the Pi (`jpark@bath-pi.local`); this
repo is the Mac/client half.

## Architecture

```
  Pi /dev/ttyACM0 ─► lidar_server.py ─(TCP :8628, raw 64-value frames)─►  this client
                     (always on, holds the serial port, broadcasts)

  stream.orient ─► classify (per frame) ─► engine (temporal, debounced) ─► state + timeline
                                        └► record (labeled clips) ─► validate / train
```

Reading the broadcast stream means no serial cold start and no contention: the
heatmap, the live readout, and a recording all share it and never steal the
port from the server. Frames are oriented locally (row 0 = top, col 0 = column A).

## Two layers

- **Per-frame classifier** (`bathroom/classify.py`): one oriented frame → one
  state, baseline-relative (a body reads closer than the empty room at whatever
  it occludes; the DOOR column H is excluded because it swings on its own).
- **Temporal engine** (`bathroom/engine.py`): debounces the classifier with
  hysteresis, adopts the first observation as its initial state, and emits a
  **timeline of stable segments**. Segments are exactly what a sequence model
  is trained to reproduce.

### Ground-truth states (`zones.State`)

| State | What | How detected (hardened against boundary cases) |
|---|---|---|
| `EMPTY` | no body | almost no occluded cells, or a tiny co-located noise cluster |
| `WALKING` | body crossing close in front of the sensor | many very-near cells, **or** near cells spanning ≥4 columns (robust to the exact near cutoff) |
| `STANDING_AT_TOILET` | upright between toilet and door | top rows occluded **deeply** (one deep top cell over a real footprint is enough) |
| `SITTING_AT_TOILET` | seated on the toilet | top clear (no deep top occlusion), occluded cells cluster low; the toilet zone **alone** is not sufficient |
| `UNKNOWN` | body present, no clean posture | returned instead of a confident wrong label |

Door state is separate: `open%` from column H (1000mm = 0%, ≥1400 = 100%).

## Use

Run from the repo root (so `import bathroom` resolves). Stdlib only for
capture; `numpy` only for the training loader.

```
python3 heatmap_client.py            # live heatmap + zone panel + STATE + sparkline
python3 -m bathroom.live             # text live readout: state, door%, occupancy sparkline, state strip
python3 -m bathroom.engine <clip>    # static render of a recorded clip (timeline + sparkline)
```

### Capture labeled clips

```
python3 -m bathroom.record --label sitting  --seconds 30
python3 -m bathroom.record --label walking                # until Ctrl-C
python3 -m bathroom.record --baseline                     # empty-room baseline
```

`--label` is your verbatim ground truth. The recorder prints the current corpus
mix so you can balance classes, writes the clip atomically, and preserves the
**full time series** (per-frame `mm`, `state_auto`, debounced `state_engine`,
`door_open_pct`) plus the `timeline`, the `baseline` used, and `orientation`.

### Inspect / correct a clip

```
python3 -m bathroom.review data/clips/sitting_123.json                 # render
python3 -m bathroom.review <clip> --set-label standing                 # relabel whole clip
python3 -m bathroom.review <clip> --relabel 0:90=standing,90:240=sitting
```

Per-frame `corrections` (frame-index ranges) become per-frame ground truth via
`dataset.effective_labels`, so a clip with a mid-capture transition trains correctly.

## Grounding & validation (the model harness)

One tool scores any predictor against the labeled snapshots — the engine today,
the trained model later, with identical metrics:

```
python3 -m bathroom.validate                              # temporal engine
python3 -m bathroom.validate --predictor frame            # per-frame classifier
python3 -m bathroom.validate --frame-level                # per-frame metrics too
python3 -m bathroom.validate --predictions model.jsonl    # score a MODEL's output
python3 -m bathroom.validate --min-accuracy 0.9           # regression gate (nonzero exit if below)
```

Model predictions are JSONL keyed by sample id: `{"id": "sitting_seed", "pred":
"SITTING_AT_TOILET"}`. Output is accuracy, per-class precision/recall/F1, and a
confusion matrix (`bathroom/metrics.py`). Labeled snapshots come from
`fixtures/labeled/` (seed set, the four operator captures) + `data/clips/`
(`bathroom/ingest.py`).

### Corpus tools

```
python3 -m bathroom.dataset --summary       # class balance over the corpus
python3 -m bathroom.dataset --split 0.2      # deterministic train/val clip ids
python3 -m bathroom.dataset --vocab          # write data/vocab.json (label source of truth)
python3 -m bathroom.dataset --validate       # scan for corrupt clips
python3 -m bathroom.calibrate                # threshold sweep over labeled data
python3 -m bathroom.baselines --age          # baseline staleness
python3 -m bathroom.baselines --check-drift  # vs newest empty clip (or --live N)
```

### Training loader

```python
from bathroom import dataset
for clip in dataset.iter_clips():            # corrupt files skipped + warned
    tr = dataset.to_training(clip)
    x        = tr["x"]          # (T, 8, 8) int32 oriented distances
    mask     = tr["mask"]       # (T, 8, 8) bool, True at no-target (4000) dropouts
    y        = tr["y"]          # operator label for the clip
    y_frames = tr["y_frames"]   # per-frame labels (corrections applied)
    base     = tr["baseline"]   # (8, 8) reference the auto states used
    # also: times, dt, gaps, door_open_pct, state_auto, state_engine, hz
```

## Tests

```
python3 tests/run.py     # classifier + adversarial + pipeline + grounding gates
```

`test_classify.py` pins the four labeled captures; `test_classify_adversarial.py`
pins four boundary cases an independent review surfaced (sparse-top standing,
toilet-only noise, walking at the near cutoff, upper-row noise);
`test_pipeline.py` pins the clip round-trip, corrections, split, and the
corrupt-input fallbacks; the grounding gates require the engine and the frame
classifier to score 1.0 on the labeled fixtures.

## Critique (tickle harness)

This project is registered as a target in `~/crit-o-matic/tickle`
(`targets/matrixlidar/`), the operator's POC-or-GTFO critique harness. Grounding
the engine against the labeled snapshots surfaced a real functional break —
**F-M001**: the temporal engine dropped the occupancy event for any clip shorter
than `commit_after` frames (a single-frame labeled snapshot read EMPTY
regardless of truth). Demonstrated (0.25 → 1.00 on the reproducer), fixed
(bootstrap the initial state), validated CONFIRMED, and pinned by the grounding
gate.

## Layout

```
heatmap_client.py        live heatmap + zone panel + STATE + sparkline
baselines/empty.json     empty-room reference for the classifier
bathroom/
  zones.py               zones + ground-truth State vocabulary + label aliases
  stream.py              warm-TCP reader -> oriented frames (auto-reconnect)
  detect.py              door open% + per-zone values
  classify.py            per-frame ground-truth classifier (baseline-relative)
  engine.py              temporal state machine (debounce, timeline) + CLI
  sparkline.py           occupancy sparkline + colored state strip
  record.py              time-based labeled clip capture (CLI)
  live.py                live state readout (CLI)
  review.py              inspect / per-frame relabel a clip (CLI)
  validate.py            ground engine/frame/model vs labeled snapshots (CLI)
  metrics.py             confusion matrix + precision/recall/F1
  calibrate.py           threshold sweep over the labeled corpus (CLI)
  baselines.py           baseline age + drift checks (CLI)
  ingest.py              labeled snapshots -> uniform grounding corpus
  dataset.py             load clips for training + corpus CLI (summary/split/vocab)
  config.py              paths + robust baseline loading
data/clips/              recorded clips (gitignored); manifest.jsonl indexes them
fixtures/labeled/        seed labeled snapshots (the four operator captures)
tests/                   regression suite (run.py)
```
