# matrixlidar

Client for a DFRobot Matrix Lidar (SEN0628, 8x8 time-of-flight, mm) watching a
small bathroom. Reports presence states from the live sensor, runs a debounced
temporal state engine, captures labeled clips, and validates the engine (or a
trained model) against labeled snapshots.

The sensor and its always-on server run on the Pi (`bath-pi.local`); this repo
is the client half.

## Architecture

```
  Pi serial port ─► lidar_server.py ─(TCP :8628, raw 64-value frames)─►  this client
                    (always on, holds the port, broadcasts)

  stream.orient ─► classify (per frame) ─► engine (temporal, debounced) ─► state + timeline
                                        └► record (labeled clips) ─► validate / train
```

Clients read the broadcast stream, never the serial port: no cold start, no
contention between the heatmap, the live readout, and a recording. Frames are
oriented locally (row 0 = top, col 0 = column A).

## Two layers

- Per-frame classifier (`bathroom/classify.py`): one oriented frame to one
  state, baseline-relative (a body reads closer than the empty room at whatever
  it occludes; the DOOR column H is excluded because it swings on its own).
- Temporal engine (`bathroom/engine.py`): debounces the classifier with
  hysteresis, adopts the first observation as its initial state, and emits a
  timeline of stable segments.

### States (`zones.State`)

| State | What | How detected |
|---|---|---|
| `EMPTY` | no body | almost no occluded cells, or a tiny co-located noise cluster |
| `WALKING` | body crossing close in front of the sensor | many very-near cells, or near cells spanning >=4 columns |
| `STANDING_AT_TOILET` | upright between toilet and door | headroom row occluded deeply, with occupied torso cells below the head |
| `SITTING_AT_TOILET` | seated on the toilet | no supported head-over-torso signal, occluded cells cluster low; the toilet zone alone is not sufficient |
| `UNKNOWN` | body present, no clean posture | returned instead of a confident wrong label |

Door state is separate: `open%` from column H (1090mm = 0%, >=1400mm = 100%).
A body in the doorway splits the column; those frames hold the last coherent
reading instead of moving the %. `doorcal.py` is the live calibration readout.

### Room occupancy (`bathroom/occupancy.py`)

A third layer on top of the posture engine: `UNKNOWN` / `EMPTY` / `OCCUPIED`,
shown as the `ROOM:` line in both front-ends. The door is the only exit; any
door-side vanish marks the room EMPTY (exits are deltas and cannot be counted).
A vanish off the left FOV edge (sink/tub/shower) or mid-field keeps OCCUPIED.
A door-side entry that penetrates the field while already occupied increments
the occupant count. Boot and stream gaps are UNKNOWN, never assumed EMPTY.
Events: `ENTRY_CONFIRMED`, `ENTRY_ABORTED`, `EXIT`, `GAP_RESET`. Design:
`docs/occupancy-design.md`.

Committed transitions are also reported back to the Pi (`bathroom/archive.py`
sends `MARK` lines on the stream socket). The server snapshots its latest raw
frame with each mark into an in-memory ring (255 transitions, RAM only,
nothing on the SD card), grouped into sessions by the room state. `replay.py`
fetches the archive (`DUMP`) and replays it by time.

## Use

Run from the repo root (so `import bathroom` resolves). Stdlib only for
capture; `numpy` only for the training loader.

```
python3 heatmap_client.py            # live heatmap (cm) + zone panel + STATE + ROOM + sparkline
python3 -m bathroom.live             # text live readout: room, state, door%, sparkline, state strip
python3 -m bathroom.engine <clip>    # static render of a recorded clip (timeline + sparkline)
python3 replay.py                    # list archived transition sessions (from the Pi's RAM ring)
python3 replay.py --session N        # replay one session's raw-frame snapshots by time
```

### Capture labeled clips

```
python3 -m bathroom.record --label sitting  --seconds 30
python3 -m bathroom.record --label walking                # until Ctrl-C
python3 -m bathroom.record --baseline                     # empty-room baseline
```

`--label` is verbatim ground truth. The recorder prints the current corpus mix,
writes the clip atomically, and preserves the full time series (per-frame `mm`,
`state_auto`, debounced `state_engine`, `door_open_pct`) plus the `timeline`,
the `baseline` used, and `orientation`.

### Inspect / correct a clip

```
python3 -m bathroom.review data/clips/sitting_123.json                 # render
python3 -m bathroom.review <clip> --set-label standing                 # relabel whole clip
python3 -m bathroom.review <clip> --relabel 0:90=standing,90:240=sitting
```

Per-frame `corrections` (frame-index ranges) become per-frame ground truth via
`dataset.effective_labels`.

## Validation

One tool scores any predictor against the labeled snapshots, engine or model,
with identical metrics:

```
python3 -m bathroom.validate                              # temporal engine
python3 -m bathroom.validate --predictor frame            # per-frame classifier
python3 -m bathroom.validate --frame-level                # per-frame metrics too
python3 -m bathroom.validate --predictions model.jsonl    # score a model's output
python3 -m bathroom.validate --min-accuracy 0.9           # regression gate (nonzero exit if below)
```

Model predictions are JSONL keyed by sample id: `{"id": "sitting_seed", "pred":
"SITTING_AT_TOILET"}`. Output is accuracy, per-class precision/recall/F1, and a
confusion matrix (`bathroom/metrics.py`). Labeled snapshots come from
`fixtures/labeled/` and `data/clips/` (`bathroom/ingest.py`).

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
    y        = tr["y"]          # label for the clip
    y_frames = tr["y_frames"]   # per-frame labels (corrections applied)
    base     = tr["baseline"]   # (8, 8) reference the auto states used
    # also: times, dt, gaps, door_open_pct, state_auto, state_engine, hz
```

## Tests

```
python3 tests/run.py     # classifier + adversarial + pipeline + grounding gates
```

`test_classify.py` pins the four labeled captures; `test_classify_adversarial.py`
pins four boundary cases (sparse-top standing, toilet-only noise, walking at
the near cutoff, upper-row noise); `test_pipeline.py` pins the clip round-trip,
corrections, split, and the corrupt-input fallbacks; the grounding gates require
the engine and the frame classifier to score 1.0 on the labeled fixtures.

## Layout

```
heatmap_client.py        live heatmap (cm) + zone panel + STATE + ROOM + sparkline
replay.py                fetch + replay the Pi's in-memory transition archive
doorcal.py               live door-panel readout for calibrating column H
baselines/empty.json     empty-room reference for the classifier
docs/occupancy-design.md room occupancy engine design
bathroom/
  zones.py               zones + State vocabulary + label aliases
  stream.py              warm-TCP reader -> oriented frames (auto-reconnect)
  detect.py              door open% + per-zone values
  classify.py            per-frame classifier (baseline-relative)
  engine.py              temporal state machine (debounce, timeline) + CLI
  occupancy.py           room occupancy engine (UNKNOWN/EMPTY/OCCUPIED + events)
  archive.py             forward committed transitions to the Pi's RAM archive
  sparkline.py           occupancy sparkline + colored state strip
  record.py              time-based labeled clip capture (CLI)
  live.py                live state readout (CLI)
  review.py              inspect / per-frame relabel a clip (CLI)
  validate.py            score engine/frame/model vs labeled snapshots (CLI)
  metrics.py             confusion matrix + precision/recall/F1
  calibrate.py           threshold sweep over the labeled corpus (CLI)
  baselines.py           baseline age + drift checks (CLI)
  ingest.py              labeled snapshots -> uniform grounding corpus
  dataset.py             load clips for training + corpus CLI (summary/split/vocab)
  config.py              paths + robust baseline loading
data/                    calibration captures; clips/ + manifest are gitignored
fixtures/labeled/        seed labeled snapshots (the four reference captures)
tests/                   regression suite (run.py)
```
