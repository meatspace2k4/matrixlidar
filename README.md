# matrixlidar — bathroom presence interface

Client-side interface to a DFRobot Matrix Lidar (SEN0628, 8x8 time-of-flight,
mm) watching a small bathroom. It reports **ground-truth presence states** from
the live sensor and **captures labeled time-series clips** for training a
sequence (transformer) model.

The sensor and its always-on server live on the Pi (`jpark@bath-pi.local`).
This repo is the Mac/client half: it reads the warm stream, classifies, draws
the heatmap, and records.

## Architecture

```
  Pi  /dev/ttyACM0 ──► lidar_server.py ──(TCP :8628, raw 64-value frames)──►  this client
                       (always on, holds the serial port, broadcasts)
```

Reading the broadcast stream means **no serial cold start and no contention**:
the heatmap, the live readout, and a recording can all run at once and never
steal the port from the server (unlike the Pi's direct-serial snapshot tool).
Each frame is oriented locally into display orientation (row 0 = top, col 0 =
column A) — the same calibrated transform the heatmap uses.

## Ground-truth states

`bathroom/classify.py` maps each (denoised) frame to one of these
(`bathroom/zones.py:State`):

| State | What it is | How it's detected |
|---|---|---|
| `EMPTY` | no body in the field | almost no cells closer than the empty baseline |
| `WALKING` | body crossing close in front of the sensor; toilet obscured | many cells very near (< 650 mm), large area |
| `STANDING_AT_TOILET` | body upright between toilet and door | occlusion reaches the **top** rows (tall vertical band) |
| `SITTING_AT_TOILET` | body seated on the toilet | top rows stay at baseline; occlusion clusters **low** / at the toilet |
| `UNKNOWN` | ambiguous / transitional | none of the above fire (never forced to a guess) |

Detection is **baseline-relative**: a body reads closer than the empty room at
whatever it occludes (the sensor sits between the user and the toilet). We
measure per-cell depression below a captured empty baseline rather than using
absolute thresholds, because the empty field is a non-uniform "bowl". The
**DOOR column (H) is excluded from presence** — it swings 1000→4000 mm on its
own and says nothing about a body. Door state is reported separately as
`open%` (column H median, 1000 mm = 0%, ≥1400 mm = 100%).

The thresholds (`classify.Params`) are calibrated against the four labeled
captures; `tests/test_classify.py` pins them. Re-run it after any tuning:

```
python3 tests/test_classify.py     # all four labeled states must PASS
```

## Use

Run everything from the repo root (so `import bathroom` resolves). Python 3
stdlib only for capture; `numpy` only for the training loader.

```
python3 heatmap_client.py                 # live heatmap + zone panel + STATE line
python3 -m bathroom.live                  # text-only live state readout
python3 -m bathroom.live --verbose        # + feature dump on each state change
```

### Capture labeled clips (for training)

```
python3 -m bathroom.record --label sitting  --seconds 30
python3 -m bathroom.record --label standing --seconds 30
python3 -m bathroom.record --label walking                 # until Ctrl-C
python3 -m bathroom.record --label empty    --seconds 60
```

`--label` is **your verbatim ground truth** for the whole clip. Watch
`heatmap_client.py` in another pane while you record to see what's happening.
Each clip preserves the **full time series** (nothing is collapsed to a
median, unlike the Pi snapshot tool): one self-describing JSON under
`data/clips/`, plus a one-line summary appended to `data/manifest.jsonl`.

### Empty-room baseline

```
python3 -m bathroom.record --baseline       # leave the room; ~6 s median
```

Writes `baselines/empty.json`, which the classifier prefers over the built-in
default. Best captured with the **room empty and the door closed**. A clean
default (real data from this install) already ships.

## Clip format (`schema: bathroom-clip/1`)

```jsonc
{
  "clip_id": "sitting_1782787142",
  "label": "sitting",                 // operator ground truth (verbatim)
  "started": 1782787142.4, "ended": ..., "n": 440, "hz": 14.7,
  "orientation": "display: rotate_cw -> flip_x -> flip_y; row0=top, col0=A",
  "baseline": [[...8x8...]],          // baseline used for the auto states
  "auto_state_counts": {"SITTING_AT_TOILET": 430, "UNKNOWN": 10},
  "frames": [
    {"t": 0.0, "mm": [64 ints, row-major oriented],
     "door_open_pct": 39.0, "state_auto": "SITTING_AT_TOILET"},
    ...
  ]
}
```

`mm` is the oriented frame, row-major (reshape to 8x8). `state_auto` is the
classifier's per-frame guess (advisory, denoised over a 5-frame window);
`label` is the authoritative target. Load for training:

```python
from bathroom import dataset
for clip in dataset.iter_clips():
    x, y, auto = dataset.to_arrays(clip)   # x: (T, 8, 8) int32, y: label str
```

## Layout

```
heatmap_client.py        live heatmap + zone panel + STATE
baselines/empty.json     empty-room reference for the classifier
bathroom/
  zones.py               zones + ground-truth State vocabulary
  stream.py              warm-TCP reader -> oriented frames (auto-reconnect)
  detect.py              door open% + per-zone values
  classify.py            the ground-truth state classifier  <-- the gap, filled
  record.py              time-based labeled clip capture (CLI)
  live.py                live state readout (CLI)
  dataset.py             load clips into training arrays
  config.py              paths + baseline loading
data/clips/              recorded clips (gitignored); manifest.jsonl indexes them
tests/test_classify.py   calibration test on the four labeled captures
```

`zones.py` and `classify.py` are kept identical to the copies in the Pi's
`~/matrixlidar/bathroom/` so the offline snapshot tool and this client report
the same states.
