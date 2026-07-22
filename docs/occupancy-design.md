# Room occupancy engine — design

Design only; no implementation in this document. Sits ON TOP of the existing
stack (stream → classify → StateEngine) and consumes their outputs. It never
touches sensing, orientation, zones, or the posture classifier.

```
stream → classify (posture, per frame) → StateEngine (debounced posture)
                                       └→ OccupancyEngine (this design)
                                            → RoomState + events
```

## 1. Invariants (from the operator)

- **I1 Sticky occupancy.** Once a person has verifiably entered, only an
  observed exit through the door yields EMPTY. A body vanishing out of the
  60° FOV (sink/tub/shower side, left edge) is NOT an exit. A body vanishing
  mid-field (occlusion, stillness) is NOT an exit.
- **I2 Anti-latch.** Presence that never amounted to a real entry (hallway
  noise through an open door, a step-in-step-out) must settle back to EMPTY,
  or the next real entry's EMPTY→OCCUPIED edge never fires.
- **I3 Honest boot.** Boot state is UNKNOWN, never EMPTY: the Pi can restart
  while someone is out of view. Same on stream gaps.
- **I4 Guarded calibration.** The door jamb is always in view and is the
  calibration reference, but recalibration may only run after the room has
  been verifiably EMPTY for a quiet period.
- **I5 Exit certainty.** The sensor is close and cannot be defeated: a body
  moving doorward and past the jamb IS an exit, regardless of door open%.
  Door state is never consulted for occupancy.

## 2. Inputs (all already available)

| Input | Source |
|---|---|
| instant posture state + features (`occ_cells`, `occ_count`, `min_mm`) | `classify.features` via `engine.Update` |
| committed posture state | `StateEngine` |
| centroid column / drift | derived from `occ_cells` (as `live._direction` does) |
| frame timestamps, gap detection | `stream.frames` wall clock |
| door open% (debounced), column H / jamb values | `detect` |

## 3. Geometry bands (config, display orientation)

- `DOOR_BAND` = columns G,H (indices 6,7): approach + jamb. A track must be
  here to enter or exit.
- `LEFT_BAND` = column A (index 0): the FOV boundary toward sink/tub/shower.
  Vanishing here means out-of-FOV, still in the room.
- `FIELD` = everything else.

## 4. Track model

Single-track model: one centroid, so simultaneous bodies are not separable.
Entries are individually observable (each is a door-side appear + penetration),
so an entry while OCCUPIED increments an occupant count. Exits are deltas and
cannot be attributed, so any exit empties the room (see section 5). Per frame,
the track is the centroid + bounding box of
`occ_cells`, smoothed over K=3 frames; velocity is centroid-column drift per
second (the same computation the WALKING direction arrow uses). Track
primitives:

- `APPEAR(side)` — occupied frames after quiet; side = door / left / mid from
  the first smoothed centroid.
- `VANISH(side)` — occupied → empty; side from the last L=4 frames' centroid
  plus velocity direction (a doorward-moving vanish near the door is
  door-side even if the final centroid is a column short).
- `CROSS_IN` — `APPEAR(door)` whose centroid then leaves `DOOR_BAND` into
  `FIELD` (penetration = a real entry, not hallway shimmer).
- `CROSS_OUT` — centroid in `FIELD` drifts doorward by ≥ `D_EXIT` columns,
  overlaps `DOOR_BAND`, then `VANISH(door)`.

## 5. Room state machine

External states: `UNKNOWN`, `EMPTY`, `OCCUPIED` (a separate `RoomState` enum;
the posture `State` enum is untouched). Internal fourth state
`MAYBE_OCCUPIED`, reported externally as OCCUPIED so consumers react
instantly, but it can decay (that decay is the anti-latch).

| From | To | Condition |
|---|---|---|
| UNKNOWN | OCCUPIED | body evidence sustained ≥ `T_CONFIRM` (present at boot counts; they never "entered") |
| UNKNOWN | EMPTY | observed `CROSS_OUT` exit only. Never by timeout: a shower out of FOV is indefinitely long |
| EMPTY | MAYBE_OCCUPIED | any body evidence (instant ≠ EMPTY) |
| MAYBE_OCCUPIED | OCCUPIED | `CROSS_IN` observed, or body persists ≥ `T_CONFIRM` |
| MAYBE_OCCUPIED | EMPTY | anti-latch: evidence gone, total presence < `T_CONFIRM`, track never left `DOOR_BAND`, and `T_DECAY` of empty frames since |
| OCCUPIED | EMPTY | exit only: ANY door-side vanish empties the room and zeroes the count, regardless of occupants (exits are deltas; they cannot be attributed) |
| OCCUPIED | OCCUPIED | on `VANISH(left)` or `VANISH(mid)`: keep state, record `last_seen` side + time (I1) |
| OCCUPIED | OCCUPIED | on a new `CROSS_IN`: occupants += 1 |
| any | UNKNOWN | stream gap > `T_GAP` or server restart (I3) |

Anti-latch cases mapped to the table:
1. Hallway noise through an open door: blob confined to `DOOR_BAND`, short →
   MAYBE decays to EMPTY.
2. Step-in, step-out: `CROSS_IN` then `CROSS_OUT` → EMPTY via the exit rule.
3. Door swing itself: never presence (door column already excluded from the
   classifier; jamb is static).
4. Safety valve against a permanent latch: a `VANISH` whose last position was
   inside `DOOR_BAND` within `T_VANISH` of having been in the band counts as
   an exit candidate even if the drift threshold was missed (a fast exit
   between frames), still requiring `T_EXIT_CONFIRM` of empty. Left-side
   vanishes never qualify.

The EMPTY→OCCUPIED edge is the consumer trigger; anti-latch guarantees the
edge re-arms after every false start.

## 6. Door calibration (jamb-referenced, guarded)

- Maintain a rolling median of the jamb cells (always structural, always in
  view) as the reference `J`.
- Recalibrate (door closed/open thresholds and the empty baseline) only when
  ALL hold: state == EMPTY continuously ≥ `T_RECAL`, no MAYBE excursion in
  that window, jamb spread < `J_EPS` (nothing moving). UNKNOWN or OCCUPIED
  never recalibrates (I4): a towel on the door must not poison the baseline.
- If `J` shifts beyond `J_MOVED` versus the stored reference, the sensor was
  bumped: raise `SENSOR_MOVED`, drop to UNKNOWN, refuse recalibration until
  re-attested empty.

## 7. Parameters (one config block, sweepable by calibrate.py over clips)

| Param | Default | Meaning |
|---|---|---|
| `T_CONFIRM` | 2.0 s | presence needed to confirm occupancy without penetration |
| `T_DECAY` | 0.7 s | empty time before an unconfirmed MAYBE reverts |
| `T_EXIT_CONFIRM` | 1.0 s | empty time after a door-side vanish before EMPTY |
| `T_VANISH` | 0.5 s | door-band recency window for the fast-exit safety valve |
| `T_GAP` | 5 s | stream gap that forces UNKNOWN |
| `T_RECAL` | 60 s | continuous EMPTY required before recalibration |
| `D_EXIT` | 1.5 cols | net doorward drift for a deliberate exit approach |
| `K` | 3 frames | centroid smoothing window |

## 8. Interface

New module `bathroom/occupancy.py`:
`OccupancyEngine.update(u: engine.Update, t) → RoomUpdate(room_state, event, meta)`.
Events: `ENTRY_CONFIRMED`, `ENTRY_ABORTED`, `EXIT`, `GAP_RESET`,
`RECALIBRATED`, `SENSOR_MOVED`. `live.py` and `heatmap_client.py` add one
`ROOM: OCCUPIED (3m12s)` line; `record.py` logs `room_state` per frame so
clips validate the machine.

## 9. Validation plan (before any live trust)

Labeled clips asserting the final room state: normal enter/exit cycle;
enter → vanish left (shower) → reappear → exit; hallway noise with door open;
step-in-step-out; pause-at-door exit; fast exit; boot with person mid-room;
boot into truly empty room (must stay UNKNOWN until first observed exit).

## 10. Non-goals

Counting bodies on exit (delta-only; any exit empties the room), fall
detection, door% accuracy changes, any Pi-side or sensing changes.

## 11. Implementation

`bathroom/occupancy.py` implements sections 4, 5, and 7 with events
`ENTRY_CONFIRMED`, `ENTRY_ABORTED`, `EXIT`, `GAP_RESET`; scenarios from
section 9 are pinned in `tests/test_occupancy.py`. Section 6 (guarded
recalibration, `RECALIBRATED`, `SENSOR_MOVED`) is not implemented.
