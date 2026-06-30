"""Ground-truth state classifier for the bathroom matrix lidar.

This is the gap the existing code deferred (snapshot.py reported
"Status: DEFERRED"). It turns an oriented 8x8 frame into one of the operator's
labeled states (zones.State) deterministically and explainably, so it can serve
as ground truth for recorded clips.

How it decides
--------------
Distances are mm; smaller = closer. The sensor sits between the user and the
toilet, so a body reads CLOSER than the empty-room baseline at whatever it
occludes. We measure occlusion as a per-cell *depression* below a captured
empty baseline (the field is a non-uniform bowl, so absolute thresholds are
wrong). The DOOR column (H) is excluded from presence: it swings 1000->4000mm
on its own and says nothing about a body.

Discriminators, calibrated on the four labeled captures AND hardened against
the boundary failures an adversarial review surfaced (see tests/):
  EMPTY   almost no occluded cells, OR a tiny co-located noise cluster.
  WALKING a body filling the near field in front of the sensor: many very-near
          cells, OR near cells spanning >=4 columns (so it is robust to the
          exact VNEAR cutoff: a body at ~650mm across the width still reads
          WALKING instead of flipping to STANDING).
  STANDING upright body -> it occludes the TOP rows DEEPLY (>= DEEP_DELTA below
          baseline). One deep top cell over a real body footprint is enough,
          so a narrow/off-center stance is not misread as sitting.
  SITTING low/seated body -> top stays at baseline (no deep top occlusion) while
          occluded cells cluster low. The toilet zone alone is NOT sufficient
          (door swing / dust on those 4 cells would otherwise read SITTING).
  UNKNOWN a body is present but matches no posture cleanly. Returned instead of
          a confident wrong label, so auto-labels stay trustworthy as ground
          truth.

Tuned constants live in `Params`; calibrate.py sweeps them over the labeled
corpus. The classifier is pure (no I/O).
"""
import statistics

from .zones import N, NO_TARGET, DOOR_COL, ZONE_CELLS, Zone, State


class Params:
    DELTA_MM = 300          # cell is "occupied" if this many mm closer than baseline
    DEEP_DELTA = 500        # cell is "deeply" occluded (a body, not an edge graze)
    BASELINE_CAP = 1500     # no-target baseline cells capped here for delta math
    VNEAR_MM = 650          # "very near" => body essentially at the sensor lens
    VNEAR_MIN = 8           # WALKING via this many very-near cells
    NEAR_SOFT = 850         # softer "near" band, used for the column-spread rule
    NEAR_MIN = 10           # WALKING via this many near cells ...
    NEAR_COLS_MIN = 4       # ... spread across at least this many columns
    EMPTY_MAX = 2           # EMPTY if at most this many occupied cells
    NOISE_MAX = 4           # a cluster this small that fits in a 2x2 box is noise
    MIN_BODY = 5            # fewer occupied cells than this is not a posture
    UPPER_ROWS = (0, 1)     # the top of the field (occluded when upright)
    LOWER_ROWS = (5, 6, 7)  # the bottom of the field (occluded when seated)
    DEEP_UPPER_MIN = 1      # STANDING needs this many deeply-occluded top cells
    LOWER_MIN = 3           # SITTING needs this many occupied bottom cells


# Default empty-room baseline: the labeled door-CLOSED capture
# (~/snapshots/labeled.jsonl), a clean unoccupied frame. The single 4000 is a
# dropout, handled by BASELINE_CAP. Operators capture their own with
# `python3 -m bathroom.record --baseline`; this is the fallback.
DEFAULT_BASELINE = [
    [1115, 1185, 1290, 1400, 1318, 1163, 1155, 1044],
    [1210, 1343, 1453, 1496, 1303, 1155, 1152, 1050],
    [1342, 1496, 1641, 1516, 1285, 1159, 1150, 1059],
    [1392, 1626, 1734, 1513, 1290, 1168, 1154, 1060],
    [1297, 1444, 1701, 1488, 1302, 1158, 1145, 1066],
    [1266, 1342, 4000, 1535, 1292, 1143, 1149, 1048],
    [1335, 973, 966, 1497, 1266, 1153, 1130, 1055],
    [1342, 967, 943, 1402, 1291, 1131, 1129, 1045],
]


def _eff_base(b):
    return b if b < NO_TARGET else Params.BASELINE_CAP


def median_window(frames):
    """Element-wise median over oriented frames -> denoised frame. Kills
    isolated 4000 dropouts and per-frame jitter. `frames` must be non-empty."""
    if not frames:
        raise ValueError("median_window requires at least one frame")
    return [[int(statistics.median(f[r][c] for f in frames))
             for c in range(N)] for r in range(N)]


def features(matrix, baseline=None):
    """Presence features used by the decision rule. Plain dict so it can be
    logged alongside the state for audit. DOOR column excluded throughout."""
    base = baseline or DEFAULT_BASELINE
    occ = []          # (r, c) occupied cells (closer than baseline by DELTA_MM)
    deep = set()      # of those, the deeply occluded ones (>= DEEP_DELTA)
    vnear = 0         # cells essentially at the sensor (< VNEAR_MM)
    near_cells = []   # cells in the softer near band (< NEAR_SOFT)
    for r in range(N):
        for c in range(N):
            if c == DOOR_COL:
                continue
            v = matrix[r][c]
            if v >= NO_TARGET:
                continue
            if v < Params.VNEAR_MM:
                vnear += 1
            if v < Params.NEAR_SOFT:
                near_cells.append((r, c))
            depression = _eff_base(base[r][c]) - v
            if depression >= Params.DELTA_MM:
                occ.append((r, c))
                if depression >= Params.DEEP_DELTA:
                    deep.add((r, c))

    occ_rows = sorted({r for (r, _c) in occ})
    occ_cols = sorted({c for (_r, c) in occ})
    upper = [rc for rc in occ if rc[0] in Params.UPPER_ROWS]
    lower = [rc for rc in occ if rc[0] in Params.LOWER_ROWS]
    deep_upper = [rc for rc in upper if rc in deep]
    near_cols = {c for (_r, c) in near_cells}
    occ_mm = [matrix[r][c] for (r, c) in occ]
    noise_cluster = (
        0 < len(occ) <= Params.NOISE_MAX
        and (max(occ_rows) - min(occ_rows) <= 1)
        and (max(occ_cols) - min(occ_cols) <= 1)
    ) if occ else False

    return {
        "occ_count": len(occ),
        "vnear_count": vnear,
        "near_count": len(near_cells),
        "near_cols": len(near_cols),
        "upper_count": len(upper),
        "deep_upper": len(deep_upper),
        "lower_count": len(lower),
        "toto_occluded": any(rc in occ for rc in ZONE_CELLS[Zone.TOTO]),
        "noise_cluster": noise_cluster,
        "row_spread": (max(occ_rows) - min(occ_rows) + 1) if occ_rows else 0,
        "col_spread": (max(occ_cols) - min(occ_cols) + 1) if occ_cols else 0,
        "min_mm": min(occ_mm) if occ_mm else None,
        "top_row": occ_rows[0] if occ_rows else None,
        "occ_cells": occ,
    }


def classify(matrix, baseline=None):
    """Map an oriented (denoised) frame to a (State, features) pair.

    Order matters: EMPTY/noise and WALKING are resolved before the posture
    split, because a body up close also occludes the upper field. A present-
    but-unclassifiable body returns UNKNOWN rather than a confident guess.
    """
    f = features(matrix, baseline)
    min_mm = f["min_mm"] if f["min_mm"] is not None else NO_TARGET

    if f["occ_count"] <= Params.EMPTY_MAX or f["noise_cluster"]:
        return State.EMPTY, f
    # WALKING: a body filling the near field in front of the sensor.
    if f["vnear_count"] >= Params.VNEAR_MIN:
        return State.WALKING, f
    if (f["near_cols"] >= Params.NEAR_COLS_MIN
            and f["near_count"] >= Params.NEAR_MIN
            and min_mm < Params.NEAR_SOFT + 100):
        return State.WALKING, f
    # below WALKING but too little structure to call a posture
    if f["occ_count"] < Params.MIN_BODY:
        return State.UNKNOWN, f
    # STANDING: upright body occludes the top rows DEEPLY.
    if f["deep_upper"] >= Params.DEEP_UPPER_MIN:
        return State.STANDING_AT_TOILET, f
    # SITTING: low body, top clear (no deep top occlusion), bottom occupied.
    if f["lower_count"] >= Params.LOWER_MIN and f["deep_upper"] == 0:
        return State.SITTING_AT_TOILET, f
    return State.UNKNOWN, f


def classify_window(frames, baseline=None):
    """Denoise a window of frames, then classify. `frames` must be non-empty."""
    if not frames:
        raise ValueError("classify_window requires at least one frame")
    return classify(median_window(frames), baseline)
