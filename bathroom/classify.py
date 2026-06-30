"""Ground-truth state classifier for the bathroom matrix lidar.

This is the gap the existing code deferred: snapshot.py reports
"Status: DEFERRED" and detect.py stops at zone values. Here we turn an
oriented 8x8 frame into one of the operator's labeled states (see
zones.State) deterministically and explainably, so it can serve as ground
truth for recorded clips.

How it decides
--------------
Distances are in mm; smaller = closer to the sensor. The sensor sits between
the user and the toilet, so a body reads CLOSER than the empty-room baseline
at whatever it occludes. We measure occlusion as a per-cell *depression*
below a captured empty baseline rather than absolute thresholds, because the
empty field is a non-uniform "bowl" (center columns naturally read ~1600mm,
edges ~1100mm). The DOOR column (H) is excluded from presence entirely: it
swings 1000->4000mm on its own and says nothing about a body.

Discriminators, derived from the four labeled captures:
  EMPTY               almost no occluded cells.
  WALKING             a body right at the sensor: many cells very near
                      (< VNEAR_MM, ~350-460mm in the sample), large area.
  STANDING_AT_TOILET  body upright -> it occludes the TOP rows of the field
                      (a tall vertical band, near values reach rows 1-2).
  SITTING_AT_TOILET   body low -> the top rows stay at baseline while the
                      occluded cells cluster in the bottom rows / toilet.

Tuned constants live in `Params`; recalibrate from labeled clips with
`tune.py` if the geometry changes. The classifier is pure (no I/O) so the Pi
snapshot tool and the live client share identical logic.
"""
import statistics

from .zones import N, NO_TARGET, DOOR_COL, ZONE_CELLS, Zone, State


class Params:
    # a cell counts as occupied if it reads this many mm CLOSER than baseline
    DELTA_MM = 300
    # baseline cells reading no-target are unreliable; cap them here for delta
    BASELINE_CAP = 1500
    # "very near" => body essentially at the sensor lens
    VNEAR_MM = 650
    # WALKING needs at least this many very-near cells (a whole body up close)
    VNEAR_MIN = 8
    # EMPTY if at most this many cells are occupied (tolerates sensor specks)
    EMPTY_MAX = 2
    # STANDING: this many occupied cells in the top rows (upright body occludes
    # the upper field)
    UPPER_ROWS = (0, 1)
    UPPER_MIN = 2
    # SITTING: occupied cells clustered low when the top is clear
    LOWER_ROWS = (5, 6, 7)
    LOWER_MIN = 3


# Default empty-room baseline: the labeled door-CLOSED capture
# (~/snapshots/labeled.jsonl), a clean unoccupied frame. The single 4000 is a
# dropout and is handled by BASELINE_CAP. Operators should capture their own
# with `python3 -m bathroom.record --baseline`; this is the fallback.
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
    """Element-wise median over a list of oriented frames -> denoised frame.

    Kills isolated 4000 dropouts and per-frame jitter the same way the
    snapshot tool does. `frames` must be non-empty.
    """
    return [[int(statistics.median(f[r][c] for f in frames))
             for c in range(N)] for r in range(N)]


def features(matrix, baseline=None):
    """Compute the presence features used by the decision rule.

    Returns a plain dict so it can be logged alongside the state for audit.
    The DOOR column is excluded from every presence measure.
    """
    base = baseline or DEFAULT_BASELINE
    occ = []          # occupied (occluded) cells, excluding the door column
    vnear = 0         # cells essentially at the sensor
    for r in range(N):
        for c in range(N):
            if c == DOOR_COL:
                continue
            v = matrix[r][c]
            if v >= NO_TARGET:
                continue
            if v < Params.VNEAR_MM:
                vnear += 1
            if v < _eff_base(base[r][c]) - Params.DELTA_MM:
                occ.append((r, c))

    upper = sum(1 for (r, _c) in occ if r in Params.UPPER_ROWS)
    lower = sum(1 for (r, _c) in occ if r in Params.LOWER_ROWS)
    toto_occ = any((r, c) in occ for (r, c) in ZONE_CELLS[Zone.TOTO])
    occ_mm = [matrix[r][c] for (r, c) in occ]

    return {
        "occ_count": len(occ),
        "vnear_count": vnear,
        "upper_count": upper,
        "lower_count": lower,
        "toto_occluded": toto_occ,
        "min_mm": min(occ_mm) if occ_mm else None,
        "top_row": min((r for (r, _c) in occ), default=None),
        "occ_cells": occ,
    }


def classify(matrix, baseline=None):
    """Map an oriented (already denoised) frame to a (State, features) pair.

    Order matters: EMPTY and WALKING are checked before the posture split,
    because a body up close also occludes the upper field.
    """
    f = features(matrix, baseline)

    if f["occ_count"] <= Params.EMPTY_MAX:
        return State.EMPTY, f
    if f["vnear_count"] >= Params.VNEAR_MIN:
        return State.WALKING, f
    if f["upper_count"] >= Params.UPPER_MIN:
        return State.STANDING_AT_TOILET, f
    if f["toto_occluded"] or f["lower_count"] >= Params.LOWER_MIN:
        return State.SITTING_AT_TOILET, f
    return State.UNKNOWN, f


def classify_window(frames, baseline=None):
    """Denoise a window of frames, then classify. `frames` must be non-empty."""
    return classify(median_window(frames), baseline)
