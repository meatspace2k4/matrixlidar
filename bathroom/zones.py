"""Zone + ground-truth state definitions for the bathroom sensor.

DISPLAY orientation (matches the calibrated on-screen heatmap and the Pi's
`bathroom/zones.py`):
  columns A-H  ->  indices 0-7, left -> right
  rows    1-8  ->  indices 0-7, top  -> bottom (row 1 = top)

After `stream.orient()` a frame is indexed matrix[row][col] with row 0 = top,
col 0 = column A.

Named regions (operator-supplied geometry, small bathroom):
  TOTO       = columns B,C x rows 7,8   (the toilet; bottom-left green block)
  DOOR       = entire column H (the far-right range). Closed -> door panel
               ~1000-1160mm; open -> sensor sees into the hallway, ~1400 up to
               4000/no-target. Column G is the jamb/handle, NOT the door.
  LEFT_EDGE  = column A (sink/tub/shower out of frame to the left; a body
               entering from that side first appears here)

This file is kept byte-for-byte in sync with the Pi's bathroom/zones.py up to
the State enum, which here carries the operator's posture/position ground-truth
vocabulary (the thing the classifier reports), not the deferred temporal phases.
"""
from enum import IntEnum

N = 8
NO_TARGET = 4000  # sensor value meaning "nothing in range" (mm)

# the calibrated orientation chain every frame is stored in; clips/baselines
# record this string so a loader can refuse data from a differently-oriented setup
ORIENTATION = "display: rotate_cw -> flip_x -> flip_y; row0=top, col0=column A"


class Zone(IntEnum):
    NONE = 0       # no blob / empty
    TOTO = 1       # toilet
    DOOR = 2       # door column
    LEFT_EDGE = 3  # leftmost column, sink/tub/shower side
    FIELD = 4      # anywhere else in view


class State(IntEnum):
    """Ground-truth presence states, exactly the postures the operator labeled.

    These are what the classifier reports and what a recorded clip is tagged
    with. They are physical positions, not the deferred temporal phase machine
    (Entered/Idle/Exit): a transformer is trained on these stable states and
    can learn the transitions itself.

      EMPTY               no body in the field (door open or closed)
      WALKING             body crossing close in front of the sensor; the
                          toilet is obscured (seen only through the legs)
      STANDING_AT_TOILET  body upright between the toilet and the door; it
                          occludes a tall vertical band of the field
      SITTING_AT_TOILET   body low/seated on the toilet; the upper field is
                          clear and the near cells sit in the bottom rows
      UNKNOWN             ambiguous / transitional; never forced to a guess so
                          that auto-labels stay trustworthy as ground truth
    """
    UNKNOWN = -1
    EMPTY = 0
    WALKING = 1
    STANDING_AT_TOILET = 2
    SITTING_AT_TOILET = 3


def _col(letter):
    return ord(letter) - ord("A")


def _cells(cols, rows):
    """Build (row_idx, col_idx) list from 1-based row numbers / letter cols."""
    return [(r - 1, _col(c)) for r in rows for c in cols]


ALL_ROWS = [1, 2, 3, 4, 5, 6, 7, 8]

ZONE_CELLS = {
    Zone.TOTO: _cells(["B", "C"], [7, 8]),
    Zone.DOOR: _cells(["H"], ALL_ROWS),
    Zone.LEFT_EDGE: _cells(["A"], ALL_ROWS),
}

DOOR_COL = _col("H")  # presence detection ignores this column (door swing != body)

# reverse lookup for the structural zones (FIELD is the default elsewhere)
CELL_ZONE = {}
for _z, _cs in ZONE_CELLS.items():
    for _rc in _cs:
        CELL_ZONE[_rc] = _z


# operator label strings (verbatim, from --label) -> ground-truth State.
# Maps the casual words a person types while capturing to the enum.
_LABEL_ALIASES = {
    "empty": State.EMPTY, "idle": State.EMPTY, "none": State.EMPTY, "away": State.EMPTY,
    "walking": State.WALKING, "walk": State.WALKING, "passing": State.WALKING,
    "infront": State.WALKING, "in-front": State.WALKING, "crossing": State.WALKING,
    "standing": State.STANDING_AT_TOILET, "stand": State.STANDING_AT_TOILET,
    "standing-at-toilet": State.STANDING_AT_TOILET, "at-toilet": State.STANDING_AT_TOILET,
    "sitting": State.SITTING_AT_TOILET, "sit": State.SITTING_AT_TOILET, "seated": State.SITTING_AT_TOILET,
    "sitting-at-toilet": State.SITTING_AT_TOILET, "toilet": State.SITTING_AT_TOILET,
    "unknown": State.UNKNOWN, "ambiguous": State.UNKNOWN,
}


def from_label(s):
    """Normalize an operator label string to a State, or None if unrecognized.

    Accepts the enum names too (e.g. 'SITTING_AT_TOILET'). Ground-truth labels
    are noisy free text; this is the single place that decides what they mean."""
    if s is None:
        return None
    if isinstance(s, State):
        return s
    key = str(s).strip().lower().replace(" ", "-")
    if key in _LABEL_ALIASES:
        return _LABEL_ALIASES[key]
    for st in State:
        if st.name.lower() == key.replace("-", "_"):
            return st
    return None


def label(r, c):
    """Human cell label, e.g. (6,1) -> 'B7'."""
    return "%s%d" % (chr(ord("A") + c), r + 1)


def zone_of(r, c):
    return CELL_ZONE.get((r, c), Zone.FIELD)
