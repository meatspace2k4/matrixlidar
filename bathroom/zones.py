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


def label(r, c):
    """Human cell label, e.g. (6,1) -> 'B7'."""
    return "%s%d" % (chr(ord("A") + c), r + 1)


def zone_of(r, c):
    return CELL_ZONE.get((r, c), Zone.FIELD)
