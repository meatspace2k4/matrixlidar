"""Deterministic zone-derived values (door open%, per-zone summaries).

Mirrors the Pi's bathroom/detect.py so the client reports identical numbers.
These are explicit, requested mappings of zone distances, not presence
heuristics; presence/state lives in classify.py.
"""
import statistics

from .zones import Zone, ZONE_CELLS

# Door swing reads on column H from 1000mm (shut) to 1400mm. Past 1400 the door
# has swung clear and the sensor sees through (up to 4000/no-target), still
# fully open -> the >=1400 clamp maps it to 100%.
DOOR_CLOSED_MM = 1090
DOOR_OPEN_MM = 1400


def zone_values(matrix, zone):
    """Per-zone distance summary: median / min / max over the zone's cells."""
    vals = [matrix[r][c] for (r, c) in ZONE_CELLS[zone]]
    return {
        "median": int(statistics.median(vals)),
        "min": min(vals),
        "max": max(vals),
    }


def door_open_pct(matrix):
    """Column H median mapped 1000mm=0% .. 1400mm=100%; >=1400 (incl. 4000
    no-target, door fully swung clear) clamps to 100%."""
    d = zone_values(matrix, Zone.DOOR)["median"]
    if d >= DOOR_OPEN_MM:
        return 100.0
    if d <= DOOR_CLOSED_MM:
        return 0.0
    span = float(DOOR_OPEN_MM - DOOR_CLOSED_MM)
    return round((d - DOOR_CLOSED_MM) / span * 100.0, 1)
