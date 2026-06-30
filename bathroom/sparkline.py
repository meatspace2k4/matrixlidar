"""Terminal sparklines for the state engine.

Two strips, both fed from a rolling history:
  - `sparkline(values)`  : occupancy/proximity over time as block glyphs
  - `state_strip(states)`: one colored cell per frame, by committed state

Used live (engine CLI, heatmap client) and for a static render of a recorded
clip so a labeled capture can be eyeballed before training on it.
"""
from .zones import State

BLOCKS = " ▁▂▃▄▅▆▇█"

# 24-bit colors per state (match the heatmap palette's intent)
STATE_RGB = {
    State.EMPTY: (60, 60, 80),
    State.WALKING: (240, 120, 20),
    State.STANDING_AT_TOILET: (240, 220, 40),
    State.SITTING_AT_TOILET: (40, 190, 60),
    State.UNKNOWN: (120, 120, 120),
}
STATE_ABBR = {
    State.EMPTY: "·",
    State.WALKING: "W",
    State.STANDING_AT_TOILET: "S",
    State.SITTING_AT_TOILET: "T",
    State.UNKNOWN: "?",
}


def sparkline(values, lo=None, hi=None):
    """Block-glyph sparkline. Scales to [lo, hi] (auto from data if None)."""
    vals = list(values)
    if not vals:
        return ""
    lo = min(vals) if lo is None else lo
    hi = max(vals) if hi is None else hi
    span = (hi - lo) or 1
    out = []
    for v in vals:
        f = (v - lo) / span
        f = 0.0 if f < 0 else 1.0 if f > 1 else f
        out.append(BLOCKS[int(f * (len(BLOCKS) - 1))])
    return "".join(out)


def state_strip(states, color=True):
    """One cell per state. Colored blocks when color=True, else abbreviations."""
    out = []
    for s in states:
        s = s if isinstance(s, State) else State[s]
        if color:
            r, g, b = STATE_RGB.get(s, (120, 120, 120))
            out.append("\x1b[48;2;%d;%d;%dm \x1b[0m" % (r, g, b))
        else:
            out.append(STATE_ABBR.get(s, "?"))
    return "".join(out)


def legend(color=True):
    parts = []
    for s in (State.EMPTY, State.WALKING, State.STANDING_AT_TOILET,
              State.SITTING_AT_TOILET, State.UNKNOWN):
        if color:
            r, g, b = STATE_RGB[s]
            parts.append("\x1b[48;2;%d;%d;%dm \x1b[0m%s" % (r, g, b, s.name))
        else:
            parts.append("%s=%s" % (STATE_ABBR[s], s.name))
    return "  ".join(parts)


def render_clip(per_frame_states, signal, width=72):
    """Static two-line render of a recorded clip: signal sparkline + state strip,
    downsampled to `width` columns. Returns a printable string."""
    def bucket(seq, agg):
        n = len(seq)
        if n <= width:
            return list(seq)
        step = n / float(width)
        return [agg(seq[int(i * step):max(int(i * step) + 1, int((i + 1) * step))])
                for i in range(width)]

    sig = bucket(signal, lambda c: sum(c) / len(c) if c else 0)
    # for the state strip, take the most common state in each bucket
    def mode(c):
        if not c:
            return State.EMPTY
        t = {}
        for s in c:
            t[s] = t.get(s, 0) + 1
        return max(t, key=t.get)
    sts = bucket(list(per_frame_states), mode)
    return "  %s\n  %s" % (sparkline(sig), state_strip(sts))
