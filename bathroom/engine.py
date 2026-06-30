"""End-to-end temporal state engine.

classify.py answers "what does THIS frame look like"; the engine answers
"what state is the room actually in, over time". It debounces the per-frame
classifier with hysteresis so a single flicker frame can't flip the committed
state, tracks dwell time, and emits a timeline of stable segments. This is the
temporal machine the original package deferred.

  raw stream -> orient -> classify_window -> StateEngine -> committed state
                                                          + transition events
                                                          + timeline

A segment is one stable stretch of a single state: (state, t_start, t_end,
n_frames). The timeline is the closed-segment history; `current` is the open
one. Segments are exactly the spans a sequence model is trained to reproduce,
so the engine's timeline doubles as auto-labels to check the model against.
"""
from collections import deque

from .classify import classify_window
from .zones import State


class Segment:
    __slots__ = ("state", "t_start", "t_end", "n_frames")

    def __init__(self, state, t_start):
        self.state = state
        self.t_start = t_start
        self.t_end = t_start
        self.n_frames = 0

    @property
    def duration(self):
        return max(0.0, self.t_end - self.t_start)

    def as_dict(self):
        return {
            "state": self.state.name,
            "t_start": round(self.t_start, 3),
            "t_end": round(self.t_end, 3),
            "duration": round(self.duration, 3),
            "n_frames": self.n_frames,
        }


class Update:
    """One engine step's result (what live views and recorders consume)."""
    __slots__ = ("committed", "instant", "changed", "event", "features", "t")

    def __init__(self, committed, instant, changed, event, features, t):
        self.committed = committed      # the debounced, stable state
        self.instant = instant          # this frame's raw classifier state
        self.changed = changed          # True on the frame a transition commits
        self.event = event              # the Segment that just closed, or None
        self.features = features        # classifier features (drives sparklines)
        self.t = t


class StateEngine:
    """Debounced state machine over the per-frame classifier.

    window       frames the per-frame classifier denoises over (matches live).
    commit_after a candidate state must hold this many consecutive frames
                 before the committed state flips (hysteresis against flicker).
    """

    def __init__(self, baseline=None, window=5, commit_after=4, start=State.EMPTY):
        self.baseline = baseline
        self.window = window
        self.commit_after = commit_after
        self._win = deque(maxlen=window)
        self.committed = start
        self.current = None              # open Segment
        self.timeline = []               # closed Segments
        self._cand = None
        self._cand_n = 0

    def update(self, matrix, t):
        self._win.append(matrix)
        instant, feats = classify_window(list(self._win), self.baseline)

        if self.current is None:
            # Bootstrap: adopt the first observation as the initial state. There
            # is no prior to debounce against, so committing immediately is
            # correct AND it means a clip shorter than commit_after frames (e.g.
            # a single-frame labeled snapshot) still reports its true state
            # instead of being stuck at the EMPTY start assumption. Hysteresis
            # below guards only subsequent transitions.
            self.committed = instant
            self.current = Segment(instant, t)

        changed = False
        event = None

        if instant == self.committed:
            self._cand, self._cand_n = None, 0
        else:
            if instant == self._cand:
                self._cand_n += 1
            else:
                self._cand, self._cand_n = instant, 1
            if self._cand_n >= self.commit_after:
                # commit the transition: close the open segment, open a new one
                self.current.t_end = t
                self.timeline.append(self.current)
                event = self.current
                self.committed = instant
                self.current = Segment(instant, t)
                self._cand, self._cand_n = None, 0
                changed = True

        self.current.t_end = t
        self.current.n_frames += 1
        return Update(self.committed, instant, changed, event, feats, t)

    def close(self, t=None):
        """Finalize the open segment (call at end of stream/clip)."""
        if self.current is not None and self.current.n_frames:
            if t is not None:
                self.current.t_end = t
            self.timeline.append(self.current)
            self.current = None
        return self.timeline

    def segments(self):
        """All segments incl. the still-open one, as dicts (read-only view)."""
        segs = [s.as_dict() for s in self.timeline]
        if self.current is not None and self.current.n_frames:
            segs.append(self.current.as_dict())
        return segs

    def dominant_state(self):
        """The state the engine spent the most frames in (clip-level label)."""
        tally = {}
        for s in self.segments():
            tally[s["state"]] = tally.get(s["state"], 0) + s["n_frames"]
        if not tally:
            return State.EMPTY
        name = max(tally, key=tally.get)
        return State[name]


def run_clip(frames, times=None, baseline=None, window=5, commit_after=4):
    """Run the engine over a recorded clip (list of oriented matrices).

    Returns (per_frame_committed_states, timeline_dicts). `times` defaults to a
    synthetic 15 Hz clock if not supplied.
    """
    eng = StateEngine(baseline=baseline, window=window, commit_after=commit_after)
    per_frame = []
    occ = []
    for i, m in enumerate(frames):
        t = times[i] if times else i / 15.0
        u = eng.update(m, t)
        per_frame.append(u.committed)
        occ.append(u.features["occ_count"])
    eng.close()
    return per_frame, eng.segments(), occ


def _main():
    """`python3 -m bathroom.engine [clip.json]` — render a recorded clip's
    state timeline + sparkline, or run the live readout with no argument."""
    import sys
    if len(sys.argv) < 2:
        from . import live
        return live.main()
    from . import config, ingest, sparkline
    sample = ingest._load_file(sys.argv[1])
    baseline = config.load_baseline()
    per_frame, segs, occ = run_clip(sample.frames, baseline=baseline)
    print("clip %s  label=%s  frames=%d" % (sample.id, sample.label, sample.n))
    print(sparkline.render_clip(per_frame, occ))
    print()
    print("timeline:")
    for s in segs:
        print("  %-18s %6.2fs  %4d frames" % (s["state"], s["duration"], s["n_frames"]))
    print()
    print(sparkline.legend(color=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
