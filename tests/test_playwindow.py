#!/usr/bin/env python3
"""PlayWindow regression: the track fades on its own, a wave holds it open,
the second wave ends it. Driven at 10 Hz.

  python3 tests/test_playwindow.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bathroom.playwindow import (AUTO_FADE_S, FADE_OUT, HOLD,      # noqa: E402
                                 PlayWindow)

HZ = 10.0
FAILS = []


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


def drive(w, t, seconds, wave_at=()):
    """Run `seconds` of frames; wave_at is a list of times to wave at (each
    matched on the first frame at or after it). Returns (t, [(t, action)])."""
    out = []
    pending = list(wave_at)
    end = t + seconds
    while t < end:
        waved = bool(pending) and t >= pending[0]
        if waved:
            pending.pop(0)
        a = w.update(t, waved)
        if a is not None:
            out.append((round(t, 1), a))
        t += 1.0 / HZ
    return t, out


def main():
    n = 0

    # 1. no wave: the track fades on its own at the window
    w = PlayWindow(auto_fade_s=25.0)
    w.start(0.0)
    t, ev = drive(w, 0.0, 30.0)
    check([a for _t, a in ev] == [FADE_OUT], "unheld playback fades once")
    check(abs(ev[0][0] - 25.0) < 0.2, "and it fades AT the window (25s)")
    check(not w.playing, "the window knows playback is over")
    n += 3

    # 2. a wave during playback holds it open, indefinitely
    w = PlayWindow(auto_fade_s=25.0)
    w.start(0.0)
    t, ev = drive(w, 0.0, 120.0, wave_at=[10.0])
    check([a for _t, a in ev] == [HOLD], "a wave holds; no auto-fade after it")
    check(w.playing and w.held, "still playing, and marked held")
    n += 2

    # 3. the second wave ends it
    w = PlayWindow(auto_fade_s=25.0)
    w.start(0.0)
    t, ev = drive(w, 0.0, 60.0, wave_at=[5.0, 40.0])
    check([a for _t, a in ev] == [HOLD, FADE_OUT],
          "second wave fades out")
    check(abs(ev[1][0] - 40.0) < 0.2, "at the second wave, not on a timer")
    check(not w.playing, "and playback is over")
    n += 3

    # 4. a wave landing just before the window still cancels it
    w = PlayWindow(auto_fade_s=25.0)
    w.start(0.0)
    t, ev = drive(w, 0.0, 40.0, wave_at=[24.8])
    check([a for _t, a in ev] == [HOLD],
          "a wave at 24.8s beats the 25s timer")
    n += 1

    # 5. a wave in the very frame playback starts is not a hold: it is the
    #    tail of whatever gesture started things, not an instruction
    w = PlayWindow(auto_fade_s=25.0)
    w.start(3.0)
    check(w.update(3.0, True) is None, "a wave in the starting frame is ignored")
    check(not w.held, "and does not hold")
    n += 2

    # 6. nothing happens when nothing is playing
    w = PlayWindow(auto_fade_s=25.0)
    _, ev = drive(w, 0.0, 60.0, wave_at=[5.0, 20.0])
    check(ev == [], "waves with no playback do nothing")
    n += 1

    # 7. ended() disarms: the walk-away fade stopped the track, so a later
    #    wave must not fade a track that is not playing
    w = PlayWindow(auto_fade_s=25.0)
    w.start(0.0)
    _, ev1 = drive(w, 0.0, 5.0)
    w.ended()
    _, ev2 = drive(w, 5.0, 60.0, wave_at=[10.0, 30.0])
    check(ev1 == [] and ev2 == [], "after ended(), nothing fires")
    n += 1

    # 8. a fresh play after a held one starts unheld again
    w = PlayWindow(auto_fade_s=25.0)
    w.start(0.0)
    t, _ = drive(w, 0.0, 30.0, wave_at=[5.0])
    check(w.held, "first playback is held")
    w.ended()
    w.start(t)
    t, ev = drive(w, t, 30.0)
    check([a for _t, a in ev] == [FADE_OUT],
          "the next playback is a fresh window, not still held")
    n += 2

    check(AUTO_FADE_S == 30.0, "the shipped window is 30s")
    n += 1

    print("\n%d checks, %d failures" % (n, len(FAILS)))
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
