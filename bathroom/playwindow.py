"""How long the track plays: a short window by default, held open by a wave.

The sit starts the track. From there the wave is a HOLD, not a stop:

  no wave            the track fades out after AUTO_FADE_S of playback. Most
                     visits want a few seconds of music, not the whole song.
  wave during play   cancel that timer and keep playing, indefinitely. The
                     wave says "I am still here and I want this".
  wave while held    fade out. The second wave is the stop, and it is the
                     only stop this machine has.

Pure and frame-driven like SitLatch and DoneWith: update() takes the clock
and whether a wave landed this frame, and answers with an action. No I/O, no
audio, no knowledge of the player.

WHY A WAVE IS NOT A STOP ANY MORE

It was, and the inverse is what the room actually needs: the common case is
"play a little and stop on its own", and the rare case is "keep going". A
gesture is the right control for the rare case, and a timer for the common
one. The wave detector itself is unchanged (bathroom/wave4d.py); only what
the daemon does with a WAVE moved.

The two waves cannot be one gesture seen twice: wave4d already refuses to
fire again inside its refractory window (2s), and this machine additionally
ignores a wave in the same frame it started playing.
"""

AUTO_FADE_S = 30.0   # unheld playback fades after this much. Measured
                     # against nothing: it is a taste setting, and the one
                     # number here that is meant to be argued with. The
                     # matching CLI flag is --auto-fade.
FADE_S = 2.0         # the fade itself, matching the walk-away fade so the
                     # track always ends the same way it always has

HOLD = "HOLD"        # a wave landed mid-play: the timer is off, play on
FADE_OUT = "FADE_OUT"


class PlayWindow:
    """Feed it (t, waved) once per frame while the daemon runs."""

    __slots__ = ("auto_fade_s", "_started", "_held", "_playing")

    def __init__(self, auto_fade_s=AUTO_FADE_S):
        self.auto_fade_s = auto_fade_s
        self._started = None   # t the current playback began
        self._held = False     # a wave has held this playback open
        self._playing = False

    @property
    def playing(self):
        return self._playing

    @property
    def held(self):
        return self._held

    def start(self, t):
        """The track just started. Arms the auto-fade timer."""
        self._started = t
        self._held = False
        self._playing = True

    def ended(self):
        """Playback stopped for a reason this machine did not decide: the
        walk-away fade, the visit ending, a stream loss. Disarms everything,
        so a later wave cannot fade a track that is not playing."""
        self._started = None
        self._held = False
        self._playing = False

    def update(self, t, waved):
        """One frame. Returns HOLD, FADE_OUT, or None."""
        if not self._playing:
            return None
        if waved and t > self._started:
            if self._held:
                self._playing = False
                return FADE_OUT
            self._held = True
            return HOLD
        if not self._held and t - self._started >= self.auto_fade_s:
            self._playing = False
            return FADE_OUT
        return None
