"""Done with the toilet: standing at it, then something that is not sitting.

Sitting is using it. Standing at it is still using it: that is the stand to
wipe, the stand to flush, the stand before sitting back down. Leaving STANDING
for anything that is not SITTING is being done with it.

Not an exit. Not entering WALKING or UNKNOWN from a sit. Not a distance. The
transition out of STANDING_AT_TOILET into anything that is not
SITTING_AT_TOILET.
"""
from .zones import State

FADE = "FADE"


class DoneWith:
    """Feed it the committed posture. It says when the toilet is done with."""

    __slots__ = ("_prev", "done")

    def __init__(self):
        self._prev = None
        self.done = False

    def update(self, state):
        prev, self._prev = self._prev, state
        if state == State.SITTING_AT_TOILET:
            self.done = False           # sat back down: using it again
            return None
        if prev == State.STANDING_AT_TOILET and state != State.STANDING_AT_TOILET:
            if not self.done:
                self.done = True
                return FADE
        return None
