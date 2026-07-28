"""Shim: the scene model was promoted to bathroom.scene; the lab validator
imports through here unchanged."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bathroom.scene import (  # noqa: E402,F401
    Params, SceneAutomaton, SceneReading, _Debounced, door_panel_state,
    passage_band, read_frame,
)
