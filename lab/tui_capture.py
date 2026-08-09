#!/usr/bin/env python3
"""Run the REAL heatmap client against the dev server, stdout captured.

No reimplementation of the TUI: this imports heatmap_client verbatim and
only repoints where it connects (the same monkeypatch replay_demo.py uses on
the Pi). Its ANSI repaints go to stdout, which the dev server reads and
republishes to the browser, so what the 3D page renders is byte-for-byte
what the terminal renders.

  python3 lab/tui_capture.py <tcp_port>
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from bathroom import stream                                     # noqa: E402

port = int(sys.argv[1])
_real = stream.connect_with_retry
stream.connect_with_retry = (
    lambda host=None, p=None, on_wait=None: _real("127.0.0.1", port, on_wait))

import heatmap_client                                           # noqa: E402

raise SystemExit(heatmap_client.main())
