#!/usr/bin/env python3
"""Harvest the Pi's RAM transition ring into fixtures/misdetections/.

The ring holds raw-frame snapshots of real (mis)detected sessions and dies on
a server restart; this saves each session as one self-describing JSON file so
they can drive regression tests. Run from the repo root:

  python3 scripts/harvest_ring.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from replay import fetch  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "fixtures", "misdetections")


def main():
    dump = fetch(query=b"DUMP")
    sessions = dump["sessions"]
    if not sessions:
        print("RAM ring is empty; nothing to harvest")
        return 1
    os.makedirs(OUT, exist_ok=True)
    for s in sessions:
        path = os.path.join(OUT, "session_%03d.json" % s["id"])
        with open(path, "w") as f:
            json.dump(s, f)
        n_frames = sum(1 for tr in s["transitions"] if tr.get("frame"))
        print("saved %s  transitions=%d  with_frames=%d"
              % (path, len(s["transitions"]), n_frames))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
