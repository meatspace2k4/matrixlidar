#!/usr/bin/env python3
"""Harvest the Pi's RAM transition ring into fixtures/misdetections/.

The ring holds raw-frame snapshots of real (mis)detected sessions and dies on
a server restart; this saves each session as one self-describing JSON file so
they can drive regression tests. Run from the repo root:

  python3 scripts/harvest_ring.py

Filenames carry the session's START TIMESTAMP, not just its id: the Pi's
session counter restarts whenever the server restarts, so id alone collides
and silently overwrites earlier captures (it clobbered the original session
6 once). Existing files are never overwritten.
"""
import json
import os
import sys
import time

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
        trs = s.get("transitions") or []
        if not trs:
            continue
        start = min(t["ts"] for t in trs)
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(start))
        path = os.path.join(OUT, "session_%s_id%03d.json" % (stamp, s["id"]))
        if os.path.exists(path):
            print("exists, not overwriting: %s" % path)
            continue
        with open(path, "w") as f:
            json.dump(s, f)
        n_frames = sum(1 for tr in trs if tr.get("frame"))
        print("saved %s  transitions=%d  with_frames=%d"
              % (path, len(trs), n_frames))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
