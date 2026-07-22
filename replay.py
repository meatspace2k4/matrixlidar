#!/usr/bin/env python3
"""Fetch the Pi's in-memory transition archive and replay it.

The server keeps raw-frame snapshots of committed transitions (see
bathroom/archive.py), grouped into sessions. This rebuilds the timeline by
time and replays the snapshots on the client, rendered like the live heatmap.

  python3 replay.py                  # list sessions
  python3 replay.py --session 3      # replay one session's snapshots
  python3 replay.py --session 3 --delay 0.8
"""
import argparse
import json
import sys
import time

from bathroom import stream
from bathroom.zones import N, NO_TARGET
from heatmap_client import CELL_W, color_for


def fetch(host=stream.HOST, port=stream.PORT):
    sock = stream.connect_with_retry(host, port)
    if sock is None:
        raise ConnectionError("could not reach %s:%d" % (host, port))
    try:
        sock.sendall(b"DUMP\n")
        with sock.makefile("r") as f:
            for line in f:
                if line.startswith("ARCHIVE "):
                    return json.loads(line[8:])
    finally:
        try:
            sock.close()
        except OSError:
            pass
    raise ConnectionError("stream ended before the ARCHIVE reply")


def _grid(frame_csv):
    flat = [int(x) for x in frame_csv.split(",")]
    m = stream.orient([flat[i * N:(i + 1) * N] for i in range(N)])
    rows = []
    for r in range(N):
        cells = []
        for c in range(N):
            v = m[r][c]
            br, bg, bb = color_for(v)
            txt = "" if v >= NO_TARGET else str((v + 5) // 10)  # cm
            cells.append("\x1b[48;2;%d;%d;%dm\x1b[97m%s\x1b[0m"
                         % (br, bg, bb, txt.center(CELL_W)))
        rows.append("  " + "".join(cells))
    return "\n".join(rows)


def _label(tr):
    m = tr["mark"]
    when = time.strftime("%H:%M:%S", time.localtime(tr["ts"]))
    ev = " [%s]" % m["event"] if m.get("event") else ""
    return "%s  t=%7.2fs  %-7s %s -> %s%s" % (
        when, m["t"], m["kind"], m["from"], m["to"], ev)


def main():
    ap = argparse.ArgumentParser(description="Replay the Pi's transition archive")
    ap.add_argument("--session", type=int, help="replay this session id")
    ap.add_argument("--delay", type=float, default=0.0,
                    help="pause between snapshots (seconds)")
    a = ap.parse_args()
    dump = fetch()
    sessions = dump["sessions"]
    if not sessions:
        print("archive is empty")
        return 0

    if a.session is None:
        for s in sessions:
            trs = s["transitions"]
            t0 = time.strftime("%H:%M:%S", time.localtime(trs[0]["ts"]))
            t1 = time.strftime("%H:%M:%S", time.localtime(trs[-1]["ts"]))
            print("session %-3d %s - %s  %d transitions" % (s["id"], t0, t1, len(trs)))
        return 0

    for s in sessions:
        if s["id"] != a.session:
            continue
        trs = sorted(s["transitions"], key=lambda tr: tr["ts"])
        print("session %d  (%d transitions)\n" % (s["id"], len(trs)))
        for tr in trs:
            print(_label(tr))
            if tr["frame"]:
                print(_grid(tr["frame"]))
            print()
            if a.delay:
                time.sleep(a.delay)
        return 0
    print("no session %d in the archive" % a.session)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
