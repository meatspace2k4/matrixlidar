#!/usr/bin/env python3
"""Fetch the Pi's transition archive and replay it.

Two sources, different content:
  DUMP (default) the live RAM ring: raw-frame snapshots per transition, lost
                 on a server restart. Replay renders the frame as a grid.
  LOG (--log)    the persisted event log: just {ts, kind, from, to, event,
                 boundary} per transition, no sensor values, survives a
                 crash/power cycle. Replay prints the event line only.

  python3 replay.py                  # list sessions in the live RAM ring
  python3 replay.py --log            # list sessions from the persisted log
  python3 replay.py --log --session 3
  python3 replay.py --session 3 --delay 0.8
"""
import argparse
import json
import sys
import time

from bathroom import stream
from bathroom.zones import N, NO_TARGET
from heatmap_client import CELL_W, color_for


def fetch(host=stream.HOST, port=stream.PORT, query=b"DUMP"):
    sock = stream.connect_with_retry(host, port)
    if sock is None:
        raise ConnectionError("could not reach %s:%d" % (host, port))
    try:
        sock.sendall(query + b"\n")
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


def _normalize(session):
    """Both archive shapes -> a uniform list of transition dicts:
    {ts, kind, from, to, event, boundary, frame}. DUMP (RAM ring) sessions
    carry a raw-frame snapshot per transition; LOG (persisted) sessions are
    events only, so frame is always None there."""
    if "transitions" in session:  # DUMP: RAM ring, has frame snapshots
        out = []
        for tr in session["transitions"]:
            m = tr["mark"]
            out.append({"ts": tr["ts"], "t": m.get("t"), "kind": m.get("kind"),
                       "from": m.get("from"), "to": m.get("to"),
                       "event": m.get("event"), "boundary": m.get("boundary"),
                       "frame": tr.get("frame")})
        return out
    return [dict(e, frame=None) for e in session["events"]]  # LOG: events only


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
    when = time.strftime("%H:%M:%S", time.localtime(tr["ts"]))
    ev = " [%s]" % tr["event"] if tr.get("event") else ""
    t = "t=%7.2fs  " % tr["t"] if tr.get("t") is not None else ""
    if tr["kind"] == "ir":
        # an ir mark packs address/command/scancode into from/to/event so
        # they survive the server's persisted-log field whitelist
        return "%s  %sir      addr=%s cmd=%s%s" % (when, t, tr["from"],
                                                   tr["to"], ev)
    return "%s  %s%-7s %s -> %s%s" % (when, t, tr["kind"], tr["from"], tr["to"], ev)


def main():
    ap = argparse.ArgumentParser(description="Replay the Pi's transition archive")
    ap.add_argument("--session", type=int, help="replay this session id")
    ap.add_argument("--delay", type=float, default=0.0,
                    help="pause between snapshots (seconds)")
    ap.add_argument("--log", action="store_true",
                    help="read the persisted session log instead of the RAM ring")
    ap.add_argument("--count", type=int, default=20,
                    help="with --log, how many recent sessions to fetch")
    a = ap.parse_args()
    query = ("LOG %d" % a.count).encode() if a.log else b"DUMP"
    dump = fetch(query=query)
    sessions = dump["sessions"]
    if not sessions:
        print("%s is empty" % ("session log" if a.log else "RAM archive"))
        return 0

    if a.session is None:
        for s in sessions:
            trs = _normalize(s)
            t0 = time.strftime("%H:%M:%S", time.localtime(trs[0]["ts"]))
            t1 = time.strftime("%H:%M:%S", time.localtime(trs[-1]["ts"]))
            # id 0 is the between-sessions bucket: marks that carry no
            # occupancy (an ir press at an empty room), persisted by the
            # server as orphan records rather than sessions
            name = ("orphan     " if s.get("orphan") or s["id"] == 0
                    else "session %-3d" % s["id"])
            print("%s %s - %s  %d transitions" % (name, t0, t1, len(trs)))
        return 0

    # several orphan groups can share id 0, so replay every group with the
    # requested id rather than stopping at the first
    picked = [s for s in sessions if s["id"] == a.session]
    if not picked:
        print("no session %d in the archive" % a.session)
        return 1
    for s in picked:
        trs = sorted(_normalize(s), key=lambda tr: tr["ts"])
        name = "orphan" if s.get("orphan") or s["id"] == 0 else "session %d" % s["id"]
        print("%s  (%d transitions)\n" % (name, len(trs)))
        for tr in trs:
            print(_label(tr))
            if tr["frame"]:
                print(_grid(tr["frame"]))
            print()
            if a.delay:
                time.sleep(a.delay)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
