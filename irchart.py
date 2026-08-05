#!/usr/bin/env python3
"""IR presses charted against the sensor timeline, in the terminal.

A labeling aid: shows WHEN each remote button was pressed relative to what
the sensor saw, so a recorded clip can be trusted as a (movement, button)
sample before anything trains on it. Read-only client, same as replay.py:
it consumes a recorded clip or the Pi's RAM ring over the wire and derives
nothing the daemon didn't already publish.

  python3 irchart.py --clip data/clips/sitting_X.json   # a recorded clip
  python3 irchart.py                                    # list ring sessions
  python3 irchart.py --session 28                       # one ring session

Output is the terminal, nothing else. NOTHING IS WRITTEN unless --out is
given, and then it is exactly one atomic write of the finished render (this
runs on the Pi's SD card; a chart is not worth wear). The render is plain
text either way, so the file greps and diffs.

The three strips share one time axis, one column per bucket:

  move   ▁▂▇█▅   mean per-cell change between consecutive frames (clips
                 only; the ring keeps sparse snapshots, not a stream)
  state  ··WSSTT the committed posture the daemon recorded, forward-filled
  ir     ··1··2· which button, by table index. A lone repeat shows as '.',
                 a wrong-length fragment as 'x' (LED feedback, kept but
                 never counted as a press).
"""
import argparse
import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bathroom import ir as ir_mod                              # noqa: E402
from bathroom.sparkline import STATE_ABBR, sparkline           # noqa: E402
from bathroom.zones import State                               # noqa: E402
from replay import fetch, _normalize                           # noqa: E402

IDX = "123456789abcdefghijklmnopqrstuvwxyz"


def _clip_series(doc):
    """(t0, t1, move[(t,energy)], states[(t,name)], ir_events) from a clip."""
    frames = doc["frames"]
    move = []
    prev = None
    for fr in frames:
        mm = fr["mm"]
        if prev is not None:
            move.append((fr["t"],
                         sum(abs(a - b) for a, b in zip(mm, prev)) / len(mm)))
        prev = mm
    states = [(fr["t"], fr["state_engine"]) for fr in frames]
    irs = doc.get("ir") or []
    ts = [fr["t"] for fr in frames] + [e.get("t", 0.0) for e in irs]
    return min(ts), max(ts), move, states, irs


def _session_series(session):
    """Same shape from a ring session's sparse transition marks."""
    trs = sorted(_normalize(session), key=lambda tr: tr["ts"])
    t0 = trs[0]["ts"]
    states, irs = [], []
    for tr in trs:
        t = tr["ts"] - t0
        if tr["kind"] == "posture":
            states.append((t, tr["to"]))
        elif tr["kind"] == "ir":
            irs.append({"t": t, "fp": tr["from"], "repeat": tr.get("event") == "repeat"})
    return 0.0, trs[-1]["ts"] - t0, [], states, irs


def _bucket(t, t0, t1, width):
    span = (t1 - t0) or 1e-9
    b = int((t - t0) / span * width)
    return min(max(b, 0), width - 1)


def _presses(irs):
    """Collapse repeats onto their press: [(t, fp, n_repeats, fragment)]."""
    out = []
    for ev in sorted(irs, key=lambda e: e.get("t", 0.0)):
        if ev.get("repeat") and out and out[-1][1] == ev.get("fp"):
            t, fp, n, frag = out[-1]
            out[-1] = (t, fp, n + 1, frag)
        else:
            out.append([ev.get("t", 0.0), ev.get("fp", "?"), 0,
                        not ir_mod.is_expected_length(ev)])
    return [tuple(p) for p in out]


def _state_at(states, t):
    cur = "?"
    for st, name in states:
        if st > t:
            break
        cur = name
    return cur


def _strip_of_states(states, t0, t1, width):
    cells = [" "] * width
    for i in range(width):
        t = t0 + (t1 - t0) * (i + 0.5) / width
        name = _state_at(states, t)
        try:
            cells[i] = STATE_ABBR[State[name]]
        except KeyError:
            cells[i] = "?"
    return "".join(cells)


def render(t0, t1, move, states, irs, width, title):
    presses = _presses(irs)
    fps = []                       # distinct fingerprints, in press order
    for _t, fp, _n, frag in presses:
        if not frag and fp not in fps:
            fps.append(fp)

    lines = ["%s   %.1fs   %d ir events, %d presses"
             % (title, t1 - t0, len(irs), sum(1 for p in presses if not p[3]))]

    if move:
        buckets = [0.0] * width
        for t, e in move:
            b = _bucket(t, t0, t1, width)
            buckets[b] = max(buckets[b], e)
        lines.append(" move  |%s|" % sparkline(buckets, lo=0))
    if states:
        lines.append(" state |%s|" % _strip_of_states(states, t0, t1, width))

    ir_cells = [" "] * width
    for t, fp, _n, frag in presses:
        b = _bucket(t, t0, t1, width)
        if frag:
            ir_cells[b] = "x" if ir_cells[b] == " " else ir_cells[b]
        else:
            ir_cells[b] = IDX[fps.index(fp) % len(IDX)]
    lines.append(" ir    |%s|" % "".join(ir_cells))
    lines.append("       %-*s%s" % (width // 2, "0.0s", "%.1fs" % (t1 - t0)))

    lines.append("")
    lines.append(" #  t         fp                n  state at press")
    for t, fp, n, frag in presses:
        tag = "x" if frag else IDX[fps.index(fp) % len(IDX)]
        rep = "+%d rpt" % n if n else ""
        lines.append(" %s  %8.2fs  %-16s %-6s %s%s"
                     % (tag, t - t0, fp, rep, _state_at(states, t),
                        "  (fragment)" if frag else ""))
    if not presses:
        lines.append("    (no ir in this source)")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(
        description="Chart IR presses against the sensor timeline")
    ap.add_argument("--clip", help="recorded clip JSON instead of the ring")
    ap.add_argument("--session", type=int, help="ring session id")
    ap.add_argument("--width", type=int, default=0,
                    help="chart columns (default: fit the terminal)")
    ap.add_argument("--out", help="ALSO write the render to this file, one "
                                  "atomic write; without it nothing touches "
                                  "the filesystem")
    a = ap.parse_args()
    width = a.width or max(24, min(100, shutil.get_terminal_size().columns - 10))

    if a.clip:
        with open(a.clip) as f:
            doc = json.load(f)
        series = _clip_series(doc)
        title = doc.get("clip_id", a.clip)
    else:
        dump = fetch()
        sessions = dump["sessions"]
        if a.session is None:
            for s in sessions:
                trs = _normalize(s)
                n_ir = sum(1 for tr in trs if tr["kind"] == "ir")
                t0 = time.strftime("%H:%M:%S", time.localtime(trs[0]["ts"]))
                print("session %-3s %s  %3d marks, %d ir"
                      % (s.get("id", "?"), t0, len(trs), n_ir))
            return 0
        match = [s for s in sessions if s.get("id") == a.session]
        if not match:
            print("no session %d in the ring" % a.session)
            return 1
        series = _session_series(match[0])
        title = "session %d" % a.session

    text = render(*series, width=width, title=title)
    sys.stdout.write(text)
    if a.out:
        tmp = a.out + ".tmp"
        with open(tmp, "w") as f:
            f.write(text)
        os.replace(tmp, a.out)
        sys.stderr.write("wrote %s (%d bytes, one write)\n"
                         % (a.out, len(text.encode())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
