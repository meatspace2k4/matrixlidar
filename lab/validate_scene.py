#!/usr/bin/env python3
"""Replay the misdetection fixture sessions through the per-frame scene
model (lab/scene_model.py) and print every DOORWAY_EVENT, every
ENTERED/EXITED/PRESENCE_NO_ENTRY verdict, and the final occupancy verdict,
followed by a scene-model vs live-marks vs ground-truth comparison table.

Faithfulness rules (stated in the output, not hidden):
  - Fixture sessions are transition-ring snapshots, not full streams.
    Between consecutive snapshots <= BRIDGE_MAX_S apart, the earlier frame
    is repeated at 10 Hz, ONLY so the 2-frame debounce has contiguous
    evidence to chew on; every bridged span is listed.
  - Gaps > BRIDGE_MAX_S are unobserved: the debounce history is reset and
    the next snapshot is read fresh. Every reset is listed (counted, for
    the long sessions).
  - Where the data cannot support a verdict, that is printed explicitly.

Thresholds are global (scene_model.Params), never tuned per session.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bathroom.stream import orient  # noqa: E402
from lab.scene_model import Params, SceneAutomaton, read_frame  # noqa: E402

FIXDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "fixtures", "misdetections")
SESSIONS = ["001", "010", "005", "006", "008"]
HZ = 10.0
BRIDGE_MAX_S = 1.0

GROUND_TRUTH = {
    "001": "real entry then real exit: must read ENTERED then EXITED",
    "010": "empty room, mid-field artifact: PRESENCE_NO_ENTRY only, "
           "final not-occupied",
    "005": "tail holds the OLD false EXIT (near-lens direction flip); "
           "no real doorway transit at that moment",
    "006": "sparse multi-minute session; report without overclaiming",
    "008": "sparse multi-minute session (old engine stuck OCCUPIED for "
           "hours); report without overclaiming",
}


def to_matrix(frame_csv):
    flat = [int(x) for x in frame_csv.split(",")]
    return orient([flat[i * 8:(i + 1) * 8] for i in range(8)])


def load_session(sid):
    with open(os.path.join(FIXDIR, "session_%s.json" % sid)) as f:
        doc = json.load(f)
    seen = set()
    out = []
    for tr in sorted(doc["transitions"], key=lambda t: t["ts"]):
        key = (tr["ts"], tr["frame"])
        if key in seen:  # ring fixtures carry a few exact duplicates
            continue
        seen.add(key)
        out.append(tr)
    return out


def marks_summary(trs):
    """What the OLD engine did live: its room transitions + final room."""
    room = [m for m in (t["mark"] for t in trs) if m["kind"] == "room"]
    if not room:
        return "no room transitions recorded", "(unchanged)"
    parts = ["%s->%s%s" % (m["from"], m["to"],
                           "[%s]" % m["event"] if m.get("event") else "")
             for m in room]
    return "; ".join(parts), room[-1]["to"]


def fmt_cells(cells, limit=8):
    if not cells:
        return "-"
    if len(cells[0]) == 4:
        body = ", ".join("%s=%dmm(band %d-%d)" % c for c in cells[:limit])
    else:
        body = ", ".join("%s=%dmm" % c for c in cells[:limit])
    if len(cells) > limit:
        body += ", +%d more" % (len(cells) - limit)
    return body


def replay(sid):
    trs = load_session(sid)
    t0 = trs[0]["ts"]
    auto = SceneAutomaton()
    events = []          # (t, kind, text)
    vetoes = []          # (t, text) per real snapshot
    bridges = []         # (dt_from, dt_to)
    resets = []          # dt
    last_reading = None

    def emit(evs):
        for e in evs:
            t = e["t"]
            if e["kind"] == "DOORWAY_EVENT":
                txt = ("DOORWAY_EVENT      cells: %s  panel=%s"
                       % (fmt_cells(e["cells"]), e["panel"]))
            elif e["kind"] == "ENTERED":
                txt = ("ENTERED            %.2fs after DOORWAY_EVENT%s"
                       % (e["since_event"],
                          "  (already occupied)" if e["already_occupied"] else ""))
            elif e["kind"] == "EXITED":
                txt = ("EXITED             %.2fs after DOORWAY_EVENT  panel=%s%s"
                       % (e["since_event"], e["panel"],
                          "" if e["was_occupied"] else "  (was not occupied)"))
            else:
                txt = ("PRESENCE_NO_ENTRY  cells: %s  panel=%s"
                       % (fmt_cells(e["cells"]), e["panel"]))
            events.append((t, e["kind"], txt))

    for i, tr in enumerate(trs):
        t = tr["ts"] - t0
        reading = read_frame(to_matrix(tr["frame"]))
        fresh = i == 0 or (t - (trs[i - 1]["ts"] - t0)) > BRIDGE_MAX_S
        if fresh and i > 0:
            resets.append(t)
        if reading.vetoed:
            vetoes.append((t, "doorway reading VETOED (near-lens smear): "
                              "in-band %s BUT %d sub-floor cells %s "
                              "(floor=%.0f%% of ray baseline)"
                           % (fmt_cells(reading.doorway_cells, 5),
                              len(reading.subfloor_cells),
                              fmt_cells(reading.subfloor_cells, 5),
                              Params.BAND_LO * 100)))
        emit(auto.step(t, reading, fresh=fresh))
        # bridge toward the next snapshot when the gap is small enough
        if i + 1 < len(trs):
            t_next = trs[i + 1]["ts"] - t0
            if 0 < t_next - t <= BRIDGE_MAX_S:
                bridges.append((t, t_next))
                tb = t
                while tb + 1.0 / HZ < t_next:
                    tb += 1.0 / HZ
                    emit(auto.step(tb, reading, fresh=False))
        last_reading = reading

    # ---------- printing ----------
    dur = trs[-1]["ts"] - t0
    print("=" * 78)
    print("session_%s  (%d snapshots over %.1fs; baseline = "
          "classify.DEFAULT_BASELINE == baselines/empty.json)" % (sid, len(trs), dur))
    print("ground truth: %s" % GROUND_TRUTH[sid])
    if bridges:
        span = "; ".join("%.2f->%.2f" % b for b in bridges)
        if len(span) > 400:
            span = span[:400] + "... (list truncated for print only)"
        print("bridging: %d gaps <=%.1fs bridged at %.0fHz by repeating the "
              "earlier frame (debounce feed only): %s"
              % (len(bridges), BRIDGE_MAX_S, HZ, span))
    else:
        print("bridging: none used")
    print("gaps >%.1fs: %d (unobserved; debounce reset, next frame read "
          "fresh)" % (BRIDGE_MAX_S, len(resets)))

    merged = sorted([(t, "EV", txt) for (t, _k, txt) in events]
                    + [(t, "DIAG", txt) for (t, txt) in vetoes])
    if not merged:
        print("  (no events, no vetoed doorway readings)")
    for t, tag, txt in merged:
        print("  t=%8.2fs  %s%s" % (t, "" if tag == "EV" else "[diag] ", txt))

    verdict = "OCCUPIED" if auto.occupied else "NOT OCCUPIED"
    print("final occupancy verdict: %s" % verdict)

    # ---------- honesty notes, derived from generic conditions ----------
    kinds = [k for (_t, k, _x) in events]
    if auto.first_interior_t is not None and (
            auto.first_doorway_t is None
            or auto.first_interior_t < auto.first_doorway_t):
        print("  note: interior presence precedes any DOORWAY_EVENT; the "
              "entry (if any) was not captured in these snapshots, so "
              "occupancy before t=%.2fs cannot be determined."
          % auto.first_interior_t)
    if auto.pending is not None:
        print("  note: an unresolved DOORWAY_EVENT (t=%.2fs) is still "
              "pending at session end." % auto.pending)
    if last_reading is not None and auto.interior.value:
        if last_reading.interior_body:
            print("  note: interior presence is still present at the final "
                  "snapshot; the session ends unresolved.")
        else:
            print("  note: the final snapshot itself reads no interior "
                  "body, but the 2-frame debounce had not yet cleared at "
                  "session end.")
    if auto.occupied:
        after = [t for (t, txt) in vetoes
                 if any(k == "ENTERED" and et <= t
                        for (et, k, _x) in events)]
        print("  note: final OCCUPIED rests on the out-of-FOV rule "
              "(interior disappearance without a doorway transit is not an "
              "exit). The data cannot distinguish out-of-FOV from an exit "
              "the passage band never saw%s."
              % (" (vetoed near-lens doorway smears at %s and panel=%s at "
                 "the end suggest exactly such an exit)"
                 % (", ".join("t=%.2fs" % t for t in after),
                    last_reading.door_panel)
                 if after and last_reading is not None else ""))
    if ("PRESENCE_NO_ENTRY" in kinds and "ENTERED" not in kinds
            and not auto.occupied):
        print("  note: presence appeared and vanished with no doorway "
              "transit; the model reports it without latching occupancy. "
              "It cannot verify whether the room is truly empty.")

    mk, mk_final = marks_summary(trs)
    return {
        "sid": sid,
        "verdict": verdict,
        "kinds": kinds,
        "events": events,
        "vetoes": vetoes,
        "marks": mk,
        "marks_final": mk_final,
    }


def main():
    print(__doc__.strip())
    print()
    print("passage band: [%.0f%%, %.0f%%] of each doorway ray's baseline "
          "(cols F,G,H); body margin %d mm; smear veto on any sub-floor "
          "doorway-ray return; interior body >= %d cells (baseline drift at "
          "B4/C4/C5/D6/D7/D8 reaches 6 cells corpus-wide, attested bodies "
          ">= 9); debounce N=%d."
          % (Params.BAND_LO * 100, Params.BAND_HI * 100, Params.MARGIN_MM,
             Params.MIN_INTERIOR_CELLS, Params.DEBOUNCE_N))
    print()
    results = [replay(sid) for sid in SESSIONS]

    print()
    print("=" * 78)
    print("COMPARISON: scene model  vs  recorded live marks  vs  ground truth")
    print("=" * 78)
    for r in results:
        verdicts = []
        noop_entries = 0
        doorway_n = 0
        for (t, k, txt) in r["events"]:
            if k == "DOORWAY_EVENT":
                doorway_n += 1
            elif k == "ENTERED" and "(already occupied)" in txt:
                noop_entries += 1
            elif k in ("ENTERED", "EXITED", "PRESENCE_NO_ENTRY"):
                verdicts.append("%s@%.2fs" % (k, t))
        model = " -> ".join(verdicts) if verdicts else "(no verdicts)"
        extra = "; %d doorway events" % doorway_n
        if noop_entries:
            extra += " (%d re-sightings resolved ENTERED while already occupied)" % noop_entries
        if r["vetoes"]:
            extra += "; %d vetoed near-lens doorway smears" % len(r["vetoes"])
        print("session_%s" % r["sid"])
        print("  scene model : %s; final %s%s" % (model, r["verdict"], extra))
        print("  live marks  : %s; final room %s" % (r["marks"], r["marks_final"]))
        print("  ground truth: %s" % GROUND_TRUTH[r["sid"]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
