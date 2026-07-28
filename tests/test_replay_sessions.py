#!/usr/bin/env python3
"""Replay real fixture sessions through the occupancy engine.

Fixtures are transition snapshots, not full streams. This replay reconstructs
continuous evidence by repeating each snapshot's frame at 10 Hz until the next
snapshot (only across gaps <= 1.0s), then appends 60s of the final frame to
encode the attested-empty aftermath.

  python3 tests/test_replay_sessions.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bathroom.classify import DEFAULT_BASELINE  # noqa: E402
from bathroom.occupancy import OccupancyEngine, RoomState  # noqa: E402
from bathroom.stream import orient  # noqa: E402
from bathroom.zones import N  # noqa: E402

HZ = 10.0


def load_fixture(path):
    """Load and validate a fixture file."""
    try:
        with open(path) as f:
            data = json.load(f)
    except FileNotFoundError:
        raise AssertionError(f"{path}: file not found")
    except json.JSONDecodeError as e:
        raise AssertionError(f"{path}: invalid JSON: {e}")

    trs = sorted(data.get("transitions", []), key=lambda tr: tr["ts"])
    if len(trs) < 5:
        raise AssertionError(
            f"{path}: fixture has {len(trs)} transitions, expected >= 5"
        )
    return trs


def to_matrix(frame_csv):
    """Convert comma-separated flat frame to oriented 8x8 matrix."""
    flat = [int(x) for x in frame_csv.split(",")]
    mat = orient([flat[i * N:(i + 1) * N] for i in range(N)])
    return mat


def replay_with_bridge(transitions, baseline=None):
    """Replay transitions with bridging.

    Repeats each snapshot's frame at HZ until the next snapshot, but only
    across gaps <= 1.0s (longer gaps are handled by the engine's own
    GAP_RESET logic). Appends 60s of the final frame as attested-empty
    aftermath.

    Returns (list of (t, event, room) tuples, engine)."""
    if not transitions:
        return [], OccupancyEngine(baseline=baseline)

    eng = OccupancyEngine(baseline=baseline)
    t0 = transitions[0]["ts"]
    events = []

    def step(matrix, t):
        """Feed one matrix frame, collect any event."""
        ru = eng.update(matrix, t, door_pct=None)
        if ru.event:
            events.append((t, ru.event, ru.room))

    # Real snapshots at real relative timestamps with bridging
    for i, tr in enumerate(transitions):
        t_here = tr["ts"] - t0
        mat_here = to_matrix(tr["frame"])
        step(mat_here, t_here)

        # Bridge to next snapshot if gap is attestable (<=1.0s)
        if i + 1 < len(transitions):
            t_next = transitions[i + 1]["ts"] - t0
            gap = t_next - t_here
            if gap <= 1.0:
                # Repeat current frame at HZ until next snapshot
                t_b = t_here
                while t_b + 1.0 / HZ < t_next:
                    t_b += 1.0 / HZ
                    step(mat_here, t_b)

    # Attested-empty aftermath: repeat the final frame for 60s
    last = transitions[-1]
    t = transitions[-1]["ts"] - t0
    mat_last = to_matrix(last["frame"])
    tail_seconds = 60.0
    n = int(tail_seconds * HZ)
    for _ in range(n):
        t += 1.0 / HZ
        step(mat_last, t)

    return events, eng


def check(cond, msg):
    """Check a condition and print result."""
    print(("PASS " if cond else "FAIL ") + msg)
    return cond


def main():
    fixture_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "fixtures", "misdetections"
    )
    fixture_010 = os.path.join(fixture_dir, "session_010.json")
    fixture_001 = os.path.join(fixture_dir, "session_001.json")

    baseline = DEFAULT_BASELINE
    failures = []

    # Load and validate fixtures
    try:
        trs_010 = load_fixture(fixture_010)
    except AssertionError as e:
        print("FAIL " + str(e))
        return 1

    try:
        trs_001 = load_fixture(fixture_001)
    except AssertionError as e:
        print("FAIL " + str(e))
        return 1

    # Test session_010: artifact never latches
    # Transitions are present but no ENTRY_CONFIRMED should be emitted.
    # Room stays UNKNOWN, occupants = 0.
    events_010, eng_010 = replay_with_bridge(trs_010, baseline)
    event_names_010 = [ev for _, ev, _ in events_010]

    if not check("ENTRY_CONFIRMED" not in event_names_010,
                 "session_010: no ENTRY_CONFIRMED"):
        failures.append("session_010 has ENTRY_CONFIRMED")

    if not check("EXIT" not in event_names_010,
                 "session_010: no EXIT"):
        failures.append("session_010 has EXIT")

    # The PRESENCE_NO_ENTRY / stays-UNKNOWN assertions are gone. They pinned
    # the rule that a body VISIBLE IN THE ROOM with no observed entry must not
    # latch occupancy, which is the rule that had the operator sitting on the
    # toilet under `ROOM: UNKNOWN(0) 8m31s`. Seeing someone is evidence that
    # someone is there. What still matters here, and is kept above, is that a
    # mid-field artifact must not be reported as somebody coming through the
    # DOOR: no ENTRY_CONFIRMED, no EXIT.

    # session_001's entry/exit assertions are gone, and the reason is in what
    # this replay actually feeds. The archive stores ONE FRAME PER COMMITTED
    # TRANSITION, and replay_with_bridge holds each of those frames constant
    # until the next one. Entry and exit are a body CROSSING THE THRESHOLD
    # PLANE now (bathroom/threshold.py), confirmed by travel across it, and a
    # body that never moves never crosses. There is no crossing in this data to
    # find: at every ENTRY_CONFIRMED the archive ever recorded, the body is
    # already 555-871mm inside the room, and its one archived EXIT frame has no
    # body in view at all. The old logic never saw a crossing either; it
    # inferred one afterwards from the door's state.
    #
    # Entry and exit are gated on continuous recorded captures instead, in
    # tests/test_crossing.py, where the body is actually seen moving through.
    events_001, eng_001 = replay_with_bridge(trs_001, baseline)
    event_names_001 = [ev for _, ev, _ in events_001]

    if not check("ENTRY_CONFIRMED" not in event_names_001,
                 "session_001: no entry claimed from held transition frames"):
        failures.append("session_001 claimed an entry from a static frame")

    print("ALL PASS" if not failures else "SOME FAILED")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
