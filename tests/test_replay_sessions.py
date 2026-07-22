#!/usr/bin/env python3
"""Replay real fixture sessions through the occupancy engine using bridged
playback (repeat each snapshot at 10Hz until the next) and verify against
known outcomes.

  python3 tests/test_replay_sessions.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bathroom.classify import classify
from bathroom.detect import door_open_pct                # noqa: E402
from bathroom.config import load_baseline             # noqa: E402
from bathroom.occupancy import OccupancyEngine, RoomState  # noqa: E402
from bathroom.stream import orient                    # noqa: E402

HZ = 10.0


class Shim:
    """Adapter: a classified frame -> the .instant/.features surface the
    occupancy engine consumes."""

    def __init__(self, state, features):
        self.instant = state
        self.features = features


def load_fixture(path):
    """Load and validate a fixture file."""
    with open(path) as f:
        data = json.load(f)
    trs = sorted(data.get("transitions", []), key=lambda tr: tr["ts"])
    if len(trs) < 5:
        raise AssertionError(
            f"{path}: fixture has {len(trs)} transitions, expected >= 5"
        )
    return trs


def replay_with_bridge(transitions, baseline):
    """Replay transitions with bridging: repeat each snapshot's frame at HZ
    until the next snapshot, then append 60s of the final frame.
    Returns collected (event, room) tuples."""
    if not transitions:
        return []

    def to_update(frame_csv):
        flat = [int(x) for x in frame_csv.split(",")]
        mat = orient([flat[i * 8:(i + 1) * 8] for i in range(8)])
        st, feats = classify(mat, baseline)
        # feed door% as the live clients do; without it the door-motion
        # exit rule is structurally disabled and exits cannot replay
        return Shim(st, feats), door_open_pct(mat)

    eng = OccupancyEngine()
    t0 = transitions[0]["ts"]
    events = []

    def step(pair, t):
        u, pct = pair
        ru = eng.update(u, t, door_pct=pct)
        if ru.event:
            events.append((ru.event, ru.room))

    # Real snapshots at real relative timestamps with bridging
    for i, tr in enumerate(transitions):
        t_here = tr["ts"] - t0
        step(to_update(tr["frame"]), t_here)
        if i + 1 < len(transitions):
            t_next = transitions[i + 1]["ts"] - t0
            t_b = t_here
            while t_b + 1.0 / HZ < t_next:
                t_b += 1.0 / HZ
                step(to_update(tr["frame"]), t_b)

    # Attested-empty aftermath: repeat the final frame for 60s at HZ
    last = transitions[-1]
    t = transitions[-1]["ts"] - t0
    tail_seconds = 60.0
    n = int(tail_seconds * HZ)
    for i in range(n):
        t += 1.0 / HZ
        step(to_update(last["frame"]), t)

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

    baseline = load_baseline()
    failures = []

    # Load and validate fixtures
    try:
        trs_010 = load_fixture(fixture_010)
    except (FileNotFoundError, json.JSONDecodeError, AssertionError) as e:
        print("FAIL " + str(e))
        return 1

    try:
        trs_001 = load_fixture(fixture_001)
    except (FileNotFoundError, json.JSONDecodeError, AssertionError) as e:
        print("FAIL " + str(e))
        return 1

    # Test session_010: mid-field artifact in empty room
    # Should abort the tentative entry, not confirm it
    events_010, eng_010 = replay_with_bridge(trs_010, baseline)
    event_names_010 = [ev for ev, _ in events_010]

    if not check("ENTRY_CONFIRMED" not in event_names_010,
                 "session_010: no ENTRY_CONFIRMED"):
        failures.append("session_010 has ENTRY_CONFIRMED")
    if not check("ENTRY_ABORTED" in event_names_010,
                 "session_010: has ENTRY_ABORTED"):
        failures.append("session_010 missing ENTRY_ABORTED")
    if not check(eng_010.room == RoomState.UNKNOWN,
                 "session_010: final room is UNKNOWN"):
        failures.append(f"session_010 room is {eng_010.room}, not UNKNOWN")
    if not check(eng_010.occupants == 0,
                 "session_010: final occupants = 0"):
        failures.append(f"session_010 occupants = {eng_010.occupants}, not 0")

    # Test session_001: real walk-in/walk-out through door
    # Should confirm entry and exit cleanly
    events_001, eng_001 = replay_with_bridge(trs_001, baseline)
    event_names_001 = [ev for ev, _ in events_001]

    if not check("ENTRY_CONFIRMED" in event_names_001,
                 "session_001: replay confirms the door entry"):
        failures.append("session_001 missing ENTRY_CONFIRMED")
    # The exit is replay-proven via the door-motion rule: the body cannot
    # leave except through the door, and the captured frames record the door
    # state changing after the vanish. This requires the replay to feed
    # door_pct into update(), as the live clients do.
    if not check("EXIT" in event_names_001,
                 "session_001: replay derives the EXIT"):
        failures.append("session_001 missing EXIT")
    if not check(eng_001.room == RoomState.EMPTY,
                 "session_001: final room is EMPTY"):
        failures.append("session_001 room is %s, not EMPTY" % eng_001.room)
    if not check(eng_001.occupants == 0,
                 "session_001: final occupants = 0"):
        failures.append("session_001 occupants = %d" % eng_001.occupants)
    # and the live-recorded marks agree (attested live behavior)
    recorded_room = [tr["mark"] for tr in trs_001 if tr["mark"]["kind"] == "room"]
    if not check(any(m.get("event") == "EXIT" and m.get("to") == "EMPTY"
                     for m in recorded_room),
                 "session_001: live-recorded marks contain EXIT -> EMPTY"):
        failures.append("session_001 recorded marks missing EXIT")

    print("ALL PASS" if not failures else "SOME FAILED")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
