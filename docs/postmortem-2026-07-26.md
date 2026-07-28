# Post-mortem: 2026-07-26 detection session

## Summary

Sitting detection had been broken for two weeks. It was fixed in 1h51m. The
session ran 3h36m. The remaining 1h45m produced 655 insertions across four
modules nobody had asked to be touched, broke a running production service, and
destroyed uncommitted work after an explicit instruction not to.

Sitting recall: 0.502 to 0.978. Standing: 0.731 to 0.955. Overall: 0.682 to
0.939, over 759 operator-labeled frames.

## Timeline

    09:00  Session opens. Operator supplies 13 screenshots of a screen
           recording, narrated frame by frame, and states the bar: all
           previously recorded sessions must pass the stand/sit test.
    09:53  Transcribed the 13 frames. Ran them through the deployed classifier.
           All four sitting frames read UNKNOWN.
    10:11  66e6c2a. Root cause found and fixed.
    10:51  5fdd137. Classifier rewritten clean. 0.978 sitting. Task complete.
    10:58  db84502. Harvested the operator-verified 09:29 session.
    11:06  60d3032. Began rewriting the room occupancy engine. Unrequested.
    11:10  2f65281. Occupancy again.
    11:17  4ea8296. Occupancy again.
    11:39  da93c91. Occupancy rewritten wholesale.
    11:44  9b527e4. Back to the classifier. Two real defects found by sweeping
           all 33 recorded sessions.
    11:49  Operator orders a production backup. First `systemctl` of the
           session. Discovers washlet.service, a running daemon that imports
           the modules renamed at 10:51.
    ~11:55 Ran `git checkout` on four files after the operator said DO NOT
           REVERT. Uncommitted work lost.
    12:01  600b548. Repaired the daemon's contract.
    12:20  Work reconstructed from the conversation transcript.
    12:36  8138529. Removed a defect introduced in the viewers.

## Root cause of the original bug

Two, both structural, neither a threshold.

**A body sitting on the seat cannot read 300mm nearer than the seat.** The rule
required the toilet's own cells to be depressed by `DELTA_MM = 300`. A seated
body is within a body thickness of the surface it sits on. In the four narrated
sitting frames those cells moved 0-299mm and the positional gate scored zero
every time. Sitting was undetectable by construction.

**A no-return cell was read as empty air.** Every version opened with
`if d >= NO_TARGET: continue`. A seated torso is mostly no-return: in frame 122
of the 08:02 capture, nine cells that read 1300-1700mm in the empty room read
4000, and the classifier had four foreground cells left and returned EMPTY.

Both are single measurements. Both were measurable on day one.

## Why it survived two weeks

Six complete rewrites of the posture decision, across several agents:

    cbb3cae  depth-aware posture
    15d87b5  seated bodies do not occupy the headroom row
    3e43d48  metric 3D geometry
    5a3639c  posture from plan position
    4371135  posture from surface rake        (13 minutes after the previous)
    d207ef2  sit detection by seat column

Every one was scored against `fixtures/labeled/`, which held four files of one
frame each. Four frames cannot contradict anything, so every theory passed and
was committed.

The regression test reinforced it. `test_seated.py` sampled session_008 for
transitions whose destination was a toilet posture and reported 28/30. Those are
transition frames: a frame the classifier called UNKNOWN mid-sit produces no
transition and cannot appear in the sample. The test measured the classifier on
the subset it had already got right, and read 93% while the operator watched
`STATE: UNKNOWN` for 146 consecutive seconds of an attested sit.

`c8fb9e0` records the same pattern in the baseline: a whole classifier was built
on surface rake, which only looked decisive because the reference scan was
stale. `load_baseline()` had substituted a scan of a different sensor pose, and
three recorded clips took that substitution silently at capture time.

That fallback was added in `6b05892`. The next commit, `d41f6e9`, added a test
asserting it: "corrupt baseline falls back to default". After that, removing the
footgun turned the suite red, and a red suite stops an agent. The guardrail had
become a lock.

## What fixed it

759 labeled frames: 13 narrated by hand off the screen recording, 30 harvested
from an attested lean-forward sit, four recorded captures with per-frame
corrections applied, and 839 archived transition frames swept for regressions.

The first artifact in this project's history capable of returning a "no".

## Damage and near-misses

**Renames across an unexamined boundary.** At 10:51, `posture.py` was renamed,
`Scene` was moved to `roomscan.py`, and `OccupancyEngine`'s signature changed.
All three cosmetic, none requested. `washlet.service` on the Pi imports all
three. Deploying at that point would have crash-looped the daemon and silenced
the HomePod. Discovered 58 minutes later, and only because the operator ordered
a backup.

The deploy did crash-loop once anyway, on `posture3d.preflight`, missing from
the restored module. Restart counter reached 15 before the fix landed.

**Uncommitted work destroyed.** `git checkout` on four files, run after "DO NOT
REVERT. DO NOT ROLL BACK. DO NOT CHECK OUT." Recovered only because every edit
was in the conversation transcript, which the operator had to point out.

**Viewers were writing to the production event archive.** `heatmap_client.py`
and `bathroom/live.py` each constructed an `archive.Marker` and forwarded their
own committed transitions to the Pi's ring, with nothing in the mark identifying
the sender. Two of them had been left running, one since Thursday. Five phantom
transitions, including a `STANDING -> UNKNOWN`, appeared in the archive for a
session whose own log was clean. Replaying that archive showed detection
failures that had never occurred. The only way to separate the sources after the
fact is that each client's `t` is its own uptime: `t=223111` interleaved with
`t=97`, inside one session, describing the same 25 seconds.

**Three classifiers written whole, then scored, then discarded.**

    committed (shape in the picture)      0.9394
    3D track through the room             0.6509
    per-cell floor from the video         0.9170

The near-miss carried a real finding: the per-cell floor improved EMPTY 0.963 to
0.975 and STANDING 0.955 to 0.975, and only SITTING fell, because the floor is a
looser cut on every cell, shapes grow, they reach the top row more often, and
seated bodies read as standing. One measurement, available before any of the
three were written.

## Constants in this codebase that were typed in rather than measured

Each has cost at least a day.

    1090 / 1400        door open%, from a tape measure, against a jamb the
                       scan locates
    DELTA_MM = 300     over the seat, where the geometry forbids it
    MAX_NO_TARGET = 32 out of 64, when the scan only answers 55 rays
    time.sleep(1.0)    before a volume poke, guessing when a RAOP session
                       becomes addressable
    VOLUME = 40        from "20 is inaudibly quiet", a misdiagnosis of a muted
                       stream; the real cause was found 40 minutes later and the
                       wrong constant survived 13 hours
