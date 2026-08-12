"""Washlet jukebox daemon: entry sound on walk-in, the track on the visit's sit.

  python3 -m bathroom.washlet             # live
  python3 -m bathroom.washlet --dry-run   # log decisions, no audio

Consumes the same stream -> StateEngine -> OccupancyEngine stack as
bathroom.live. The main track (--media) plays on sit, once per VISIT.
An OPTIONAL entry sound (--queue-media) can play on walk-in; it is
DISABLED by default (empty), and with it disabled nothing plays on entry.
Decisions:

- QUEUE: the room becomes OCCUPIED with the latch clear (including a body
  already present at daemon start) -> stream the entry sound at --volume
  if one is configured, otherwise no audio (the machinery still runs).
- PLAY: the committed posture goes STANDING_AT_TOILET -> SITTING_AT_TOILET,
  the act of sitting down -> kill whatever is streaming and start the MAIN
  track from the top at --volume. Happens once per VISIT: the latch blocks
  replays on continued sitting, stand-up/sit-down cycles, and the track
  ending; the next entry clears it. There is no dwell and no delay: the
  music starts as the body lands.
- UNQUEUE: occupancy ends with no sit (entry aborted, stream gap, or exit)
  -> stop the entry sound.
- STOP (vacancy): room EMPTY while posture also reads EMPTY -> stop
  playback. The latch is NOT touched: an empty room has nobody to rearm
  for, and a room misread under a seated body can never cut-and-replay.
- REARM (entry): the room becomes OCCUPIED -> clear the latch, then QUEUE
  as before. A new body in the room is a new visit, and the new visit gets
  its own play. There is no clock anywhere: the once-a-day 05:00 reset
  this replaces is gone, not disabled.

HOW LONG THE TRACK PLAYS (bathroom/playwindow.py)

- by default it fades out after --auto-fade seconds (30) of playback: most
  visits want a few seconds of music, not the whole song
- a hand WAVED NEAR THE SENSOR during playback HOLDS it open: the timer is
  cancelled, the track plays on indefinitely, and the volume RAMPS up to
  --hold-volume (20%) over half a second. Holding it open is a request for
  the music, so it gets louder; the default stays quiet because most plays
  are background
- the NEXT wave fades it out, from whatever level it is at. That is the
  only stop the wave has now
- either way the fade is 2s, the same ending the walk-away fade gives, and
  the latch stays spent until the next entry

The wave itself is unchanged (wave4d: the trajectory of the near-field
cluster in metric 3D over time; normal movement, standing up and walking
never enter the near-field, and a passing body translates instead of
oscillating). Only what this daemon does with a WAVE moved: a gesture is
the right control for the rare case (keep going), a timer for the common
one (stop on its own). The walk-away fade is untouched and still ends
playback whatever the window is doing.

IR is recorded, never obeyed. The daemon opens the Pi's local receiver
(bathroom/ir.py; absent hardware is logged once and costs nothing) and every
frame FINGERPRINT is printed and MARKed into the session archive on the
lidar server, where it persists with the session and comes back out of
replay.py. Signals arriving on the stream as "IR <json>" lines
(a replayed clip's recorded log) are printed the same way. Neither source
triggers audio or touches the latch: the remote is not a control surface for
this daemon, it is something the room heard.

Every stream start gets the volume poke (a second atvremote, retried
across the startup window because a set accepted mid-handshake can be
overridden when audio starts); the session's own starting volume is
receiver state we do not trust. Both media files must exist at startup
(fail loud, no fallbacks).

THE LATCH IS IN MEMORY ONLY, DELIBERATELY. Nothing is written to disk and
nothing is read at startup, so a restart -- a crash, a redeploy, a power
cut -- comes up rearmed and the next sit plays. Worst case is one extra
play into the visit a restart lands in.
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time

from collections import deque

from . import archive, classify, config, detect, ir as ir_mod, occupancy, player as player_mod, playwindow, posture3d, scene as scene_mod, stream, threshold, wave4d
from .donewith import DoneWith, FADE
from .engine import StateEngine
from .live import DRIFT_FRAMES, _direction
from .occupancy import RoomState
from .zones import State

VOLUME = 10         # HomePod volume % applied as soon as the session accepts it
HOLD_VOLUME = 20    # ... and what a HOLD raises it to, over HOLD_RAMP_S.
                    # Holding the track open is a request for the music, so it
                    # gets louder; the default stays quiet because most plays
                    # are background.
HOLD_RAMP_S = 0.5   # the lift is a ramp, not a jump: a step change on a quiet
                    # track in a small tiled room is startling, and every other
                    # volume move here is a ramp already.
POKE_TRIES = 8      # set_volume attempts against the young stream session
POKE_INTERVAL_S = 1.5
STOP_GRACE_S = 3.0  # SIGTERM -> SIGKILL grace for the player child
PUBLISH_EVERY_S = 1.0  # heartbeat for the STATE feed, on top of every change

DEF_HOMEPOD = os.environ.get("WASHLET_HOMEPOD", "192.168.10.240")
DEF_MEDIA = os.environ.get(
    "WASHLET_MEDIA", os.path.expanduser("~/washlet/media/days_of_blues.mp3"))
DEF_QUEUE_MEDIA = os.environ.get("WASHLET_QUEUE_MEDIA", "")
DEF_ATVREMOTE = os.environ.get(
    "WASHLET_ATVREMOTE", os.path.expanduser("~/washlet/venv/bin/atvremote"))

QUEUE = "QUEUE"
PLAY = "PLAY"
UNQUEUE = "UNQUEUE"
STOP = "STOP"
REARM = "REARM"


def _posture3d_window(frames, baseline):
    """posture3d's verdict, with classify.py's FEATURES left alone.

    Same signature as classify.classify_window, so it drops into
    StateEngine(classifier=...). Only the posture verdict is swapped: the
    feature dict still comes from classify, because OccupancyEngine consumes
    it (occ_cells, body_mm and friends) and the room layer is not what is
    broken here. The posture is.
    """
    m = classify.median_window(frames)
    _legacy_state, feats = classify.classify(m, baseline)
    state, _f3d = posture3d.classify(m, baseline)
    return state, feats


POSTURE = {"3d": _posture3d_window, "legacy": None}


class SitLatch:
    """Room-queued sit trigger, latched until the NEXT ENTRY (pure, no I/O).

    update() consumes (committed posture, room state, t) once per frame and
    returns QUEUE (occupancy began: latch cleared for the new visit, entry
    sound), PLAY (the act of sitting down: the main track, once for this
    visit), UNQUEUE (occupancy evaporated with no sit), STOP (vacancy: cut
    the audio, latch untouched), or None. REARM is folded into the entry
    edge: a new body in the room is what rearms, never a clock and never the
    empty room itself.

    PLAY IS A TRANSITION, NOT A DWELL. It fires on the committed posture
    going STANDING_AT_TOILET -> SITTING_AT_TOILET: the act of sitting down,
    which is the thing the track is for. The 2s sit dwell it replaces was a
    debounce standing in for exactly this. It cost two seconds of music on
    every real sit and still admitted any sitting-shaped reading that
    persisted, which is what a stand misreading as a sit does.

    Coming from STANDING is what makes it a sit rather than a shape. Both
    states are already debounced by the engine's commit_after, so this is a
    transition between two committed states, not two noisy frames.
    """

    __slots__ = ("latched", "_prev_room", "_last_posture", "_played")

    def __init__(self):
        self.latched = False
        self._prev_room = None
        self._last_posture = None  # last KNOWN committed posture (see below)
        self._played = False       # THIS visit started the main track, so THIS
                                   # visit's exit has audio to stop

    def update(self, committed, room, t):
        prev, self._prev_room = self._prev_room, room

        # The transition is tracked across UNKNOWN, not through it. UNKNOWN
        # means "cannot tell", never "the body moved", and a stand-to-sit is
        # routinely split by one: the body is mid-motion exactly when the
        # classifier is least sure. Remembering the last KNOWN posture makes
        # STANDING -> UNKNOWN -> SITTING the same event as STANDING ->
        # SITTING, which is what it physically is.
        was = self._last_posture
        if committed != State.UNKNOWN:
            self._last_posture = committed

        # THE NEXT ENTRY IS WHAT REARMS: an OCCUPIED edge coming FROM EMPTY.
        # From-EMPTY is what makes it a new visit. A gap does not qualify:
        # UNKNOWN -> OCCUPIED is the stream recovering around the same seated
        # body, and rearming on it replayed mid-visit (the gap regression in
        # tests/test_washlet.py). The departing body cannot rearm either: its
        # own exit leads to EMPTY with nobody left to make the edge. No clock
        # is consulted anywhere.
        if (self.latched and room == RoomState.OCCUPIED
                and prev == RoomState.EMPTY):
            self.latched = False
            return REARM   # the QUEUE edge for this entry is deliberately
                           # consumed by the rearm: one event per frame, and
                           # the entry sound belongs to a latch-clear visit,
                           # which this now is (the next frame is not an edge,
                           # so no double QUEUE)

        # Exit condition: BOTH layers must agree the room is vacated. Room
        # EMPTY alone must not cut the audio (a misread under a still-seated
        # body would cut mid-sit); posture EMPTY alone must not (a body out of
        # FOV at the sink/shower is still in the room). Only the visit that
        # actually started the track has anything to stop, so later visits --
        # latched, silent -- exit without a word.
        if (self._played and room == RoomState.EMPTY
                and committed == State.EMPTY):
            self._played = False
            return STOP

        if not self.latched:
            # the act of sitting down: STANDING at the toilet, then SITTING on
            # it. A body that is merely sitting-SHAPED never fires this,
            # because it never came from standing at the fixture, and someone
            # walking past never does either.
            if (committed == State.SITTING_AT_TOILET
                    and was == State.STANDING_AT_TOILET):
                self.latched = True
                self._played = True
                return PLAY
            if room == RoomState.OCCUPIED and prev != RoomState.OCCUPIED:
                return QUEUE   # occupancy began (or daemon started into it)
            if prev == RoomState.OCCUPIED and room != RoomState.OCCUPIED:
                return UNQUEUE  # aborted entry, gap, or sit-less exit
        return None


class Player:
    """The atvremote child process (at most one). Never blocks the frame loop."""

    def __init__(self, atvremote, homepod, volume, dry_run=False):
        self.atvremote = atvremote
        self.homepod = homepod
        self.volume = volume
        self.dry_run = dry_run
        self._proc = None
        self._dry_running = False  # virtual child so --dry-run takes the same
                                   # branches as live

    def running(self):
        if self.dry_run:
            return self._dry_running
        return self._proc is not None and self._proc.poll() is None

    def reap(self):
        """Return the exit code once, on the poll after a natural end."""
        if self._proc is not None and self._proc.poll() is not None:
            rc = self._proc.returncode
            self._proc = None
            return rc
        return None

    def _poke_volume(self, proc):
        """From a worker thread: unmute the session. The RAOP session starts
        effectively muted regardless of the prior device volume; a set_volume
        against the ACTIVE session is honored, but one sent while it is still
        handshaking can fail. So retry from ~1s after launch until one lands:
        a short sit must become audible in seconds, not after a fixed delay
        (a 5s fixed poke lost the whole 10s window of a brief real sit)."""
        time.sleep(1.0)
        landed = 0
        for attempt in range(1, POKE_TRIES + 1):
            if proc.poll() is not None:
                break  # stream over; nothing left to unmute
            try:
                r = subprocess.run(
                    [self.atvremote, "-s", self.homepod,
                     "set_volume=%d" % self.volume],
                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL, timeout=15)
                if r.returncode == 0:
                    landed += 1
                    if landed == 1:
                        _log("volume poke landed (attempt %d)" % attempt)
                    # keep re-asserting: a set accepted during the handshake
                    # can be overridden when stream audio actually starts, so
                    # one success is not proof the session stays unmuted
            except (subprocess.SubprocessError, OSError):
                pass
            time.sleep(POKE_INTERVAL_S)
        if not landed:
            _log("volume poke gave up after %d attempts" % POKE_TRIES)

    def start(self, media):
        """Spawn a stream child for `media` (from the top), volume poked."""
        if self.dry_run:
            self._dry_running = True
            return
        if self.running():
            return
        self._proc = subprocess.Popen(
            [self.atvremote, "-s", self.homepod, "stream_file=%s" % media],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.unmute()

    def unmute(self):
        """Raise the session to the target volume (idempotent, best-effort)."""
        if self.dry_run:
            return
        proc = self._proc
        if proc is None or proc.poll() is not None:
            return
        threading.Thread(target=self._poke_volume, args=(proc,),
                         daemon=True).start()

    def stop(self):
        if self.dry_run:
            self._dry_running = False
            return
        if not self.running():
            return
        self._proc.terminate()
        try:
            self._proc.wait(timeout=STOP_GRACE_S)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait()
        self._proc = None


def _log(msg):
    print("%s %s" % (time.strftime("%m-%d %H:%M:%S"), msg), flush=True)


def main():
    ap = argparse.ArgumentParser(
        description="Sit-triggered AirPlay daemon (one play per visit, latch "
                    "clears on the next entry)")
    ap.add_argument("--auto-fade", type=float, default=playwindow.AUTO_FADE_S,
                    metavar="SECONDS",
                    help="playback fades out after this long unless a wave "
                         "holds it open (default %.0f)" % playwindow.AUTO_FADE_S)
    ap.add_argument("--hold-volume", type=int, default=HOLD_VOLUME,
                    help="volume %% the first wave raises playback to "
                         "(default: %(default)s)")
    ap.add_argument("--volume", type=int, default=VOLUME,
                    help="HomePod volume %% applied shortly into each play")
    ap.add_argument("--homepod", default=DEF_HOMEPOD, help="HomePod IP")
    ap.add_argument("--media", default=DEF_MEDIA,
                    help="main track, streamed on the sit trigger")
    ap.add_argument("--queue-media", default=DEF_QUEUE_MEDIA,
                    help="optional entry sound streamed when occupancy "
                         "begins; empty (default) plays nothing on entry")
    ap.add_argument("--atvremote", default=DEF_ATVREMOTE,
                    help="path to pyatv's atvremote")
    ap.add_argument("--dry-run", action="store_true",
                    help="log PLAY/STOP/REARM decisions without starting audio")
    ap.add_argument("--posture", choices=sorted(POSTURE), default="3d",
                    help="posture source: 3d (metric, derived from the current "
                         "baseline) or legacy (classify.py's row bands)")
    ap.add_argument("--commit-after", type=int, default=4,
                    help="frames a candidate must hold before the state flips")
    ap.add_argument("--stream-host", default=stream.HOST,
                    help="frame broadcast host (a replay server for testing)")
    ap.add_argument("--stream-port", type=int, default=stream.PORT,
                    help="frame broadcast port")
    args = ap.parse_args()

    if not args.dry_run:
        for f in (args.media, args.queue_media):
            if f and not os.path.isfile(f):
                _log("FATAL media file missing: %s" % f)
                return 1

    baseline = config.load_baseline()
    if args.posture == "3d":
        _log("scene: %s" % posture3d.preflight(baseline))
    eng = StateEngine(baseline=baseline, commit_after=args.commit_after,
                      classifier=POSTURE[args.posture])
    room_eng = occupancy.OccupancyEngine()
    # A body seen crossing the doorway outward is the exit. The door rules
    # alone stick OCCUPIED when someone leaves with the door already open.
    sc = scene_mod.Scene(baseline)
    crossing = threshold.Crossing()
    latch = SitLatch()
    player = Player(args.atvremote, args.homepod, args.volume,
                    dry_run=args.dry_run)
    # one pyatv session: volume set before the stream, and a fade that can
    # reach the session that is actually playing
    # The fade needs pyatv, which lives in the venv. If this interpreter cannot
    # reach it, say so at startup and keep the old cut-on-exit behaviour rather
    # than silently cutting where a fade was asked for.
    # THE DAEMON MARKS. Nothing did, until now: archive.Marker was only ever
    # constructed by heatmap_client and bathroom.live, so the Pi's frame ring
    # was filled by whichever viewer happened to be running, and when those were
    # made read-only the archive went silent. A false SITTING at 13:37:05 burned
    # the play latch and left no frame to replay.
    marker = archive.Marker()
    # The room's own IR receiver. This daemon runs on the Pi the sensor is
    # wired to, so it is the process that hears the remote; what it hears goes
    # into the same session archive as the transitions, via the marker.
    irx = ir_mod.Receiver()
    irx.start()
    _log("ir: %s" % ir_mod.describe(irx.source))
    fader = None
    if not args.dry_run:
        if player_mod.available():
            fader = player_mod.Playback()
        else:
            _log("FADE UNAVAILABLE: pyatv not importable by %s. The main track "
                 "will cut instead of fading. Point the unit's ExecStart at "
                 "~/washlet/venv/bin/python." % sys.executable)
    done = DoneWith()
    waver = wave4d.WaveTracker()
    window = playwindow.PlayWindow(auto_fade_s=args.auto_fade)

    # Say it out loud at every start: the latch came up CLEAR. A restart is
    # a rearm, and after it the next sit plays.
    _log("washlet daemon up: posture=%s vol=%d homepod=%s media=%s "
         "queue=%s%s | latch REARMED by this start; one play per visit on "
         "standing->sitting, cleared by the next entry"
         % (args.posture, args.volume, args.homepod, args.media,
            args.queue_media or "(disabled)",
            " (DRY RUN)" if args.dry_run else ""))
    t0 = None
    last_pub = None
    last_pub_t = 0.0
    # This service derives EVERYTHING the front-ends show. Door reading and
    # walking direction were still computed by each viewer against its own
    # local baseline; they are published here now, from the one process that
    # owns the scene.
    door = detect.DoorReading(baseline)
    col_hist = deque(maxlen=DRIFT_FRAMES)
    try:
        for m in stream.frames(host=args.stream_host, port=args.stream_port,
                               on_wait=_waiting):
            now = time.monotonic()
            t0 = t0 if t0 is not None else now
            t = now - t0
            u = eng.update(m, t)
            cross = crossing.update(m, sc, t)
            door.update(m)
            ru = room_eng.update(u, t, door_pct=door.pct,
                                 crossed_out=(cross == threshold.EXIT))
            cells = u.features["occ_cells"]
            if cells:
                col_hist.append(sum(c for _r, c in cells) / float(len(cells)))
            else:
                col_hist.clear()

            if ru.event is not None:
                _log("room -> %-9s (%s)" % (ru.room, ru.event))
            if u.changed and u.event is not None:
                _log("state %-18s -> %s (%.1fs)"
                     % (u.event.state.name, u.committed.name, u.event.duration))

            # publish: this daemon is the one process that derives state, and
            # every front-end displays what it publishes rather than deriving
            # its own copy off the same frames.
            snap = (u.committed.name, str(ru.room), ru.occupants)
            # on change, and otherwise every PUBLISH_EVERY_S: a front-end that
            # connects (or reconnects after a server restart) must be able to
            # get current state without waiting for the next transition
            if snap != last_pub or now - last_pub_t >= PUBLISH_EVERY_S:
                last_pub, last_pub_t = snap, now
                stream.send("STATE " + json.dumps(
                    {"posture": u.committed.name, "room": str(ru.room),
                     "occupants": ru.occupants, "t": round(t, 2),
                     "event": ru.event,
                     "since": round(ru.since, 2), "blocked": ru.blocked,
                     "door": None if door.pct is None else round(door.pct),
                     "occ": u.features["occ_count"],
                     "dir": _direction(col_hist).strip()}))

            # IR from two sources, one treatment: print the fingerprint, mark
            # into the session archive, trigger nothing. The local receiver is
            # the live bathroom (this daemon runs on the Pi the sensor is wired
            # to); the stream's "IR <json>" lines are a replayed clip's
            # recorded log arriving from evals/stub_pi.py. The live server
            # sends no IR lines, so nothing is heard twice.
            for ev in irx.drain():
                # kernel wall timestamp -> engine t, anchored on this frame:
                # the event is (now - its ts) old, and the frame is at t
                ev_t = t - max(0.0, time.time() - ev["ts"])
                _log("IR    t=%7.2fs  %s" % (ev_t, ir_mod.format_event(ev)))
                marker.ir(ev, ev_t)
            for ev in stream.drain_ir():
                ev_t = ev.get("t", t)
                _log("IR    t=%7.2fs  %s" % (ev_t, ir_mod.format_event(ev)))
                marker.ir(ev, ev_t)

            rc = player.reap()
            if rc is not None:
                _log("player ended rc=%d" % rc)

            marker.update(u, ru, t)
            # A wave HOLDS the track open; the second wave ends it. Playback
            # otherwise fades on its own after --auto-fade seconds. The wave
            # detector is unchanged; only what this daemon does with a WAVE
            # lives here (see bathroom/playwindow.py).
            waved = waver.update(m, t) == wave4d.WAVE
            act = window.update(t, waved)
            if act == playwindow.HOLD:
                _log("HOLD  wave during playback: auto-fade off, playing on "
                     "at %d%%%s" % (args.hold_volume,
                                    " (dry run, no audio)" if args.dry_run
                                    else ""))
                if fader is not None:
                    fader.set_volume(args.hold_volume, seconds=HOLD_RAMP_S)
                else:
                    # the atvremote path pokes its own volume attribute
                    player.volume = args.hold_volume
                    player.unmute()
            elif act == playwindow.FADE_OUT:
                _log("WAVE  second wave, %.0fs out" % playwindow.FADE_S
                     if waved else
                     "AUTO  %.0fs of playback, %.0fs out"
                     % (args.auto_fade, playwindow.FADE_S))
                if fader is not None:
                    fader.fade(seconds=playwindow.FADE_S)
                else:
                    player.stop()
            if done.update(u.committed) == FADE:
                _log("FADE done with the toilet, 2s out")
                window.ended()
                if fader is not None:
                    fader.fade(seconds=2.0)
                else:
                    player.stop()
            action = latch.update(u.committed, ru.room, t)
            dry = " (dry run, no audio)" if args.dry_run else ""
            if action == QUEUE:
                # WARM THE AIRPLAY SESSION WHILE THEY WALK IN. Scanning and
                # connecting cost 0.46s against this HomePod, and they used
                # to be spent after the body had already sat down. Nothing
                # about them needs the music. Returns immediately; the sit
                # then only has to start the stream.
                if fader is not None and fader.warm(args.homepod):
                    _log("QUEUE body in room, warming the session%s" % dry)
                if args.queue_media:
                    _log("QUEUE body in room -> entry sound at %d%%%s"
                         % (args.volume, dry))
                    player.start(args.queue_media)
                elif fader is None:
                    _log("QUEUE body in room (entry sound disabled)")
            elif action == PLAY:
                _log("PLAY  sat down (standing->sitting) -> main track from "
                     "the top at %d%%%s" % (args.volume, dry))
                player.stop()
                window.start(t)
                if fader is not None:
                    # one pyatv session, volume set before the first sample, and
                    # the same object fades it when the toilet is done with.
                    # "warm" means the entry already paid for scan+connect.
                    how = fader.start(args.media, args.homepod,
                                      volume=args.volume)
                    if how == "cold":
                        _log("      (cold session: connecting now, "
                             "the entry warm-up did not land)")
                else:
                    player.start(args.media)
            elif action == UNQUEUE:
                _log("UNQUEUE occupancy ended with no sit%s"
                     % (", stopping entry sound" if player.running() else ""))
                player.stop()
            elif action == STOP:
                _log("STOP  the played visit ended%s; latch holds until the "
                     "next entry"
                     % (", stopping player" if player.running() else ""))
                window.ended()
                player.stop()
                if fader is not None:
                    fader.stop()
            elif action == REARM:
                _log("REARM new entry, latch cleared for this visit")
    except KeyboardInterrupt:
        _log("interrupted; stopping")
        player.stop()
        if fader is not None:
            fader.stop()
        return 0
    except ConnectionError as e:
        _log("stream lost: %s" % e)
        player.stop()
        return 1
    return 0


def _waiting(attempt, total):
    if attempt == 1 or attempt % 10 == 0:
        _log("waiting for %s:%d ... %d/%d"
             % (stream.HOST, stream.PORT, attempt, total))


if __name__ == "__main__":
    raise SystemExit(main())
