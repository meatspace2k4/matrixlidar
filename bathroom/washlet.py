"""Washlet jukebox daemon: entry sound on walk-in, the track on sit.

  python3 -m bathroom.washlet             # live
  python3 -m bathroom.washlet --dry-run   # log decisions, no audio

Consumes the same stream -> StateEngine -> OccupancyEngine stack as
bathroom.live. The main track (--media) plays on sit, once per session.
An OPTIONAL entry sound (--queue-media) can play on walk-in; it is
DISABLED by default (empty), and with it disabled nothing plays on entry.
Decisions:

- QUEUE: the room becomes OCCUPIED with the latch clear (including a body
  already present at daemon start) -> stream the entry sound at --volume
  if one is configured, otherwise no audio (the machinery still runs).
- PLAY: committed SITTING_AT_TOILET holds --hold seconds -> kill whatever
  is streaming and start the MAIN track from the top at --volume. Happens
  once per session: the latch blocks replays on continued sitting,
  stand-up/sit-down cycles, and the track ending.
- UNQUEUE: occupancy ends with no sit (entry aborted, stream gap, or exit)
  -> stop the entry sound.
- REARM (exit condition): room EMPTY while posture also reads EMPTY -> stop
  playback, clear the latch. The posture guard means a room misread under a
  seated body can never rearm-and-replay; a gap (UNKNOWN) holds the latch,
  and a latched (audible) stream is never stopped by a mere gap.

Every stream start gets the volume poke (a second atvremote, retried
across the startup window because a set accepted mid-handshake can be
overridden when audio starts); the session's own starting volume is
receiver state we do not trust. Both media files must exist at startup
(fail loud, no fallbacks). State is in-memory only; a daemon restart
forgets the latch (worst case: one extra play into an occupied session).
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time

from . import archive, classify, config, detect, occupancy, player as player_mod, posture3d, scene as scene_mod, stream, threshold
from .donewith import DoneWith, FADE
from .engine import StateEngine
from .occupancy import RoomState
from .zones import State

SIT_HOLD_S = 2.0    # SITTING must be OBSERVED for this long before the track
                    # fires. This is the debounce, not a delay: a false sit that
                    # lasts under 2s never reaches the audio and never spends the
                    # one-shot latch. A false SITTING inside a stand ran 1.1s and
                    # burned the latch 45 seconds before the real sit; at
                    # hold=0.0 the first committed frame fires, so a single
                    # misread costs the whole visit. Dwell only accumulates on
                    # frames actually observed as SITTING (see _sit_time), so a
                    # seated body that jitters through STANDING keeps its clock
                    # rather than restarting it.
                    #
                    # was 0.0: fire the moment the posture commits to
                    # SITTING. The engine's commit_after already requires
                    # several consecutive frames, so this is not one noisy frame.
VOLUME = 20         # HomePod volume % applied as soon as the session accepts it
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
    """Room-queued one-shot sit trigger with an exit latch (pure, no I/O).

    update() consumes (committed posture, room state, t) once per frame and
    returns QUEUE (occupancy began: entry sound), PLAY (sit held: main
    track, once), UNQUEUE (occupancy evaporated with no sit), REARM (exit
    observed, latch cleared), or None.
    """

    __slots__ = ("hold", "latched", "_sit_time", "_prev_room", "_last_t")

    def __init__(self, hold=SIT_HOLD_S):
        self.hold = hold
        self.latched = False
        self._sit_time = 0.0     # seconds actually OBSERVED as committed SITTING
        self._prev_room = None
        self._last_t = None

    def update(self, committed, room, t):
        prev, self._prev_room = self._prev_room, room

        # The dwell counts seconds OBSERVED as sitting. UNKNOWN NEVER touches
        # it: UNKNOWN means "cannot tell", not "the body left", and treating a
        # gap as evidence of absence is what lost real sits. On 07-25 someone
        # sat for six minutes reading SITTING 1.5s, UNKNOWN, SITTING 1.6s,
        # UNKNOWN, and a dwell that restarted on every gap never reached 2.0s.
        #
        # There is no cap on how long a gap may last. A dwell that survives a
        # long gap costs nothing: the track is one-shot per occupancy, so the
        # worst case is one play into a session that already has an occupant,
        # and the room latch clears on exit anyway. Losing a real sit costs the
        # entire feature.
        dt = 0.0 if self._last_t is None else max(0.0, t - self._last_t)
        self._last_t = t
        if committed == State.SITTING_AT_TOILET:
            self._sit_time += dt
        elif committed != State.UNKNOWN:
            # DRAIN, do not reset. A single non-sitting frame used to zero the
            # clock, so a seated body that flickers to STANDING every ninth
            # frame never reached the dwell and the track never played. Draining
            # at the same rate it fills means a real sit nets positive through
            # jitter, and a stand that misreads as sitting one frame in three
            # nets negative and never fires.
            #
            #   1.1s false sit inside a stand                nothing
            #   a real sit, held                             PLAY at 3.4s
            #   a real sit jittering 1 frame in 9            PLAY at 3.6s
            #   a real sit jittering 1 frame in 3            PLAY at 7.0s
            #   a stand misreading as sitting 1 frame in 3   nothing
            self._sit_time = max(0.0, self._sit_time - dt)
        # leaving the room is the only thing that wipes the slate
        if room != RoomState.OCCUPIED:
            self._sit_time = 0.0

        # Exit condition: BOTH layers must agree the room is vacated. Room
        # EMPTY alone must not rearm (a misread under a still-seated body
        # would rearm-and-replay every frame); posture EMPTY alone must not
        # rearm (a body out of FOV at the sink/shower is still in the room).
        if (self.latched and room == RoomState.EMPTY
                and committed == State.EMPTY):
            self.latched = False
            return REARM

        if not self.latched:
            # committed SITTING is required, not just a satisfied dwell: with
            # hold=0 the dwell test alone is true on the first frame of any
            # posture, which fired the track at someone walking past.
            if (committed == State.SITTING_AT_TOILET
                    and self._sit_time >= self.hold):
                self.latched = True
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
        description="Sit-triggered one-shot AirPlay daemon (latched until exit)")
    ap.add_argument("--hold", type=float, default=SIT_HOLD_S,
                    help="seconds committed SITTING must hold before firing")
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
                    help="log PLAY/REARM decisions without starting audio")
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
    latch = SitLatch(hold=args.hold)
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
    fader = None
    if not args.dry_run:
        if player_mod.available():
            fader = player_mod.Playback()
        else:
            _log("FADE UNAVAILABLE: pyatv not importable by %s. The main track "
                 "will cut instead of fading. Point the unit's ExecStart at "
                 "~/washlet/venv/bin/python." % sys.executable)
    done = DoneWith()

    _log("washlet daemon up: posture=%s hold=%.1fs vol=%d homepod=%s media=%s queue=%s%s"
         % (args.posture, args.hold, args.volume, args.homepod, args.media,
            args.queue_media or "(disabled)",
            " (DRY RUN)" if args.dry_run else ""))
    t0 = None
    last_pub = None
    last_pub_t = 0.0
    try:
        for m in stream.frames(host=args.stream_host, port=args.stream_port,
                               on_wait=_waiting):
            now = time.monotonic()
            t0 = t0 if t0 is not None else now
            t = now - t0
            u = eng.update(m, t)
            cross = crossing.update(m, sc, t)
            ru = room_eng.update(u, t, door_pct=detect.door_open_pct(m),
                                 crossed_out=(cross == threshold.EXIT))

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
                     "event": ru.event, "sit_s": round(latch._sit_time, 2)}))

            rc = player.reap()
            if rc is not None:
                _log("player ended rc=%d" % rc)

            marker.update(u, ru, t)
            if done.update(u.committed) == FADE:
                _log("FADE done with the toilet, 2s out")
                if fader is not None:
                    fader.fade(seconds=2.0)
                else:
                    player.stop()
            action = latch.update(u.committed, ru.room, t)
            dry = " (dry run, no audio)" if args.dry_run else ""
            if action == QUEUE:
                if args.queue_media:
                    _log("QUEUE body in room -> entry sound at %d%%%s"
                         % (args.volume, dry))
                    player.start(args.queue_media)
                else:
                    _log("QUEUE body in room (entry sound disabled)")
            elif action == PLAY:
                _log("PLAY  sit held %.1fs -> main track from the top at %d%%%s"
                     % (args.hold, args.volume, dry))
                player.stop()
                if fader is not None:
                    # one pyatv session, volume set before the first sample, and
                    # the same object fades it when the toilet is done with
                    fader.start(args.media, args.homepod, volume=args.volume)
                else:
                    player.start(args.media)
            elif action == UNQUEUE:
                _log("UNQUEUE occupancy ended with no sit%s"
                     % (", stopping entry sound" if player.running() else ""))
                player.stop()
            elif action == REARM:
                _log("REARM exit observed, latch cleared%s"
                     % (", stopping player" if player.running() else ""))
                player.stop()
                if fader is not None:
                    fader.stop()
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
