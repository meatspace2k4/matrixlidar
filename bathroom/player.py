"""Play a file on the HomePod at a known volume, in one session.

    asyncio.run(play("/path/track.mp3", "192.168.10.240", volume=20))

WHY THIS EXISTS

The volume is set BEFORE stream_file, on the same connection, which is the case
pyatv is written for. From pyatv/protocols/raop, inside stream_file:

    if not self.audio.has_changed_volume and "initialVolume" in client.info:
        context.volume = client.info["initialVolume"]
    else:
        ...                      # use the volume the caller already set

So a session that has had set_volume called on it streams at that volume from
the first sample. There is no window during which the audio is playing at the
wrong level, so there is nothing to poll for and nothing to time.

The daemon this replaces spawned `atvremote ... stream_file` as one subprocess
and `atvremote ... set_volume` as another. Two processes are two pyatv sessions,
and the second cannot touch the first's volume until that session exists and is
past its handshake. That is unknowable from outside, so it slept 1.0s and then
retried set_volume up to 8 times, 15s timeout each, continuing to fire after one
succeeded because a set accepted mid-handshake could still be overridden. All of
that is the cost of using two processes where one session does the job.
"""
import asyncio
import time


def _pyatv():
    """pyatv, imported on use.

    The washlet unit runs /usr/bin/python3 and pyatv is installed in
    ~/washlet/venv, so importing it at module scope would crash the daemon on
    import the moment anything in the package imports this module. Imported here
    instead, so the failure lands on the caller that actually needs it, with a
    message naming the interpreter.
    """
    try:
        import pyatv
        from pyatv.const import Protocol
        return pyatv, Protocol
    except ImportError as e:
        raise PlayError(
            "pyatv is not importable by this interpreter (%s): %s. The unit's "
            "ExecStart must be the venv python that has it."
            % (sys.executable, e))


import sys


CONNECT_WAIT_S = 6.0   # how long a fade waits for the session to come up


class PlayError(Exception):
    """The HomePod could not be reached or the track could not be streamed."""


async def _connect(address, loop, timeout):
    pyatv, Protocol = _pyatv()
    confs = await pyatv.scan(loop, hosts=[address], timeout=timeout)
    if not confs:
        raise PlayError("no AirPlay device at %s" % address)
    conf = confs[0]
    if not conf.get_service(Protocol.RAOP):
        raise PlayError("%s advertises no RAOP service" % address)
    return await pyatv.connect(conf, loop)


async def ramp_volume(atv, frm, to, seconds, steps=20):
    """Walk the volume from `frm` to `to` over `seconds`, inclusive of both
    ends. One implementation for every volume move, so a fade down and a
    lift up cannot drift apart."""
    for i in range(steps + 1):
        await atv.audio.set_volume(frm + (to - frm) * i / steps)
        await asyncio.sleep(seconds / steps)


async def fade_out(atv, from_volume, seconds=2.0, steps=20):
    """Ramp the volume to zero over `seconds`, then stop."""
    await ramp_volume(atv, from_volume, 0, seconds, steps)


async def play(path, address, volume=20, timeout=5, on_start=None,
               on_connect=None):
    """Stream `path` to `address` at `volume`, returning when it finishes.

    `on_connect(atv)` is called with the live connection before streaming, which
    is how a fade reaches the session that is actually playing. Without it the
    fade has nothing to set the volume on.

    Cancel the task to stop playback: stream_file is cancellable and the
    connection is closed on the way out either way.
    """
    loop = asyncio.get_running_loop()
    atv = await _connect(address, loop, timeout)
    if on_connect:
        on_connect(atv)
    try:
        # BEFORE stream_file, on this connection. This is the whole point.
        await atv.audio.set_volume(float(volume))
        if on_start:
            on_start()
        await atv.stream.stream_file(path)
    finally:
        atv.close()
        await asyncio.sleep(0)      # let the transport close cleanly


class Playback:
    """A single play, startable and stoppable from synchronous code.

    The daemon's loop is synchronous and must not block on a track that runs for
    minutes, so the stream owns a thread with its own event loop, and stop()
    cancels it.
    """

    __slots__ = ("_thread", "_loop", "_task", "_done", "_atv", "_volume",
                 "_warm_addr", "_parked")

    def __init__(self):
        self._thread = None
        self._loop = None
        self._task = None
        self._done = None
        self._atv = None
        self._volume = 20
        self._warm_addr = None   # address the held-open session is against
        self._parked = False     # the loop is in run_forever (warm), not
                                 # run_until_complete (cold): they end
                                 # differently, so stop() must know which

    # --- the warm session -------------------------------------------------
    #
    # Scanning and connecting cost 0.46s against this HomePod (0.09 scan,
    # 0.37 connect, measured on the Pi), and they used to be spent AFTER the
    # body had already sat down. Nothing about them needs the music: they can
    # be spent while the person is still walking in, so the sit only has to
    # start the stream.

    def warm(self, address, timeout=5):
        """Open the session ahead of the music. Returns immediately.

        Idempotent for the same address, and a no-op while something is
        playing (the session in use is already warm). Failure is reported and
        left alone: the next `start` simply connects the old way, which is
        the same work in the same place it always happened.
        """
        if self.is_playing():
            return False
        if self._atv is not None and self._warm_addr == address:
            return True
        self.stop()
        self._spawn_loop()
        loop = self._loop

        def _took(f):
            try:
                self._atv = f.result()
                self._warm_addr = address
            except Exception as e:
                sys.stderr.write("warm: %s\n" % e)

        asyncio.run_coroutine_threadsafe(
            _connect(address, loop, timeout), loop).add_done_callback(_took)
        return True

    def is_warm(self, address=None):
        return (self._atv is not None
                and (address is None or self._warm_addr == address))

    def _spawn_loop(self):
        """A thread with a PARKED event loop, and nothing on it yet."""
        import threading
        ready = threading.Event()

        def run():
            # the loop is held in a LOCAL: stop() clears self._loop, and this
            # thread must still be able to close the one it owns
            loop = asyncio.new_event_loop()
            self._loop = loop
            self._parked = True
            asyncio.set_event_loop(loop)
            ready.set()
            loop.run_forever()
            loop.close()

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()
        ready.wait(2.0)

    def start(self, path, address, volume=20, on_start=None):
        import threading

        # WARM PATH: the session is already open, so this is only the stream.
        if self._atv is not None and self._warm_addr == address and self._loop:
            self._volume = volume
            self._done = threading.Event()
            atv = self._atv

            async def stream_warm():
                try:
                    await atv.audio.set_volume(float(volume))
                    if on_start:
                        on_start()
                    await atv.stream.stream_file(path)
                finally:
                    try:
                        atv.close()
                    except Exception:
                        pass
                    self._atv = None
                    self._warm_addr = None
                    self._done.set()

            self._task = asyncio.run_coroutine_threadsafe(stream_warm(),
                                                          self._loop)
            return "warm"

        # COLD PATH: scan, connect, then stream, exactly as before.
        self.stop()
        self._done = threading.Event()

        def run():
            loop = asyncio.new_event_loop()   # local: see _spawn_loop
            self._loop = loop
            self._parked = False
            asyncio.set_event_loop(loop)
            self._volume = volume
            self._task = loop.create_task(
                play(path, address, volume=volume, on_start=on_start,
                     on_connect=self._took_connection))
            try:
                loop.run_until_complete(self._task)
            except (asyncio.CancelledError, PlayError, Exception):
                pass
            finally:
                self._atv = None
                self._warm_addr = None
                loop.close()
                self._done.set()

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()
        return "cold"

    def _took_connection(self, atv):
        self._atv = atv

    def _cancel_task(self):
        """Cancel whatever start() left behind: an asyncio.Task on the cold
        path, a concurrent Future from run_coroutine_threadsafe on the warm
        one. Both cancel the same coroutine; only the call differs."""
        task, self._task = self._task, None
        if task is None:
            return
        try:
            if isinstance(task, asyncio.Task):
                if not task.done() and self._loop:
                    self._loop.call_soon_threadsafe(task.cancel)
            else:
                task.cancel()
        except Exception:
            pass

    def is_playing(self):
        return bool(self._done) and not self._done.is_set()

    def fade(self, seconds=2.0, timeout=5.0):
        """Fade the track out over `seconds`, then stop it.

        Waits for the connection first. `start` returns before the scan and
        connect finish, so a fade arriving early found `_atv` unset, skipped the
        ramp, and cut: measured at exactly 5.00s, which is stop()'s join
        timeout, not a fade. Being silent about that is what hid it twice.
        """
        deadline = time.time() + CONNECT_WAIT_S
        while self._atv is None and self.is_playing() and time.time() < deadline:
            time.sleep(0.05)
        if self._atv is None:
            sys.stderr.write("fade: no session after %.1fs; cutting\n"
                             % CONNECT_WAIT_S)
            self.stop(timeout)
            return
        fut = asyncio.run_coroutine_threadsafe(
            fade_out(self._atv, self._volume, seconds), self._loop)
        try:
            fut.result(timeout=seconds + 2)
        except Exception as e:
            sys.stderr.write("fade: ramp failed (%s); cutting\n" % e)
        self.stop(timeout)

    def set_volume(self, volume, seconds=0.5):
        """Ramp the playing session to `volume` over `seconds`.

        A ramp, not a jump: a step change on a quiet track in a small tiled
        room is startling, and the rest of this daemon's volume moves are
        ramps already.

        Does NOT wait for the ramp: the caller is the frame loop, and
        blocking it for half a second drops sensor frames. The coroutine runs
        on the playback thread's loop and reports its own failure. `_volume`
        is updated up front, so a fade arriving mid-ramp still ramps down
        from the level being aimed at rather than the one left behind.

        False means there was no session to ramp (playback already over, or
        the connect has not landed yet); the volume is remembered either way,
        so the next play starts where it was asked to.
        """
        frm, self._volume = self._volume, volume
        atv, loop = self._atv, self._loop
        if atv is None or loop is None:
            sys.stderr.write("set_volume: no live session\n")
            return False
        fut = asyncio.run_coroutine_threadsafe(
            ramp_volume(atv, frm, volume, seconds), loop)

        def _report(f):
            exc = f.exception()
            if exc is not None:
                sys.stderr.write("set_volume: ramp failed (%s)\n" % exc)

        fut.add_done_callback(_report)
        return True

    def stop(self, timeout=5.0):
        """Cancel the stream, drop the session, and tear the thread down.

        A warm session is torn down too: stop means stop, and a connection
        held open past the end of a visit is one the receiver may have closed
        under us anyway. The next warm re-opens it.

        The cancelled coroutine is given a moment to UNWIND before the loop
        is stopped. Stopping first strands its finally clause, which is where
        the receiver connection is closed: the fake-receiver test caught
        exactly that, reporting no stream_cancelled and a pending task at
        interpreter exit.
        """
        parked, loop, done = self._parked, self._loop, self._done
        self._cancel_task()
        if done is not None and loop is not None and loop.is_running():
            # Wait on the coroutine's OWN completion event, which its finally
            # sets. Waiting on the concurrent Future instead does not work:
            # cancel() marks that Future cancelled immediately and returns
            # while the coroutine is still unwinding, so the loop got stopped
            # out from under the close.
            done.wait(min(timeout, 2.0))
        self._loop = None
        if parked and loop is not None and loop.is_running():
            loop.call_soon_threadsafe(loop.stop)   # cold path ends by itself
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout)
        self._thread = None
        self._parked = False
        self._atv = None
        self._warm_addr = None


def available():
    """Can this interpreter drive the HomePod? No fallback, just the answer."""
    try:
        _pyatv()
        return True
    except PlayError:
        return False
