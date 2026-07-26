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


async def fade_out(atv, from_volume, seconds=2.0, steps=20):
    """Ramp the volume to zero over `seconds`, then stop."""
    for i in range(steps, -1, -1):
        await atv.audio.set_volume(from_volume * i / steps)
        await asyncio.sleep(seconds / steps)


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

    __slots__ = ("_thread", "_loop", "_task", "_done", "_atv", "_volume")

    def __init__(self):
        self._thread = None
        self._loop = None
        self._task = None
        self._done = None
        self._atv = None
        self._volume = 20

    def start(self, path, address, volume=20, on_start=None):
        import threading
        self.stop()
        self._done = threading.Event()

        def run():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._volume = volume
            self._task = self._loop.create_task(
                play(path, address, volume=volume, on_start=on_start,
                     on_connect=self._took_connection))
            try:
                self._loop.run_until_complete(self._task)
            except (asyncio.CancelledError, PlayError, Exception):
                pass
            finally:
                self._atv = None
                self._loop.close()
                self._done.set()

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

    def _took_connection(self, atv):
        self._atv = atv

    def is_playing(self):
        return bool(self._done) and not self._done.is_set()

    def fade(self, seconds=2.0, timeout=5.0):
        """Fade the track out over `seconds` and stop it."""
        if self._loop and self._atv is not None:
            fut = asyncio.run_coroutine_threadsafe(
                fade_out(self._atv, self._volume, seconds), self._loop)
            try:
                fut.result(timeout=seconds + 2)
            except Exception:
                pass
        self.stop(timeout)

    def stop(self, timeout=5.0):
        if self._loop and self._task and not self._task.done():
            self._loop.call_soon_threadsafe(self._task.cancel)
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout)
        self._thread = self._loop = self._task = None


def available():
    """Can this interpreter drive the HomePod? No fallback, just the answer."""
    try:
        _pyatv()
        return True
    except PlayError:
        return False
