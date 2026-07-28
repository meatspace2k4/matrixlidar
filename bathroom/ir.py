"""IR receiver: fingerprint the raw pulse train, decode nothing.

The bathroom remote is a toilet remote. It shares the NEC family's bit
TIMING (~560us pulses, ~560us/~1690us spaces) but not the NEC frame: wrong
bit count, no address/inverse-address structure, so the kernel's protocol
decoders (built for TV remotes) validate every burst, fail, and emit
nothing. Scancodes do not exist for this remote and never will.

So this module does not ask the kernel to understand the signal. It reads
the raw demodulated stream from /dev/lirc0 (LIRC_MODE2: one u32 per pulse,
space, or timeout) and reduces each frame to a FINGERPRINT:

  - a frame is the pulse/space run between quiet gaps (>= FRAME_GAP_US or
    the receiver's own timeout marker)
  - within a frame, only the SPACES carry information in this timing
    family (pulses are all ~560us); each space quantizes to a bit,
    short=0 long=1
  - the fingerprint is the bit count plus the bits packed big-endian into
    hex: "84:0d0a0300c380440107..."

Same button -> same fingerprint, different button -> different fingerprint,
which is everything recording and replay need. What the bits MEAN is the
remote's business.

No root anywhere: /dev/lirc0 is group video and the daemon user is in it.
No ir-keytable, no protocol enabling, no keymaps. The gpio-ir overlay in
config.txt is the only system dependency.

  python3 -m bathroom.ir      # print fingerprints live, to check wiring

Every entry point fails soft and says why in `source`: a machine with no
receiver records lidar exactly as before, and the clip states the reason
instead of carrying an empty lie.
"""
import os
import struct
import threading
import time

from collections import deque

# LIRC_MODE2 stream: u32 little-endian, top byte the kind, low 24 bits usec.
MODE2_SPACE = 0x00
MODE2_PULSE = 0x01
MODE2_FREQ = 0x02
MODE2_TIMEOUT = 0x03

FRAME_GAP_US = 10000   # a space this long ends a frame. Frames within a held
                       # press arrive ~20-45ms apart; intra-frame spaces top
                       # out ~1.7ms. An order of magnitude sits between.
BIT_SPLIT_US = 1100    # short/long space boundary: the observed clusters are
                       # ~560us and ~1690us, this is the midpoint gap.
MIN_FRAME_BITS = 8     # anything shorter is a glitch, not a button

# A held button re-sends its frame every few tens of ms. Same fingerprint
# within this window is the button still down, not a second press. Logged
# either way, flagged, because "12 presses" and "one long press" are very
# different readings of the same twelve lines.
REPEAT_WINDOW_S = 0.25

QUEUE_MAX = 4096       # an undrained receiver leaks nothing worse than this

RC_CLASS = "/sys/class/rc"


def fingerprint(spaces):
    """bits:hex fingerprint from a frame's intra-frame space durations."""
    bits = ["1" if us > BIT_SPLIT_US else "0" for us in spaces]
    n = len(bits)
    if n < MIN_FRAME_BITS:
        return None
    val = int("".join(bits), 2)
    return "%d:%0*x" % (n, (n + 3) // 4, val)


class FrameAssembler:
    """Feed it mode2 words; it yields one fingerprint per completed frame.

    Pure and incremental: a frame split across two os.read() chunks is the
    normal case, not an edge case, so state lives here rather than in the
    read loop.
    """

    __slots__ = ("_spaces", "_saw_pulse", "_tail")

    def __init__(self):
        self._spaces = []
        self._saw_pulse = False
        self._tail = b""    # partial trailing word from the previous feed

    def feed(self, data):
        """Consume raw bytes (any split), return fingerprints completed."""
        out = []
        data = self._tail + data
        cut = len(data) & ~3
        data, self._tail = data[:cut], data[cut:]
        for (word,) in struct.iter_unpack("<I", data):
            kind, usec = word >> 24, word & 0xFFFFFF
            if kind == MODE2_PULSE:
                self._saw_pulse = True
            elif kind == MODE2_SPACE and self._saw_pulse:
                if usec >= FRAME_GAP_US:
                    self._flush(out)
                else:
                    self._spaces.append(usec)
            elif kind == MODE2_TIMEOUT:
                self._flush(out)
            # frequency reports and leading idle spaces carry nothing
        return out

    def _flush(self, out):
        if self._spaces:
            fp = fingerprint(self._spaces)
            if fp is not None:
                out.append(fp)
        self._spaces = []
        self._saw_pulse = False


def find_device(rc_root=RC_CLASS):
    """(lirc_path, info) for the raw IR receiver, or (None, info) with why.

    Walks /sys/class/rc rather than trusting a device number: HDMI CEC
    endpoints are rc devices too. The receiver is the rc device that offers
    more than bare cec and exposes a lircN node.
    """
    info = {"searched": rc_root}
    try:
        names = sorted(os.listdir(rc_root))
    except OSError as e:
        info["reason"] = "cannot read %s: %s" % (rc_root, e)
        return None, info

    rejected = []
    for rc in names:
        try:
            with open(os.path.join(rc_root, rc, "protocols")) as f:
                protos = [p.strip("[]") for p in f.read().split()]
        except OSError:
            continue
        if protos and set(protos) <= {"cec"}:
            rejected.append("%s (cec)" % rc)
            continue
        lirc = [d for d in sorted(os.listdir(os.path.join(rc_root, rc)))
                if d.startswith("lirc")]
        if not lirc:
            rejected.append("%s (no lirc node)" % rc)
            continue
        dev = "/dev/" + lirc[0]
        name = "?"
        try:
            with open(os.path.join(rc_root, rc, "device_name")) as f:
                name = f.read().strip()
        except OSError:
            pass
        return dev, {"searched": rc_root, "rc": rc, "name": name,
                     "protocols": protos}

    info["reason"] = ("no IR receiver among the rc devices"
                      + (" (rejected: %s)" % ", ".join(rejected) if rejected
                         else ""))
    return None, info


class Receiver:
    """Fingerprints raw IR frames off /dev/lirc0 in a background thread.

    The read blocks, so it lives on its own thread and events land in a
    queue the recorder drains once per lidar frame. The frame loop must
    never wait on a remote nobody pressed.
    """

    __slots__ = ("device", "source", "_fd", "_thread", "_q", "_stop",
                 "_asm", "_last_fp", "_last_ts")

    def __init__(self, device=None):
        # An explicit device beats discovery, and IR_DEVICE beats both.
        self.device = device or os.environ.get("IR_DEVICE") or None
        self.source = {"device": self.device, "state": "unstarted"}
        self._fd = None
        self._thread = None
        self._q = deque(maxlen=QUEUE_MAX)
        self._stop = threading.Event()
        self._asm = FrameAssembler()
        self._last_fp = None
        self._last_ts = 0.0

    def start(self):
        """Open the receiver and start reading. False (never an exception)
        if there is nothing to open; `source` then carries the reason."""
        if self.device is None:
            dev, info = find_device()
            self.source.update(info)
            if dev is None:
                self.source["state"] = "absent"
                return False
            self.device = dev
            self.source["device"] = dev
        try:
            self._fd = os.open(self.device, os.O_RDONLY)
        except OSError as e:
            self.source["state"] = "unreadable"
            self.source["reason"] = "%s: %s" % (self.device, e)
            return False
        self.source["state"] = "open"
        self.source.pop("reason", None)
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        return True

    def _read_loop(self):
        fd = self._fd
        while not self._stop.is_set():
            try:
                data = os.read(fd, 4096)
            except OSError:
                break              # closed under us by close(), or unplugged
            if not data:
                break              # EOF: device gone (or a plain file in the
                                   # tests, which ends instead of blocking)
            # The lirc chardev carries durations, not wall times; the frame
            # is stamped when its terminating gap is read, which is within
            # one gap (~tens of ms) of the truth.
            ts = time.time()
            for fp in self._asm.feed(data):
                repeat = (fp == self._last_fp
                          and ts - self._last_ts <= REPEAT_WINDOW_S)
                self._last_fp, self._last_ts = fp, ts
                self._q.append({"ts": round(ts, 6), "fp": fp,
                                "bits": int(fp.split(":", 1)[0]),
                                "repeat": repeat})

    def drain(self, t0=None):
        """Every fingerprint since the last drain, oldest first.

        With `t0` (the recording's wall-clock start) each event also gets
        `t`, seconds into the recording, on the frames' axis.
        """
        out = []
        while True:
            try:
                ev = self._q.popleft()
            except IndexError:
                break
            if t0 is not None:
                ev["t"] = round(ev["ts"] - t0, 3)
            out.append(ev)
        return out

    def close(self):
        self._stop.set()
        fd, self._fd = self._fd, None
        if fd is not None:
            try:
                os.close(fd)       # unblocks the read in the thread
            except OSError:
                pass
        self.source["state"] = "closed"


def describe(source):
    """One line for a human: what the IR side of a recording actually was."""
    state = source.get("state")
    if state == "open":
        return "IR %s (%s, raw fingerprinting)" % (
            source.get("device"), source.get("name", "?"))
    return "IR unavailable (%s): %s" % (state, source.get("reason", "-"))


def format_event(ev):
    """One line for a human, for the replay printer and the recorder alike."""
    return "fp=%s%s" % (ev.get("fp", "?"),
                        " (repeat)" if ev.get("repeat") else "")


def _main():
    """python3 -m bathroom.ir -- print what the receiver sees, live."""
    rx = Receiver()
    ok = rx.start()
    print(describe(rx.source), flush=True)
    if not ok:
        return 1
    print("listening; Ctrl-C to stop", flush=True)
    t0 = time.time()
    try:
        while True:
            for ev in rx.drain(t0):
                print("  t=%7.2fs  %s" % (ev["t"], format_event(ev)),
                      flush=True)
            time.sleep(0.05)
    except KeyboardInterrupt:
        return 0
    finally:
        rx.close()


if __name__ == "__main__":
    raise SystemExit(_main())
