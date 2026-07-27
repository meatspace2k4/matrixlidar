"""IR receiver: the kernel decodes, this records what it decoded.

Same part and the same path as the ir ipad bridge on aircheck-pi: a TSOP38238
on a GPIO pin, the `gpio-ir` device-tree overlay, and rc-core doing the NEC
decode in kernel space. Nothing here touches a GPIO or times a pulse. The
kernel hands over a finished scancode on an input device and this module does
three things with it: find that device, split the scancode into the NEC
address and command, and hold the events until a recorder drains them.

  from bathroom import ir
  rx = ir.Receiver()
  rx.start()                 # False if there is no receiver; see rx.source
  ...
  for ev in rx.drain(t0):    # [{t, ts, address, command, scancode, repeat}]

WHY THE ADDRESS IS THE POINT

A NEC frame is an address and a command. The address says which device the
remote is talking to and is stable across every button on that remote; the
command says which button. A log of commands alone cannot tell two remotes
apart, which is the question a bathroom recording is going to be asked.

FINDING THE DEVICE IS NOT "event0"

The bridge hardcodes IR_DEVICE=/dev/input/event0 and gets away with it. Here
that would read an HDMI connector: bath-pi enumerates four input devices
before any IR exists (vc4-hdmi-0, its ALSA jack, vc4-hdmi-1, its jack), and
two of them sit under /sys/class/rc as CEC receivers, so even "it is an rc
device" is not enough of a test. The discriminator is the protocol list:
a CEC endpoint offers [cec] and an IR receiver offers nec / rc-5 / and
friends. See `find_device`.

NOTHING HERE IS REQUIRED

Every entry point fails soft and says why in `source`. A recording made from
a machine with no receiver wired to it -- a laptop, or a Pi whose overlay is
not loaded -- gets an empty IR log and a stated reason, never a crash and
never a fabricated event.
"""
import os
import re
import struct
import threading
import time

from collections import deque

# struct input_event: struct timeval (two longs), __u16 type, __u16 code,
# __s32 value. Same unpack the bridge uses, on the same 64-bit ARM kernel.
EVENT_FORMAT = "llHHi"
EVENT_SIZE = struct.calcsize(EVENT_FORMAT)

EV_MSC = 0x04
MSC_SCAN = 0x04

# A held NEC button repeats about every 110ms. Anything closer than this to
# the previous frame of the SAME scancode is the remote still being held, not
# a second press. Recorded either way (see the module docstring: the log is
# what arrived), but flagged, because "12 presses" and "one long press" are
# very different readings of the same twelve lines.
REPEAT_WINDOW_S = 0.25

QUEUE_MAX = 4096    # a drained queue never approaches this; the cap exists so
                    # an undrained receiver leaks nothing worse than 4096 dicts

INPUT_DEVICES = "/proc/bus/input/devices"
RC_CLASS = "/sys/class/rc"

_SYSFS_RC = re.compile(r"/rc/(rc\d+)")
_EVENT_HANDLER = re.compile(r"\b(event\d+)\b")


def split_scancode(scancode):
    """(address, command) from an rc-core NEC scancode.

    rc-core packs every NEC width the same way: the command is the low byte
    and everything above it is the address. 0x7902 is address 0x79 command
    0x02 (8-bit NEC); extended NEC just makes the address 16 bits wide and
    NEC32 24, and the split does not move. Yamaha's remote in the bridge's
    map reads 0x79/0x7A, which is that same low-byte rule.
    """
    return scancode >> 8, scancode & 0xFF


def _blocks(text):
    """/proc/bus/input/devices split into per-device blocks of {letter: [lines]}."""
    out = []
    cur = {}
    for line in text.splitlines():
        if not line.strip():
            if cur:
                out.append(cur)
            cur = {}
            continue
        if len(line) > 2 and line[1] == ":":
            cur.setdefault(line[0], []).append(line[3:].strip())
    if cur:
        out.append(cur)
    return out


def rc_protocols(rc_name, root=RC_CLASS):
    """The protocol list of /sys/class/rc/<rc_name>, e.g. ['nec'] or ['cec'].

    The enabled ones are bracketed in that file ("rc-5 [nec] sony"); the
    brackets are stripped here and the enabled set is reported separately by
    `enabled_protocols`.
    """
    try:
        with open(os.path.join(root, rc_name, "protocols")) as f:
            raw = f.read().split()
    except OSError:
        return []
    return [p.strip("[]") for p in raw]


def enabled_protocols(rc_name, root=RC_CLASS):
    """Just the bracketed (enabled) protocols. Empty means the kernel will
    hand over nothing: the receiver is there and no decoder is running on it,
    which is what `ir-keytable -p nec` turns on."""
    try:
        with open(os.path.join(root, rc_name, "protocols")) as f:
            raw = f.read().split()
    except OSError:
        return []
    return [p[1:-1] for p in raw if p.startswith("[") and p.endswith("]")]


def find_device(devices_path=INPUT_DEVICES, rc_root=RC_CLASS):
    """(event_path, info) for the IR receiver, or (None, info) with a reason.

    Walks the input device table rather than trusting an event number: the
    numbering depends on what else the kernel enumerated first, and on this
    Pi that is four HDMI devices. A candidate must be an rc-core device whose
    protocol list is something other than bare CEC.
    """
    info = {"searched": devices_path}
    try:
        with open(devices_path) as f:
            text = f.read()
    except OSError as e:
        info["reason"] = "cannot read %s: %s" % (devices_path, e)
        return None, info

    rejected = []
    for b in _blocks(text):
        name = (b.get("N") or [""])[0]
        name = name[6:-1] if name.startswith('Name="') and name.endswith('"') else name
        handlers = " ".join(b.get("H") or [])
        sysfs = (b.get("S") or [""])[0]
        sysfs = sysfs[6:] if sysfs.startswith("Sysfs=") else sysfs
        m_ev = _EVENT_HANDLER.search(handlers)
        m_rc = _SYSFS_RC.search(sysfs)
        if not m_ev or not m_rc:
            continue
        rc_name = m_rc.group(1)
        protos = rc_protocols(rc_name, rc_root)
        if protos and set(protos) <= {"cec"}:
            rejected.append("%s (%s: cec)" % (name, rc_name))
            continue
        return "/dev/input/" + m_ev.group(1), {
            "searched": devices_path, "name": name, "rc": rc_name,
            "protocols": protos, "enabled": enabled_protocols(rc_name, rc_root),
        }

    info["reason"] = ("no IR receiver among the input devices"
                      + (" (rejected: %s)" % ", ".join(rejected) if rejected
                         else ""))
    return None, info


class Receiver:
    """Reads decoded IR scancodes off the kernel in a background thread.

    The read is blocking, so it lives on its own thread and the events land in
    a queue the recorder drains once per frame. The frame loop must never wait
    on a remote nobody pressed.
    """

    __slots__ = ("device", "source", "_fd", "_thread", "_q", "_stop",
                 "_last_code", "_last_ts")

    def __init__(self, device=None):
        # An explicit device beats discovery, and IR_DEVICE beats both -- the
        # bridge sets exactly that variable in its unit file.
        self.device = device or os.environ.get("IR_DEVICE") or None
        self.source = {"device": self.device, "state": "unstarted"}
        self._fd = None
        self._thread = None
        self._q = deque(maxlen=QUEUE_MAX)
        self._stop = threading.Event()
        self._last_code = None
        self._last_ts = 0.0

    def start(self):
        """Open the receiver and start reading. False (never an exception) if
        there is nothing to open; `source` then carries the reason."""
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
                data = os.read(fd, EVENT_SIZE)
            except OSError:
                break              # closed under us by close(), or unplugged
            if not data:
                break              # EOF: the device went away (and a plain
                                   # file, which is how the tests feed this,
                                   # ends rather than blocking). Returning to
                                   # the read would spin on nothing.
            if len(data) < EVENT_SIZE:
                continue
            sec, usec, ev_type, ev_code, ev_value = struct.unpack(
                EVENT_FORMAT, data)
            if ev_type != EV_MSC or ev_code != MSC_SCAN:
                continue           # EV_KEY/EV_SYN of the same press, or noise
            # The timeval is the kernel's own CLOCK_REALTIME stamp for the
            # frame. Use it rather than "when this thread woke up": the wake
            # can be late and the recording is meant to say when the signal
            # arrived, not when Python noticed.
            ts = sec + usec / 1e6
            scancode = ev_value & 0xFFFFFFFF
            address, command = split_scancode(scancode)
            repeat = (scancode == self._last_code
                      and ts - self._last_ts <= REPEAT_WINDOW_S)
            self._last_code, self._last_ts = scancode, ts
            self._q.append({
                "ts": round(ts, 6),
                "scancode": scancode,
                "hex": "0x%04X" % scancode,
                "address": address,
                "address_hex": "0x%02X" % address,
                "command": command,
                "repeat": repeat,
            })

    def drain(self, t0=None):
        """Every event since the last drain, oldest first.

        With `t0` (the recording's wall-clock start) each event also gets `t`,
        seconds into the recording, on the same axis as the frames.
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
        enabled = source.get("enabled")
        return "IR %s (%s%s)" % (
            source.get("device"), source.get("name", "?"),
            ", decoding %s" % "/".join(enabled) if enabled
            else ", NO decoder enabled: run ir-keytable -p nec")
    return "IR unavailable (%s): %s" % (state, source.get("reason", "-"))


def format_event(ev):
    """One line for a human, for the replay printer and the recorder alike."""
    return "addr=%s cmd=0x%02X scancode=%s%s" % (
        ev.get("address_hex", "0x%02X" % ev.get("address", 0)),
        ev.get("command", 0), ev.get("hex", "?"),
        " (repeat)" if ev.get("repeat") else "")


def _main():
    """python3 -m bathroom.ir -- print what the receiver sees, to check wiring."""
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
                print("  t=%7.2fs  %s" % (ev["t"], format_event(ev)), flush=True)
            time.sleep(0.05)
    except KeyboardInterrupt:
        return 0
    finally:
        rx.close()


if __name__ == "__main__":
    raise SystemExit(_main())
