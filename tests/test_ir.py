#!/usr/bin/env python3
"""IR capture regression: scancode -> address, device discovery, and the
round trip from a clip's recorded log to the line the washlet daemon prints.

No IR hardware is involved. The kernel interface is a file of struct
input_event and a text table under /proc, so both are built here as real
files and read through the real code path.

  python3 tests/test_ir.py
"""
import json
import os
import struct
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bathroom import ir                                          # noqa: E402

FAILS = []


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


def event_bytes(sec, usec, ev_type, code, value):
    return struct.pack(ir.EVENT_FORMAT, sec, usec, ev_type, code, value)


# The four HDMI input devices bath-pi enumerates before any IR exists, plus a
# gpio-ir receiver. Verbatim shape of /proc/bus/input/devices.
DEVICES = """\
I: Bus=0019 Vendor=0000 Product=0000 Version=0000
N: Name="vc4-hdmi-0"
P: Phys=vc4-hdmi-0/input0
S: Sysfs=/devices/platform/soc/fef00700.hdmi/rc/rc0/input0
U: Uniq=
H: Handlers=kbd event0
B: PROP=0

I: Bus=0000 Vendor=0000 Product=0000 Version=0000
N: Name="vc4-hdmi-0 HDMI Jack"
P: Phys=ALSA
S: Sysfs=/devices/platform/soc/fef00700.hdmi/sound/card0/input1
U: Uniq=
H: Handlers=event1
B: PROP=0

I: Bus=0019 Vendor=0001 Product=0001 Version=0100
N: Name="gpio_ir_recv"
P: Phys=gpio_ir_recv/input0
S: Sysfs=/devices/platform/ir-receiver@19/rc/rc2/input5
U: Uniq=
H: Handlers=kbd event5
B: PROP=0
"""


def fake_rc(root):
    """/sys/class/rc laid out the way bath-pi and aircheck-pi have it: two
    CEC endpoints from the HDMI blocks, one real receiver with nec enabled."""
    for name, protocols in (("rc0", "[cec]"), ("rc1", "[cec]"),
                            ("rc2", "rc-5 [nec] sony sharp")):
        os.makedirs(os.path.join(root, name))
        with open(os.path.join(root, name, "protocols"), "w") as f:
            f.write(protocols + "\n")
    return root


def main():
    n = 0

    # --- the address is the low-byte split ----------------------------------
    # Yamaha codes from the bridge's own SCANCODE_MAP: address 0x79 for the CD
    # transport, 0x7A for main volume, and one 0x7E.
    for scancode, addr, cmd in ((0x7902, 0x79, 0x02), (0x7955, 0x79, 0x55),
                                (0x7A1A, 0x7A, 0x1A), (0x7E2A, 0x7E, 0x2A)):
        a, c = ir.split_scancode(scancode)
        check((a, c) == (addr, cmd),
              "0x%04X -> address 0x%02X command 0x%02X" % (scancode, addr, cmd))
        n += 1
    # extended NEC: a 16-bit address keeps the same split
    check(ir.split_scancode(0x1234AB) == (0x1234, 0xAB),
          "extended NEC keeps the command in the low byte")
    n += 1

    # --- discovery rejects HDMI/CEC and finds the receiver ------------------
    with tempfile.TemporaryDirectory() as d:
        devices = os.path.join(d, "devices")
        with open(devices, "w") as f:
            f.write(DEVICES)
        rc_root = fake_rc(os.path.join(d, "rc"))

        dev, info = ir.find_device(devices, rc_root)
        check(dev == "/dev/input/event5",
              "discovery finds the gpio-ir receiver, not HDMI event0")
        check(info.get("name") == "gpio_ir_recv", "it names what it found")
        check(info.get("enabled") == ["nec"], "and reports the enabled decoder")
        n += 3

        # the same table with the receiver removed: no guess, a stated reason
        with open(devices, "w") as f:
            f.write(DEVICES.split('I: Bus=0019 Vendor=0001')[0])
        dev, info = ir.find_device(devices, rc_root)
        check(dev is None, "no receiver -> no device")
        check("cec" in info.get("reason", ""),
              "and the reason names the CEC endpoints it rejected")
        n += 2

    # --- reading the device -------------------------------------------------
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "event")
        t0 = time.time()
        with open(path, "wb") as f:
            # a press, its EV_KEY/EV_SYN company (ignored), then a repeat
            # frame 0.11s later (a held button), then a genuine second press
            f.write(event_bytes(int(t0), 250000, ir.EV_MSC, ir.MSC_SCAN, 0x7902))
            f.write(event_bytes(int(t0), 250000, 0x01, 0x0207, 1))
            f.write(event_bytes(int(t0), 250000, 0x00, 0x00, 0))
            f.write(event_bytes(int(t0), 360000, ir.EV_MSC, ir.MSC_SCAN, 0x7902))
            f.write(event_bytes(int(t0) + 3, 250000, ir.EV_MSC, ir.MSC_SCAN, 0x7A1A))

        rx = ir.Receiver(device=path)
        check(rx.start(), "an explicit device opens without discovery")
        for _ in range(50):                    # let the reader thread drain it
            time.sleep(0.01)
            if len(rx._q) >= 3:
                break
        evs = rx.drain(t0=int(t0))
        rx.close()
        n += 1

        check(len(evs) == 3, "only EV_MSC/MSC_SCAN becomes an event (3 of 5)")
        check([e["address_hex"] for e in evs] == ["0x79", "0x79", "0x7A"],
              "each event carries the address it was addressed to")
        check([e["repeat"] for e in evs] == [False, True, False],
              "a frame 0.11s after the same code is a held button, not a press")
        check(evs[0]["command"] == 0x02 and evs[2]["command"] == 0x1A,
              "the command rides along with the address")
        check(abs(evs[0]["t"] - 0.25) < 0.001 and abs(evs[2]["t"] - 3.25) < 0.001,
              "t comes from the kernel timestamp, not from when we woke up")
        check(rx.drain() == [], "a second drain is empty: events are consumed")
        n += 6

    # --- absent hardware is a fact, not a crash -----------------------------
    rx = ir.Receiver(device="/nonexistent/event99")
    check(rx.start() is False, "a missing device fails soft")
    check(rx.source["state"] == "unreadable" and "reason" in rx.source,
          "and says why, for the clip to record")
    check(rx.drain() == [], "an unstarted receiver drains empty")
    n += 3

    # --- the replay round trip ----------------------------------------------
    # A clip's ir log -> stub_pi's wire lines -> stream's parser -> the line
    # the washlet daemon prints. This is the whole feature, end to end, with
    # the socket left out.
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "evals"))
    import stub_pi                                               # noqa: E402
    from bathroom import stream                                  # noqa: E402

    with tempfile.TemporaryDirectory() as d:
        clip = os.path.join(d, "clip.json")
        with open(clip, "w") as f:
            json.dump({
                "schema": "bathroom-clip/2",
                "frames": [{"t": 0.0, "mm": [100] * 64},
                           {"t": 0.1, "mm": [100] * 64},
                           {"t": 0.2, "mm": [100] * 64}],
                "ir": [{"t": -0.02, "ts": 1.0, "scancode": 0x7902,
                        "hex": "0x7902", "address": 0x79, "address_hex": "0x79",
                        "command": 0x02, "repeat": False},
                       {"t": 0.15, "ts": 1.17, "scancode": 0x7A1A,
                        "hex": "0x7A1A", "address": 0x7A, "address_hex": "0x7A",
                        "command": 0x1A, "repeat": True}],
            }, f)
        frames, irlog = stub_pi.load_frames([clip, clip])
        check(len(frames) == 6 and len(irlog) == 4,
              "two clips concatenate both logs")
        check([round(e["t"], 2) for e in irlog] == [0.0, 0.15, 0.2, 0.35],
              "the second clip's presses land during the second clip's frames")
        n += 2

        # what the stub would write, fed to the parser the daemon reads with
        for ev in irlog:
            stream._ir.append(json.loads(json.dumps(ev)))
        got = stream.drain_ir()
        check(len(got) == 4 and stream.drain_ir() == [],
              "the daemon drains each signal exactly once")
        check(ir.format_event(got[0]) ==
              "addr=0x79 cmd=0x02 scancode=0x7902",
              "the printed line leads with the address")
        check(ir.format_event(got[1]).endswith("(repeat)"),
              "a held button is marked in the printed line")
        n += 3

    # a clip recorded before IR existed replays as it always did
    with tempfile.TemporaryDirectory() as d:
        clip = os.path.join(d, "old.json")
        with open(clip, "w") as f:
            json.dump({"schema": "bathroom-clip/1",
                       "frames": [{"t": 0.0, "mm": [100] * 64}]}, f)
        frames, irlog = stub_pi.load_frames([clip])
        check(len(frames) == 1 and irlog == [],
              "a schema/1 clip with no ir log replays with no ir")
        n += 1

    print("\n%d checks, %d failures" % (n, len(FAILS)))
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
