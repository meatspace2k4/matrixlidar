#!/usr/bin/env python3
"""IR fingerprint regression: raw mode2 words -> stable per-button IDs, and
the round trip from a clip's recorded log to the line the daemon prints.

No IR hardware is involved. The kernel interface is a stream of u32 words
and a directory tree under /sys/class/rc, so both are built here as real
bytes and real directories and read through the real code path. The pulse
timings are the bathroom remote's own, measured off /dev/lirc0 on bath-pi:
~560us pulses, ~560/~1690us spaces, >=20ms quiet between frames.

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


def w(kind, usec):
    return struct.pack("<I", (kind << 24) | usec)


def frame(bits, jitter=0):
    """Raw mode2 bytes for one frame carrying `bits` (a '01' string), using
    the measured timings. `jitter` shifts every duration, because the real
    receiver wobbles +-40us and the fingerprint must not."""
    out = b""
    for b in bits:
        out += w(ir.MODE2_PULSE, 560 + jitter)
        out += w(ir.MODE2_SPACE, (1690 if b == "1" else 560) + jitter)
    out += w(ir.MODE2_PULSE, 560 + jitter)      # closing mark
    out += w(ir.MODE2_TIMEOUT, 22000)
    return out


BUTTON_A = "0110100101101001010110010110100101011010"  # 40 bits
BUTTON_B = "0110100101101001010110010110100101011001"  # differs in last bit


def main():
    n = 0

    # --- the fingerprint itself ---------------------------------------------
    asm = ir.FrameAssembler()
    fps = asm.feed(frame(BUTTON_A))
    check(len(fps) == 1, "one frame -> one fingerprint")
    fp_a = fps[0]
    check(fp_a.startswith("40:"), "the fingerprint states its bit count")
    n += 2

    check(ir.FrameAssembler().feed(frame(BUTTON_A, jitter=40)) == [fp_a],
          "+-40us receiver wobble does not move the fingerprint")
    fp_b = ir.FrameAssembler().feed(frame(BUTTON_B))[0]
    check(fp_b != fp_a, "one flipped bit is a different fingerprint")
    n += 2

    # --- segmentation -------------------------------------------------------
    # two frames separated by a long SPACE (no timeout word): still two
    asm = ir.FrameAssembler()
    blob = frame(BUTTON_A)[:-4] + w(ir.MODE2_SPACE, 28000) + frame(BUTTON_B)
    fps = asm.feed(blob)
    check(fps == [fp_a, fp_b], "a >=10ms space splits frames like a timeout")
    n += 1

    # a frame split across arbitrary read() boundaries reassembles
    blob = frame(BUTTON_A)
    for cut in (4, 30, len(blob) - 8):
        asm = ir.FrameAssembler()
        fps = asm.feed(blob[:cut]) + asm.feed(blob[cut:])
        check(fps == [fp_a], "chunk split at byte %d reassembles" % cut)
        n += 1

    # leading idle space (the 16.7s max-usec word lirc emits first) is noise
    asm = ir.FrameAssembler()
    fps = asm.feed(w(ir.MODE2_SPACE, 0xFFFFFF) + frame(BUTTON_A))
    check(fps == [fp_a], "leading idle space is not a frame")
    glitch = ir.FrameAssembler().feed(
        w(ir.MODE2_PULSE, 560) + w(ir.MODE2_SPACE, 560)
        + w(ir.MODE2_PULSE, 560) + w(ir.MODE2_TIMEOUT, 22000))
    check(glitch == ["1:0"] and not ir.is_expected_length(glitch[0]),
          "a 1-bit glitch is captured and fails the length gate")
    n += 2

    # --- capture keeps everything; the gate is applied when READING ---------
    # Every short frame seen on the real receiver, by bit count. These are the
    # receiver's LED feeding back into its own photodiode and chopping up a
    # real press, and they are RECORDED: their rate is the diagnostic for that
    # coupling, and a capture-time filter cannot be un-applied.
    junk_bits = (8, 10, 12, 13, 15, 16, 19, 21, 22, 28, 30, 31, 35)
    junk = b"".join(frame("1" * b) for b in junk_bits)
    fps = ir.FrameAssembler().feed(junk)
    check(len(fps) == 13, "every wrong-length burst is still captured")
    check([ir.fp_bits(f) for f in fps] == list(junk_bits),
          "and captured at its real length, not normalized")
    n += 2

    # ... and none of them passes the length gate when read
    check([f for f in fps if ir.is_expected_length(f)] == [],
          "no short frame passes the length gate")
    check(ir.is_expected_length(fp_a) and ir.is_expected_length({"fp": fp_a}),
          "a 40-bit frame passes, dict or bare string")
    n += 2

    # the gate must not cost a real press: same stream, remote frame appended
    fps = ir.FrameAssembler().feed(junk + frame(BUTTON_A))
    check([f for f in fps if ir.is_expected_length(f)] == [fp_a],
          "a real 40-bit press survives a burst of LED feedback")
    check(len(fps) == 14, "and the fragments are in the log alongside it")
    n += 2

    # want_bits=0 passes everything, for an unknown remote
    check([f for f in fps if ir.is_expected_length(f, want_bits=0)] == fps,
          "want_bits=0 passes every frame length")
    n += 1

    # the length is the gate, not the content: a 40-bit frame of all zeros
    # still passes
    zeros = ir.FrameAssembler().feed(frame("0" * 40))
    check(zeros == ["40:" + "0" * 10] and ir.is_expected_length(zeros[0]),
          "a 40-bit all-zero frame passes (length gates, content does not)")
    n += 1

    # --- a bad WASHLET_IR_BITS must not take the recorder down --------------
    # An unguarded int() here crashed record._open_receiver, whose entire job
    # is to not crash when the IR setup is broken.
    old = os.environ.get("WASHLET_IR_BITS")
    try:
        os.environ["WASHLET_IR_BITS"] = "auto"
        check(ir._want_bits() == ir.FRAME_BITS,
              "WASHLET_IR_BITS=auto falls back instead of raising")
        rx = ir.Receiver(device="/nonexistent/lirc99")
        check(rx.start() is False, "and a Receiver still constructs and fails soft")
        os.environ["WASHLET_IR_BITS"] = "0"
        check(ir._want_bits() == 0, "WASHLET_IR_BITS=0 is honoured, not falsy-defaulted")
        os.environ["WASHLET_IR_BITS"] = "-3"
        check(ir._want_bits() == ir.FRAME_BITS, "a negative width falls back")
    finally:
        os.environ.pop("WASHLET_IR_BITS", None)
        if old is not None:
            os.environ["WASHLET_IR_BITS"] = old
    n += 4

    # --- discovery walks /sys/class/rc and rejects CEC ----------------------
    with tempfile.TemporaryDirectory() as d:
        for rc, protos, lirc in (("rc0", "rc-5 [nec] sony [lirc]", "lirc0"),
                                 ("rc1", "[cec]", None),
                                 ("rc2", "[cec]", None)):
            os.makedirs(os.path.join(d, rc))
            with open(os.path.join(d, rc, "protocols"), "w") as f:
                f.write(protos + "\n")
            if lirc:
                os.makedirs(os.path.join(d, rc, lirc))
                with open(os.path.join(d, rc, "device_name"), "w") as f:
                    f.write("gpio_ir_recv\n")
        dev, info = ir.find_device(d)
        check(dev == "/dev/lirc0", "discovery finds the receiver's lirc node")
        check(info.get("name") == "gpio_ir_recv", "and names what it found")
        n += 2

        for rc in ("rc0",):
            os.rename(os.path.join(d, rc), os.path.join(d, "gone"))
        import shutil
        shutil.rmtree(os.path.join(d, "gone"))
        dev, info = ir.find_device(d)
        check(dev is None and "cec" in info.get("reason", ""),
              "no receiver -> no device, and the reason names the CEC rejects")
        n += 1

    # --- reading a device ---------------------------------------------------
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "lirc")
        with open(path, "wb") as f:
            f.write(frame(BUTTON_A))            # press
            f.write(frame(BUTTON_A))            # held: same fp, right away
            f.write(frame(BUTTON_B))            # a different button
        rx = ir.Receiver(device=path)
        check(rx.start(), "an explicit device opens without discovery")
        for _ in range(100):
            time.sleep(0.01)
            if len(rx._q) >= 3:
                break
        t0 = time.time() - 1.0
        evs = rx.drain(t0)
        rx.close()
        check([e["fp"] for e in evs] == [fp_a, fp_a, fp_b],
              "the stream fingerprints in order")
        check([e["repeat"] for e in evs] == [False, True, False],
              "an identical frame within 250ms is a held button")
        check(all(e["bits"] == 40 for e in evs), "bit count rides along")
        check(all(e["t"] >= 0.9 for e in evs), "t is anchored on the drain origin")
        check(rx.drain() == [], "a second drain is empty: events are consumed")
        n += 6

    # --- absent hardware is a fact, not a crash -----------------------------
    rx = ir.Receiver(device="/nonexistent/lirc99")
    check(rx.start() is False, "a missing device fails soft")
    check(rx.source["state"] == "unreadable" and "reason" in rx.source,
          "and says why, for the clip to record")
    check(rx.drain() == [], "an unstarted receiver drains empty")
    n += 3

    # --- the replay round trip ----------------------------------------------
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
                "ir": [{"t": -0.02, "ts": 1.0, "fp": fp_a, "bits": 40,
                        "repeat": False},
                       {"t": 0.15, "ts": 1.17, "fp": fp_a, "bits": 40,
                        "repeat": True}],
            }, f)
        frames, irlog = stub_pi.load_frames([clip, clip])
        check(len(frames) == 6 and len(irlog) == 4,
              "two clips concatenate both logs")
        check([round(e["t"], 2) for e in irlog] == [0.0, 0.15, 0.2, 0.35],
              "the second clip's presses land during the second clip's frames")
        n += 2

        for ev in irlog:
            stream._ir.append(json.loads(json.dumps(ev)))
        got = stream.drain_ir()
        check(len(got) == 4 and stream.drain_ir() == [],
              "the daemon drains each signal exactly once")
        check(ir.format_event(got[0]) == "fp=" + fp_a,
              "the printed line is the fingerprint")
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
