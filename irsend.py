#!/usr/bin/env python3
"""Transmit a recorded IR fingerprint back out a gpio-ir-tx LED.

The receive side (bathroom/ir.py) reduces a frame to `bits:hex`, keeping
only the intra-frame SPACES and treating every pulse as ~560us. This rebuilds
a sendable pulse train from that: one 560us pulse per bit, then a 560us space
for 0 or a 1690us space for 1, at a 38kHz carrier. A frame sent this way
re-fingerprints identically on the receiver, which is the whole point of the
loopback test: press SEAT2 in software, see it land as 40:a00800f6f6.

It does NOT reproduce the remote's exact pulse widths or any leader the RX
path discarded, because the fingerprint did not keep them. Good enough to
reproduce a fingerprint, not to be bit-exact with the OEM remote.

  python3 irsend.py --list                 # show lirc nodes, which can send
  python3 irsend.py --fp 40:a00800f6f6 --dry-run   # build + print, send nothing
  python3 irsend.py --button SEAT2 --send  # actually transmit (needs a TX node)

Nothing transmits without --send. --dry-run prints the ir-ctl file it would
send and exits. This is a fork-branch client and touches no live daemon: the
washlet holds the RX path and derives state, it does not transmit.
"""
import argparse
import glob
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bathroom import ir as ir_mod                              # noqa: E402

PULSE_US = 560
SHORT_US = 560
LONG_US = 1690
CARRIER_HZ = 38000
GAP_US = 40000     # quiet after the frame, so a repeat is a repeat


def fp_to_bits(fp):
    """'40:a00800f6f6' -> a 40-char '01' string. Raises on a malformed fp."""
    n_str, hex_str = fp.split(":", 1)
    n = int(n_str)
    val = int(hex_str, 16)
    return format(val, "0%db" % n)


def build_pulses(fp):
    """[(pulse_us, space_us), ...] for a fingerprint, RX-reconstruction rules."""
    bits = fp_to_bits(fp)
    return [(PULSE_US, LONG_US if b == "1" else SHORT_US) for b in bits]


def ir_ctl_file(fp):
    """The ir-ctl --send text: alternating 'pulse N'/'space N', us."""
    lines = ["carrier %d" % CARRIER_HZ]
    for pulse, space in build_pulses(fp):
        lines.append("pulse %d" % pulse)
        lines.append("space %d" % space)
    lines.append("space %d" % GAP_US)
    return "\n".join(lines) + "\n"


def _sendable(dev):
    """True if `ir-ctl -f -d dev` reports a device that can send."""
    try:
        out = subprocess.run(["ir-ctl", "-f", "-d", dev],
                             capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    tail = out.split("Send features", 1)[-1]
    return "can send" in tail.lower() and "cannot send" not in tail.lower()


def find_tx(nodes=None):
    """(dev, [(dev, sendable), ...]) for the first sendable lirc node.

    Auto-detected rather than hardcoded: adding the gpio-ir-tx overlay creates
    a new /dev/lircN whose number depends on probe order, so the discriminator
    is the device's own send feature, not its name.
    """
    nodes = nodes or sorted(glob.glob("/dev/lirc*"))
    survey = [(d, _sendable(d)) for d in nodes]
    for d, ok in survey:
        if ok:
            return d, survey
    return None, survey


def main():
    ap = argparse.ArgumentParser(description="Transmit a recorded IR fingerprint")
    ap.add_argument("--fp", help="fingerprint to send, e.g. 40:a00800f6f6")
    ap.add_argument("--button", help="a name from ir.BUTTONS, e.g. SEAT2")
    ap.add_argument("--device", help="TX lirc node (default: auto-detect)")
    ap.add_argument("--list", action="store_true",
                    help="list lirc nodes and which can send, then exit")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the ir-ctl frame that would be sent, send nothing")
    ap.add_argument("--send", action="store_true",
                    help="actually transmit (required; nothing sends without it)")
    a = ap.parse_args()

    if a.list:
        _dev, survey = find_tx()
        for d, ok in survey:
            print("%s  %s" % (d, "CAN SEND" if ok else "receive only"))
        if not survey:
            print("no /dev/lirc* nodes (gpio-ir-tx overlay not loaded?)")
        return 0

    fp = a.fp
    if a.button:
        rev = {v: k for k, v in ir_mod.BUTTONS.items()}
        if a.button not in rev:
            print("unknown button %r; known: %s"
                  % (a.button, ", ".join(sorted(rev))))
            return 1
        fp = rev[a.button]
    if not fp:
        ap.error("give --fp or --button")

    try:
        frame = ir_ctl_file(fp)
    except (ValueError, IndexError) as e:
        print("bad fingerprint %r: %s" % (fp, e))
        return 1

    label = ir_mod.button_name(fp)
    name = "" if label == fp else " (%s)" % label
    if a.dry_run or not a.send:
        sys.stderr.write("would send %s%s, %d bits:\n"
                         % (fp, name, len(fp_to_bits(fp))))
        sys.stdout.write(frame)
        if not a.dry_run:
            sys.stderr.write("\nnothing sent: pass --send to transmit\n")
        return 0

    dev = a.device
    if dev is None:
        dev, survey = find_tx()
        if dev is None:
            sys.stderr.write("no sendable lirc node found:\n")
            for d, ok in survey:
                sys.stderr.write("  %s  %s\n" % (d, "can send" if ok else "rx only"))
            sys.stderr.write("the gpio-ir-tx overlay is not loaded, or this "
                             "user cannot open the TX node.\n")
            return 1
    with tempfile.NamedTemporaryFile("w", suffix=".ir", delete=False) as f:
        f.write(frame)
        path = f.name
    try:
        r = subprocess.run(["ir-ctl", "-d", dev, "--send", path],
                           capture_output=True, text=True)
    finally:
        os.unlink(path)
    if r.returncode != 0:
        sys.stderr.write("ir-ctl failed on %s: %s\n" % (dev, r.stderr.strip()))
        return 1
    sys.stderr.write("sent %s%s out %s\n" % (fp, name, dev))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
