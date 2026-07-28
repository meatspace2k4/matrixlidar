#!/usr/bin/env python3
"""Live door readout for calibration. Run it and move the door.

Shows what the door measurement is actually made of: the panel column's 8 rays,
the plane they fit, and that plane's angle off the jamb plane taken from the
empty-room scan. No column medians: a median over 8 rays at 8 elevations is not
a distance to a surface. Ctrl+C to stop.
"""
import sys

sys.path.insert(0, "/Users/johusha/Claude/matrixlidar")
from bathroom import config, detect, stream
from bathroom.zones import N, NO_TARGET


def main():
    try:
        baseline = config.load_baseline()
    except config.BaselineError as e:
        sys.stderr.write("%s\n" % e)
        return 1

    jamb = detect.jamb_plane(baseline)
    if jamb is None:
        sys.stderr.write("the scan has no jamb in column G; recapture it with "
                         "the room empty\n")
        return 1
    sys.stderr.write("jamb plane, from the scan: %.0fmm from the lens, "
                     "%.1fmm rms. This is the reference; it does not move.\n"
                     % (jamb[1], jamb[2]))

    reading = detect.DoorReading(baseline)
    for m in stream.frames():
        colH = [m[r][detect.PANEL_COL] for r in range(N)]
        seen = [v for v in colH if 0 < v < NO_TARGET]
        panel = detect._plane(m, detect.PANEL_COL)
        reading.update(m)

        if panel is None:
            state = "\x1b[42;30m  SEES THROUGH  \x1b[0m %d/%d rays" % (len(seen), N)
            fit = " " * 24
        else:
            state = "\x1b[46;30m     PANEL     \x1b[0m %d/%d rays" % (len(seen), N)
            fit = " plane=%4.0fmm rms=%4.1fmm" % (panel[1], panel[2])
            if panel[2] > detect.PLANE_RMS_MAX:
                fit += " \x1b[41;97mNOT ONE SURFACE\x1b[0m"

        if reading.angle is None:
            ang = "  --"
        else:
            ang = "%4.1f deg -> %3.0f%%%s" % (
                reading.angle, reading.pct, "" if reading.coherent else " (held)")

        sys.stdout.write("\r\x1b[K %s%s  off jamb: %-24s H=%s "
                         % (state, fit, ang, ",".join("%4d" % v for v in colH)))
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
