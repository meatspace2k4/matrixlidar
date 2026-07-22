#!/usr/bin/env python3
"""Live door-panel detection readout for calibration. Run it, watch the line
while you move the door: it shows whether the sensor currently detects the
panel in column H and the raw distances. Ctrl+C to stop."""
import statistics
import sys

sys.path.insert(0, "/Users/johusha/Claude/matrixlidar")
from bathroom import stream
from bathroom.zones import NO_TARGET

for m in stream.frames():
    colH = [m[r][7] for r in range(8)]
    seen = [v for v in colH if v < NO_TARGET]
    if seen:
        status = "\x1b[42;30m  DOOR DETECTED  \x1b[0m cells=%d  median=%4dmm" % (
            len(seen), int(statistics.median(seen)))
    else:
        status = "\x1b[41;97m  NOT DETECTED   \x1b[0m cells=0"
    sys.stdout.write("\r\x1b[K %s   H=%s " % (status, ",".join("%4d" % v for v in colH)))
    sys.stdout.flush()
