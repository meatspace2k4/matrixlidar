#!/usr/bin/env python3
"""Run the whole regression suite. Exit 0 iff everything passes.

  python3 tests/run.py

Covers: classifier calibration (4 labeled), adversarial boundary cases, the
data pipeline (clip round-trip / corrections / split / robustness), and the
grounding gates (engine + frame must score 1.0 on the labeled fixtures).
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

SCRIPTS = ["test_classify.py", "test_classify_adversarial.py", "test_pipeline.py",
           "test_occupancy.py", "test_depth.py", "test_confirmation.py",
           "test_replay_sessions.py"]
GATES = [
    ["python3", "-m", "bathroom.validate", "--predictor", "engine", "--no-clips", "--min-accuracy", "1.0"],
    ["python3", "-m", "bathroom.validate", "--predictor", "frame", "--no-clips", "--min-accuracy", "1.0"],
]


def main():
    fails = []
    for s in SCRIPTS:
        r = subprocess.run(["python3", os.path.join(HERE, s)], cwd=ROOT)
        print("%-32s %s" % (s, "ok" if r.returncode == 0 else "FAIL"))
        if r.returncode != 0:
            fails.append(s)
    for g in GATES:
        r = subprocess.run(g, cwd=ROOT, stdout=subprocess.DEVNULL)
        name = "grounding:%s" % g[4]
        print("%-32s %s" % (name, "ok" if r.returncode == 0 else "FAIL"))
        if r.returncode != 0:
            fails.append(name)
    print("\n%s" % ("ALL GREEN" if not fails else "FAILURES: " + ", ".join(fails)))
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
