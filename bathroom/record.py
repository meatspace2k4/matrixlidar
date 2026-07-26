"""Time-based labeled clip capture for transformer training.

Records a CLIP: the full time series of oriented frames over a window, tagged
with the operator's verbatim ground-truth label, plus a per-frame auto state
from the classifier (advisory) and the door open%. Unlike the Pi snapshot
tool, NOTHING is collapsed to a median: the temporal dimension is preserved,
which is the whole point of training a sequence model.

  python3 -m bathroom.record --label sitting --seconds 30
  python3 -m bathroom.record --label walking          # until Ctrl-C / Enter
  python3 -m bathroom.record --baseline               # capture empty baseline

Each clip is one self-describing JSON file under data/clips/, and a one-line
summary is appended to data/manifest.jsonl for indexing. Run the heatmap
client alongside to watch what you are labeling; both share the warm stream.
"""
import argparse
import json
import os
import sys
import time

from . import config
from . import detect
from . import stream
from .classify import median_window
from .engine import StateEngine
from .zones import N, ORIENTATION

SCHEMA = "bathroom-clip/1"
BASELINE_SECONDS = 6


def _flat(matrix):
    return [matrix[r][c] for r in range(N) for c in range(N)]


def _write_json(path, doc):
    """Atomic write: temp file in the same dir, then rename. A crash mid-write
    leaves the temp file, never a half-written clip the loader would choke on."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(doc, f)
    os.replace(tmp, path)


def _corpus_hint():
    """One line summarizing the current label mix, so the operator can balance
    classes as they capture. Reads the manifest; silent if absent/unreadable."""
    counts = {}
    try:
        with open(config.MANIFEST) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                counts[d.get("label", "?")] = counts.get(d.get("label", "?"), 0) + 1
    except (OSError, ValueError):
        return
    if counts:
        mix = ", ".join("%s=%d" % kv for kv in sorted(counts.items()))
        fewest = min(counts, key=counts.get)
        sys.stderr.write("corpus so far: %s   (lightest: %s)\n" % (mix, fewest))


def _status(state, n, elapsed, door):
    sys.stderr.write("\r\x1b[K  rec %-18s frames=%-4d  %5.1fs  door=%3.0f%%"
                     % (state.name, n, elapsed, door))
    sys.stderr.flush()


def _capture(seconds, on_frame):
    """Pull frames from the warm stream until `seconds` elapse (0 = until the
    user interrupts). Returns the wall-clock start time. Ctrl-C ends cleanly."""
    start = time.time()
    deadline = start + seconds if seconds > 0 else None
    try:
        for m in stream.frames(on_wait=_waiting):
            on_frame(m, time.time() - start)
            if deadline and time.time() >= deadline:
                break
    except KeyboardInterrupt:
        pass
    return start


def _waiting(attempt, total):
    sys.stderr.write("\r\x1b[K  waiting for %s:%d ... %d/%d"
                     % (stream.HOST, stream.PORT, attempt, total))
    sys.stderr.flush()


def record_clip(label, seconds, notes=None):
    config.ensure_dirs()
    # A clip stores the baseline it was captured against and downstream that
    # field IS the scene, so this must be a real scan. config.load_baseline
    # raises rather than substituting one; three clips in this corpus predate
    # that and had to be repaired by hand.
    baseline = config.load_baseline()
    eng = StateEngine(baseline=baseline)
    frames = []

    def on_frame(m, t):
        u = eng.update(m, t)
        frames.append({
            "t": round(t, 3),
            "mm": _flat(m),
            "door_open_pct": detect.door_open_pct(m),
            "state_auto": u.instant.name,        # per-frame classifier (raw)
            "state_engine": u.committed.name,    # debounced temporal engine
        })
        _status(u.committed, len(frames), t, frames[-1]["door_open_pct"])

    _corpus_hint()
    sys.stderr.write("recording label=%r (%s). Ctrl-C to stop.\n"
                     % (label, "%ds" % seconds if seconds else "open-ended"))
    start = _capture(seconds, on_frame)
    sys.stderr.write("\n")

    if not frames:
        sys.stderr.write("no frames captured; is %s:%d up?\n"
                         % (stream.HOST, stream.PORT))
        return 1

    eng.close()
    end = start + frames[-1]["t"]
    dur = frames[-1]["t"] or 1e-9
    clip_id = "%s_%d" % (_slug(label), int(start))
    auto_counts = {}
    for fr in frames:
        auto_counts[fr["state_auto"]] = auto_counts.get(fr["state_auto"], 0) + 1
    doc = {
        "schema": SCHEMA,
        "clip_id": clip_id,
        "label": label,                 # operator ground truth, verbatim
        "started": round(start, 3),
        "ended": round(end, 3),
        "n": len(frames),
        "hz": round(len(frames) / dur, 2),
        "orientation": ORIENTATION,
        "baseline": baseline,
        "auto_state_counts": auto_counts,
        "timeline": eng.segments(),     # debounced state segments over the clip
        "notes": notes,
        "frames": frames,
    }
    path = "%s/%s.json" % (config.CLIPS_DIR, clip_id)
    _write_json(path, doc)
    # the clip on disk is authoritative (dataset globs the dir); the manifest is
    # a convenience index, so a failed append warns but never loses the clip.
    try:
        with open(config.MANIFEST, "a") as f:
            f.write(json.dumps({
                "clip_id": clip_id, "label": label, "n": len(frames),
                "hz": doc["hz"], "started": doc["started"],
                "auto_state_counts": auto_counts, "file": path, "notes": notes,
            }) + "\n")
    except OSError as e:
        sys.stderr.write("warning: clip saved but manifest update failed: %s\n" % e)

    sys.stderr.write("saved %d frames @ %.1f Hz -> %s\n"
                     % (len(frames), doc["hz"], path))
    sys.stderr.write("auto-state mix: %s\n"
                     % ", ".join("%s=%d" % kv for kv in sorted(auto_counts.items())))
    return 0


def record_baseline(seconds):
    config.ensure_dirs()
    frames = []
    sys.stderr.write("capturing empty-room baseline for %ds. Leave the room.\n" % seconds)
    _capture(seconds, lambda m, t: frames.append(m))
    sys.stderr.write("\n")
    if not frames:
        sys.stderr.write("no frames captured; is %s:%d up?\n"
                         % (stream.HOST, stream.PORT))
        return 1
    matrix = median_window(frames)
    doc = {"captured": round(time.time(), 3), "frames": len(frames),
           "orientation": ORIENTATION, "matrix": matrix}
    _write_json(config.BASELINE_FILE, doc)
    sys.stderr.write("baseline (%d frames, median) -> %s\n"
                     % (len(frames), config.BASELINE_FILE))
    return 0


def _slug(s):
    return "".join(c if c.isalnum() else "-" for c in s.lower()).strip("-") or "clip"


def main():
    ap = argparse.ArgumentParser(description="Capture labeled lidar clips for training")
    ap.add_argument("--label", help="verbatim ground-truth label for the clip "
                    "(e.g. sitting, standing, walking, empty)")
    ap.add_argument("--seconds", type=int, default=0,
                    help="capture window; 0 = until Ctrl-C")
    ap.add_argument("--notes", default=None, help="free-text note stored with the clip")
    ap.add_argument("--baseline", action="store_true",
                    help="capture an empty-room baseline instead of a clip")
    args = ap.parse_args()
    if args.baseline:
        return record_baseline(args.seconds or BASELINE_SECONDS)
    if not args.label or not args.label.strip():
        ap.error("--label must be a non-empty ground-truth label (or use --baseline)")
    return record_clip(args.label.strip(), args.seconds, args.notes)


if __name__ == "__main__":
    raise SystemExit(main())
