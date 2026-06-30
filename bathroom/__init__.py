"""Bathroom presence sensing: client-side interface to the DFRobot Matrix
Lidar (SEN0628), reading the Pi's always-on warm stream.

Modules:
  zones     - zone definitions + ground-truth State vocabulary (display orient)
  stream    - warm-TCP reader -> oriented 8x8 frames, auto-reconnecting
  detect    - deterministic zone values (door open%, per-zone summaries)
  classify  - the ground-truth state classifier (EMPTY / WALKING /
              STANDING_AT_TOILET / SITTING_AT_TOILET), baseline-delta based
  record    - time-based labeled clip capture for training (CLI)
  live      - live state readout (CLI)
  dataset   - load recorded clips into training arrays
  config    - project paths + baseline loading

The classifier fills the gap the Pi package deferred: it turns a frame into a
labeled state deterministically so recorded clips carry ground truth.
"""
