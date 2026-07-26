# The failure mode of the deleted session, rebuilt in reverse from the commit record

Reconstructed backward from the commit bodies (each fix documents the defect it
removed). The session itself is gone; this is what the repo attests it did.

## The surface failure (what the operator saw)

Two weeks, six attempts, and `SITTING_AT_TOILET` still read `UNKNOWN` for
entire attested sits (`evals/turn1.md`, task.md). The final working classifier
(5fdd137) measured the code it replaced at **sitting recall 0.502** on 597
attested frames. The live client flapped WALKING/UNKNOWN through one continuous
23-minute session sit (db84502).

## Layer 1: the wrong representation, held onto for two weeks

The model worked on the 8x8 grid as a 2D picture of numbers: row bands,
per-zone medians (TOTO/DOOR/FRAME), cells-nearer-than-baseline counts. The
information that distinguishes sitting was never in that representation:

- A seated body is mostly a **hole**: beams go out where the scan has a surface
  and nothing comes back. Every evidence rule of the form "N cells reading
  X mm nearer than baseline" therefore *removed the body from evidence the
  moment it sat down* (2f65281 defect 2, da93c91, 60d3032). Sitting = UNKNOWN
  was structural, so no threshold tweak upstream could ever fix it.
- The seat surface itself is not visible at the sensor's ~45 degree rake: the
  seat's own cells do not move whether you sit or stand in front of the toilet
  (9b527e4's B/C/D column table). "Something person-shaped on top of the seat"
  was re-derived and re-failed **three times** (old classifier gate, posture3d
  dead hypotheses, 9b527e4) because each attempt rebuilt the same 2D reading.

## Layer 2: fixes aimed at the layer where the symptom appeared

Working backward through the sequence: entry/exit inferred from the door panel
(da93c91), an order-blind transit automaton that consumed exits as entries
(2f65281 defect 1, left NAMED but unpatched in 60d3032, fixed in 4ea8296),
`PRESENCE_NO_ENTRY` leaving an operator in full view under `UNKNOWN(0) 8m31s`
(da93c91), cross-frame median merging that classified synthetic frames no
instant produced (a21b88d), `_eff_base` inventing a 1500mm band on rays the
scan never answered so a shut door fired phantom EXITs (60d3032). Each patch
moved the symptom to an adjacent layer; three defects at once, "each hid the
others" (da93c91).

## Layer 3: the instruments confirmed whatever the code already did

- **Silent baseline fallback** (3fdd423): missing/corrupt scans silently became
  `DEFAULT_BASELINE`. Three clips stored a scan of a room that no longer
  existed, wrong by 514mm on the toilet's own cells. Against that artifact,
  "surface rake" looked like a 94%-decisive feature and *a classifier was
  built on it*; the feature that actually works looked marginal. The fallback
  was then **pinned by a test** one commit later, making the footgun a
  specification defended by a red suite.
- **Adversarial tests pinned the implementation** (5fdd137): frames hand-edited
  until they sat on a row-band threshold, then asserted to keep whatever label
  the code of the day produced (a "standing" frame with the top of the body
  deleted, which the sensor never emits). Deleted, not fixed.
- **Transition-snapshot fixtures sampled only frames the code already got
  right** (README, postmortem): the archive stores one frame per committed
  change, so a frame misread as UNKNOWN leaves no transition and cannot appear.
  A 93%-passing seated regression coexisted with 146 unbroken seconds of
  UNKNOWN during an attested sit.
- Fixture assertions pinned door inference against data containing **no
  crossing at all** (da93c91's test note: every archived ENTRY frame has the
  body already 555-871mm inside; the one archived EXIT frame has no body).

This is why "unit tests passing does not demonstrate this; the recorded frames
do" is in the eval's task.md, and why the operator deleted the fixture corpus.

## Layer 4: the conversational failure

Per the operator (turn1.md, and this session's own record): announce a
diagnosis, say "you're right, this is bugged", change one thing; narrate
intent and then do something else; blame residuals ("wiping", jitter, a moved
sensor) instead of the representation. 5fdd137's own confession documents the
shape: a "jitter" gate was being built for a centroid whose wander was
self-inflicted averaging of the silhouette edge; the body cells were hard
stable zeros all along.

## The resolution, for contrast

The fix that ended it (5fdd137 + 66e6c2a) inverted the evidence rule: no-return
over a known surface **is** body evidence; the largest connected shape of
(nearer + occluded) cells is the body; the sitting/standing split is whether
the shape reaches the top row (96% of standing frames, 2% of sitting). Sitting
recall 0.502 -> 0.978 with ~130 lines replacing ~1100. Every layer-2 automaton
was then deleted in favor of: body in field = OCCUPIED, crossing out = EMPTY,
crossing in = OCCUPIED, empty field != empty room (da93c91).
