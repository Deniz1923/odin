# Learning the classical odin pipeline

## Follow one observation

`Service` receives camera or replay frames. `HaarDetector` locates faces.
`prepare_face` extracts, resizes, quality-checks and equalizes a grayscale crop.
`describe` computes an LBP histogram descriptor. `match` compares it with authorized
reference descriptors. `Pipeline` associates boxes and accumulates time-spaced
identity evidence. Enabled, fresh confirmations propose an appearance event;
`Store` persists it after eligibility and cooldown checks. The browser displays
the preview, decisions and event history.

## What we wrote and what we use

| Component | Ownership |
|---|---|
| Haar features/cascade inference | OpenCV, using its pinned pretrained XML |
| LBP bit extraction and spatial histograms | Our NumPy code in `classical/lbp.py` |
| Histogram distance, per-identity search and rejection | Our NumPy/Python code |
| Tracking, temporal confirmation and event policy | Our basic Python baseline |
| Image/video decode, resize, grayscale, equalization | OpenCV/Pillow primitives |
| Encryption and database engine | cryptography and SQLite; application lifecycle is ours |

Classical vision can use machine learning. Haar feature formulas are fixed, but a
Haar cascade's selected features, thresholds and stages are learned from face/nonface
examples using classical methods. Loading a pretrained cascade is not deep learning.
Our LBP extraction has no trained network weights. Enrollment stores labeled
reference descriptors; it does not train a neural model.

## LBP by hand

Take this patch; the center intensity is five:

```text
0 9 0
9 5 9
0 9 0
```

Visit NW, N, NE, E, SE, S, SW, W. Set bit `i` if that neighbor is at least as bright
as the center. Here only N/E/S/W set bits 1/3/5/7: the code is 170. Equal-intensity
neighbors count as one. Exclude the outer pixel border rather than invent neighbors.
The unit test checks these literal values independently of the implementation.

`describe` divides the code image into an 8 by 8 grid. Each cell gets a normalized
256-bin histogram. Concatenating 64 histograms produces 16,384 numbers. Keeping
cells separate preserves rough location: cheek texture is not freely interchangeable
with forehead texture. This implementation uses basic LBP, not a uniform-pattern
or rotation-invariant variant. Its descriptor space is versioned accordingly.

## Matching and unknown people

For two histograms, the chi-square-style distance sums
`0.5 * (a-b)^2 / (a+b+epsilon)`. We average over the normalized cells. Identical
descriptors give zero; smaller means more similar. Different eyes, pose or occlusion
can change many bins, even when the real person is the same.

For each enrolled person, keep their minimum sample distance. Compare the best
person with the next **different person**. A distance above the selected limit is
unknown; a near-tie is insufficient evidence. An empty gallery is insufficient.
With one enrolled identity there is no second-person margin, so only the distance
limit applies. That case needs validation too.

An unknown person still has a nearest neighbor. The rejection limit is essential.
The default 0.35 and margin 0.02 are starting settings, not established performance.
Use validation recordings with genuine, wrong-known and unknown observations before
claiming a useful operating point. Adding references or people can change score
distributions and requires rechecking the threshold.

## Tracking and time

`Pipeline` uses box intersection-over-union for a deliberately basic spatial tracker.
Each detection can match at most one track. Ambiguous association resets all involved
evidence; missed tracks expire after a bounded interval. This is less capable than
the larger architecture's proposed motion tracker, especially for fast motion and
crossings, and should be replaced or extended if measured failures justify it.

Positive observations are sampled at least 0.25 seconds apart. The default requires
three accepted observations spanning at least 0.5 seconds inside a two-second window.
Strong contradictory/unknown evidence is checked even between those sample times.
Stale evidence demotes the decision; prediction alone is not new identity evidence.

The current quality gates reject tiny, badly exposed or low-texture/blurred crops.
They are simple engineering heuristics, not an occlusion detector or calibrated
quality probability. A mask can cause a missed detection, rejection, or a wrong
match; the final study must measure these separately.

## Events and privacy

Enrollment and alert eligibility are distinct. Alerts default off. A fresh temporal
confirmation can create one event per track/person; persistent source/person
cooldown suppresses short fragmentation. A changed identity must confirm independently.
No events are authorized for unknown/insufficient states. Acknowledgement records
operator receipt, not verified recognition correctness.

Descriptors are encrypted in a private SQLite gallery; original enrollment photos
are not saved. Deletion serializes with processing, removes linked events/templates,
resets the in-memory gallery and clears preview/evidence. The UI discards old preview
blobs and in-flight snapshots when an operator mutation invalidates them.

The capture helper retains one pending frame; a slower pipeline replaces stale
frames and counts the drops. EOF is a terminal signal that must wait behind the
last pending frame. Stop invalidates the session so old work cannot publish. A stuck
native decoder is isolated in a daemon helper; restarting is refused while it has
not released. Full process-level recovery is a future improvement.

## How this contributes to the larger project

The gallery/operator workflows, source service, tracking/event contracts and evaluation
protocol can be reused for the neural methods. Their descriptor spaces and matching
rules must remain separate: LBPH distance is not ArcFace cosine similarity. Future
comparisons can run the same detector/crops/recordings where appropriate, alongside
full-pipeline results using each method's required preprocessing. Fusion should use
candidate-specific calibrated evidence, not average unrelated raw winning scores.

The next useful learning exercise is to enroll your own authorized samples, inspect
distance changes on a separate recording, and trace one accepted, one rejected and
one insufficient observation through the functions above. Then collect the independent
sessions needed to compare methods and report difficult-condition results honestly.
