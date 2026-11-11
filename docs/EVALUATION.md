# odin: evaluation protocol and evidence

This document owns the study protocol. Current completed checks and report readiness
are in [project status](PROJECT_STATUS.md); runnable commands are in the
[README](../README.md#evaluate-recorded-data).

## Data and separation

The study includes public research datasets, our own recordings of team members and
consenting friends, and public-person photos/videos, especially actors. Controlled
participant recordings support recognition and unknown-person evaluation as well as
tracking, alerts and live/runtime checks. Actor media supports appearance-change
comparisons. ChokePoint and ROF are reserved for later. See [dataset planning](DATASETS.md)
for acquisition status, held-out appearances and pretrained-data limitations. The
participant roster and recordings are not yet complete. Report results and exposure
separately for each source.

Keep enrollment, validation and final-test capture sessions independent. Known people
may appear in all partitions, but their images/recordings must differ. Final-test
unknown people stay outside both enrollment and unknown-person validation. Keep entire
capture sessions together: randomly split adjacent frames or augmented copies leak data.
Choose thresholds, quality gates, temporal settings and cooldown on validation only.
Freeze configuration and the test manifest before examining final-test scores.

Include frontal/turned pose; bright/dim/backlit illumination; face-size/distance
bands; real masks and partial occlusion; blur; crossings; disappearance/reappearance;
enabled/disabled enrolled people; unknown-only intervals; and empty scenes.
Synthetic occlusion and repeated photographs are supplementary fixtures.
Annotate boxes, identities/unknown labels, condition tags and encounter windows.
Tracking evaluation additionally needs ground-truth trajectories; have another team
member review a representative subset, especially crossings and event boundaries.

Externally sourced media needs source/terms, acquisition date, label provenance,
hashes, transformations and partition groups. Do not pool it silently with volunteer
exposure. WIDER FACE concerns detection; video-pair verification benchmarks do not
directly establish watchlist/event performance. Potential pretrained-data overlap
must be disclosed where it cannot be ruled out.

## Implemented runner

The [example manifest](../experiments/classical.example.json) uses schema version 1,
split `validation`, `test` or `smoke`, authorized enrollment images, video paths,
optional frame annotations and optional encounters. Paths are relative to the manifest.
No annotation means unscored, not a negative label; an explicitly empty face list
labels a negative frame. Boxes use `[x, y, width, height]`.

The CLI processes every decoded frame; operator replay is paced and may drop frames.
Both assume constant frame rate and use frame index/FPS. They do not validate true
variable-frame-rate presentation timestamps or support network sources. Report which
mode was used rather than comparing their frame counts as equivalent exposure.

Outputs go into a new directory. `predictions.jsonl` records tracks for each decoded
frame. `report.json` includes input/manifest hashes, policy, detector hash, descriptor
space, per-condition counts, annotated-frame denominators, video exposure, events
and median/p95 pipeline processing time. Failures after output creation are reported.
Repeated video bytes within a manifest are rejected. Hashes do not enforce independent
capture sessions or disjoint people across separate manifests; the study must do that.
Record command, Git revision, lockfile, hardware and environment alongside runs;
the current runner does not automatically capture all of this provenance.

## Threshold selection

`--calibrate` requires `split: validation` and usable known and unknown observations.
It selects a single-observation distance threshold under an empirical unknown
false-accept constraint (default 0.05), preferring more correct-known outcomes,
then fewer wrong-known/unknown accepts and a lower threshold. Matching also uses
the configured distinct-person margin. This is not probability calibration or a
calibration of the full temporal/event policy.

The runner accepts an injected detector/recognizer in Python; its CLI still composes
Haar plus LBP. Enrollment, prediction and calibration all use the selected recognizer's
extraction and matching. Calibration respects its declared score direction/range;
the default remains lower-is-better LBP distance. Reports/predictions/events now use
`score` and `score_kind`, recording raw values rather than converting between scales.

Copy the selected threshold into operator settings and the frozen test manifest.
Adding templates/identities or changing preparation changes score distributions and
requires rechecking applicability. One-person galleries have no runner-up margin.
Keep each recognizer's gallery, preparation and threshold separate.

## Metrics and interpretation

Frame scoring greedily matches current non-missing predictions one-to-one with truth
at IoU at least 0.3. Confirmed identities count as accepted; candidates/insufficient
states remain unresolved.

| Layer | Report |
|---|---|
| Detection | Visible truth faces, matched faces, false detections, condition/size breakdown |
| Known identity | Correct, wrong identity, rejected, unresolved and detection-missed outcomes |
| Unknown identity | False acceptance, rejection and unresolved outcomes; misses are unresolved |
| Events | Correct, false, missed, duplicate events, encounter denominator and successful-event delays |
| Coverage | Usable/accepted observations relative to all visible faces/encounters, including misses |
| Runtime | Mode, input/processed/preview cadence, drops, frame age, delay and resource use on named hardware |

The implemented event scorer matches identity and media-time window one-to-one within
each video's supplied encounters. It counts subsequent events in an already matched
window as duplicates. Provide eligible alert-enabled encounters explicitly; eligibility
and encounter boundaries are not inferred by the scorer. Latencies for successful
events must be accompanied by missed encounters. Acknowledgement is not a correctness label.

The current runner does not calculate tracking IDF1/switches, uncertainty intervals,
calibration reliability, stage-level timing or end-to-end display latency. Collect
these separately if claiming them. Avoid unsupported AP, probability or extreme
false-accept-rate claims. Report numerators, denominators and unique video-hours;
repeated playback adds endurance time, not independent recognition exposure.

## Comparisons and remaining measurements

Compare selected recognition approaches on the same identities and recording splits
with separate validation thresholds. Where practical, report both matched-observation
recognition and complete-pipeline behavior to expose detector/preparation differences.
ArcFace and the selected FaceNet checkpoint differ in architecture, training provenance
and crop preparation; this is not a controlled loss-function-only comparison.

Temporal comparisons should hold observations, gallery, tracker, evidence budget and
event rules constant where possible. Single-frame versus current temporal confirmation
is useful; quality weighting and embedding aggregation remain planned experiments.
If studied, fit on development data, select on validation, freeze for test, and report
coverage/delay/cost as well as correctness. A negative result remains useful evidence.

Declare a hardware profile before sustained live benchmarking: resolution/FPS, faces,
gallery size, processing cadence, frame age, latency and memory limits. Old proposed
15 FPS/250 ms targets are not measured capabilities. Test source disconnect/restart,
deletion and overload, then endurance within the available project time. Short neural
stage timings and the generated smoke video do not establish application FPS or
real-world robustness. The progress report may honestly present these as pending work.
