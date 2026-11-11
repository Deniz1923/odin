# odin: project status and progress-report preparation

Updated 11 November 2026. The current planning horizon is **approximately eight
weeks**, with three team members. Earlier 10- and 12-week plans are historical.
Exact project start/end dates and remaining weeks must come from the team's calendar.

The immediate deliverable is a **progress report due 20 November 2026 at 23:59**,
according to the [supplied assignment](../CENG391_Project_Report_Template/progress_report.txt).
It records completed work, preliminary results, challenges and planned work.
Independent final evaluation is remaining project work, not a prerequisite for
an honest progress report. The final submission deadline is not established here.

## Scope and requirement coverage

The supplied course definition asks for real-time face detection, mask/occlusion
handling, authorized watchlist matching, tracking, scored alerts, comparison of at
least two recognition approaches, difficult-condition evaluation, a simple operator
interface and privacy-aware stored face data.

| Requirement | Current implementation/evidence | Remaining work |
|---|---|---|
| Video pipeline | Single webcam/replay service, bounded capture, timestamps and preview | Live webcam and sustained runtime measurements on team hardware |
| Detection | Haar in the app; SCRFD in neural utilities | Annotated detection results by condition |
| Masks/occlusion | Size/exposure/blur gates and abstention | Physical-mask/occlusion recordings; these gates are not an occlusion classifier |
| Watchlist matching | Authorized multi-sample enrollment, distance/margin rejection and alert toggle | Independent known/unknown validation and thresholds |
| Tracking | Greedy IoU, ambiguity/conflict resets, missing/expiry behavior | Crossing/occlusion switches and fragmentation measurements |
| Scored alerts | Temporal confirmation, persisted events, deduplication/cooldown and acknowledgement | Event correctness, misses and delay on real recordings; scores are not probabilities |
| Two recognition approaches | LBP application; ArcFace and FaceNet execute in preparation utilities | Integrate/evaluate another recognition path under a shared protocol |
| Robustness evaluation | Manifest runner, condition counts and validation threshold selection | Independent pose/light/size/occlusion study and frozen comparisons |
| Operator interface | Monitor, gallery, events/export and settings | Team usability and live-source demonstration |
| Privacy | Encrypted descriptors, local guards, transient photos/previews, retention and deletion | Verify actual study recordings, exports and backup lifecycle |

The earlier planned pair was ArcFace versus FaceNet; the classical baseline now
also exists. LBP versus one neural encoder is a possible smaller two-method study.
Record the team's selected comparison explicitly; neither a still-image execution
probe nor two temporal variants constitutes a completed two-recognizer comparison.

## Preliminary evidence available for the report

Checks rerun on 11 November 2026:

| Check | Observed result | What it supports |
|---|---|---|
| Python suite | 70 passed | Algorithm, state, API and utility regression behavior, including the recognizer boundary and legacy data compatibility |
| Backend Ruff lint/format | Passed | Static/style checks |
| Frontend lint/build | Passed | Linting, type checking and production compilation |
| Classical smoke | 25 offline frames, one correct fixture event, deletion checked | Real Haar/LBP replay and service integration |
| Explicit CPU neural regression | SCRFD single/blank/two-copy counts 1/0/2; finite `[1,512]` encoder outputs | Model execution and preprocessing consistency |

Fresh local smoke reports were written to
`/tmp/odin-recognizer-smoke-20261111/evaluation/report.json` and
`/tmp/odin-recognizer-cpu-20261111/report.json`. These temporary files are not
durable or cloned evidence. Reproduce retained runs under ignored `runs/` with the
[README checks](../README.md#checks-and-learning) before citing them in submission.

The generated video repeats one public photograph for 20 frames, then five blank
frames. It is plumbing evidence, not independent participant accuracy, unknown-person
robustness or 20 independent samples. CPU differences from historical CUDA vectors
were about `9.07e-6` (ArcFace) and `1.40e-7` (FaceNet), measuring numerical consistency.
GPU/live-camera/endurance checks were not rerun in this review.

The shared pipeline now accepts a recognizer that owns preparation, descriptor
validation and matching. Enrollment and evaluation use the same extraction contract;
storage and UI carry labeled raw scores. Tests exercise different crop/color
preparation, signed two-value descriptors and higher-is-better matching, plus legacy
LBP database/policy compatibility. This is an interface regression test double,
not an integrated neural method or a completed second recognition approach. All 25
LBP smoke predictions and aggregate metrics matched the earlier baseline after
the `distance` → `score` field rename.

Historical model preparation includes CUDA execution profiles and FaceNet export
parity on eleven numerical inputs. See the [archive index](archive/README.md) for
dated reports and limitations. Old 18/35/60 test counts describe different milestones;
never present historical timings as current full-application FPS.

## Next work within the eight-week project

This is a dependency order, not an invented calendar or assignment of contributors.

1. Prepare the progress report from actual implementation and preliminary evidence.
   Agree on named task owners, planned dates, current status and contributions.
2. Collect a manageable pilot with authorized participants, independent enrollment
   and validation sessions, unknown people, and required condition labels.
3. Calibrate LBP on validation recordings and inspect failure cases.
4. Integrate the selected neural recognition path with separate model-specific
   preparation, templates and thresholds; compare on the same study inputs.
5. Freeze test choices, complete held-out condition/event results and sustained live
   measurements, and document limitations for the later final submission.

Previously discussed event clips, stronger motion tracking and process isolation
remain unimplemented targets/improvements. Quality-weighted temporal aggregation is
a research hypothesis. Fusion and central multi-camera inference remain independent,
equal-priority optional extensions after required workflows/evaluation have capacity.
No positive fusion gain, new model training, remote accounts or cross-camera tracking
is required by this plan. Preserve earlier preferences in the archive without turning
every proposal into an immediate implementation obligation.

## Progress-report checklist

Preserve the supplied report structure and formatting. Use the working report in
[the report directory](../CENG391_Project_Report_Template/); this cleanup does not
edit its files.

- Problem/dataset: distinguish the current public fixture from planned participant data.
- Methodology: describe the implemented classical pipeline and separate neural preparation.
- Preliminary results: use the evidence above with dates, denominators and claim limits.
- Challenges/next steps: include calibration, independent data, neural integration,
  Haar/tracker limitations and outstanding hardware checks.
- Mandatory Gantt chart: actual named owners, planned start/end dates, status or
  completion percentage, and completed/ongoing/upcoming activities.
- Team contributions: actual individual work; commit dates are not proof of authorship
  or sufficient task dates. Do not infer contributions from proposed role splits.
- GenAI disclosure: actual tool names, explicit versions, purposes, verification and
  use/revisions, based on session records and the supplied course policy.

No participant results, task ownership, percentages or completed experiments should
be invented to fill a report section. The [evaluation guide](EVALUATION.md) defines
how remaining experiments should produce defensible evidence.
