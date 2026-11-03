# odin: project review and study guide

Updated 3 November 2026. Start here if you are joining the project or deciding what
to work on next. This is shared guidance, not a finished course report or a record
of individual student contributions.

## 1. What the project is trying to do

The supplied project definition is:

> Develop a real-time video pipeline that detects faces, handles masks/occlusion,
> matches faces against an authorized watchlist, tracks recognized subjects across
> frames, and generates alerts with confidence scores. The system should compare
> at least two recognition approaches and evaluate robustness under pose,
> illumination, distance, and partial occlusion. Include a simple operator
> interface and privacy-aware handling of stored face data.

The practical goal is one understandable, working local system plus a defensible
comparison of recognition approaches. A large ensemble, multiple cameras, model
training and a polished surveillance platform are additional possibilities. They
are not prerequisites for the required comparison.

The project now has a classical baseline using pretrained Haar detection and a
custom NumPy LBP recognizer. SCRFD, ArcFace-family and FaceNet-family models have
also been prepared and checked, but their recognition pipelines are not integrated
into the operator application yet.

## 2. Read the repository in this order

1. [README](../README.md): installation, operation, data handling and commands.
2. This guide: overall concepts, current evidence and remaining decisions.
3. [Classical walkthrough](CLASSICAL_PIPELINE.md): follow the implemented algorithm.
4. [LBP source](../backend/src/face_surveillance/classical/lbp.py), then
   [pipeline](../backend/src/face_surveillance/classical/pipeline.py),
   [service](../backend/src/face_surveillance/classical/service.py) and
   [storage](../backend/src/face_surveillance/classical/storage.py).
5. [Evaluation runner](../backend/src/face_surveillance/classical/evaluate.py) and
   [example manifest](../experiments/classical.example.json).
6. [Environment handoff](ENVIRONMENT.md) and
   [model feasibility notes](MODEL_FEASIBILITY.md) for the neural preparation work.
7. [Architecture](ARCHITECTURE.md): future design options and historical decisions.
   Check proposed features against actual source before treating them as implemented.

The [initial review](archive/2026-10-31-initial-project-review.md) is preserved as a
dated snapshot. It described the earlier scaffold and is not the current status.

## 3. Current requirement coverage

| Requirement | Implemented or demonstrated | Work still needed |
|---|---|---|
| Video pipeline | One-source webcam/replay service, timestamps, bounded capture, preview and offline replay | Verify the actual webcam and sustained operating profile on team hardware |
| Face detection | Pretrained Haar detector in the app; SCRFD in preparation utilities | Annotated detection results on independent recordings |
| Masks/occlusion | Basic quality rejection and insufficient-evidence states | Physical-mask/occlusion recordings and measured failure rates; current gates are not an occlusion classifier |
| Authorized watchlist | Enrollment, per-person reference descriptors, distance/margin rejection, alert toggle | Validate enrollment quality and unknown rejection on real participants |
| Tracking | Basic spatial IoU association, ambiguity reset, missing/expiry behavior | Measure switches, fragmentation and identity persistence in crossings/occlusion |
| Alerts with scores | Temporal confirmation, deduplication/cooldown, persistent events and acknowledgement | Calibrate policy and evaluate alert correctness, misses and delay |
| Two recognition approaches | Classical LBP path works; two neural encoders execute on fixtures | Integrate/evaluate at least one neural recognition path under the same study protocol |
| Difficult-condition evaluation | Manifest-based runner, per-condition counts and validation threshold selection | Collect and annotate independent pose/light/size/occlusion recordings; run frozen comparisons |
| Operator interface | Monitor, enrollment/gallery, events, source controls and distance settings | Team usability testing and demonstrations with real sources |
| Privacy-aware storage | Encrypted descriptors, local access guards, transient images, deletion and event retention | Verify the study's actual storage, exports and cleanup practices |

Implementation and evaluation are separate milestones. An implemented feature can
still need empirical validation. No independent participant study has been supplied
for this guide; private recordings outside the repository must be documented by the team.

## 4. Understand the processing chain

```text
authorized reference images → face preparation → per-person gallery
                                                    ↓
video → face detection → preparation → descriptor → matching/rejection
       → track association + temporal evidence → eligible event → operator review
```

### Detection locates a face

Detection returns boxes. It does not identify people. The classical app uses a Haar
cascade; the SCRFD preparation utility additionally returns five facial landmarks.
A detection score is evidence for a face, not evidence for a person's identity.

Classical vision can include machine learning. Haar feature calculations are fixed,
but the cascade's feature choices, thresholds and stages were learned from face and
nonface examples. We use OpenCV's pretrained XML; we do not train that detector here.

### Preparation establishes the recognizer's input

The current classical recognizer receives a grayscale crop resized to 128 by 128,
checked for basic size/exposure/blur problems, then histogram-equalized. It uses box
cropping, not neural landmark alignment. Those choices form part of its versioned
descriptor space.

The prepared neural paths use different contracts: ArcFace takes a landmark-aligned
112px crop; FaceNet takes a 160px RGB crop with `(pixel - 127.5) / 128`
standardization. Model-specific preparation matters even when all outputs are finite.

### LBP describes local texture

Our LBP code compares each pixel's eight neighbors with the center. A neighbor at
least as bright as the center sets a bit. In clockwise order from NW, this patch
sets bits 1, 3, 5 and 7, producing code 170:

```text
0 9 0
9 5 9
0 9 0
```

The outer border is excluded. The code image is divided into an 8 by 8 spatial grid;
each cell gets a normalized 256-bin histogram. Concatenating them produces 16,384
numbers. These are basic LBP histograms, not neural embeddings or a trained network.

### Neural embeddings are a different representation

The selected ArcFace-family and FaceNet-family encoders each produce 512 numbers.
They represent appearance in separate learned spaces. Equal vector length does not
make those spaces interchangeable. Neither model directly produces a participant's
name; matching against an authorized gallery provides that association.

ArcFace names a training loss as well as a family label in this project. The chosen
FaceNet-family checkpoint lineage is documented as softmax-trained, whereas the
original FaceNet paper describes triplet training. Describe exact checkpoints,
architectures and preprocessing when comparing; do not claim a controlled loss-only
experiment.

### Matching proposes a candidate; rejection decides whether to accept it

The classical matcher uses a chi-square-style histogram distance, averaged over the
spatial cells. Smaller means closer. It takes the closest sample per person, then
checks the best identity's distance and its margin from the next distinct identity.
An empty gallery is insufficient. A one-person gallery has no runner-up margin.

Neural matching commonly uses normalized embeddings and cosine similarity:

```text
normalized_embedding = embedding / ||embedding||
cosine_similarity(query, reference) = query · reference
```

Cosine similarity is larger for closer vectors; LBP distance is smaller for closer
histograms. Their scales and directions differ. Do not mix model galleries or
average raw winning scores as if they were comparable.

A nearest neighbor exists even for an unknown person. Validation-derived rejection
is essential. Keep these outcomes separate:

- **Candidate:** an enrolled identity is plausible; confirmation is incomplete.
- **Confirmed:** current evidence meets the temporal acceptance policy.
- **Unknown:** a usable observation fails the enrolled-identity acceptance rule.
- **Insufficient:** poor quality, ambiguity, stale evidence or missing gallery prevents a decision.
- **Unavailable:** the source/system cannot process input; this is an error state.

The current defaults are engineering starting settings. LBP distance 0.35 is not a
universal threshold, and a score is not a percentage probability of correctness.

### Tracking connects observations, not names

`track_7` is a temporary video trajectory; `person_A` is an enrolled identity.
The current tracker greedily associates boxes using overlap. It is intentionally
basic. The architecture's Kalman/Hungarian design remains a possible improvement.

Ambiguous association resets involved identity evidence. Contradictory usable
observations can also reset a confirmation. A missing/predicted face does not provide
fresh recognition evidence. Reappearance can create a new track ID.

### Temporal decisions and events have separate roles

The default policy needs three accepted observations, sampled at least 0.25 seconds
apart, spanning at least 0.5 seconds inside a bounded window. Evidence expires.
Agreement between adjacent frames is not independent probability evidence.

Only enabled identities with fresh confirmations can generate appearance events.
Per-track/person suppression and source/person cooldown limit duplicate events.
Acknowledgement means an operator received an event; it does not prove the predicted
identity was correct. Unknown identity does not imply suspicious behavior.

Quality-weighted temporal aggregation and ensemble fusion are still hypotheses to
evaluate. Neither needs to be implemented before collecting useful baseline data.

## 5. Where each implemented responsibility lives

| File | Responsibility |
|---|---|
| [classical/lbp.py](../backend/src/face_surveillance/classical/lbp.py) | Custom LBP codes, regional histograms, distance, identity scoring and rejection |
| [classical/detection.py](../backend/src/face_surveillance/classical/detection.py) | Hash-checked Haar model, detection, crop preparation and quality heuristics |
| [classical/pipeline.py](../backend/src/face_surveillance/classical/pipeline.py) | Spatial association, bounded evidence, confirmation and event proposals |
| [classical/storage.py](../backend/src/face_surveillance/classical/storage.py) | SQLite gallery/events, encrypted descriptors, deletion, retention and acknowledgement |
| [classical/service.py](../backend/src/face_surveillance/classical/service.py) | Enrollment, one-source capture, preview, session invalidation and workflow coordination |
| [api/main.py](../backend/src/face_surveillance/api/main.py) | Local operator routes, bounded command execution and mutation guards |
| [App.tsx](../frontend/src/App.tsx) | Monitor, gallery/enrollment, event review and settings UI |
| [classical/evaluate.py](../backend/src/face_surveillance/classical/evaluate.py) | Exhaustive replay, annotated-frame metrics, encounter scoring and validation-only threshold selection |
| [vision/model_probe.py](../backend/src/face_surveillance/vision/model_probe.py) | Original mixed PyTorch/ONNX technical feasibility utility |
| [vision/export_facenet.py](../backend/src/face_surveillance/vision/export_facenet.py) | FaceNet ONNX export and numerical parity checks |
| [vision/onnx_probe.py](../backend/src/face_surveillance/vision/onnx_probe.py) | Strict CUDA shared-model execution check, not a watchlist application |

## 6. Setup and privacy in practice

Use the [README commands](../README.md#run-the-operator-application). Normal
`uv sync --locked` manages Python dependencies and editable installation. The project
uses `opencv-contrib-python`, replacing the main OpenCV wheel; both share `cv2` and
must not be installed together. No manual wheel installation or `PYTHONPATH` override
is required by the documented workflow.

The Haar model is a separate, hash-pinned asset acquired by a script. Neural weights,
study data, keys and generated run outputs are intentionally ignored by Git. Teammates
need their own permitted local assets; cloning the repository does not clone those files.

The classical app stores encrypted reference descriptors, not original enrollment
photographs. Unknown observations and previews are transient. Uploaded replay copies
are removed after processing; private study recordings retained separately have their
own lifecycle. Deletion clears linked templates/events and active pipeline evidence.

Encryption keys live locally with restricted permissions. This is a trusted local
operator design, not protection against a process running as the same OS user.
Exported reports and backups need separate handling; SQLite deletion is not a physical
SSD-erasure guarantee. Model code licenses do not automatically establish weight/data
usage rights; existing provenance limitations remain in the feasibility notes.

## 7. What the current evidence proves

The last recorded verification passed 60 software tests, backend lint/formatting,
frontend lint/build, a real Haar/LBP replay fixture and CPU checks of the three neural
models. Treat this as dated evidence, and rerun the README checks on your checkout.

The generated replay uses one public photograph repeated across 20 face frames,
followed by five blank frames: 25 frames over 2.5 seconds. The saved run detected all
20 face instances, confirmed the identity on 14 frames after six collecting-evidence
frames, and emitted one event at 0.6 seconds with no duplicate event.

This demonstrates pipeline integration and temporal behavior on that fixture. It is
not 20 independent face samples, an unknown-person experiment, or a general accuracy
estimate. In particular, do not report 14/20 as recognition accuracy on new people.

The saved CPU neural check produced finite 512-element embeddings. Maximum component
differences from historical CUDA outputs were approximately `9.07e-6` for ArcFace and
`1.40e-7` for FaceNet. Export parity and repeatability establish numerical behavior
on the tested inputs; they do not establish identity discrimination or mask robustness.

Historical GPU timings are short instrumented model-stage measurements, not current
full-system FPS. The implementation session could not access a working GPU driver or
webcam inside its sandbox. Team hardware/live-source checks remain necessary.

Local evidence includes `runs/uv-clean-smoke-01/evaluation/report.json` and
`runs/uv-clean-cpu-01/report.json`; these ignored files will not appear in a fresh
clone. Reproduce comparable checks using
[classical_smoke.py](../scripts/classical_smoke.py) and
[check_vision_cpu.py](../scripts/check_vision_cpu.py). Use a new output directory.

## 8. Decide the next work in this order

1. **Reproduce and explain the baseline.** Each teammate should run it, trace one
   known/unknown/insufficient observation, inspect a score, and explain an event.
2. **Agree on a manageable study.** Choose available authorized participants, camera,
   recording sessions, required conditions and annotation responsibilities. Prioritize
   independent sessions and some unknown identities over large frame counts.
3. **Collect a pilot and calibrate LBP.** Use separate reference and validation
   recordings. Inspect accepted, rejected and ambiguous score distributions.
4. **Integrate the next recognition approach.** Reuse source/track/event/evaluation
   infrastructure, but keep model-specific preparation, galleries and thresholds.
5. **Freeze and run comparisons.** Use the same independent test observations and
   report both recognition-only and complete-pipeline behavior where practical.
6. **Investigate measured failures.** Improve tracking or quality handling when data
   shows a concrete problem. Consider fusion/multiple cameras after required evidence
   and working workflows are secure.

For the required two-method comparison, LBP versus one neural encoder is a useful
classical/deep pair. The earlier plan selected ArcFace versus FaceNet; the team can
retain that comparison and add LBP, provided time permits. Record the chosen study
scope explicitly. Two temporal variants of one recognizer do not replace two
recognition approaches. Confirm any more specific assessor interpretation if one exists.

The team still needs to decide the actual participants/splits, next encoder,
condition definitions, task owners and calendar. The guide does not assign names or
invent completed student contributions.

## 9. Make the experiments defensible

Keep enrollment, validation and final test capture sessions separate. Known identities
can appear in gallery and test, but their images/recordings must differ. Keep final-test
unknown identities out of enrollment and unknown-person calibration. Randomly splitting
adjacent frames, or putting augmented copies of one photo in different splits, leaks data.

Include frontal/turned pose, bright/dim/backlit scenes, measured face-size/distance
bands, real masks/partial occlusion, crossings, disappearance/reappearance, disabled
alert identities, unknown-only intervals and empty scenes. Synthetic occlusion and
repeated photos are supplementary fixtures, not substitutes for those recordings.

Choose each model's threshold on validation only and freeze it before test. Hold
gallery identities, reference/probe splits and event rules constant across comparisons.
Record preprocessing and detector differences rather than claiming a loss-only comparison.

| Measurement | What to count or explain |
|---|---|
| Detection recall | Annotated visible faces found, broken down by condition |
| Known identification | Correct accepted, wrong identity, rejected, unresolved and detection-missed outcomes |
| Unknown false acceptance | Unknown observations incorrectly accepted as enrolled identities |
| Coverage | How often the pipeline has enough usable evidence; include rejected/missed cases |
| Tracking | Switches, fragments and identity inheritance errors in annotated sequences |
| Events | Correct, false, missed and duplicate events; delay alongside never-confirmed encounters |
| Runtime | Processing/preview cadence, drops, frame age, processing delay and confirmation delay |

State numerators, denominators and unique recording exposure. Report per-condition
outcomes; an aggregate can hide complete failure on masks or small faces. A correct
abstention is preferable to a forced identity but still counts as a missed recognition
opportunity. Zero observed false alerts over a short clip does not establish zero risk.

## 10. Use the progress report as a record of understanding

The progress report is due **20 November 2026 at 23:59**, according to the supplied
assignment. Use the provided CENG391 template, preserve its formatting and section
structure, and incorporate the assignment's additional mandatory material.

| Template/assignment area | Team material to prepare |
|---|---|
| Abstract and Introduction | Problem, current scope and actual progress; avoid promised results |
| Related Work | Explain the methods actually selected and verify relevant citations |
| Dataset and Problem Definition | Current fixtures versus participant study, labels, conditions, partitions and limitations |
| Methodology | Implemented steps, preprocessing, parameters, score semantics and data lifecycle |
| Experiments and Results | Preliminary functional/model checks clearly separated from independent recognition evaluation |
| Discussion / Challenges | Haar limits, threshold selection, tracking ambiguity, environment constraints and study validity |
| Conclusion / Next Steps | Completed work and specific remaining tasks |
| Mandatory Gantt chart | Individual tasks, planned start/end dates, actual named owners, status/progress and completed/ongoing/upcoming work |
| Team Contributions | Actual individual responsibilities and work performed so far |
| GenAI disclosure | Actual tools, explicit model versions, purposes and how outputs were checked/used/revised |

Use commit dates as evidence of repository milestones, not as invented task start/end
dates or proof of individual authorship. The team should agree on its real Gantt chart
and contribution statements. Proposed future dates must be labeled planned. The final
course-project deadline has not been supplied here; do not infer it from the progress
report deadline. Submit one group PDF using the actual group ID, such as `G11.pdf`.

This implementation and these notes involved substantial AI assistance. Read the course
policy and establish what assistance is permitted; record actual verified student
contributions and tool/model versions from session records. Each teammate should be
able to explain and reproduce submitted material. This guide supports that learning
and planning process; it does not establish authorization or replace the team's report.

## 11. Questions every teammate should be able to answer

1. How does a Haar detection differ from an identity match?
2. Which parts use pretrained models and which algorithms are implemented here?
3. How does the example patch produce LBP code 170 and a face produce 16,384 values?
4. Why does a nearest-neighbor result still need unknown rejection?
5. Why can LBP distance, ArcFace similarity and FaceNet embeddings not be mixed directly?
6. What separates a track ID, person ID, candidate, confirmation and appearance event?
7. What happens when evidence conflicts, two tracks overlap, a source stops or a person is deleted?
8. What does the generated replay prove, and what evidence is still missing?
9. How will enrollment, validation and test sessions be kept independent?
10. What concrete experiment will determine whether the next planned improvement helps?

Answer these with the code and real observations. Where the answer depends on data
the team has not collected, record that as the next experiment rather than a conclusion.
