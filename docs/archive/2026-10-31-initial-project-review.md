> **Historical snapshot:** this review was written on 31 October 2026, before
> the classical pipeline was implemented. Its missing-feature statements describe
> that earlier checkout. Use the [current project and study guide](../PROJECT_REVIEW_AND_STUDY_GUIDE.md)
> for the current implementation and learning route.

# odin: project review and study guide

Reviewed 31 October 2026 against the project definition supplied in this conversation.

## 1. Assessment

odin currently contains a tested model preparation toolkit and a detailed proposed architecture. It does not yet contain the requested video recognition application. The useful work is real: pretrained models are available locally, preprocessing has been investigated, FaceNet has been converted to ONNX and numerically checked, and there are defensive utility tests. The remaining work includes almost all application behavior and the actual recognition study.

The main source of overload is that the documentation describes a much larger future system than the code implements. Reading a proposed component contract can feel like reading an implemented capability. Always ask: Where is the code? What input did it process? What output and evaluation prove the claim?

This review adds this guide only. Application source, dependencies, models, original specifications and report templates were not intentionally changed. Frontend verification regenerated ignored build outputs.

## 2. What I examined

- The original 388-line project specification.
- The 709-line architecture, environment handoff, model feasibility notes and both historical implementation plans.
- All Python application, model utility, GPU-check and test source files.
- Frontend source, configuration and package manifests; Python packaging and local model manifests.
- Active model/export artifacts and selected historical run reports, including source/lock hash comparisons. Weight binaries were checked by digest and exercised through inference, rather than treated as readable source.
- Both the LaTeX and extracted PDF report template, its bibliography, and the purpose of its supporting logo/style files.

Installed third-party library trees, generated bundles and every raw profiler event were not individually audited. This is a review of odin and its evidence, not an audit of all its dependencies.

## 3. Requirement coverage today

| Supplied requirement | Existing evidence | Remaining gap |
|---|---|---|
| Real-time video pipeline | Still-image inference utilities | No capture/replay loop, timestamps, sustained video results or integrated preview |
| Detect faces | SCRFD runs on an astronaut, blank image and two-copy image | No annotated video detection evaluation |
| Handle masks/occlusion | Proposed quality/abstention policies | No implemented quality policy or real-mask/occlusion study |
| Match an authorized watchlist | Two encoders produce embeddings | No enrollment, identity gallery, matcher or unknown rejection |
| Track recognized subjects | Proposed tracking contracts | No tracker or temporal identity implementation |
| Alerts with confidence scores | Proposed event and score semantics | No alert engine, score calibration, event store or alert interface |
| Compare two recognition approaches | ArcFace-family and FaceNet-family models execute | No same-protocol recognition accuracy comparison |
| Pose/light/distance/occlusion robustness | Required conditions described in documentation | No participant dataset, held-out condition results or failure analysis |
| Simple operator interface | React/Vite starter compiles | No monitor, enrollment or event workflow |
| Privacy-aware face storage | Sensitive local outputs excluded from Git; proposed policies | No implemented enrollment lifecycle, retention, deletion or verified protected study storage |

An absent capability is not automatically a defect in the completed preparation task: the historical plans explicitly stopped before video implementation. It is still a gap relative to the complete course project.

## 4. The essential ideas

### Detection: where is a face?

A detector takes an image and returns rectangles and scores. SCRFD also returns five landmarks: two eyes, nose and two mouth corners. A detection score indicates the detector's evidence for a face. It does not establish the person's identity.

A video is a sequence of images with time attached. Detection on one still image is a useful first step, but it leaves capture, timing, overload and continuity unanswered.

### Preparation: what image does the recognizer expect?

The recognizer expects a particular crop, size, channel order and numerical range. Alignment uses landmarks to put facial features into a consistent geometry. Cropping alone extracts the bounding rectangle without necessarily correcting head rotation.

Current ArcFace preparation uses a landmark-aligned 112 by 112 crop. Its adapter handles channel conversion and normalization. FaceNet uses a box crop resized to 160 by 160, converted to RGB float32, then standardized with `(pixel - 127.5) / 128`.

For FaceNet, `NCHW` means batch count, channel count, image height, image width. `[1, 3, 160, 160]` is one RGB image. OpenCV commonly supplies BGR pixels, so color conversion matters. Applying another model's preparation can produce plausible-looking vectors with poor identity behavior.

### Embeddings: how is a face represented?

Each selected encoder produces 512 numbers. This vector is an embedding: a representation intended to put observations of the same identity near each other in that model's space.

The model does not output the name Deniz. Names come from your authorized reference gallery. Nor does each coordinate have a simple meaning such as eye color. The network learned the representation before this project; enrolling a person adds reference templates rather than retraining the network.

Both models output 512 numbers, but their spaces differ. Never compare an ArcFace query with a FaceNet reference or average their vectors together.

### Enrollment and matching: whose reference is closest?

Enrollment would take several authorized reference images, detect and prepare each face, encode it, and associate its embeddings with a pseudonymous person ID.

Let `e` be an embedding. Unit normalization is `e / ||e||`. A basic person template is the normalized average of that person's normalized enrollment vectors. For normalized query `q` and template `g`, cosine similarity is simply their dot product:

```text
similarity(q, g) = q · g
```

Search returns the best candidate. Acceptance is a separate decision. Even an entirely unknown person has a nearest gallery entry. A threshold and, where useful, separation from the next identity prevent forced matching. Thresholds must be measured separately for each model and matching policy.

Hypothetical example: the best two identities score 0.71 and 0.70. The tiny gap may indicate ambiguity. A clear gap can be more useful than a high best score alone. These values are teaching examples, not recommended settings.

### Open-set recognition: can the answer be none of these people?

Closed-set recognition assumes everyone belongs to the gallery. Your project must allow people who were never enrolled, so it needs open-set rejection.

Keep these outcomes separate:

- Known: usable evidence supports an enrolled identity.
- Unknown: usable evidence supports rejecting the enrolled identities.
- Insufficient: the face is too small, blurred, obscured or ambiguous to decide.
- Unavailable: the system cannot process the input because of an error.

A hidden face does not become a confidently unknown person. A camera failure does not become a no-face result.

### Tracking: which observation belongs to the same moving face?

A tracker links detections across frames. `track_7` means a local moving object, whereas `person_A` means an enrolled identity. A tracker can follow an unknown face. A recognized person can receive a new track ID after leaving and returning.

The proposed Kalman filter predicts box position/motion; the Hungarian algorithm chooses a one-to-one assignment between tracks and new detections. Neither algorithm establishes identity. During crossings, a wrong association can transfer accumulated recognition evidence between people. Ambiguous associations should invalidate that evidence.

No tracking code exists yet. The architecture's named tracker is a proposal.

### Temporal recognition: what evidence survives across time?

Single-frame labels can flicker. A simple policy accumulates time-spaced accepted observations before confirmation, and expires evidence after it becomes stale.

An example sequence is insufficient → candidate A → confirmed A. Strong contradiction or uncertain tracking should reset the evidence. Losing the face for a while does not create fresh identity evidence.

The proposed research extension weights embeddings by observation quality:

```text
aggregate = normalize(sum(weight_i * embedding_i))
```

It must be tested against simpler voting or unweighted averaging. Adjacent frames are correlated, so repeated agreement cannot be treated as independent probabilities.

### Alerts: when does recognition become an event?

An alert is a policy decision downstream of recognition. A confirmed enrolled person should generate an appearance event only if alerts are explicitly enabled for that person and the evidence is fresh.

Continued presence should not create an alert every frame. Deduplication groups an encounter; a cooldown can help suppress tracker fragmentation. Unknown identity does not itself mean suspicious behavior.

### Confidence: what does the score mean?

Detector score, identity similarity and probability of a correct identity are different quantities. Cosine similarity 0.83 is not automatically an 83% probability.

A useful initial display can show identity similarity, decision state, number/time span of supporting observations and whether the score is calibrated. A probability percentage requires a fitted and independently checked calibration mapping. The supplied requirement asks for confidence scores but does not explicitly demand probabilities.

## 5. How the existing code works

### `backend/src/face_surveillance/api/main.py`

Creates FastAPI and one `/health` endpoint returning `{"status": "ok"}`. There is no model startup, video job, gallery, database or authentication. `/health` proves the process can respond; it cannot prove recognition readiness.

### `frontend/src/App.tsx`

Contains the generated React/Vite page with a counter and documentation links. The Vite config provides a development `/api` proxy to localhost port 8000. This is useful setup, but the interface does not use the vision utilities.

### `vision/model_probe.py`

This is the original feasibility experiment. It verifies local model/overlay hashes, loads SCRFD and ArcFace through InsightFace/ONNX Runtime, and loads FaceNet through a local PyTorch package overlay. It checks simple detection fixtures, prepares model-specific crops, executes repeated inference, validates vectors, records stage timings and inspects actual CUDA execution profiles.

Useful functions include `verify_artifact`, `facenet_input`, `normalize_embedding`, `validate_embedding_series` and `summarize_profile`. These are building blocks and experiment safeguards, not a watchlist system.

### `vision/export_facenet.py`

Converts the pinned FaceNet network/checkpoint to ONNX. It checks graph validity and compares PyTorch outputs with ONNX outputs on eleven inputs. Those inputs include a face crop, altered versions and synthetic images. It checks raw output shape, finiteness, unit norm, component differences and pairwise cosine differences. A successful export publishes a manifest bound to the finalized validation report.

This conversion addresses the older FaceNet Python package's dependency constraints. It does not train a third recognizer or prove that the recognizer knows who is in the image.

### `vision/onnx_probe.py`

This is the current shared-runtime smoke check. It verifies the model bundle and export-report hashes, blocks export-only imports, then runs the detector and both recognizers through ONNX Runtime CUDA. Its fixture is still a photograph, not video. It does not compare identities against a gallery.

The CLI is CUDA-required. The successful CPU diagnostic does not give this CLI a CPU option; an eventual application must implement an explicit device choice.

### Configuration, artifacts and tests

`configs/model_probe.json` describes the original source assets and overlay. `configs/onnx_models.json` selects the active derived FaceNet model and the two original ONNX models. SHA-256 hashes identify exact bytes; they do not establish accuracy, privacy protection or permission to use an asset.

`models/` holds local weights and export evidence. `runs/` holds generated diagnostics and embeddings. `data/` and `experiments/` have placeholders rather than a study dataset. `.gitignore` keeps several sensitive/generated areas out of ordinary Git additions; it does not encrypt or delete them.

The 35 collected tests consist of 34 model utility cases and one API scaffold test. They check invalid vectors, corrupt artifacts, preprocessing, import isolation, export comparisons and related failure paths. They do not test capture, tracking, recognition accuracy, alerts or privacy lifecycle, because those features do not exist yet.

## 6. What verification established

Fresh checks during this review:

| Check | Outcome |
|---|---|
| Model utility tests, explicitly using `PYTHONPATH=backend/src` | 34 passed |
| Ruff lint and formatting | Passed |
| Frontend lint and production build | Passed |
| Active model files and export-report hash links | Passed |
| CPU inference with current source and active models | Detector returned 1/0/2 faces; both encoders returned finite `[1,512]` outputs with valid normalized vectors |
| Full pytest suite | Timed out after 34 cases; API TestClient case stalls |
| Isolated AnyIO blocking portal without application | Also stalled; this localizes the symptom below application endpoint logic |

The deeper cause of the AnyIO/host interaction was not established. It is consistent with the earlier sandbox observation, but the complete suite cannot be reported as passing in this session. Direct `health()` and OpenAPI generation succeed. No dependency changes were made to investigate the stall.

The CPU check emitted an upstream alignment deprecation warning; it did not prevent inference. This is a maintenance consideration when updating dependencies, rather than a reason to alter the frozen environment during review.

Historical GPU evidence was inspected, not remeasured. `runs/shared-onnx-final-merge/report.json` is passed, its probe/helper/lock hashes match current files, and its profiles record CUDA node execution for all three models. Its median stage times were approximately 8.79 ms detector, 3.46 ms ArcFace and 11.03 ms FaceNet. They exclude substantial application work and cannot establish full-system FPS.

The active FaceNet export report records eleven numerical cases, maximum component difference about `2.72e-7` and maximum pairwise cosine difference about `8.49e-7`. This supports conversion equivalence on those inputs. It is not a recognition accuracy result. Repeating one astronaut or placing two copies side by side does not test discrimination between people.

## 7. Review findings and complexity

### Highest priority: application and evaluation are absent

The strongest next evidence would be one annotated recording progressing through detection, preparation, recognition and eventually tracking/events. More infrastructure documentation does not close that gap. Data collection must progress alongside coding: a working pipeline without independent enrollment/validation/test recordings still cannot answer the course's robustness question.

### Scope authority is difficult to follow

The original specification assumes 12 weeks and a larger team/dataset. The later architecture records 10 weeks, three people and approximately 8–10 available volunteers. The original document describes application authentication and encrypted payloads; the later core chooses a trusted local OS operator and encrypted study storage. Some older paragraphs in the architecture still mention a separate FaceNet experiment environment even though its introduction and environment handoff select shared ONNX inference.

These are documentation consistency issues, not proof that code behaves incorrectly. For learning, use the supplied course paragraph as the requirement, actual source/artifacts as evidence, and the later environment handoff for current model setup. Historical notes about earlier user choices are useful context, not new instructions from this conversation.

### Necessary concepts versus additional design

Keep the distinctions between track and identity, unknown and insufficient, similarity and probability, validation and test, and current versus stale evidence. Those prevent fundamental mistakes even in a small system.

The literal course paragraph does not require multi-camera streaming, cross-camera trajectories, model fusion, event video pre-roll, a multi-role account system, distributed messaging or an elaborate entity model. Some are explicitly recorded as prior preferences, so they should be labeled additional scope rather than silently removed. They need not obscure understanding of the core face pipeline.

Quality-weighted temporal aggregation is an optional research hypothesis. Two recognizers and the required robustness comparison already provide a substantial course experiment. A positive improvement from a novel method is not stated as a requirement in the report template.

### Code quality and limits

The utilities make several good choices: explicit local model loading, hash checks, fail-visible CUDA behavior, model-specific preparation, output validation, fresh result directories and honest claim labels.

Their design is intentionally tied to this repository. Root resolution uses `Path(__file__).resolve().parents[4]`; the tools also expect a repository lockfile and local artifact layout. An installed wheel elsewhere would need a configurable asset root. This is a portability limit rather than a blocker for the documented local commands.

The runtime manifest records preprocessing, but the probe largely applies hardcoded crop/provider/input settings. Before using it as an application adapter, make compatibility checks executable so a changed manifest cannot merely describe a different contract while the code keeps the old behavior.

`normalize_embedding` flattens arrays and checks their element count. It would accept an unintended shape with 512 elements. That is adequate for these fixed known outputs, but an application boundary should validate batch and output shape explicitly. Runtime repeatability also normalizes before comparing vectors, so it is weaker than the export's raw-unit-norm check. Neither issue demonstrates a failure of the pinned models observed here.

Privacy remains mostly a design promise. Git exclusions are useful but leave local copies, run embeddings and derived artifacts on disk. Enrollment, retention and deletion need implemented behavior and evidence before being claimed in the report.

### Recognition terminology needs care

ArcFace names an angular-margin training loss; this checkpoint also has a particular architecture and training dataset. The original FaceNet paper uses triplet-based embedding training, while the selected InceptionResnetV1 checkpoint lineage is documented as softmax-trained. Your comparison should describe the two exact pretrained pipelines, not claim to isolate angular loss versus triplet loss. Architecture, training data and current preprocessing differ too. See the [ArcFace paper](https://arxiv.org/abs/1801.07698), [FaceNet paper](https://arxiv.org/abs/1503.03832) and [checkpoint provenance](https://github.com/davidsandberg/facenet#pre-trained-models).

InsightFace's code and pretrained assets have different usage terms; the upstream notice restricts supplied pretrained models to non-commercial research. The project's own notes leave a separate FaceNet weight grant unresolved. Conversion to ONNX does not settle that provenance question. See the [InsightFace notice](https://github.com/deepinsight/insightface#license). This is an asset follow-up already recorded by the project, not a legal clearance assessment.

## 8. Understanding a valid experiment

Use separate recordings for reference enrollment, threshold selection and final evaluation. Known people may appear in all three groups, but the samples/capture sessions must differ. Unknown test people should remain outside the gallery and outside unknown-person calibration where feasible. Randomly splitting adjacent frames creates leakage because nearly identical images end up on both sides.

For each recognizer, compare the same people and recordings, use its required preparation and choose its own validation threshold. Freeze thresholds before testing. A shared detector helps comparability, but full-pipeline comparison still includes different crop conventions. Separately examining recognition on matched observations helps distinguish detection misses from encoding failures.

Each required condition needs actual exposure: pose changes, dim/backlit illumination, different face sizes/distances, masks and other partial occlusion. Include clear known people, unknown people, disabled watchlist members, empty scenes, crossings and reappearances. Start with understandable condition labels; report the participant/session limits honestly.

Useful measurements answer different questions:

| Measurement | Question |
|---|---|
| Detection recall | Of annotated visible faces, how many were detected? |
| Correct known identification | Of eligible known probes/encounters, how many received the correct accepted identity? |
| Unknown false acceptance | Of unknown probes/encounters, how many were accepted as an enrolled person? |
| Unresolved/coverage rate | How often did the system have usable evidence or abstain? |
| Track switches/fragmentation | Did continuity fail or transfer between people? |
| Alert precision and recall | Were emitted events correct, and were eligible appearances detected? |
| Duplicate alerts | Did one encounter repeatedly generate events? |
| Processing and alert delay | Was computation timely, and how long did evidence accumulation take? |

Always state denominators. Correctly identifying 90 of 100 clear crops says little about end-to-end recall if another 300 visible faces were rejected or missed. Report them too. Zero false alerts over a short recording is an observation over limited exposure, not a guarantee. Looping a video adds endurance time, not independent recognition data.

## 9. How the report template maps to odin

The template specifies presentation structure. Its example robotics and drone citations are placeholders, not relevant face-recognition literature. Neither the PDF nor LaTeX adds a requirement to train a model, invent an ensemble or demonstrate improvement.

| Section | Content supported by a completed odin project |
|---|---|
| Abstract, approximately 250 words maximum | Problem, actual data, two methods, principal numerical findings and conclusion; write once results exist |
| Introduction | Watchlist recognition task, course objectives, operating limits and team contribution |
| Related Work | Detector, exact recognition methods, chosen tracking and temporal techniques with relevant citations |
| Dataset and Problem Definition | Participants, recordings, labels, unknown protocol, condition coverage, partitions and data limitations |
| Methodology | Implemented flow, preprocessing, galleries, thresholds, tracking, alerts, interface and privacy controls |
| Experiments and Results | Same-protocol model comparison, condition tables, failures, alerts and runtime with real denominators |
| Discussion | Why errors occur, robustness/coverage tradeoffs, limited data and compute costs |
| Conclusion and Future Work | Achieved objectives and remaining requirements; future extensions clearly labeled |
| Team Contributions | Actual work performed by each member |
| Bibliography | Papers, implementation/checkpoint sources and dataset references actually used |

At today's state you can report model feasibility and export parity as preliminary engineering results. You cannot report watchlist accuracy, tracked encounters, mask robustness, real-time operation or privacy lifecycle as completed results.

## 10. Learning route and self-check

Read in this order: the supplied requirement → this guide's pipeline concepts → `api/main.py` and `App.tsx` → the small preprocessing/normalization helpers → `_execute` in `onnx_probe.py` → the export comparison functions → environment notes → the architecture as a future design reference.

Use one hypothetical enrolled person and one unknown person to explain the whole pipeline aloud. Trace the enrolled person's reference photos into templates, then trace a new frame into a query embedding, a candidate score, a temporally confirmed identity and an eligible alert. Repeat with an unknown person, an obscured face and a camera failure. Each should produce a different reason or state.

Check your understanding:

1. Does a detector score identify the person? No; it concerns face detection.
2. Does enrollment train the pretrained network? No; it creates reference templates.
3. Can two 512-number vectors from different encoders be matched directly? No.
4. Why does nearest-neighbor search need rejection? It returns a nearest candidate even for an unknown person.
5. Can a tracker follow someone whose name is unknown? Yes; continuity and identity are separate.
6. Does cosine 0.9 mean 90% correct? No; that requires independent probability calibration.
7. Does an obscured face prove unknown identity? No; the evidence may be insufficient.
8. What does export parity prove? Numerical agreement between implementations on the tested inputs.
9. Do 34 passing utility cases establish recognition accuracy? No; labeled independent recognition evaluation is absent.
10. What is the first missing connection? Processing a real recording with timestamps through the prepared detector/encoders; gallery matching and the remaining application then need to be added and evaluated.

The project is understandable once its layers are separated: model readiness, identity decisions, video continuity, operator behavior and scientific evidence. Today, most implemented work belongs to the first layer.
