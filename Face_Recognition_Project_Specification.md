# Face Recognition and Surveillance Alerting

## Project specification — 12-week course project

Version: 1.0 proposed design · Date: 12 October 2026

The team has selected project #1 from the supplied CENG391 project pool. The user confirmed a 12-week timeline. Team size, hardware, submission rubric, and participant availability are not supplied. Resource planning assumes a team of 3–4, access to a webcam and ordinary desktop/laptop, and optional NVIDIA GPU access. These assumptions change workload and performance targets, not the core project definition. This document defines the project; it does not report an implemented or validated system.

## 1. Project definition

Build a local, real-time computer vision application that detects and tracks faces in video, compares usable face observations against a deliberately enrolled and authorized identity gallery, rejects identities absent from that gallery, and generates reviewable alerts when a configured enrolled identity is recognized with sufficient temporal evidence. The application presents a live view, track status, interpretable match evidence, event history, enrollment management, and measured system health.

The primary research question is: **Can quality-aware temporal recognition reduce false identification and false alerts while maintaining useful recognition recall and response time under pose, lighting, distance, mask, and occlusion changes?**

The primary deliverable is one complete vertical system: enrollment → video ingestion → detection → tracking → quality/alignment → embedding/matching → temporal identity decision → alert → operator review. A reproducible experiment suite is equally important to the interface.

### Terms

- Detection: locate a face, without identifying it.
- Tracking: associate observations with a local video track. A track ID is not a real-world identity.
- Verification: compare an observation against one claimed identity; used as a supporting experiment.
- Identification: choose among enrolled identities.
- Open-set identification: identification with the ability to reject a person absent from the enrolled gallery.
- Gallery: enrolled identities, embeddings/templates, model versions, and alert policies.
- Watchlist: the subset of enrolled identities explicitly configured to produce an alert. Enrollment alone does not make a person an alert subject.
- Identity alert: an observation event for an enrolled identity, not an assertion about that person's behavior or intentions.

## 2. Course requirement coverage

| Requirement in supplied definition | Design response | Evidence at submission |
|---|---|---|
| Real-time video pipeline | Webcam, video replay, timestamped bounded processing pipeline | Live demonstration and hardware-specific profiling |
| Face detection | Landmark-capable detector with explicit missed-detection analysis | Detection results by face size, pose, and occlusion |
| Masks/occlusion | Evaluate physical masks and partial occlusion; reject unusable observations | Condition-specific recognition, coverage, and failure results |
| Authorized watchlist matching | Controlled enrollment plus policy-defined alert subset | Enrollment workflow and positive/negative scenarios |
| Tracking recognized subjects | Face tracks maintain separately managed identity hypotheses | Annotated crossing, disappearance, and occlusion sequences |
| Alerts with confidence | Calibrated identity evidence and event records | Threshold provenance, alert metrics, and event display |
| At least two recognition approaches | ArcFace-family embeddings versus FaceNet-family embeddings | Controlled comparison and dataset/model provenance |
| Pose, illumination, distance, occlusion robustness | Deliberately recorded evaluation matrix | Per-condition tables with denominators and uncertainty |
| Simple operator interface | Live view, gallery, event review, configuration, metrics | Complete user workflows |
| Privacy-aware stored face data | Minimized storage, access control, retention, deletion, protected templates | Tested deletion/access/retention behavior |

Handling occlusion means detecting it where possible, measuring its effect, and abstaining when necessary. It does not promise identification through every mask or fully hidden face.

## 3. Scope and boundaries

### Required release

1. Single-camera live processing and reproducible recorded-video replay.
2. Multiple visible faces tracked concurrently within one camera.
3. Enrollment from several accepted samples; add, replace, deactivate, and delete identities.
4. At least two recognition methods behind the same conceptual interface.
5. Open-set rejection, low-quality abstention, and temporally confirmed identity decisions.
6. Configurable alerts with deduplication, operator acknowledgement, and event history.
7. Full difficult-condition evaluation and controlled ablations.
8. Local authenticated operator interface and privacy-aware data lifecycle.
9. CPU execution path, optional GPU execution, stage-level profiling, and reproducible installation.
10. Recoverable source failures, bounded queues, clean shutdown, and stable long runs.

### Separately evaluated extensions

Select at most one substantial extension after the core is stable: (a) two simultaneous camera streams and cross-camera identity evidence; (b) print/replay anti-spoofing; (c) C++/ONNX deployment of the inference path; or (d) occlusion-focused model adaptation. An extension requires its own data, metrics, and limitations. Independent camera streams do not by themselves demonstrate cross-camera tracking. Blink detection does not by itself establish liveness. Default preference: deployment optimization; choose anti-spoofing only if the use case becomes access-control verification.

### Explicitly outside the required release

- Searching the internet or unrestricted databases to discover a person's identity.
- Public-camera deployment or collecting uninvolved people for the demo.
- Inferring emotions, age, ethnicity, personality, intentions, or suspiciousness.
- Automatic door control, punitive decisions, or dispatching messages to external recipients.
- Guaranteed recognition at arbitrary distances, under complete occlusion, or in all conditions.
- A certified security product or a claim that the system is production-ready.
- Training a large face-recognition foundation model from scratch.
- Nationwide tracking, distributed camera infrastructure, mobile applications, or cloud-scale operation.

## 4. Design approaches and selection

| Approach | Benefit | Limitation | Decision |
|---|---|---|---|
| Library-led image-recognition demo | Fast baseline and easier debugging | Limited temporal/system contribution | Use only as the first baseline |
| Modular local video system using pretrained encoders plus original temporal decision logic | Complete application, interpretable experiments, manageable compute | Requires careful data collection and integration | Recommended architecture |
| Custom-trained face model plus distributed multi-camera platform | Large research/system scope | Large data/compute demands; multiple independent research questions | Beyond the required release |

The selected design uses pretrained representation models, with the team's contribution centered on evaluation, quality selection, temporal identity decisions, false-alert management, and efficient integration. Optional small-scale adaptation is an extension. Enrollment adds templates; it does not require retraining a classifier.

## 5. User workflows

### Enrollment

An authorized administrator creates a pseudonymous identity, records authorization status, and submits images or short video. The system checks that each sample contains exactly one detectable face, reports quality problems, aligns accepted faces, computes templates for each enabled recognition model, and displays the accepted sample count. Aim for 5–10 diverse accepted samples per identity. Duplicate samples are not considered independent evidence. The administrator explicitly chooses whether this identity generates alerts. Invalid enrollment leaves the existing gallery unchanged.

### Live operation

An operator selects a configured source and sees camera health, timestamped video, face boxes, track IDs, identity decision state, and relevant evidence. The interface distinguishes unknown identity from insufficient-quality evidence. An eligible, confirmed watchlist match produces an event. The operator can acknowledge it, mark it as a confirmed or incorrect match, and inspect associated evidence. Review labels enter a separate evaluation/review store and never silently update templates or thresholds.

### Evaluation replay

An experiment command runs named videos using an immutable gallery snapshot, model identifiers, and configuration. It writes predictions, per-stage timings, events, and metrics. Offline exhaustive processing and live low-latency processing are separate modes; results state which mode was used.

### Data lifecycle

An administrator can deactivate an identity immediately or delete it. Deactivation removes it from matching/alert eligibility. Deletion removes templates, configured source samples, associated thumbnails, and identifiable event associations; minimal non-identifying audit records may remain. In-memory galleries refresh atomically, so deleted identities cannot continue producing new matches.

## 6. Architecture and component contracts

### Processing and service boundaries

Video ingestion workers feed bounded queues. The vision worker performs detection, tracking, face preparation, recognition, and temporal decisions. An event writer persists policy-approved events. FastAPI exposes state and workflows; the web interface displays previews and metadata. Slow preview clients cannot block inference. Do not run heavy inference directly inside API request handlers.

| Component | Input | Output / responsibility |
|---|---|---|
| Source adapter | Webcam/video/configuration | FramePacket(source_id, sequence, capture_time, image); reconnect/end-of-file status |
| Detector | FramePacket | Boxes, detector scores, landmarks, model version |
| Tracker | Detections and elapsed time | Camera-local track IDs, association state, missing/expired state |
| Face preparer | Face region, landmarks, image | Model-specific aligned crop; blur, size, exposure, pose/occlusion proxies |
| Recognition adapter | Accepted crop | Normalized embedding and encoder/version identifier |
| Gallery matcher | Embedding plus matching model's gallery | Best and second-best identity scores; threshold/margin evidence |
| Temporal decision engine | Track observations and quality | Candidate/confirmed/unknown/unresolved states and decision reason |
| Alert engine | Decision plus identity policy | Deduplicated event or suppression reason |
| Persistence | Identity/templates/events/configuration | Transactional storage, retention/deletion, schema version |
| Operator service | Authenticated actions and state subscriptions | Enrollment, configuration, event review, preview, health |
| Evaluation runner | Dataset manifest, config, gallery snapshot | Predictions, metrics, timings, experiment manifest |

Track state and identity state are distinct. A changed track association can invalidate accumulated recognition evidence. On an ambiguous association, reset identity evidence or terminate/restart the track; do not inherit a confirmed identity solely from spatial proximity.

### Operating modes

- Live mode: prioritize recent frames; bounded queues may discard stale frames. Preserve timestamps and count drops. Tracking uses actual elapsed time.
- Offline mode: process every selected frame deterministically; wall-clock throughput may differ from live responsiveness.
- Demo mode: replay an explicitly labeled recording; never present prerecorded output as live inference.

### Persistence entities

Identity; per-model Template; EnrollmentSample metadata; Source; Policy; TrackSummary; AlertEvent; ReviewLabel; Experiment; AuditEvent. Templates carry identity ID, encoder name, checkpoint hash, preprocessing version, and creation time. Embeddings from different models or preprocessing configurations are never mixed. Model changes require explicit template regeneration and threshold recalibration.

### API surface

Authenticated endpoints cover identity create/list/update/delete, sample enrollment, source start/stop/status, alert list/detail/acknowledgement/review, configuration validation, metrics, and experiment exports. A WebSocket publishes track/event/health metadata; preview frames use a separately rate-limited image stream. Identity mutations are administrator-only. Operator access is limited to observation/review. Configuration changes are versioned and audited.

## 7. Recognition and identity decision design

### Baselines

Use an ArcFace-family encoder through InsightFace and a FaceNet-family InceptionResnetV1 implementation through facenet-pytorch. These are two different pretrained recognition pipelines, not a controlled comparison of loss functions alone: architecture and training data also differ. An optional LBPH baseline can contextualize classical recognition on a modest controlled subset; it does not replace the two main methods.

Use one shared landmark-capable detector where practical. Compare recognition on the same face observations while retaining each encoder's required crop size and normalization. Report both fixed-crop recognition results and complete-pipeline results. Alignments must follow each model's expected landmark/crop convention; uncontrolled preprocessing changes make comparisons misleading.

### Gallery and scoring

Compute normalized embeddings for accepted enrollment samples. Evaluate one identity-centroid representation against a small multi-template representation. For a query, return top identity similarity and the margin to the second distinct identity. Aggregate templates per identity before computing this margin; two templates of the same identity are not competing identities.

Nearest-neighbor similarity always returns a candidate, even for an unknown person. Therefore, acceptance requires a validation-tuned threshold, sufficient separation from the next identity, acceptable observation quality, and stable temporal evidence. Thresholds are specific to model, template policy, and gallery size.

### Decision states

- Unresolved: evidence is too small, blurry, occluded, ambiguous, or insufficient in time.
- Candidate: an enrolled identity is plausible, but confirmation requirements are incomplete.
- Confirmed: the identity meets all configured observation and temporal conditions.
- Unknown: enough usable evidence exists, but no enrolled identity passes the acceptance criteria.

Track lifecycle (active, missing, expired) is orthogonal. A confirmed label can briefly persist through missed detections only with a visible stale-evidence indicator and no new alert based solely on stale evidence. Strong contradictory evidence demotes or resets the identity. Expiration discards the track evidence.

### Temporal contribution

Compare: single-frame decisions; unweighted temporal voting/aggregation; and quality-weighted temporal aggregation. Quality features include measurable face pixel size, sharpness, exposure, landmark/pose consistency, and available occlusion signals. Use time-separated observations; dozens of nearly identical adjacent frames do not establish independent confidence. Choose the observation spacing, time window, minimum evidence, acceptance margin, and demotion behavior using validation data.

The initial quality-weighted method forms a normalized mean of normalized embeddings within one model and one track: normalize(sum(q_i * e_i) / sum(q_i)), where e_i is an embedding and q_i is a nonnegative quality weight. Unweighted aggregation uses q_i = 1. No accepted observations or zero total weight yields unresolved. Quality weights are derived from the declared image-quality features with a versioned, validation-selected mapping. Match the aggregate against that model's gallery. Require the same identity to pass the configured acceptance test across time-spaced updates; averaging alone does not confirm identity. Reset the aggregate on identity/association conflicts rather than mixing two people.

Starting policy values, to be tuned on validation and frozen before final testing: at most one recognition observation per track per 250 ms; a two-second evidence window with at most 16 stored observations; at least three accepted observations spanning at least 500 ms before confirmation; a one-second stale-evidence limit; track expiration after two seconds without association; and a 30-second camera/identity alert cooldown. Use actual timestamps. Identity similarity/margin thresholds and image-quality floors have no universal numeric defaults and are derived separately for each model using validation data. Publish both initial and final policies; these starting values are not validated performance claims.

### Confidence semantics

Detector confidence, identity similarity, and track-level confidence are different quantities. Cosine similarity is not a percentage probability. Display identity similarity as a labeled score. If a probability-like confidence is included, fit and validate a calibration mapping on held-out validation observations under the declared gallery/protocol, report calibration error/reliability plots, and flag operating conditions outside that calibration. Alerts retain raw scores, quality, supporting observation count/span, threshold/configuration version, and calibration version.

## 8. Alert policy

Required event type: temporally confirmed appearance of a watchlist-enabled enrolled identity. Unknown people do not generate identity alarms by default. Optional configured zones may restrict where an enrolled-identity appearance event is created, but do not imply suspicious behavior.

Event eligibility requires an enabled policy, an active identity, fresh evidence, and a confirmed identity decision. Deduplication uses camera, identity, encounter, and cooldown/rearming rules. Define an encounter as a continuous tracked appearance; brief missed detections within the configured gap do not start a new encounter. Cross-track duplicate suppression can use a short camera/identity time window, with missed-event tradeoffs measured. Reappearance after the rearming interval permits a new event.

Each event records source, timestamps, track/identity pseudonyms, model/configuration versions, raw/calibrated evidence, quality reason, optional retained thumbnail, and acknowledgement/review status. No external email/SMS integration is required. Human review does not erase the original model prediction.

## 9. Data collection and experimental protocol

### Primary system dataset

Planning target: approximately 20 enrolled participants plus 12 non-enrolled participants, with 2–4 hours of distinct mixed footage and multiple recording sessions. Participation must be explicitly authorized. Actual counts and hours are reported; limited recruitment limits generalization claims. Use pseudonymous identifiers in experiments and the public report.

For enrolled participants, separate enrollment, validation, and final test sessions. For unknown participants, separate validation identities from final test identities (e.g. six in each partition). Final-test unknown identities never enter the gallery. If adapting an encoder, use a separate development dataset/identities where feasible and state any overlap; enrollment samples are not model-training examples.

Record controlled and mixed scenarios: frontal/turned faces; bright, dim, and backlit conditions; varying distance/face pixel size; glasses; physical masks; partial occlusion; motion blur; multiple faces crossing; appearance/disappearance; two enrolled identities in one scene; unknown people; watchlist-disabled enrolled people; no faces; camera loss. Measure face pixel size rather than presenting meters alone as a universal operating limit.

Physically worn masks and real occlusion are required for the robustness evaluation. Synthetic occlusion can supplement experiments but must be reported separately. A mask detector is optional; segmentation of masks is not mandatory for recognition abstention.

Annotate a representative subset with face boxes, track IDs, ground-truth identities/unknown labels, condition tags, and alert encounter boundaries. Include negative-only footage. Have a second person review a subset of annotations and report disagreements. Longer repeated playback is useful for runtime testing but does not increase independent recognition/false-alert evidence.

### Public supporting data

WIDER FACE can evaluate detection under varied scale, pose, and occlusion. It is a detection benchmark, not a substitute for the identity/alert dataset. Public verification benchmarks may support encoder checks if accessible under their actual terms; they are not the completion dependency. Standard pretrained benchmarks do not establish performance for this camera setup, participant population, or gallery.

### Experimental controls

Freeze the final dataset manifest before final evaluation. Tune detection thresholds, quality rules, identity thresholds, margins, temporal windows, and cooldowns only on development/validation data. Preserve complete capture-session/video groups in partitions; do not randomly distribute adjacent frames. Evaluate the same recordings and gallery snapshots for each method. Record checkpoint hashes, seeds, software versions, hardware, gallery size, preprocessing, and configuration.

## 10. Metrics and required experiments

| Layer | Metrics / reporting |
|---|---|
| Detection | Precision/recall and AP on annotated subsets; recall by face pixel size, pose, mask, occlusion |
| Verification support | Genuine/impostor score distributions, ROC, TAR at feasible measured FAR operating points |
| Open-set identification | Correct enrolled-identity rate, wrong-identity rate, unknown false-accept rate, known rejection rate; state the per-probe denominator |
| Abstention | Usable-observation coverage; correctness versus coverage; unresolved rate and duration |
| Tracking | ID switches and fragmentation; IDF1 on annotated sequences where ground truth is adequate |
| Events | Alert precision/recall, false identity alerts/hour, duplicate alerts/encounter, missed encounters, time to correct alert |
| Calibration | Reliability plot and calibration error when confidence is presented as probability-like |
| Runtime | Input/processed/preview FPS, dropped frames, queue age, p50/p95 latency, per-stage time, CPU/GPU/RAM use |

Define an eligible positive encounter as a ground-truth watchlist identity appearance meeting the predeclared visible-duration/visibility protocol. Report total positive encounters and those excluded by eligibility criteria. A correct alert must match camera and identity within the encounter's defined window. Wrong identity, unknown acceptance, duplicate, and stale post-departure events receive separate counts. Report unsuccessful encounters alongside latency for successful ones to avoid hiding misses.

Report confidence intervals or session-level bootstrap estimates where feasible. Very small samples cannot substantiate extremely low false-accept rates. Zero observed false alerts is a count over stated exposure, not proof of a zero error rate. Frames within a clip are correlated; uncertainty should use encounter/session units where possible.

Required experiment set:

1. ArcFace-family versus FaceNet-family under matched protocols.
2. Single-frame versus temporal decisions.
3. Unweighted versus quality-weighted temporal decisions.
4. Single-centroid versus multi-template gallery.
5. Per-condition robustness and mixed difficult scenes.
6. Threshold tradeoff between correct recognition, false alerts, rejection, and delay.
7. Gallery-size sensitivity, within the available identities.
8. CPU versus available GPU execution and at least one profiling-driven optimization.

Select an operating point using validation false-alert/recall constraints, then keep it fixed for final testing. Report methods that miss the constraint rather than silently replacing the test threshold.

## 11. Technology stack and resource assumptions

| Area | Selected technology | Purpose / boundary |
|---|---|---|
| Vision and research | Python, NumPy, OpenCV, PyTorch | Image preparation, experiments, adapter implementations |
| Primary detector | SCRFD through InsightFace, with checkpoint-specific landmarks | Face localization and alignment support; freeze exact checkpoint |
| Recognition A | ArcFace-family InsightFace checkpoint | Primary embedding baseline |
| Recognition B | facenet-pytorch InceptionResnetV1 | Independent recognition baseline; freeze checkpoint/provenance |
| Tracking | Kalman/Hungarian face-box baseline; ByteTrack-style association comparison if warranted | Tracking remains separately evaluated; adapt to face motion/size |
| Inference runtime | ONNX Runtime CPU; CUDA provider when compatible hardware exists | Reproducible deployment and optional acceleration |
| Backend | FastAPI, Pydantic | Typed workflows, validation, state/event APIs |
| Interface | React, TypeScript, Vite | Live view, enrollment, review, configuration, metrics |
| Persistence | SQLite with SQLAlchemy; encrypted sensitive payloads | Local transactional application; PostgreSQL only if measured concurrency needs justify it |
| Experiments | Versioned YAML configs, JSON/CSV outputs, local MLflow tracking if useful | Comparable runs and reproducible provenance |
| Quality checks | pytest, formatting/linting, Git, CI | Algorithm and workflow regression checks |
| Packaging | Documented environment plus Docker Compose where hardware support is tested | Repeatable run; GPU setup documented separately |

InsightFace code and pretrained models have different terms: the official project states that supplied pretrained models are restricted to non-commercial research. This fits an academic baseline but is not an unrestricted commercial redistribution/deployment assumption. Record each model/data license and distribute download instructions rather than unauthorized weights/data. Disable unrelated age/gender analysis modules. FaceNet-family implementation/checkpoint terms also need a recorded check; code licensing alone is not proof of dataset/weight rights.

Do not choose exact package versions before checking their joint Python/PyTorch/ONNX/CUDA compatibility on the team's hardware. Freeze a tested environment at milestone 1 and publish a lockfile. ONNX Runtime supports CPU and NVIDIA CUDA execution, but throughput remains model/hardware dependent.

Resource planning: 16 GB system RAM is a useful development target; an NVIDIA GPU with roughly 6–8 GB VRAM is helpful rather than a core requirement for pretrained inference. Hardware access is unconfirmed. CPU fallback must operate with an explicitly declared lighter resolution/cadence profile. Training from scratch is unnecessary. Live camera tests and video replay are required; RTSP/network-camera ingestion is an optional adapter unless equipment is available.

Provisional live benchmark profile: one 720p source, up to five visible faces, gallery of 20 identities. Initial targets are at least 15 processed FPS on the declared accelerated machine and p95 capture-to-decision processing latency below 250 ms, excluding the intentional temporal confirmation window. These are engineering goals, not promised model performance. At week 2, publish a hardware-specific achievable profile and retain results for the original profile. Report alert confirmation delay separately. Measure a lighter CPU profile rather than claiming the same performance everywhere.

## 12. Operator interface

Required views: (1) live camera/replay with identity/track states; (2) identity gallery and enrollment quality feedback; (3) alert list, details, acknowledgement, and review labels; (4) source/policy configuration; and (5) runtime metrics and experiment exports. A visible legend explains unresolved, unknown, candidate, confirmed, and stale evidence. Model similarity is labeled as similarity; probability-like confidence appears only with calibration provenance.

The interface must surface source disconnects, paused/replayed sources, dropped frames, model load failures, and gallery/model mismatches. It must never depict a stale last frame as a healthy live feed. Show last frame time and connection status. Operator review and administrator mutation permissions differ.

## 13. Stored-data protection and operational behavior

Store the minimum needed for the authorized local study. Unknown-face images/embeddings remain transient by default. Do not store full live video automatically. Event thumbnails are off by default and enabled only for specifically authorized evaluation/demo footage. Proposed event metadata retention is seven days; optional thumbnails are limited to 24 hours. Research footage uses a separately declared study retention schedule and is removed after the agreed evaluation period.

Treat embeddings as sensitive identity data, not anonymous data. Encrypt stored templates and optional face images using a maintained authenticated-encryption library, with keys held outside the database/repository in restricted configuration. Protect stored source credentials similarly. Do not log embeddings, face images, passwords, or stream credentials. Do not commit participant footage, templates, or secrets. Authenticate operator/admin sessions and enforce permissions server-side. Bind locally by default; any network deployment requires its own transport/access configuration.

If encryption keys/models/database fail, source health must show the error and identification must fail safely to unavailable/unresolved rather than fabricating an identity. A model change cannot use incompatible templates. Restart begins fresh tracks; persistent events and gallery remain intact. Database writes are transactional; failure never produces a UI-only alert that appears durably saved. Stop/shutdown closes sources and workers cleanly.

## 14. Twelve-week milestones and exit evidence

| Week | Milestone | Exit evidence |
|---|---|---|
| 1 | Requirements, architecture, environment, authorization/data protocol | Course coverage table; tested model loads; data manifest design; contributor ownership; frozen initial environment |
| 2 | End-to-end single-frame baseline | Webcam and replay produce detections, embeddings, gallery match scores; timing baseline; initial enrollment/validation recordings; hardware profile |
| 3 | Recognition comparison and enrollment | Two model adapters; transactional multi-sample gallery; same-observation comparison; first unknown-person rejection curves |
| 4 | Tracking and session data | Multi-face face tracks; crossing/occlusion annotations; resets/expiry behavior; independent enrollment/validation/test sessions planned and captured progressively |
| 5 | Quality-aware face preparation | Pixel-size/blur/exposure/pose checks; unresolved reasons; preliminary physical-mask/occlusion results |
| 6 | Temporal identity decisions | Single-frame, unweighted, quality-weighted configurations; validation threshold tradeoff; contradiction/stale-evidence tests |
| 7 | Alert engine and persistence | One event per defined encounter; cooldown/rearming; correct watchlist-disabled/unknown behavior; persisted reviewable events |
| 8 | Complete operator workflow and data lifecycle | Live/gallery/events/configuration/metrics views; authentication; deletion/deactivation; retention; disconnect/restart handling |
| 9 | Integration freeze and optional extension gate | Primary pipeline stable; final test manifest sealed; no core dependency on an extension; choose at most one extension if schedule permits |
| 10 | Runtime optimization and reliability | Stage profiling; optimization comparison; CPU fallback; 60-minute stress run; no unbounded queue/memory growth |
| 11 | Frozen evaluation | Complete model/ablation/robustness/event results; uncertainty; latency distributions; failure catalogue; no test-set threshold tuning |
| 12 | Submission and portfolio package | Reproducible repository; specification; report; demo video; results; installation guide; limitations; individual contribution records |

Participant recruitment and data collection start in week 1 and run alongside implementation; they are not postponed until the interface is ready. Tracking development and recognition experiments may be divided among team members, but integration occurs every week. A two-person team should remove the extension and reduce UI polish before dropping evaluation or required course coverage.

## 15. Team responsibilities

Suggested four-person ownership: recognition/data protocol; detection/tracking/quality; temporal decisions/alerts/evaluation; backend/interface/persistence. Each owner also reviews another module and contributes to final experiments. With three members, combine temporal decisions with recognition and share evaluation. Identify an integration owner and rotate weekly demonstrations. Record individual technical contributions and all configuration/model changes.

The project deliberately separates components so they can be developed and tested independently, but a shared data schema and early vertical baseline prevent integration from being postponed.

## 16. Acceptance and testing

### Functional acceptance

- Two recognition methods run on the published evaluation suite.
- An enabled enrolled identity can produce a temporally supported alert; a disabled identity cannot.
- Unknown and unresolved observations are distinguished and do not trigger enrolled-identity alerts by default.
- Multiple tracks operate concurrently; association ambiguity cannot silently transfer an identity.
- Alerts persist, can be reviewed, and obey declared encounter/cooldown rules.
- Deactivation/deletion takes effect in both persistent and in-memory galleries.
- Stream disconnection/end-of-file is visible, and processing can restart cleanly.
- Operator/admin permission checks, retention behavior, and protected storage are tested.

### Scientific acceptance

- The final dataset and model/configuration provenance are published without exposing unauthorized personal data.
- Required comparisons and condition-specific results exist with sample/exposure counts.
- Thresholds are selected before final testing, with a separate unknown-person validation partition.
- The temporal/quality contribution is evaluated at comparable operating constraints.
- Results include abstention, misses, false alerts, duplicates, and response time, not accuracy alone.
- Negative or inconclusive research outcomes are reported honestly; improvement is a hypothesis, not a completion guarantee.

### Engineering acceptance

- Fresh installation and an authorized sample replay work from documented steps.
- At least a 60-minute runtime test completes without unbounded memory/queue growth or silent source failure.
- Stage-level and end-to-end latency, throughput, and CPU/GPU memory are reported on named hardware.
- Model/preprocessing/template mismatch is detected.
- Algorithm tests cover threshold boundaries, unknown rejection, temporal reset, conflicting identities, deduplication, and deletion effects.
- Integration checks cover capture-to-event persistence and enrollment-to-gallery refresh. UI checks cover live status, enrollment failure, alert review, and permissions.

Do not require a fabricated recognition accuracy target before baseline data exists. Week 3 chooses explicit validation operating targets; week 11 reports whether the frozen system meets them. Functional completeness, valid evaluation, and honest limitations are unconditional requirements.

## 17. Risks and mitigation decisions

| Risk | Design response |
|---|---|
| Insufficient independent identities/sessions | Recruit early; prioritize distinct sessions and unknown identities; scale claims to observed exposure |
| Mask/pose recognition becomes unreliable | Preserve unresolved state; evaluate detection versus recognition failures separately; avoid forced matching |
| False alerts from tracker identity switches | Reset/demote evidence on ambiguous association; evaluate crossing scenes and identity-switch events |
| Correlated frames inflate confidence | Time-spaced sampling and session/encounter-level uncertainty |
| Slow embedding inference with many faces | Recognize selected track observations, cache bounded track state, batch where measured beneficial |
| UI/API blocks vision | Separate workers, bounded queues, throttled previews, explicit backpressure |
| Licensing or missing checkpoints | Verify in week 1; maintain model adapters and manifests; no assumption that code license covers weights |
| Scope grows into several research projects | Protect the complete core; permit at most one evaluated extension after week 9 |
| Unknown rejection mistakes imply suspiciousness | Keep identity status separate from behavior; alerts concern enrolled identities only |

## 18. Submission deliverables

1. Source repository with clear component boundaries, reproducible environment, and meaningful tests.
2. This specification, architecture/contracts, and requirement-to-evidence mapping.
3. Data collection/annotation protocol, partition manifests, and authorized demo inputs or safe download/replay instructions.
4. Experiment configurations, model/checkpoint provenance, prediction schemas, result tables/plots, and failure examples.
5. Final report explaining hypotheses, methods, operating thresholds, results, runtime, and limitations.
6. Short demo video covering enrollment, correct recognition, unknown rejection, difficult conditions, alert review, and failure handling.
7. Model/data licensing inventory and storage/deletion documentation.
8. Individual contribution records and a concise English project README suitable for portfolio review.

Suggested repository areas: vision (detectors, alignment, quality, encoders, tracking); identity (gallery, matching, temporal state); events; api; persistence; frontend; evaluation; configuration; tests; docs. The repository should expose a headless evaluation command independently of the web application.

## 19. Demonstration acceptance scenarios

| Scenario | Required visible outcome |
|---|---|
| Clear enrolled watchlist identity | Candidate then confirmed; one persisted appearance event |
| Clear enrolled identity with alerts disabled | Confirmed identity; no watchlist event |
| Clear non-enrolled person | Unknown after sufficient usable evidence; no enrolled-identity event |
| Blurry or largely occluded face | Unresolved with a reason; no forced identity |
| Two people cross and one becomes occluded | Track/identity evidence remains correct or resets explicitly; no silent identity transfer |
| Brief missed detection | Track may persist; stale evidence is visible; no new event from stale evidence |
| Reappearance | New event only after the declared rearming/cooldown policy permits it |
| Delete or deactivate an enrolled identity | It stops matching/alerting as configured, including active-worker state |
| Camera disconnect or video ends | Source status changes immediately; stale frame is labeled; no fake healthy-live state |
| Wrong operator permission or incompatible templates | Action is rejected with an actionable explanation |

The demo includes both successful recognition and deliberate abstention/failure handling. A missed recognition is recorded as a miss in evaluation even when the application correctly abstains.

## 20. Sources and rationale

Original requirements: supplied screenshot Screenshot_20261009_162122.png, project pool row #1.

- ArcFace paper: https://openaccess.thecvf.com/content_CVPR_2019/html/Deng_ArcFace_Additive_Angular_Margin_Loss_for_Deep_Face_Recognition_CVPR_2019_paper.html
- InsightFace implementation and licensing: https://github.com/deepinsight/insightface
- InsightFace model zoo: https://github.com/deepinsight/insightface/blob/master/python-package/docs/model_zoo.md
- FaceNet-family implementation: https://github.com/timesler/facenet-pytorch
- ByteTrack original implementation: https://github.com/FoundationVision/ByteTrack
- ONNX Runtime CUDA provider documentation: https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html
- FastAPI WebSocket documentation: https://fastapi.tiangolo.com/advanced/websockets/
- WIDER FACE benchmark: https://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/

Sources establish available methods/interfaces and their documented terms; architecture choices, data targets, workload, and performance goals are proposed design decisions, not claims made by those sources.
