# odin: teammate study guide

Updated 11 November 2026. This guide provides a reading route and concepts to explain.
Current requirement coverage, preliminary results and remaining tasks live in
[project status](PROJECT_STATUS.md), avoiding a second status ledger here.

## Reading route

1. [README](../README.md): install and run the operator application.
2. [Project status](PROJECT_STATUS.md): current scope and the immediate progress report.
3. [Classical walkthrough](CLASSICAL_PIPELINE.md): trace the algorithm and its limits.
4. [Implemented architecture](ARCHITECTURE.md): understand ownership, concurrency,
   source lifecycle, persistence and browser/API behavior.
5. Read `lbp.py`, `detection.py`, `pipeline.py`, `storage.py` and `service.py` in
   [the classical package](../backend/src/face_surveillance/classical/), then the
   [API](../backend/src/face_surveillance/api/main.py) and [UI](../frontend/src/App.tsx).
6. [Evaluation](EVALUATION.md), its [runner](../backend/src/face_surveillance/classical/evaluate.py)
   and [example manifest](../experiments/classical.example.json): distinguish integration
   fixtures from an independent study.
7. [Environment/model reproduction](ENVIRONMENT.md): understand the prepared neural
   paths before integrating another recognizer.

The [archive](archive/README.md) preserves earlier designs and preparation results.
Read it for rationale and historical evidence, not as today's component inventory.

## Concepts to keep distinct

| Concept | Meaning |
|---|---|
| Detection | Locate a face; a detector score does not identify a person |
| Preparation | Crop, check quality and apply the recognizer's exact input convention |
| Descriptor/embedding | Represent appearance; a gallery supplies the person association |
| Enrollment | Store authorized labeled reference representations; no neural retraining |
| Track ID | Temporary video trajectory, separate from persistent enrolled identity |
| Candidate | Plausible gallery identity with incomplete temporal confirmation |
| Confirmed | Fresh evidence meets the configured policy, not proof of correctness |
| Unknown | A usable observation fails the current gallery acceptance rule |
| Insufficient | Quality, ambiguity, stale evidence or an empty gallery prevents a decision |
| Unavailable | Source/system failure, separate from a recognition outcome |
| Appearance event | Enabled enrolled identity's policy-approved confirmation, not a behavioral judgment |

Raw distance/similarity, evidence count and probability are different quantities.
Adjacent video frames are correlated; repeated agreement is not independent probability
evidence. Acknowledgement means the operator received an event, not that the identity
was verified. Abstention can be appropriate behavior while still counting as a missed
recognition opportunity in evaluation.

## Neural representations

The prepared ArcFace-family and FaceNet-family encoders each produce 512 numbers,
but in incompatible learned spaces. Normalize an embedding with `e / ||e||`;
cosine similarity between normalized query/reference vectors is their dot product.
Higher cosine similarity means closer, whereas lower LBP histogram distance means
closer. Do not mix their templates or average unrelated raw winning scores.

ArcFace names a training loss as well as a model-family label. The selected
FaceNet-family checkpoint lineage is softmax-trained, while the original FaceNet
paper describes triplet training. Explain the exact checkpoints, preparation and
architectures rather than presenting a loss-only comparison. Input contracts and
provenance are maintained in the [environment guide](ENVIRONMENT.md).

## Learn by tracing observations

Run an authorized enrollment and separate replay. Trace a matching observation from
crop to descriptor, nearest-person score, track evidence and event insertion. Repeat
for rejected and insufficient observations. Inspect what happens after a conflict,
ambiguous association, missed frame, stop and person deletion.

Every teammate should be able to explain:

1. Which components are pretrained and which algorithms are implemented here?
2. How does the example patch yield LBP code 170 and a face yield 16,384 values?
3. Why does nearest-neighbor search require unknown rejection and distinct-person margin?
4. What separates candidate, confirmation, track identity and alert eligibility?
5. Why do ambiguity, contradiction and stale evidence reset decisions?
6. What do fixture replay and export parity prove, and what do they leave unmeasured?
7. How will enrollment, validation and test sessions remain independent?
8. Which submitted contributions can each teammate explain and reproduce?

The [progress-report checklist](PROJECT_STATUS.md#progress-report-checklist) covers
the mandatory task timeline, actual contributions and GenAI disclosure. This guide
supports understanding; it does not supply invented student work or study results.
