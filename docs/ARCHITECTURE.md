# odin: implemented architecture

Updated 11 November 2026. This document describes the current classical application.
See [project status](PROJECT_STATUS.md) for course coverage and planned work,
[evaluation](EVALUATION.md) for the study protocol, and the
[algorithm walkthrough](CLASSICAL_PIPELINE.md) for recognition details.

## Components and data flow

```text
React operator interface -- HTTP polling/commands --> FastAPI
                                                       |
                                       bounded command executor
                                                       |
                                                    Service
                                                   /   |   \
                              capture helper --> Pipeline  Store --> SQLite
                              latest-frame slot     |       encrypted templates
                                                JPEG preview
```

The application runs one Uvicorn process with threads. It does not currently spawn
a separate vision process. One webcam or uploaded replay is active at a time.

| Component | Implemented responsibility |
|---|---|
| [API](../backend/src/face_surveillance/api/main.py) | Lifespan, local request guards, uploads, commands, status, previews, gallery and events |
| [Service](../backend/src/face_surveillance/classical/service.py) | Source lifecycle, enrollment, synchronization, session invalidation and preview encoding |
| [Detector](../backend/src/face_surveillance/classical/detection.py) | Hash-checked Haar cascade, boxes, grayscale crop preparation and quality gates |
| [LBP](../backend/src/face_surveillance/classical/lbp.py) | Custom descriptors, histogram distance, nearest-person search and rejection |
| [Pipeline](../backend/src/face_surveillance/classical/pipeline.py) | Spatial association, bounded temporal evidence, decisions and event proposals |
| [Store](../backend/src/face_surveillance/classical/storage.py) | SQLite transactions, encrypted templates, event eligibility/cooldown, retention and deletion |
| [Frontend](../frontend/src/App.tsx) | Monitor, gallery/enrollment, appearance events, acknowledgement/export and distance settings |
| [Offline runner](../backend/src/face_surveillance/classical/evaluate.py) | Exhaustive replay, predictions, metrics and validation threshold selection |

## Enrollment and recognition

Enrollment accepts 1–32 distinct image files, each with exactly one usable face,
and requires an authorization affirmation. All images are processed before the
database transaction, so invalid enrollment leaves the gallery unchanged.
Pseudonyms are limited to 80 characters; alerts default off.

Faces are cropped, resized to 128×128 grayscale, checked for size/exposure/blur,
and equalized. Basic radius-one LBP with an 8×8 grid produces 16,384 values.
The descriptor space includes its preprocessing/version identity. Matching uses
minimum distance per person, then threshold and second-distinct-person margin.

Tracking greedily associates boxes with IoU at least 0.2. Ambiguous associations
reset involved evidence. At most 32 tracks/detections are processed per update.
Default confirmation needs three accepted observations at least 0.25 seconds apart,
spanning at least 0.5 seconds within two seconds. Evidence becomes stale after one
second; unseen tracks expire after two seconds. Contradictory usable observations
are checked between positive sampling times. These settings are uncalibrated.

Only fresh confirmations for alert-enabled identities propose events. The pipeline
suppresses repeat proposals per track/person; storage checks current eligibility,
session/track/person uniqueness, and source/person cooldown before insertion.
Acknowledgement records receipt, not verified recognition correctness.

## Concurrency and source lifecycle

API service commands use a single-thread executor with eight admitted command
slots; overload returns 429. Expensive commands and vision work run outside the
async HTTP loop. A service lock serializes processing and gallery mutations.
Reloading the gallery clears tracks and preview while preserving the next track ID.

Capture runs in a daemon helper with a one-frame pending slot. New frames replace
waiting frames and increment drop counts; terminal EOF/error messages wait behind
the last pending frame. A separate processing thread handles recognition, event
writes, annotation and JPEG encoding. Replay is paced using frame index/FPS;
webcam timestamps use monotonic host receipt time. Offline evaluation processes
every decoded frame without the operator service's drop policy.

Stop invalidates the session generation, clears preview/tracks and signals capture.
Old work cannot publish into a new session. Ten seconds without a fresh frame fails
the session. A blocked native decoder cannot be forcibly killed as a thread;
restarting is refused while its capture helper remains alive. Process isolation
and stronger recovery remain planned improvements.

## Storage and local access

SQLite uses the standard-library driver, not the proposed SQLAlchemy layer.
The three tables are `people`, `templates`, and `events`; foreign-key cascades remove
templates and events when a person is deleted. Fernet encrypts descriptor payloads.
The local key and database have restrictive permissions in the private data directory.
This protects a database-only copy, not access by the same OS user with the key.

Original enrollment photos and unknown descriptors are not persisted. Preview is
transient. Uploaded replay files are removed at stop/end/failure, and orphan upload
copies are cleaned at startup. Events expire at startup and event reads; retention
defaults to seven days. Policy is staged and replaced on disk; saving it stops the
active session. Deletion clears database records and active gallery/evidence, but
does not promise physical SSD erasure or remove separately downloaded exports.

Host checks, exact permitted mutation origins and a per-process bearer token support
one trusted local operator. There are no user accounts or roles. Bind to loopback,
use one worker, and do not reload while a source is active. Configuration variables
and operating commands are in the [README](../README.md).

## API and browser contract

Routes include `/health`, `/api/session`, `/api/status`, `/api/preview`,
`/api/people`, `/api/people/{id}/alerts`, `/api/start`, `/api/video`, `/api/stop`,
`/api/policy`, `/api/events`, and `/api/events/{id}/ack`. OpenAPI gives methods and
schemas. The API serves the built frontend when `frontend/dist` exists.

The browser polls every 600 ms, fetches preview separately, revokes old image blobs,
and invalidates in-flight snapshots on mutations. It labels source failure and stale
preview. Event export contains the currently fetched history, capped by storage at
200 events. The UI currently states seven-day retention even if the server's retention
environment variable has been changed.

## Neural preparation boundary

The `vision/` modules prepare and verify SCRFD/ArcFace/FaceNet; they are not called
by the classical application. The [environment guide](ENVIRONMENT.md) owns model
acquisition, export, preprocessing, device checks and provenance. Integrating a neural
recognizer requires its own gallery space and calibrated matching policy while
reusing the source/operator/evaluation infrastructure where practical.

The [archived architecture](archive/2026-10-31-architecture-and-delivery-plan.md)
preserves the larger proposed system and earlier decisions. Its process model,
motion tracker, event clips, expanded schemas and routes are not implemented here.
