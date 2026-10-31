# Classical Pipeline Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans inline, with a fresh whole-change reviewer at the end.

**Goal:** Deliver an inspectable classical odin application with reusable infrastructure.
**Architecture:** NumPy LBPH + Haar adapter, private SQLite gallery/events, bounded
video service, FastAPI routes, React UI and headless evaluation.
**Tech Stack:** Existing Python/React stack, OpenCV contrib, NumPy, SQLite, cryptography.
**Spec:** `docs/superpowers/specs/2026-10-31-classical-pipeline-design.md`.

## Global Constraints

- Own LBP implementation; pretrained Haar detection; no neural inference in baseline.
- No overlap between OpenCV wheels, no unrelated locked version upgrades.
- No raw enrollment/unknown images retained; local encrypted descriptors and deletion.
- Scores are uncalibrated distances until validation; no fabricated robustness claims.
- User authorized execution; use existing checkout on `feat/classical-pipeline`.

## Review Focus

- Delete/update during capture cannot leave eligible stale identity evidence.
- Occlusion/crossing cannot silently inherit confirmation; ambiguity resets evidence.
- Invalid/oversized media rejected without partial enrollment or permanent uploads.
- Stop, EOF and source failures cannot leave a falsely healthy preview/session.
- Empty/one-person gallery and identical-distance ties must not force identities.

## Task 1: Runtime and classical representation

Files: pyproject/lock; existing probes; `classical/lbp.py`, `classical/detection.py`;
`tests/test_classical_lbp.py`; pinned cascade manifest/acquisition script.
Interfaces: `lbp_codes(gray)`, `describe(gray)`, `distance(a,b)`, `match(query,templates,threshold,margin)`;
`HaarDetector.detect(bgr)`, `prepare_face(bgr,box)`.
- [x] Record RED with hand-derived codes, normalized histogram, rejection/tie tests.
- [x] Verify contrib wheel in temporary path; replace package and update lock, preserve pins.
- [x] Implement representation, detector and acquisition; verify core and existing probes on CPU.

## Task 2: Gallery, tracking and event policy

Files: `classical/storage.py`, `classical/pipeline.py`, tests for both.
Interfaces: `Store` owns encrypted descriptor persistence and event retention;
`Pipeline.process(frame,time)` returns tracks and eligible events, with reset on gallery change.
- [x] RED: transactional enrollment/delete, retention, unknown rejection, confirmation, conflicting track evidence, stale evidence and disabled/deduplicated events.
- [x] Implement bounded association, temporal state and persistence; run tests GREEN.

## Task 3: Video service and local API

Files: `classical/service.py`, `api/main.py`, `tests/test_classical_api.py`.
Interfaces: `Service.start/stop/status/enroll/delete/preview`, `create_app(data_dir)`.
- [x] RED: upload enrollment workflow, invalid media, mutation guard, stop/EOF and cleanup.
- [x] Implement bounded capture, replay pacing, session invalidation, HTTP API, preview and explicit shutdown.
- [x] Verify with ASGI requests and real generated-video integration.

## Task 4: Operator UI, evaluation and handoff

Files: React App/CSS, `classical/evaluate.py`, README and tutorial; evaluation tests.
- [x] RED: labeled evaluation counts including misses/unknowns, config/input provenance, validation threshold selection.
- [x] Implement monitor/enrollment/gallery/events, configurable distance policy, offline replay and calibration CLI.
- [x] Run full tests, lint/build, generated real detector/replay smoke and CPU deep-model regression.
- [x] Fresh reviewer; address material findings with reproducing tests; leave reviewable branch without merging/pushing.

## Execution ledger

- Baseline: 34 utility tests finish; existing TestClient case stalls in AnyIO portal,
  matching prior observation. GPU driver unavailable; no webcam device exposed.
- Ruling: work in existing checkout on feature branch; no additional worktree requested.
- Preflight: descriptor contract feeds storage/matching; storage generation invalidates
  pipeline; service owns pipeline and serializes mutations; UI consumes API schema.

- Task 1: complete. Hand-derived LBP tests observed RED then GREEN. OpenCV contrib
  wheel checked in isolation, hash-verified and installed; lock check passes.
  No existing dependency version changed; added cryptography/CFFI/pycparser only.
- Task 2: complete. Encryption/deletion/retention, confirmation, rejection, ambiguity,
  contradiction, cooldown and policy tests pass.
- Task 3: complete. Real Haar enrollment, video EOF and full asynchronous API
  enrollment → upload → persisted event → deletion workflow pass.
- Task 4: complete. UI lint/build, evaluation/calibration tests, all 60 tests and
  backend lint/format pass. Generated replay and existing neural CPU checks pass.
- Review: independent reviewer found six material edge cases. Added reproducing
  backend tests and fixed contradiction timing, all-track ambiguity reset,
  API lock blocking, EOF final-frame loss and empty-video success. Browser preview
  now clears on 204 and rejects stale in-flight snapshots after mutation; token
  refresh also fixes the reviewer’s minor backend-restart recovery finding.
- Verification: runs/classical-smoke-reviewed-01 and runs/classical-opencv-cpu-01
  are private generated evidence. CPU embedding differences from historical CUDA
  are 9.0711e-6 (ArcFace) and 1.3970e-7 (FaceNet), not accuracy measurements.
- Limits: GPU driver and webcam unavailable. Socket-level HTTP/browser verification
  is restricted by the execution sandbox; asynchronous ASGI integration and real
  replay inference were tested instead. No live hardware or robustness claim.
- Environment: direct uv network access was restricted. Exact wheel bytes were
  acquired with curl, checked against lock hashes and installed locally; offline
  uv lock validation passes. A fresh normal uv sync still needs network/cached wheels.
- Integration: retain the feature branch; no merge or remote push was requested.

- Follow-up environment repair: after permission changes, normal `uv sync --locked`
  rebuilt editable metadata and replaced all four local-wheel installs with registry
  installs at identical locked versions. A second sync reports no changes. README
  commands now use native uv execution without PYTHONPATH or manual wheel steps.
