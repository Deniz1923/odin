# Shared ONNX runtime environment plan

> **For agentic workers:** Use superpowers:executing-plans inline. The user explicitly authorized architecture/environment work through, but excluding, recorded-video recognition implementation.

**Goal:** Export the pinned FaceNet checkpoint to ONNX, verify equivalence and make all three models usable together without importing the incompatible FaceNet package at runtime.

**Architecture:** Keep the torch reference probe; add a bounded export/parity tool and an ONNX-only runtime smoke check. Add exporter tooling in a locked `model-export` dependency group; keep export-time source overlay outside normal runtime imports. Derived model bytes remain ignored and are identified by SHA-256 plus source provenance. No capture/video, gallery, tracking, matching, alerts or frontend changes.

**Spec:** `docs/ARCHITECTURE.md` plus the latest user-authorized shared-environment preparation boundary. Existing checkpoint rights are unchanged by conversion.

## Global constraints

- Preserve Python/PyTorch/CUDA/ORT versions unless concrete evidence requires a change.
- All model sessions require CUDA and executed-node profiling; no silent CPU fallback.
- Fixed FaceNet input 1×3×160×160 float32 RGB, (x−127.5)/128; output 1×512 normalized embeddings.
- Conversion acceptance: each element agrees at atol=1e-5, rtol=1e-4 with raw outputs checked before optional normalization; all vectors finite/nonzero and unit-norm within1e-4. Also compare pairwise cosine scores with max absolute delta1e-5. Freeze these gates before running export.
- Current task ends before recorded-video implementation. Export verification tests numerical parity, not recognition accuracy or masks/pose robustness.

## Review focus

- Corrupt source/derived artifacts rejected before inference.
- Invalid intermediate outputs or score-matrix changes cannot be hidden by normalization/aggregation.
- Failed conversion does not activate a derived model as validated.
- Runtime smoke check must work without exporting dependencies or facenet-pytorch imports.
- Fresh output/model directories refuse accidental overwrite.

## Tasks

### 1. Export environment and parity contracts

Files: `pyproject.toml`, `uv.lock`, `tests/test_facenet_export.py`, `vision/export_facenet.py` under the existing package.

- [x] Add only required exporter tools to a separate locked dependency group, inspect dependency diff, preserve existing core versions.
- [x] Write and run failing tests for exact output shape, nonfinite/zero/nonnormalized vectors, bad middle cases, per-element parity and pairwise-score parity.
- [x] Implement parity validation and deterministic diverse numerical fixtures (public astronaut crop variants plus synthetic inputs); run tests/lint.

### 2. Derived model acquisition-by-export

- [x] Implement explicit local hash-verified PyTorch construction/export with no download constructor; fresh output directory and failure/provenance report.
- [x] Export batch-one opset18 with modern torch exporter; validate ONNX graph, execute all parity fixtures on CUDA, check actual provider events and performance.
- [x] Activate/pin a separate runtime manifest only after validation passes; preserve source/derived hash and exporter/config/source versions.

### 3. Same-environment runtime readiness and handoff

Files: `vision/onnx_probe.py`, `configs/onnx_models.json`, environment/how-to documentation and architecture decision update.

- [x] Run all three ONNX models together on public/synthetic still fixtures in a fresh process with FaceNet imports blocked. Preserve explicit model-specific preprocessing, strict CUDA profiles, finite embeddings and repeatability checks.
- [x] Verify runtime with default locked environment (export group omitted), restore exporter group if needed for regeneration; no existing core version churn.
- [x] Run original and new tests, lint, original CUDA smoke check; get independent review and fix material findings.
- [x] Document actual results, setup/export/runtime commands, source/derived artifact provenance, locked dependencies and the exact boundary before video implementation.

## Progress and rulings

- Existing branch `feat/model-feasibility`, clean baseline `f085447`; continue here, no destructive operations or remote push.
- User's explicit attempt/setup authorization supplies execution approval; do not pause for another design-choice questionnaire.

- Task1 complete: parity/failure-publication/isolation unit tests observed RED then GREEN. Locked exporter tooling adds only onnxscript0.7.2 and onnx-ir1.0.0; no existing locked version changed. Exact sync removed8 unlocked plotting packages; documented.
- Task2 complete: two modern exports produced identical SHA-256 e5ab0c8e28c81bb6e9958d98dafd22cefb56d580b7c932a8da0782d21cbda5fb. All11 numerical cases pass raw-component, unit-norm and cosine-matrix gates; all executed export-validation ONNX nodes onCUDA.
- Task3 complete: active configs/onnx_models.json binds finalized passed report and source/derived hashes. Default locked env has no facenet-pytorch/onnxscript/onnx-ir; fresh runtime processes01/02 block their imports and pass all3CUDAmodel sessions, output checks and profiles. Full35tests, lint/format, originalCUDAcheck pass. Saved source/helper/lock hashes match final files.
- Independent review: no material findings. Minor wording about raw versus normalized runtime embeddings clarified; no code fix required.
- Final ruling: keep source overlay for regenerating/reference export only; runtime no longer bypasses FaceNet dependency constraints. Single environment and core version preservation take precedence over unproven speed claims. Recorded-video capture/recognition remains unimplemented as explicitly requested.
