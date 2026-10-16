# Model feasibility implementation plan

> **For agentic workers:** Use superpowers:executing-plans to implement this bounded probe inline. Steps use checkboxes for progress.

**Goal:** Establish real CUDA execution and preprocessing/output sanity for SCRFD, ArcFace-family and FaceNet-family pretrained inference, without implementing application workflows.

**Architecture:** A headless probe loads only local hash-verified artifacts. Existing backend dependencies remain unchanged; a reviewed facenet-pytorch wheel is installed into an ignored import overlay, with its incompatible metadata constraints explicitly recorded as an experimental override. No automatic downloads in the probe, API integration, enrollment database or recognition thresholds.

**Tech stack:** Existing Python 3.14 / torch CUDA / InsightFace / ORT CUDA / NumPy / OpenCV; isolated facenet-pytorch 2.6.0 overlay.

**Spec:** `docs/ARCHITECTURE.md`, M1. User explicitly authorized attempts, setup and model downloads after approving scope; execute directly without another approval round. Work on `feat/model-feasibility` in the existing checkout, preserving `main` at `92a1354`; no additional worktree or environment reinstall.

## Global constraints

- Keep the current core lockfile and dependencies unchanged.
- Store weights, optional overlay and face/crop/profile outputs only in ignored `models/`, `data/`, `runs/`.
- Never represent similarity as recognition probability, or this fixture as an accuracy benchmark.
- Never silently fall back to CPU; profile actual ONNX node placement.
- Preserve original specification and scaffold behavior.

## Review focus

- Missing or corrupt weights fail before runtime/model construction.
- Nonfinite/zero embeddings and incorrect dimensionality fail visibly.
- Color/layout/normalization must be model-specific, not shared accidentally.
- Missing CUDA or unexpected CPU ONNX nodes cannot produce a passing GPU report.
- Overwriting a prior output must not silently erase provenance.

## Task 1: Local asset preparation and output contracts

Files: `tests/test_model_probe.py`, `backend/src/face_surveillance/vision/model_probe.py`, `configs/model_probe.json`.

Interfaces: `verify_artifact(path: Path, expected_sha256: str) -> str`; `normalize_embedding(value: np.ndarray, dimension: int = 512) -> np.ndarray`; `facenet_input(rgb: np.ndarray) -> np.ndarray`; `summarize_profile(path: Path) -> dict`.

- [x] Write tests for missing/corrupt file, valid hash, zero/nonfinite/wrong-sized embeddings, RGB-to-NCHW fixed standardization, CUDA/CPU profile rejection and fresh output-directory behavior.
- [x] Run tests and observe missing implementation failure.
- [x] Implement contracts, prepare official local model assets, inspect wheel metadata/source and record hashes in manifest.
- [x] Run tests, lint, and original scaffold test; expected all pass without model downloads in tests.

## Task 2: Real-model CUDA probe and evidence report

Files: same probe module; `docs/MODEL_FEASIBILITY.md`; update architecture's historical implementation boundary.

Interfaces: CLI `python -m face_surveillance.vision.model_probe --manifest configs/model_probe.json --output runs/model-feasibility-<run>` consumes Task 1 manifest/helpers and writes a JSON run report plus private crops and ORT profiles.

- [x] Load SCRFD/ArcFace with CUDA-required sessions and profile real inference. Load FaceNet local state_dict with `weights_only=True`, strict keys, eval/inference mode, CUDA and no download constructor.
- [x] Use bundled scikit-image astronaut fixture (NASA/public-domain provenance), blank negative frame and repeated inference. ArcFace uses five-landmark alignment; FaceNet uses its box-crop/160px/fixed-standardization convention. Record this preprocessing distinction.
- [x] Save fixture/crop hashes, model/provider/runtime metadata, normalized 512-vectors' sanity summaries, repeated-output checks, warm load/inference timings and GPU resource measurements. Do not commit images or embeddings.
- [x] Run full checks and CUDA probe twice into distinct directories; diagnose failures rather than reinstalling the working stack.
- [x] Write actual results, reproducible setup/run instructions, metadata-override limits and unresolved checkpoint terms; distinguish technical feasibility from license clearance and recognition evaluation.
- [x] Review changes, resolve material issues, and leave application UI/API unchanged. No remote push unless separately requested.

## Progress / rulings

- Planning: no extra plan-approval pause: the user's latest instruction explicitly authorizes trying this milestone.
- Workspace: isolated branch in the current checkout; branch creation required sandbox escalation, succeeded. Main remains the clean recovery point.
- Dependencies: overlay-only compatibility experiment instead of a second multi-gigabyte CUDA environment. Passing the encoder probe will not certify the whole facenet-pytorch package or make its declared constraints compatible.

- Diagnostic ruling: initial dynamic SCRFD failed strict CUDA-only placement. A traced profile showed CPU Gather/Unsqueeze/Slice/Concat shape nodes, while convolutions ran on CUDA. Fix named spatial dimensions to 640 via ORT session overrides; diagnostic confirms all executed nodes are CUDA. Original checkpoint bytes unchanged. Wrong override would break supported input resolution, so manifest pins this square profile explicitly.

- Task 1 complete: initial missing-module RED observed; contracts passed. Schema/provenance failures observed and corrected. Core environment/lock unchanged.
- Task 2 complete: initial strict dynamic-shape failure diagnosed; fixed-session profiles all CUDA. Four successful full probes (02–05), final05 checks all34 outputs/model; source+lock hashes and cross-process repeatability verified.
- Independent review: one material finding, endpoint-only repeatability check. Regression for invalid/drifting middle outputs observed RED, series validation implemented,17 probe tests GREEN. Full suite18/18, lint/format clean. No other material findings.
- Final ruling: this completes technical model execution/preprocessing feasibility only. FaceNet overlay remains an experimental constraint override; full M1 data preparation and separate weight-rights provenance follow-up remain. No video recognition/alert feature or accuracy claim is made.
