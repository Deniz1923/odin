# Shared model runtime: environment and handoff

Verified 19 October 2026. odin is ready for the team to begin the recorded-video baseline. This preparation does not implement video capture, gallery enrollment/matching, tracking, temporal recognition, alerts or UI workflows.

## Decision and result

Use ONNX Runtime CUDA for the pinned SCRFD detector, ArcFace-family encoder and **verified FaceNet-family ONNX export** in the same Python environment/process. FaceNet's incompatible Python package is not a normal runtime dependency. Existing Python/PyTorch/torchvision/NumPy/Pillow/ORT versions were preserved; no downgrade was needed.

PyTorch remains installed for export/reference experiments, existing GPU checks and CUDA-library provisioning. This does not imply a PyTorch model runs during normal ONNX inference. Removing it would require separately validating library provisioning; no such change was needed here.

Default setup:

```bash
uv sync --locked
.venv/bin/python scripts/check_gpu.py
.venv/bin/pytest -q
.venv/bin/ruff check backend tests scripts
```

Model files are local, ignored assets. Before the first export, prepare the exact sources/overlay described in [MODEL_FEASIBILITY.md](MODEL_FEASIBILITY.md#acquisition-and-reproducibility). The earlier overlay probe remains a historical reference; use the ONNX probe below for the shared runtime.

## Export once, validate, then activate

Export tools are isolated as a dependency **group** within the same environment, not installed for default inference:

```bash
uv sync --locked --group model-export
.venv/bin/python -m face_surveillance.vision.export_facenet \
  --manifest configs/model_probe.json \
  --output models/facenet-onnx-v2
```

Use a new directory if that output already exists. Only after success, activate its generated manifest:

```bash
cp models/facenet-onnx-v2/runtime_manifest.json configs/onnx_models.json
uv sync --locked
.venv/bin/python -m face_surveillance.vision.onnx_probe \
  --manifest configs/onnx_models.json \
  --output runs/shared-onnx-runtime-new
```

Use a new run directory. The runtime checks all model hashes and the finalized export-report hash/status/provenance before inference. The report is part of the local model bundle; do not copy the ONNX file alone and omit its validation report. Regeneration changes timestamped report hashes, so activate the newly generated manifest rather than retaining a stale config. Manifest paths are repository-relative for the documented commands; the generated artifacts stay out of Git.

`export_facenet` requires the hash-pinned FaceNet source overlay only to instantiate the original architecture. It explicitly loads the local state dict with `weights_only=True` and strict keys, exports with modern `torch.onnx.export(dynamo=True)`, and validates the resulting graph and CUDA outputs. It never downloads a checkpoint. Failed export/parity produces a failure report and no runtime manifest.

The export group adds `onnxscript==0.7.2` and its locked `onnx-ir==1.0.0` dependency. Default `uv sync --locked` omits/removes both. The first exact sync also removed eight previously installed but **unlocked** plotting dependencies (matplotlib, contourpy, cycler, fonttools, kiwisolver, pyparsing, python-dateutil, six); none is required by the locked core or these probes. No existing locked package version changed. No FaceNet distribution was installed into the default environment.

## Numerical equivalence gates and evidence

Export input ABI: float32 `[1, 3, 160, 160]`, RGB, `(pixel - 127.5) / 128`, prepared by the same `facenet_input` helper as the reference. Output: float32 `[1, 512]` normalized embedding. Fixed batch one, ONNX opset18, embedded weights, TF32 disabled during parity and runtime checks. No image preprocessing is embedded in the ONNX graph; callers must apply this exact external contract.

Before export we fixed these acceptance gates: every raw output finite and unit norm within `1e-4`; each output component agrees with the PyTorch reference at `atol=1e-5, rtol=1e-4`; pairwise cosine-score matrix maximum absolute difference at most `1e-5`. We do not normalize away invalid/scaled outputs before checking them.

Both completed exports produced identical ONNX bytes:

```text
SHA-256: e5ab0c8e28c81bb6e9958d98dafd22cefb56d580b7c932a8da0782d21cbda5fb
Bytes:   94637651
Source checkpoint SHA-256:
281cebca8662831adb987a874bdcb36e73f5b1c6dc5ee5878f305e985625d99b
```

Eleven cases: a public-domain astronaut face crop; flipped, brighter, darker, blurred and grayscale variants; black, white and gray inputs; seeded noise and a gradient. Their input hashes and the reference/export vectors are in the private export bundle. Maximum component difference was `2.71946e-7`; maximum pairwise cosine difference was `8.49366e-7`. All gates passed. Synthetic/altered inputs test conversion numerics, **not** real-world recognition or occlusion robustness. Calibration on the eventual study must use the actual selected runtime/model/preprocessing rather than assume this parity test chooses recognition thresholds.

Checkpoint conversion preserves the source's usage restrictions and does not create a new independently trained recognition approach. Original FaceNet checkpoint/right-of-use limitations remain documented in [MODEL_FEASIBILITY.md](MODEL_FEASIBILITY.md#asset-terms-and-provenance); no unrestricted redistribution/commercial grant is claimed. Models/embeddings/images are not committed.

## Shared-runtime verification

`onnx_probe` checks all three models in one fresh default-environment process. It rejects already-loaded export-only libraries and actively blocks imports of `facenet_pytorch`, `onnxscript` and `onnx_ir`. None was installed in the default environment when the check passed. An actual node profile for each model contains CUDA execution only; CPU fallback is disabled.

The public astronaut/blank/two-copy still fixtures return 1/0/2 faces. Each encoder's 33 warmup/measured outputs is checked for finite, nonzero embeddings, normalized before repeatability checks. Raw FaceNet unit norm is checked separately during export parity. The two-copy fixture repeats one person; it is not an identity discrimination test.

First 30-repeat shared-runtime run, RTX 4070 Laptop GPU, driver 615.78.08:

| Stage | Median / p95 ms | Executed CUDA node events |
|---|---:|---:|
| SCRFD 640×640, with adapter preprocessing/postprocessing | 9.67 / 9.82 | 4932 |
| ArcFace, one prepared 112×112 crop | 3.73 / 3.79 | 4290 |
| FaceNet ONNX, one prepared 160×160 input | 10.61 / 10.66 | 10461 |

These are short, batch-one host-to-host timings with profiling enabled and TF32 disabled. They exclude decoding, cropping, tracking, matching, persistence and preview. They are not an FPS claim or a fair speed comparison against the earlier different-config PyTorch probe. The reason for choosing ONNX here is shared-runtime compatibility; a speed improvement has not been established. Device-wide GPU memory sampled after loading was 891 MiB versus 45 MiB before, not a peak or process-exclusive measure.

A second fresh-process runtime run also passed, with exactly matching embeddings across the two processes. Final runtime source/helper/lock hashes and the export source hash were checked against the saved reports. The original PyTorch/ORT GPU matrix checks and all 35 tests in the default environment also pass. The upstream scikit-image alignment deprecation warning remains visible; installed library code was not patched or warning-suppressed.

## Architecture boundary for the team's next implementation

- Treat `configs/onnx_models.json` as the hash-pinned local model bundle definition, preserving model-space/preprocessing identifiers when enrollment is implemented.
- The export tool and smoke checker are preparation utilities, not the application's worker or API. Refactor their proven model settings into narrow adapters when building the video runner; do not call export in HTTP requests or at routine startup.
- SCRFD fixes the pinned symbolic spatial dimensions to 640×640; ArcFace fixes batch one; FaceNet already has fixed input/output shapes. Change shape/batch profiles only through explicit re-export/revalidation.
- Keep ArcFace landmark alignment and FaceNet box-crop preprocessing distinct; both see the same detected face but this is not a controlled loss-function-only comparison.
- Next build the recorded-video replay/timestamp/sampling/measurement slice, then enrollment/calibration. This task intentionally stops before that code.

Exporter API rationale: [PyTorch 2.14 ONNX documentation](https://docs.pytorch.org/docs/2.14/onnx.html). Optional tool provenance: [ONNX Script](https://github.com/microsoft/onnxscript). Actual versions/artifact hashes are pinned by `uv.lock`, source/derived manifests and local run reports; upstream documentation alone is not the acceptance evidence.
