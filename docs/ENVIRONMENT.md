# odin: environment and model reproduction

Updated 11 November 2026. Run commands from the repository root. The
[README](../README.md) owns normal application installation, operation and checks.
This guide adds neural-model preparation; neural inference is not integrated into
the classical operator app. See [project status](PROJECT_STATUS.md) for evidence.

## Locked environment

Use Python 3.14 and Node 26 as pinned in `.python-version` and `.nvmrc`, with
`uv.lock` and the frontend npm lockfile. Normal `uv sync --locked` installs the
editable backend; no manual wheel installation or `PYTHONPATH` override is needed.

- `opencv-contrib-python==5.0.0.93` supplies `cv2` and Haar support. Do not install
  `opencv-python` alongside it; the transitive main distribution is excluded.
- `onnxruntime-gpu==1.31.0` supplies CPU and CUDA session providers in the tested
  environment. The separate CPU-only `onnxruntime` distribution is excluded.
- PyTorch/torchvision use the explicit cu130 index. PyTorch remains for reference,
  export, GPU checks and CUDA-library provisioning even for ONNX-only inference.
- `facenet-pytorch` is not a normal dependency. Its source overlay is used only for
  reference/export. The locked `model-export` group adds ONNX Script and ONNX IR.

These are repository pins, not claims of latest versions or portability to every
machine. A fresh CPU-only machine's library provisioning and live hardware performance
still require verification. Preserve the working stack when preparing FaceNet.

## Classical model

`uv run --locked python scripts/acquire_haar.py` explicitly downloads and verifies
the XML described in [haar_model.json](../configs/haar_model.json). No model is
downloaded at API startup. Its Intel/OpenCV license is embedded in the XML.
No neural model is needed to run the classical application.

## Neural assets

[model_probe.json](../configs/model_probe.json) pins source checkpoint/archive/wheel
hashes, overlay files, URLs and provenance. [onnx_models.json](../configs/onnx_models.json)
selects the active SCRFD/ArcFace/FaceNet ONNX bundle and finalized export-report hash.
Paths are repository-relative. Assets and generated runs stay in ignored local storage;
cloning does not acquire them. Hashes identify bytes, not recognition accuracy or an
independent signature of trust. Check the provenance limits below before acquisition.

## Acquire source assets for a new neural bundle

Existing verified bundles can be checked directly. Source acquisition and export
are needed when regenerating one; use the exact manifest pins.

Acquire the exact upstream artifacts, preserving filenames:

```bash
mkdir -p models/feasibility
curl -L --fail -o models/feasibility/buffalo_l.zip \
  https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip
curl -L --fail -o models/feasibility/20180402-114759-vggface2.pt \
  https://github.com/timesler/facenet-pytorch/releases/download/v2.2.9/20180402-114759-vggface2.pt
curl -L --fail -o models/feasibility/facenet_pytorch-2.6.0-py3-none-any.whl \
  https://files.pythonhosted.org/packages/ed/2e/2d56386bc2f834cdc683743903852cf1428b4e5ee119f16cf808b589d3cd/facenet_pytorch-2.6.0-py3-none-any.whl
```

Before extraction/install, verify hashes against the checked-in manifest. Extract **only** `det_10g.onnx` and `w600k_r50.onnx`; unrelated age/gender/extra-landmark models are not loaded:

```bash
uv run --locked python - <<'PY'
import json
import zipfile
from pathlib import Path
from face_surveillance.vision.model_probe import verify_artifact

manifest = json.loads(Path("configs/model_probe.json").read_text())
for asset in (manifest["archive"], manifest["assets"]["facenet"],
              manifest["assets"]["facenet_wheel"]):
    verify_artifact(Path(asset["path"]), asset["sha256"])
with zipfile.ZipFile(manifest["archive"]["path"]) as archive:
    for key in ("detector", "arcface"):
        asset = manifest["assets"][key]
        path = Path(asset["path"])
        path.write_bytes(archive.read(path.name))
        verify_artifact(path, asset["sha256"])
PY

uv pip install --no-index --no-deps \
  --target models/feasibility/python \
  models/feasibility/facenet_pytorch-2.6.0-py3-none-any.whl
```

### Isolated reference/export overlay

The 2.6.0 wheel declares NumPy `<2`, Pillow `<10.3`, torch `<2.3` and torchvision
`<0.18`. Installing only this hash-checked wheel into an ignored `--target` directory
deliberately bypasses those constraints for encoder reference/export. The tools
verify its source/data hashes before loading; normal API imports cannot see it.
This does not certify MTCNN, training or the full package. Keep the overlay out of
normal dependencies; use the verified ONNX conversion for inference. Historical
compatibility observations and acquisition provenance remain in the archive.

## Export, validate, activate

Export requires accessible CUDA for reference/parity validation. Use a new output
directory inside the repository's ignored `models/` area. Activation is an
intentional tracked config change; this documentation cleanup does not perform it.

Export tools are isolated as a dependency **group** within the same environment, not installed for default inference:

```bash
uv sync --locked --group model-export
uv run --locked --group model-export python -m face_surveillance.vision.export_facenet \
  --manifest configs/model_probe.json \
  --output models/facenet-onnx-new
```

Use a new directory if that output already exists. Only after success, activate its generated manifest:

```bash
cp models/facenet-onnx-new/runtime_manifest.json configs/onnx_models.json
uv sync --locked
uv run --locked python -m face_surveillance.vision.onnx_probe \
  --manifest configs/onnx_models.json \
  --output runs/shared-onnx-runtime-new
```

Use a new run directory. The runtime checks all model hashes and the finalized export-report hash/status/provenance before inference. The report is part of the local model bundle; do not copy the ONNX file alone and omit its validation report. Regeneration changes timestamped report hashes, so activate the newly generated manifest rather than retaining a stale config. Manifest paths are repository-relative for the documented commands; the generated artifacts stay out of Git.

`export_facenet` requires the hash-pinned FaceNet source overlay only to instantiate the original architecture. It explicitly loads the local state dict with `weights_only=True` and strict keys, exports with modern `torch.onnx.export(dynamo=True)`, and validates the resulting graph and CUDA outputs. It never downloads a checkpoint. Failed export/parity produces a failure report and no runtime manifest.

The export group adds `onnxscript==0.7.2` and locked `onnx-ir==1.0.0`.
Default `uv sync --locked` omits/removes them; normal ONNX inference needs neither
these libraries nor the source overlay. Regenerated bundles retain the source's
usage restrictions and are not new independently trained recognition methods.

## Preprocessing and verification contracts

| Model | Input contract |
|---|---|
| SCRFD-10GF | InsightFace BGR detector adapter, pinned 640×640 profile; symbolic `?` dimensions fixed to 640 |
| ArcFace-family ResNet50 | Five-landmark alignment to 112×112, adapter RGB/NCHW normalization `(x-127.5)/127.5`; symbolic batch `None` fixed to one |
| FaceNet-family InceptionResnetV1 | Box RGB crop, PIL bilinear resize to 160×160; float32 `[1,3,160,160]`, `(x-127.5)/128` |

Both encoders produce 512 values in separate spaces. FaceNet preprocessing is
external to the ONNX graph; output shape is `[1,512]`. Crop conventions differ, so
this is not a controlled loss-only comparison. New shapes/preprocessing need
explicit compatibility verification and threshold recalibration.

FaceNet export uses opset 18, embedded weights, fixed batch one and TF32 disabled.
Eleven face-variant/synthetic inputs check finite raw outputs, unit norm within
`1e-4`, per-component agreement at `atol=1e-5, rtol=1e-4`, and pairwise cosine-matrix
difference at most `1e-5`. Normalization cannot hide invalid/scaled raw outputs.
These gates check numerical conversion, not participant recognition or mask robustness.

SCRFD's dimension overrides eliminate CPU shape-subgraph placement for the pinned
square CUDA profile; weights remain unchanged. Required-CUDA checks disable CPU
fallback and inspect executed-node profiles, not merely available provider names.
The shared ONNX checker blocks `facenet_pytorch`, `onnxscript`, and `onnx_ir` imports.

## Device checks and utility roles

Use new output directories. The active ONNX bundle is required for neural checks,
but the source overlay and export dependencies are not needed by these commands:

```bash
uv run --locked python scripts/check_gpu.py
uv run --locked python -m face_surveillance.vision.onnx_probe \
  --manifest configs/onnx_models.json --output runs/shared-onnx-new
uv run --locked python scripts/check_vision_cpu.py --output runs/cpu-regression-new
```

The first two require CUDA and fail visibly without it. The CPU regression explicitly
requests CPU sessions in the GPU distribution; it does not silently convert a failed
CUDA check into a CPU pass. It checks detection fixtures and finite embeddings,
optionally comparing historical CUDA vectors when present. These are readiness checks,
not identity accuracy or live-throughput benchmarks.

`model_probe.py` retains the original mixed PyTorch/ONNX CUDA reference experiment.
To reproduce it with the source manifest and verified overlay:

```bash
uv run --locked python -m face_surveillance.vision.model_probe \
  --manifest configs/model_probe.json --output runs/model-feasibility-new
```

Its shared hash/preprocessing/embedding helpers remain used by the exporter and
current ONNX/CPU checks. Preparation tools are not the application's worker; do not
run exports in HTTP requests or routine API startup. Integrating a neural recognizer
still needs a model-specific gallery and decision policy.

## Asset terms and provenance

- InsightFace code: MIT; supplied pretrained model usage has a separate non-commercial research restriction. The present project is a research prototype; weights are not committed or redistributed. [Upstream notice](https://github.com/deepinsight/insightface#license)
- FaceNet wrapper code: MIT. Its VGGFace2/CASIA pretrained state dicts are converted from Sandberg's implementation, whose supplied InceptionResnetV1 models are documented as softmax-trained. They are not Google's original triplet-loss checkpoint. [PyTorch implementation](https://github.com/timesler/facenet-pytorch), [original checkpoint provenance](https://github.com/davidsandberg/facenet#pre-trained-models)
- A separate, comprehensive weight grant for the selected VGGFace2-derived state dict was not established. Preserve this limitation; code MIT does not prove unrestricted weight/data rights. No commercial/redistribution clearance is claimed. This remains a provenance/licensing follow-up, separate from the successful technical probe.
- Fixture: NASA astronaut Eileen Collins photograph bundled with scikit-image, documented as public domain. It is a technical fixture, not a volunteer identity or a substitute for a labeled study. [Fixture documentation](https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.astronaut)

## Historical evidence

The [archive index](archive/README.md) links dated timings, export parity observations,
resolved failures and dependency workarounds. These records preserve the original
18/35-test milestones and environment history. Current software checks are recorded
in [project status](PROJECT_STATUS.md); old stage timings do not establish current
application FPS or completed recognition evaluation.
