# Pretrained-model feasibility

16 October 2026. Scope: M1's technical model probe, not the recorded-video baseline or recognition evaluation.

Update, 19 October: this document preserves the original PyTorch/overlay probe results. The current same-environment inference choice uses a verified FaceNet ONNX export, with no FaceNet package or exporter dependencies in routine inference. See [ENVIRONMENT.md](ENVIRONMENT.md) for the active setup and handoff.

## Result

SCRFD-10GF, the `buffalo_l` ResNet50 ArcFace-family encoder, and the FaceNet-family VGGFace2 InceptionResnetV1 checkpoint execute on the existing RTX 4070 Laptop GPU. The core environment and `uv.lock` were not changed. FaceNet uses an explicitly experimental import overlay, not a supported dependency resolution.

The local probe is:

```bash
.venv/bin/python -m face_surveillance.vision.model_probe \
  --manifest configs/model_probe.json \
  --output runs/model-feasibility-new
```

Use a **new** output directory for each run. GPU access is required; on this agent host the command must execute outside the filesystem/network sandbox to reach the driver. The probe never downloads models automatically and fails on missing/hash-mismatched assets. It records failures after successful preflight in `report.json` and refuses to overwrite previous runs.

## Checks performed and limits

- Bundled `skimage.data.astronaut` gives one detected face and five landmarks; a blank frame gives zero; a two-copy composite gives two. The composite repeats one person; it does not evaluate identity discrimination.
- Crops were inspected: ArcFace is five-landmark aligned at 112×112; FaceNet follows the package's box-crop convention at 160×160. FaceNet's RGB/NCHW fixed standardization was checked against the pinned upstream `extract_face` implementation with exact equality.
- Both models return finite 512-dimensional embeddings, normalized and checked for repeated-output agreement at `rtol=1e-4, atol=1e-5`. Earlier runs 02–04 compared only the first/last output. Independent review identified that an invalid intermediate output could escape that check; the corrected final run 05 validates all 34 outputs per encoder (initial, three warmups, 30 measured), outside timing, with zero maximum difference. Regression tests cover bad middle outputs with identical endpoints. Cross-model vectors are separate embedding spaces even though both have 512 elements.
- CPU fallback remains disabled for both ONNX sessions. Actual node profiles, not just available-provider names, show only `CUDAExecutionProvider` execution after fixed-dimension overrides.
- PyTorch loads FaceNet's complete local state dict strictly with `weights_only=True`; it uses evaluation and inference modes on CUDA. No `pretrained=` download path is invoked.
- Unit tests cover corrupt/missing assets, invalid embeddings, input layout/normalization, executed-provider validation, output preservation, schema rejection, and failed-run reporting. The original health/OpenAPI test remains intact.

This does **not** establish identity accuracy, unknown rejection, mask robustness, threshold calibration, tracking quality, alert reliability, full-video throughput or multi-camera capacity. No enrolled-person gallery or application workflow was added. Same-photo repeated embeddings are a numerical sanity check, not genuine/impostor recognition evidence.

## Hardware and measurements

Observed environment: Python 3.14.8, torch 2.14.1+cu130, torchvision 0.29.1+cu130, ONNX Runtime GPU 1.31.0, InsightFace 2.1, NumPy 2.5.4, OpenCV 5.0.0.93, Pillow 12.3.0, scikit-image 0.26.0; optional overlay facenet-pytorch 2.6.0. GPU: NVIDIA GeForce RTX 4070 Laptop GPU, 8188 MiB reported total VRAM, driver 615.78.08, CUDA 13.0, cuDNN 9.24.0.

Thirty-repetition runs on the single still fixture; run 05 uses the reviewed final source with all-output validation:

| Stage | Run 02 median / p95 (ms) | Run 03 median / p95 (ms) | Final run 05 median / p95 (ms) |
|---|---:|---:|---:|
| SCRFD, fixed 640×640 detector input | 12.25 / 13.65 | 8.35 / 10.17 | 8.46 / 9.59 |
| ArcFace-family encoder, one pre-aligned crop | 3.34 / 3.37 | 3.22 / 3.26 | 3.19 / 3.22 |
| FaceNet-family encoder, one pre-cropped face | 7.13 / 7.58 | 6.41 / 6.52 | 6.52 / 7.05 |

These are synchronized batch-one **instrumented feasibility timings**, with three untimed warmup iterations and ORT profiling enabled. Detector timing includes its adapter preprocessing/postprocessing. Encoder timing uses already prepared crops; FaceNet includes tensor transfer to CUDA and output transfer to CPU. Image decode, crop preparation, tracking, gallery matching, queues, preview, events and video encoding are excluded. Do not invert these figures to claim application FPS. Variation between short runs is visible; a sustained video benchmark remains M2.

Final-run source and core-lock hashes match the current files; normalized outputs from separate processes in runs 02 and 05 agree exactly on this fixture. Run 05 also records crop hashes. The final full suite passes 18 tests (17 probe tests plus the original scaffold test); Ruff lint and formatting pass. The independent review's material finding was corrected and regression-tested; no other material findings were reported.

Run 02's separately measured model construction/load times were approximately 159 ms detector, 97 ms ArcFace and 250 ms FaceNet; these exclude Python imports and earlier CUDA initialization and are not full cold-start latency. With all three loaded, device-wide memory sampled at the end was 909 MiB versus 44 MiB before loading. This is not a device-wide peak or solely process-owned measurement. Torch's own peak allocation/reservation was 126,324,736 / 150,994,944 bytes and excludes ORT allocations. Full reports/profiles contain the observations; no model outputs or images are committed.

## SCRFD strict-CUDA failure and diagnosis

The first probe (`runs/model-feasibility-01`) failed because the legacy detector input has symbolic `?` height and width. ORT placed dynamic-shape `Gather`, `Unsqueeze`, `Slice`, and `Concat` operations on CPU, causing `session.disable_cpu_ep_fallback=1` to reject session construction. A diagnostic profile with explicit CPU/CUDA providers showed convolutions on CUDA and those shape operations on CPU. The encoder had no CPU-assigned execution.

The tested fix is **session configuration**, not edited weights or a relaxed GPU check:

- Detector: `add_free_dimension_override_by_name("?", 640)` fixes both equal spatial dimensions.
- ArcFace: its symbolic batch `None` is fixed to one.
- Original model bytes and their hashes remain unchanged.
- Strict no-CPU fallback is retained, and executed provider events are validated after inference.

Run 02 recorded 5069 CUDA detector node events and 4420 CUDA encoder node events across its setup/warmup/timed calls, with no CPU events. This result is specific to the pinned square 640×640 profile. New input sizes must be explicitly configured and reprofiled; the symbolic name is checkpoint-specific. An upstream scikit-image alignment deprecation warning is visible but does not prevent this pinned environment's inference. Do not edit installed libraries or suppress it as a compatibility fix.

## Acquisition and reproducibility

`configs/model_probe.json` records upstream locations, expected local paths, SHA-256 digests, byte counts, fixed dimensions, license/provenance notes and the files in the optional overlay. Paths in it are relative to the repository. Model and wheel downloads live under ignored `models/feasibility/`; outputs/embeddings/profiles live under ignored `runs/`. The probe pins source/lock/manifest hashes and package versions in its run report.

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
.venv/bin/python - <<'PY'
import hashlib, json, zipfile
from pathlib import Path

manifest = json.loads(Path("configs/model_probe.json").read_text())
for asset in (manifest["archive"], manifest["assets"]["facenet"],
              manifest["assets"]["facenet_wheel"]):
    with Path(asset["path"]).open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == asset["sha256"]
with zipfile.ZipFile(manifest["archive"]["path"]) as archive:
    for name in ("det_10g.onnx", "w600k_r50.onnx"):
        Path("models/feasibility", name).write_bytes(archive.read(name))
PY

uv pip install --cache-dir /tmp/odin-uv-cache --no-index --no-deps \
  --target models/feasibility/python \
  models/feasibility/facenet_pytorch-2.6.0-py3-none-any.whl
```

The wheel digest was checked against published PyPI metadata. The legacy weight hashes are pins of the bytes acquired from upstream release URLs, not independently signed attestations or evidence of model accuracy. Subsequent runs verify both weights and installed overlay source/data hashes before loading.

### Deliberate dependency override

The actual 2.6.0 wheel declares NumPy `<2`, Pillow `<10.3`, torch `<2.3`, torchvision `<0.18`. Installing it with normal resolution into the current backend would conflict with the working stack. We inspected the wheel imports, model constructor/loader, and crop implementation, then installed **only this wheel** into an ignored `--target` directory. No core dependency is installed/replaced. The probe prepends that directory only in its own process; the API and normal core imports do not see FaceNet.

This intentionally bypasses metadata constraints for a narrow compatibility experiment. Encoder initialization, strict local loading, crop equivalence and CUDA inference passed with the actual existing versions. That does not certify MTCNN/training/all APIs or provide a supported production dependency lock. Use the overlay for headless experiments; before a normal production integration, use a separately resolved environment or validate an explicitly versioned conversion/adapter. Do not add an incompatible bare `facenet-pytorch` requirement to `pyproject.toml`.

## Asset terms and provenance

- InsightFace code: MIT; supplied pretrained model usage has a separate non-commercial research restriction. The present project is a research prototype; weights are not committed or redistributed. [Upstream notice](https://github.com/deepinsight/insightface#license)
- FaceNet wrapper code: MIT. Its VGGFace2/CASIA pretrained state dicts are converted from Sandberg's implementation, whose supplied InceptionResnetV1 models are documented as softmax-trained. They are not Google's original triplet-loss checkpoint. [PyTorch implementation](https://github.com/timesler/facenet-pytorch), [original checkpoint provenance](https://github.com/davidsandberg/facenet#pre-trained-models)
- A separate, comprehensive weight grant for the selected VGGFace2-derived state dict was not established. Preserve this limitation; code MIT does not prove unrestricted weight/data rights. No commercial/redistribution clearance is claimed. This remains a provenance/licensing follow-up, separate from the successful technical probe.
- Fixture: NASA astronaut Eileen Collins photograph bundled with scikit-image, documented as public domain. It is a technical fixture, not a volunteer identity or a substitute for a labeled study. [Fixture documentation](https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.astronaut)

## Next boundary

The model execution/preprocessing feasibility subtask is demonstrated. M1's full data protocol and the actual independent volunteer recordings still need to be prepared; model execution is not the whole milestone. Next build M2's reproducible recorded-video capture/sampling/timestamp/timing runner, reusing these proven model settings while replacing a single still fixture with explicit selected-frame manifests. No tracking/alerts/ensemble features should be inferred from this probe's passing status.
