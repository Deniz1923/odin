# odin

Local face-recognition research prototype for CENG391. Three methods run through the
same pipeline, gallery, operator UI and evaluation runner:

| `ODIN_METHOD` / `--method` | Detector | Recognizer |
|---|---|---|
| `lbp` (default) | Haar | **our own NumPy LBP, spatial histograms and chi-square matching** |
| `scrfd-lbp` | SCRFD-10GF | our LBP, on SCRFD boxes (matched-detector comparison) |
| `arcface` | SCRFD-10GF | pretrained ArcFace R50 (w600k), five-landmark alignment, cosine similarity |

FaceNet is prepared (ONNX export) but not yet connected to the application.

Detection and recognition are injected into the shared pipeline. Each recognizer
owns its preprocessing, descriptors and matching settings; the enabled implementation
is `LBPRecognizer`. See the [recognizer contract](docs/ARCHITECTURE.md#enrollment-and-recognition).

## Start here as a teammate

| Document | Purpose |
|---|---|
| [Project status](docs/PROJECT_STATUS.md) | Course coverage, preliminary evidence, eight-week scope and progress-report checklist |
| [Preliminary scope and schedule](docs/SCOPE_AND_SCHEDULE.md) | Completed milestones, remaining tasks, proposed team roles and report deadlines |
| [Study guide](docs/PROJECT_REVIEW_AND_STUDY_GUIDE.md) | Reading route and concepts to explain |
| [Classical walkthrough](docs/CLASSICAL_PIPELINE.md) | Implemented recognition algorithm |
| [Architecture](docs/ARCHITECTURE.md) | Current components, concurrency and data lifecycle |
| [Evaluation](docs/EVALUATION.md) | Study splits, calibration, metrics and claim limits |
| [Datasets and actor study](docs/DATASETS.md) | Reserved benchmarks, actor media sources and appearance-change protocol |
| [Environment](docs/ENVIRONMENT.md) | Locked stack, neural assets, export and device checks |
| [Archive](docs/archive/README.md) | Superseded plans and dated preparation results |

## Run the operator application

From this directory, using the existing Python 3.14 / Node 26 setup:

```bash
uv sync --locked
uv run --locked python scripts/acquire_haar.py
npm --prefix frontend ci
npm --prefix frontend run build
uv run --locked python -m uvicorn face_surveillance.api.main:app --host 127.0.0.1 --port 8000
```

Python dependencies and the editable project are installed by `uv sync --locked`;
no manual wheel downloads or `PYTHONPATH` override are needed. For a setup check,
`uv sync --locked --dry-run` should report that it would make no changes.

To run SCRFD + ArcFace instead, acquire the neural models as in
[the environment guide](docs/ENVIRONMENT.md#acquire-source-assets-for-a-new-neural-bundle)
(only `det_10g.onnx` and `w600k_r50.onnx` are needed) and start with
`ODIN_METHOD=arcface`. `ODIN_DEVICE=auto` (default) uses CUDA when every model can run
on it and otherwise CPU; `cuda` or `cpu` forces one. The sidebar shows the device used.
Each method keeps its own gallery directory (`data/classical`, `data/scrfd-lbp`,
`data/arcface`) because descriptors from different methods cannot be compared.

Open **http://127.0.0.1:8000**. Use one Uvicorn worker; no reload while a camera is
active. The Haar XML is a local asset; acquisition is explicit,
hash checked and never happens at API startup. Its upstream license is embedded in
the XML. No other model download is needed for the classical app.

1. In **Authorized gallery**, enroll a participant with several different clear
   reference photographs, each containing exactly one face. Confirm authorization.
2. Enable appearance alerts only for participants who should generate events.
3. In **Monitor**, start a local camera or upload a constant-frame-rate video.
4. Inspect candidate/confirmed/unknown/insufficient states and matching distances.
5. Review and acknowledge appearances in **Appearance events**; export JSON if needed.
6. Delete a participant to remove their descriptors and associated event history.

Haar is primarily a frontal-face detector. Pose, small faces, masks and poor lighting
can cause detection or recognition failures. The default matching distance and
quality rules are **uncalibrated engineering settings**, not accuracy guarantees.
Distance is lower for closer matches; it is not a percentage probability.

Track/event JSON now carries `score` and `score_kind` instead of `distance`.
Existing event databases migrate on opening without changing recorded values or keys;
older code expecting the previous schema will not work with a migrated database.
Legacy LBP policy files remain readable; newly saved settings include their descriptor
space so another recognizer cannot reuse them silently.

For frontend development, run the API above and `npm --prefix frontend run dev`.
The Vite server proxies `/api` to the API without rewriting its path.

## Data handling

- Default private directory: `data/classical`, or set `ODIN_DATA_DIR` before startup.
- Descriptors are authenticated-encrypted with Fernet; the key and SQLite database
  live in the private directory with restrictive OS permissions. This protects a
  database-only copy, not a process operating as the same OS user or a stolen key.
- Original enrollment images are discarded. Preview frames and unknown descriptors
  stay transient. Full webcam video and separate face crops are not recorded.
- Uploaded replay copies are deleted on stop/end/failure; interrupted upload copies
  are cleaned at next startup. Use one application process per data directory.
- Event retention defaults to seven days; set `ODIN_EVENT_RETENTION_DAYS` (1–365)
  before startup to change it. Sweeping occurs at startup and when events are read.
- Policy settings persist locally. Saving new settings stops the active session.
- Deletion clears persistent identity/templates/events and active pipeline evidence.
  SQLite compaction is not a promise of secure physical SSD erasure. Delete any
  exported reports, footage and backups separately when required.
- API Host/Origin checks and a per-process mutation token support one trusted local
  operator. This is not a remote multi-user authentication system. Keep loopback binding.

## Evaluate recorded data

Copy and fill [the example manifest](experiments/classical.example.json). Paths are
relative to the manifest. Use separate capture sessions for enrollment, validation
and test; unknown test identities stay outside enrollment and validation.

```bash
uv run --locked python -m face_surveillance.classical.evaluate \
  --manifest experiments/my-validation.json --output runs/lbp-validation-01 --calibrate

uv run --locked python -m face_surveillance.classical.evaluate \
  --manifest experiments/my-test.json --output runs/lbp-test-01

uv run --locked python -m face_surveillance.classical.evaluate --method arcface \
  --manifest experiments/my-validation.json --output runs/arcface-validation-01 --calibrate
```

Calibrate each method separately: LBP distance (lower is closer) and ArcFace cosine
similarity (higher is closer) are different scales. The manifest's `policy.threshold`
must suit the method being run.

The calibration command requires `split: validation` and both known and unknown
usable observations. It selects a distance limit under an empirical unknown
false-accept constraint; it does not calibrate probabilities. Copy the chosen
threshold into the frozen test manifest and operator settings. Never tune on test.

The CLI processes every decoded frame; the operator replay prioritizes recency and
can drop frames. Both currently assume constant frame rate and derive media time
from frame index/FPS. Variable-frame-rate and network streams are not supported.

Output directories must be new. `predictions.jsonl` records every processed frame;
`report.json` records condition-specific counts, annotated-frame denominators,
one-to-one event/encounter outcomes when encounters are provided, input hashes,
configuration, detector/descriptor identity and processing timings. Missing
annotations are not treated as negative labels. Repeated video bytes are rejected
within one manifest to avoid inflating unique exposure.

Include actual pose, illumination, distance/face-size and physical occlusion/mask
recordings. The generated smoke fixture tests plumbing, **not** recognition accuracy,
unknown-person robustness or independent research exposure.

## Checks and learning

```bash
uv run --locked pytest -q
uv run --locked ruff check backend tests scripts
uv run --locked ruff format --check backend tests scripts
npm --prefix frontend run lint
npm --prefix frontend run build
uv run --locked python scripts/classical_smoke.py --output runs/classical-smoke-new
uv run --locked python scripts/classical_smoke.py --method arcface --output runs/arcface-smoke-new
uv run --locked python scripts/check_vision_cpu.py --output runs/cpu-regression-new
```

The neural-model regression needs the previously prepared local model bundle. Its
CPU mode is explicit; original CUDA probes remain strict and require a working GPU.

See [project status](docs/PROJECT_STATUS.md) for completed checks, remaining work and
the progress-report checklist. Historical plans and measurements are indexed in
the [archive](docs/archive/README.md).
