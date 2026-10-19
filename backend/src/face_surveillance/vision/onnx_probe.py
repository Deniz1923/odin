"""Verify three local ONNX models together; deliberately no video or gallery workflow."""

import argparse
import hashlib
import importlib.abc
import importlib.metadata as metadata
import json
import sys
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from face_surveillance.vision.model_probe import (
    create_run_directory,
    facenet_input,
    gpu_snapshot,
    summarize_profile,
    validate_embedding_series,
    verify_artifact,
)

EXPORT_ONLY = ("facenet_pytorch", "onnxscript", "onnx_ir")


@contextmanager
def runtime_only_imports():
    def blocked(name):
        return name.split(".", 1)[0] in EXPORT_ONLY

    if any(blocked(name) for name in sys.modules):
        raise RuntimeError("Export-only module already imported into runtime process")

    class BlockExports(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if blocked(fullname):
                raise RuntimeError(f"Export-only import attempted: {fullname}")
            return None

    finder = BlockExports()
    sys.meta_path.insert(0, finder)
    try:
        yield
    finally:
        sys.meta_path.remove(finder)


def verify_runtime_manifest(path: Path) -> tuple[dict, dict[str, Path]]:
    root = Path(__file__).resolve().parents[4]
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported runtime manifest schema_version")
    validation = manifest["validation"]
    report_path = root / validation["report_path"]
    verify_artifact(report_path, validation["report_sha256"])
    report = json.loads(report_path.read_text())
    if report.get("status") != "passed":
        raise ValueError("A passed export validation report is required")
    if report["derived_sha256"] != manifest["assets"]["facenet"]["sha256"]:
        raise ValueError("Manifest derived hash disagrees with export validation")
    if report["source_checkpoint_sha256"] != manifest["derived_from"]["sha256"]:
        raise ValueError("Source checkpoint provenance disagrees with export validation")
    paths = {}
    for name in ("detector", "arcface", "facenet"):
        asset = manifest["assets"][name]
        paths[name] = root / asset["path"]
        verify_artifact(paths[name], asset["sha256"])
    return manifest, paths


def run_probe(manifest_path: Path, output: Path, repeats: int = 30) -> dict:
    if repeats < 2:
        raise ValueError("At least two repetitions required")
    manifest, paths = verify_runtime_manifest(manifest_path)
    create_run_directory(output)
    report = {
        "status": "running",
        "started_utc": datetime.now(UTC).isoformat(),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "probe_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "helper_code_sha256": hashlib.sha256(
            Path(__file__).with_name("model_probe.py").read_bytes()
        ).hexdigest(),
        "core_lock_sha256": hashlib.sha256(
            (Path(__file__).resolve().parents[4] / "uv.lock").read_bytes()
        ).hexdigest(),
        "repetitions": repeats,
        "claim": "Shared-environment ONNX readiness only; no video recognition or accuracy claim",
    }
    try:
        with runtime_only_imports():
            _execute(manifest, paths, output, report, repeats)
        report["status"] = "passed"
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        report["finished_utc"] = datetime.now(UTC).isoformat()
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def _execute(manifest, paths, output, report, repeats):
    import cv2
    import onnxruntime as ort
    import torch
    from insightface.model_zoo.arcface_onnx import ArcFaceONNX
    from insightface.model_zoo.scrfd import SCRFD
    from insightface.utils.face_align import norm_crop
    from PIL import Image
    from skimage import data

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    ort.preload_dlls()
    report["environment"] = {
        "python": sys.version,
        "packages": {
            name: metadata.version(name)
            for name in (
                "torch",
                "onnxruntime-gpu",
                "insightface",
                "numpy",
                "opencv-python",
                "Pillow",
            )
        },
        "gpu_before": gpu_snapshot(),
        "blocked_imports": list(EXPORT_ONLY),
        "tf32": False,
    }
    sessions = {}
    for name, path in paths.items():
        options = ort.SessionOptions()
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        for dimension, size in manifest["onnx_dimension_overrides"].get(name, {}).items():
            options.add_free_dimension_override_by_name(dimension, size)
        options.enable_profiling = True
        options.profile_file_prefix = str(output / name)
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        sessions[name] = ort.InferenceSession(
            str(path),
            sess_options=options,
            providers=[("CUDAExecutionProvider", {"use_tf32": "0"})],
        )
        sessions[name].disable_fallback()
        if sessions[name].get_providers()[0] != "CUDAExecutionProvider":
            raise RuntimeError(f"{name}: CUDA provider required")
    detector = SCRFD(str(paths["detector"]), session=sessions["detector"])
    detector.prepare(ctx_id=0, input_size=(640, 640), det_thresh=0.5)
    arcface = ArcFaceONNX(str(paths["arcface"]), session=sessions["arcface"])
    rgb = data.astronaut()
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    boxes, landmarks = detector.detect(bgr)
    blank, _ = detector.detect(np.zeros_like(bgr))
    two_faces, _ = detector.detect(np.concatenate([bgr, bgr], axis=1))
    if len(boxes) != 1 or len(blank) != 0 or len(two_faces) != 2:
        raise ValueError("Single/blank/two-copy face fixture counts failed")
    report["face_counts"] = {"single": 1, "blank": 0, "two_copy": 2}
    arc_crop = norm_crop(bgr, landmarks[0], image_size=112)
    h, w = rgb.shape[:2]
    x1, y1, x2, y2 = boxes[0, :4]
    fn_crop = (
        Image.fromarray(rgb)
        .crop((max(0, int(x1)), max(0, int(y1)), min(w, int(x2)), min(h, int(y2))))
        .resize((160, 160), Image.Resampling.BILINEAR)
    )
    fn_input = facenet_input(np.asarray(fn_crop))
    operations = {
        "detector": lambda: detector.detect(bgr),
        "arcface": lambda: arcface.get_feat(arc_crop),
        "facenet": lambda: sessions["facenet"].run(["embedding"], {"face": fn_input})[0],
    }
    report["timings_ms"] = {}
    report["embedding_checks"] = {}
    for name, operation in operations.items():
        outputs = [operation() for _ in range(3)]
        durations = []
        for _ in range(repeats):
            start = time.perf_counter()
            value = operation()
            durations.append((time.perf_counter() - start) * 1000)
            outputs.append(value)
        report["timings_ms"][name] = {
            "median": float(np.median(durations)),
            "p95": float(np.percentile(durations, 95)),
            "samples": durations,
        }
        if name != "detector":
            value, report["embedding_checks"][name] = validate_embedding_series(outputs)
            np.save(output / f"{name}_embedding.npy", value)
    report["onnx_execution"] = {
        name: summarize_profile(Path(session.end_profiling())) for name, session in sessions.items()
    }
    report["environment"]["gpu_models_loaded"] = gpu_snapshot()
    report["export_only_modules_loaded"] = [
        name for name in sys.modules if name.split(".", 1)[0] in EXPORT_ONLY
    ]
    if report["export_only_modules_loaded"]:
        raise RuntimeError("Export-only library entered runtime process")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("configs/onnx_models.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=30)
    args = parser.parse_args()
    result = run_probe(args.manifest, args.output, args.repeats)
    print(
        json.dumps(
            {
                "status": result["status"],
                "report": str(args.output / "report.json"),
                "onnx_execution": result["onnx_execution"],
                "timings_ms": {
                    name: {key: value[key] for key in ("median", "p95")}
                    for name, value in result["timings_ms"].items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
