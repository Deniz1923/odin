"""Local-only model feasibility probe; not a recognition or accuracy benchmark."""

import argparse
import hashlib
import importlib.metadata as metadata
import json
import subprocess
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np


def verify_artifact(path: Path, expected_sha256: str) -> str:
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != expected_sha256:
        raise ValueError(f"SHA-256 mismatch: {path}")
    return digest


def normalize_embedding(value: np.ndarray, dimension: int = 512) -> np.ndarray:
    embedding = np.asarray(value, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(embedding)
    if embedding.size != dimension or not np.isfinite(embedding).all():
        raise ValueError("Nonfinite or incorrectly sized embedding")
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("Invalid embedding norm")
    return embedding / norm


def validate_embedding_series(outputs: list[np.ndarray]) -> tuple[np.ndarray, dict]:
    if not outputs:
        raise ValueError("No embedding outputs")
    initial = normalize_embedding(outputs[0])
    maximum_difference = 0.0
    for output in outputs:
        final = normalize_embedding(output)
        if not np.allclose(initial, final, rtol=1e-4, atol=1e-5):
            raise ValueError("Embedding drift beyond repeatability tolerance")
        maximum_difference = max(maximum_difference, float(np.max(np.abs(initial - final))))
    return final, {
        "dimension": final.size,
        "l2_norm": float(np.linalg.norm(final)),
        "repeat_max_absolute_difference": maximum_difference,
        "checked_outputs": len(outputs),
        "finite": True,
    }


def facenet_input(rgb: np.ndarray) -> np.ndarray:
    if rgb.shape != (160, 160, 3) or rgb.dtype != np.uint8:
        raise ValueError("FaceNet expects a 160x160 uint8 RGB crop")
    standardized = (rgb.astype(np.float32) - 127.5) / 128.0
    return np.ascontiguousarray(standardized.transpose(2, 0, 1)[None])


def summarize_profile(path: Path) -> dict[str, int]:
    events = json.loads(path.read_text())
    providers = Counter(
        event["args"]["provider"]
        for event in events
        if event.get("cat") == "Node" and event.get("args", {}).get("provider")
    )
    if not providers["CUDAExecutionProvider"]:
        raise ValueError("Profile contains no executed CUDA nodes")
    if any(provider != "CUDAExecutionProvider" for provider in providers):
        raise ValueError(f"Unexpected non-CUDA/CPU execution: {dict(providers)}")
    return dict(providers)


def create_run_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=False)


def gpu_snapshot() -> str:
    """Device-wide usage, including ORT and other processes; not torch allocator usage."""
    return subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.used,memory.free,"
            "utilization.gpu,power.draw,temperature.gpu",
            "--format=csv,noheader",
        ],
        text=True,
        timeout=10,
    ).strip()


def run_probe(manifest_path: Path, output: Path, repeats: int = 30) -> dict:
    if repeats < 2:
        raise ValueError("At least two timed repetitions required")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported manifest schema_version")
    # Paths in the checked-in manifest are relative to the repository, not shell cwd.
    root = Path(__file__).resolve().parents[4]
    paths = {}
    for key, artifact in manifest["assets"].items():
        paths[key] = root / artifact["path"]
        verify_artifact(paths[key], artifact["sha256"])
    overlay = root / manifest["overlay_path"]
    for relative, digest in manifest["overlay_files"].items():
        verify_artifact(overlay / relative, digest)
    create_run_directory(output)
    report = {
        "schema_version": 1,
        "status": "running",
        "started_utc": datetime.now(UTC).isoformat(),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "probe_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "core_lock_sha256": hashlib.sha256((root / "uv.lock").read_bytes()).hexdigest(),
        "claim": "Model execution/preprocessing sanity only; no recognition accuracy claim",
        "timing_scope": "batch=1, host-to-host, synchronized, ORT profiling enabled",
        "repetitions": repeats,
    }
    try:
        _execute_probe(manifest, paths, overlay, output, report, repeats)
        report["status"] = "passed"
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        report["finished_utc"] = datetime.now(UTC).isoformat()
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def _execute_probe(manifest, paths, overlay, output, report, repeats):
    # Heavy imports stay out of unit tests and application startup. No model constructors download.
    import cv2
    import onnxruntime as ort
    import torch
    from insightface.model_zoo.arcface_onnx import ArcFaceONNX
    from insightface.model_zoo.scrfd import SCRFD
    from insightface.utils.face_align import norm_crop
    from PIL import Image
    from skimage import data

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no CPU fallback is permitted")
    torch.set_num_threads(1)
    torch.manual_seed(0)
    ort.preload_dlls()
    sys.path.insert(0, str(overlay))
    from facenet_pytorch import InceptionResnetV1, extract_face

    report["environment"] = {
        "python": sys.version,
        "packages": {
            name: metadata.version(name)
            for name in (
                "torch",
                "torchvision",
                "onnxruntime-gpu",
                "insightface",
                "numpy",
                "opencv-python",
                "Pillow",
                "scikit-image",
                "facenet-pytorch",
            )
        },
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0),
        "gpu_before": gpu_snapshot(),
        "seed": 0,
        "torch_threads": 1,
        "dependency_override": manifest["dependency_override"],
    }
    report["assets"] = manifest["assets"]
    report["model_load_ms"] = {}

    def cuda_session(key):
        options = ort.SessionOptions()
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        options.enable_profiling = True
        options.profile_file_prefix = str(output / key)
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        # Pinned legacy SCRFD uses '?' for both spatial dimensions. Fixing the
        # declared square input folds its CPU shape subgraph during optimization.
        for symbol, size in manifest["onnx_dimension_overrides"].get(key, {}).items():
            options.add_free_dimension_override_by_name(symbol, size)
        started = time.perf_counter()
        session = ort.InferenceSession(
            str(paths[key]),
            sess_options=options,
            providers=["CUDAExecutionProvider"],
        )
        session.disable_fallback()
        if session.get_providers()[0] != "CUDAExecutionProvider":
            raise RuntimeError(f"{key}: CUDA provider not active")
        report.setdefault("onnx_inputs", {})[key] = {
            "shape": session.get_inputs()[0].shape,
            "dimension_overrides": manifest["onnx_dimension_overrides"].get(key, {}),
        }
        report["model_load_ms"][key] = (time.perf_counter() - started) * 1000
        return session

    detector_session = cuda_session("detector")
    detector = SCRFD(str(paths["detector"]), session=detector_session)
    detector.prepare(
        ctx_id=0,
        input_size=tuple(manifest["detector_input_size"]),
        det_thresh=manifest["detector_smoke_threshold"],
    )
    arc_session = cuda_session("arcface")
    arcface = ArcFaceONNX(str(paths["arcface"]), session=arc_session)
    started = time.perf_counter()
    facenet = InceptionResnetV1(
        pretrained=None,
        classify=True,
        num_classes=manifest["facenet_classes"],
    )
    facenet.load_state_dict(torch.load(paths["facenet"], map_location="cpu", weights_only=True))
    facenet.classify = False
    facenet = facenet.eval().to("cuda")
    torch.cuda.synchronize()
    report["model_load_ms"]["facenet"] = (time.perf_counter() - started) * 1000

    rgb = data.astronaut()
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    Image.fromarray(rgb).save(output / "fixture.png")
    report["fixture"] = {
        "source": "skimage.data.astronaut (NASA Eileen Collins public-domain photograph)",
        "sha256_rgb_pixels": hashlib.sha256(rgb.tobytes()).hexdigest(),
        "shape": list(rgb.shape),
    }
    boxes, landmarks = detector.detect(bgr)
    if len(boxes) != 1 or landmarks is None or landmarks.shape != (1, 5, 2):
        raise RuntimeError("Expected one astronaut face with five landmarks")
    if not np.isfinite(boxes).all() or not np.isfinite(landmarks).all():
        raise RuntimeError("Nonfinite detection or landmark coordinates")
    negative, _ = detector.detect(np.zeros_like(bgr))
    multiple, _ = detector.detect(np.concatenate([bgr, bgr], axis=1))
    if len(negative) != 0 or len(multiple) != 2:
        raise RuntimeError("Blank/two-copy fixture detection check failed")
    report["detection"] = {
        "single_faces": len(boxes),
        "blank_faces": len(negative),
        "two_copy_faces": len(multiple),
        "bbox_and_score": boxes[0].tolist(),
        "landmarks": landmarks[0].tolist(),
        "input_size": manifest["detector_input_size"],
        "smoke_threshold_not_calibrated": manifest["detector_smoke_threshold"],
    }

    arc_crop = norm_crop(bgr, landmarks[0], image_size=112)
    cv2.imwrite(str(output / "arcface_crop.png"), arc_crop)
    box = boxes[0, :4]
    left, top = max(0, int(box[0])), max(0, int(box[1]))
    right, bottom = min(rgb.shape[1], int(box[2])), min(rgb.shape[0], int(box[3]))
    fn_crop = (
        Image.fromarray(rgb)
        .crop((left, top, right, bottom))
        .resize(
            (160, 160),
            Image.Resampling.BILINEAR,
        )
    )
    fn_crop.save(output / "facenet_crop.png")
    report["crop_sha256"] = {
        name: hashlib.sha256((output / name).read_bytes()).hexdigest()
        for name in ("arcface_crop.png", "facenet_crop.png")
    }
    fn_input = facenet_input(np.asarray(fn_crop))
    # Check our explicit preprocessing against the pinned upstream extraction convention.
    upstream = (extract_face(Image.fromarray(rgb), box, image_size=160) - 127.5) / 128
    np.testing.assert_allclose(fn_input[0], upstream.numpy(), atol=0, rtol=0)
    report["preprocessing"] = {
        "arcface": "SCRFD five landmarks -> InsightFace 112px norm_crop; RGB NCHW; (x-127.5)/127.5",
        "facenet": "SCRFD box -> PIL bilinear 160px crop; RGB NCHW; (x-127.5)/128; no rotation",
        "facenet_upstream_crop_equivalence": True,
        "comparison_limit": (
            "FaceNet box crop differs from ArcFace landmark alignment; not a loss-only comparison"
        ),
    }

    def facenet_forward():
        with torch.inference_mode():
            return facenet(torch.from_numpy(fn_input).to("cuda")).cpu().numpy()

    operations = {
        "detector": lambda: detector.detect(bgr),
        "arcface": lambda: arcface.get_feat(arc_crop),
        "facenet": facenet_forward,
    }
    report["timings_ms"] = {}
    report["embedding_checks"] = {}
    for name, operation in operations.items():
        warm = time.perf_counter()
        first = operation()
        outputs = [first]
        torch.cuda.synchronize()
        warm_ms = (time.perf_counter() - warm) * 1000
        for _ in range(3):
            outputs.append(operation())
        torch.cuda.synchronize()
        durations = []
        for _ in range(repeats):
            started = time.perf_counter()
            last = operation()
            torch.cuda.synchronize()
            durations.append((time.perf_counter() - started) * 1000)
            outputs.append(last)
        report["timings_ms"][name] = {
            "first_timed_call_after_setup": warm_ms,
            "median": float(np.median(durations)),
            "p95": float(np.percentile(durations, 95)),
            "samples": durations,
        }
        if name != "detector":
            final, checks = validate_embedding_series(outputs)
            np.save(output / f"{name}_embedding.npy", final)
            report["embedding_checks"][name] = checks
    report["onnx_execution"] = {}
    for name, session in (("detector", detector_session), ("arcface", arc_session)):
        profile = Path(session.end_profiling())
        report["onnx_execution"][name] = {
            "profile": profile.name,
            "node_provider_counts": summarize_profile(profile),
        }
    report["environment"]["gpu_models_loaded"] = gpu_snapshot()
    report["environment"]["torch_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    report["environment"]["torch_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("configs/model_probe.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=30)
    args = parser.parse_args()
    report = run_probe(args.manifest, args.output, args.repeats)
    print(
        json.dumps(
            {
                "status": report["status"],
                "report": str(args.output / "report.json"),
                "timings_ms": {
                    k: {s: v[s] for s in ("median", "p95")} for k, v in report["timings_ms"].items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
