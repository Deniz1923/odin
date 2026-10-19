"""Export and numerically validate the pinned FaceNet encoder; no video processing."""

import argparse
import hashlib
import importlib.metadata as metadata
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from face_surveillance.vision.model_probe import (
    create_run_directory,
    facenet_input,
    gpu_snapshot,
    summarize_profile,
    verify_artifact,
)


def compare_embeddings(reference: np.ndarray, candidate: np.ndarray) -> dict:
    if reference.ndim != 2 or reference.shape != candidate.shape:
        raise ValueError("Embedding shape mismatch")
    if reference.shape[0] < 2 or reference.shape[1] != 512:
        raise ValueError("Expected shape (N>=2, 512)")
    for values in (reference, candidate):
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite embedding output")
        if not np.allclose(np.linalg.norm(values, axis=1), 1.0, rtol=0, atol=1e-4):
            raise ValueError("Embedding must have unit norm before comparison")
    if not np.allclose(reference, candidate, atol=1e-5, rtol=1e-4):
        raise ValueError("Embedding component parity failed")
    # Normalize only for cosine computation, after checking actual exported outputs.
    left = reference / np.linalg.norm(reference, axis=1, keepdims=True)
    right = candidate / np.linalg.norm(candidate, axis=1, keepdims=True)
    cosine_difference = float(np.max(np.abs(left @ left.T - right @ right.T)))
    if cosine_difference > 1e-5:
        raise ValueError("Pairwise cosine parity failed")
    return {
        "cases": len(reference),
        "max_component_difference": float(np.max(np.abs(reference - candidate))),
        "max_pairwise_cosine_difference": cosine_difference,
        "component_atol": 1e-5,
        "component_rtol": 1e-4,
        "unit_norm_atol": 1e-4,
        "pairwise_cosine_atol": 1e-5,
    }


def parity_inputs() -> dict[str, np.ndarray]:
    from PIL import Image, ImageFilter
    from skimage import data

    # Public-domain astronaut fixture; fixed box from the earlier SCRFD probe.
    crop = (
        Image.fromarray(data.astronaut())
        .crop((180, 59, 270, 179))
        .resize(
            (160, 160),
            Image.Resampling.BILINEAR,
        )
    )
    rgb = np.asarray(crop)
    rng = np.random.default_rng(391)
    variants = {
        "astronaut": rgb,
        "flipped": rgb[:, ::-1].copy(),
        "darker": (rgb.astype(np.float32) * 0.5).astype(np.uint8),
        "brighter": np.clip(rgb.astype(np.float32) * 1.3, 0, 255).astype(np.uint8),
        "blurred": np.asarray(crop.filter(ImageFilter.GaussianBlur(2))),
        "grayscale": np.repeat(np.asarray(crop.convert("L"))[:, :, None], 3, axis=2),
        "black": np.zeros_like(rgb),
        "white": np.full_like(rgb, 255),
        "gray": np.full_like(rgb, 128),
        "noise": rng.integers(0, 256, rgb.shape, dtype=np.uint8),
        "gradient": np.broadcast_to(np.arange(160, dtype=np.uint8)[None, :, None], rgb.shape),
    }
    return {name: facenet_input(value) for name, value in variants.items()}


def export_and_verify(manifest_path: Path, output: Path) -> dict:
    root = Path(__file__).resolve().parents[4]
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported source manifest schema_version")
    paths = {}
    for name in ("facenet", "facenet_wheel", "detector", "arcface"):
        asset = manifest["assets"][name]
        paths[name] = root / asset["path"]
        verify_artifact(paths[name], asset["sha256"])
    overlay = root / manifest["overlay_path"]
    for name, digest in manifest["overlay_files"].items():
        verify_artifact(overlay / name, digest)
    create_run_directory(output)
    report = {
        "status": "running",
        "started_utc": datetime.now(UTC).isoformat(),
        "source_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "export_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "core_lock_sha256": hashlib.sha256((root / "uv.lock").read_bytes()).hexdigest(),
        "source_checkpoint_sha256": manifest["assets"]["facenet"]["sha256"],
        "claim": "Numerical export parity on still/synthetic fixtures; not recognition accuracy",
    }
    try:
        runtime = _export(manifest, paths, overlay, output, report)
        report["status"] = "passed"
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        report["finished_utc"] = datetime.now(UTC).isoformat()
        (output / "export_report.json").write_text(json.dumps(report, indent=2) + "\n")
    runtime["validation"] = {
        "report_path": str(output / "export_report.json"),
        "report_sha256": hashlib.sha256((output / "export_report.json").read_bytes()).hexdigest(),
    }
    (output / "runtime_manifest.json").write_text(json.dumps(runtime, indent=2) + "\n")
    return report


def _export(manifest, paths, overlay, output, report):
    import onnx
    import onnxruntime as ort
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for reference and export parity validation")
    sys.path.insert(0, str(overlay))
    from facenet_pytorch import InceptionResnetV1

    torch.set_num_threads(1)
    torch.manual_seed(391)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    report["environment"] = {
        "python": sys.version,
        "packages": {
            name: metadata.version(name)
            for name in (
                "torch",
                "onnx",
                "onnxruntime-gpu",
                "onnxscript",
                "onnx-ir",
                "facenet-pytorch",
            )
        },
        "gpu": gpu_snapshot(),
        "tf32": False,
        "opset": 18,
        "dynamo": True,
        "external_data": False,
    }
    model = InceptionResnetV1(
        pretrained=None, classify=True, num_classes=manifest["facenet_classes"]
    )
    model.load_state_dict(torch.load(paths["facenet"], map_location="cpu", weights_only=True))
    model.classify = False
    model.eval()
    target = output / "facenet_vggface2.onnx"
    started = time.perf_counter()
    torch.onnx.export(
        model,
        (torch.zeros(1, 3, 160, 160),),
        str(target),
        input_names=["face"],
        output_names=["embedding"],
        opset_version=18,
        dynamo=True,
        external_data=False,
        optimize=True,
        report=True,
        artifacts_dir=str(output),
    )
    report["export_seconds"] = time.perf_counter() - started
    onnx.checker.check_model(str(target), full_check=True)
    report["derived_sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
    report["derived_bytes"] = target.stat().st_size
    ort.preload_dlls()
    options = ort.SessionOptions()
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    options.enable_profiling = True
    options.profile_file_prefix = str(output / "facenet_parity")
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(
        str(target), sess_options=options, providers=[("CUDAExecutionProvider", {"use_tf32": "0"})]
    )
    session.disable_fallback()
    if session.get_providers()[0] != "CUDAExecutionProvider":
        raise RuntimeError("CUDA execution provider required")
    if session.get_inputs()[0].shape != [1, 3, 160, 160]:
        raise ValueError("Unexpected exported input shape")
    if session.get_outputs()[0].shape != [1, 512]:
        raise ValueError("Unexpected exported output shape")
    model.to("cuda")
    reference, candidate = [], []
    inputs = parity_inputs()
    for name, value in inputs.items():
        with torch.inference_mode():
            reference.append(model(torch.from_numpy(value).to("cuda")).cpu().numpy()[0])
        candidate.append(session.run(["embedding"], {"face": value})[0][0])
    report["parity"] = compare_embeddings(np.stack(reference), np.stack(candidate))
    report["input_sha256"] = {
        name: hashlib.sha256(value.tobytes()).hexdigest() for name, value in inputs.items()
    }
    np.savez(output / "reference_and_export.npz", reference=reference, candidate=candidate)
    report["executed_providers"] = summarize_profile(Path(session.end_profiling()))
    # A separate runtime manifest is emitted only after every validation gate passes.
    runtime = {
        "schema_version": 1,
        "assets": {key: manifest["assets"][key] for key in ("detector", "arcface")},
        "onnx_dimension_overrides": manifest["onnx_dimension_overrides"],
        "facenet_preprocessing": "RGB float32 NCHW (x-127.5)/128, fixed 160x160, batch1",
        "derived_from": manifest["assets"]["facenet"],
    }
    runtime["assets"]["facenet"] = {
        "path": str(target.resolve().relative_to(Path(__file__).resolve().parents[4])),
        "sha256": report["derived_sha256"],
        "bytes": report["derived_bytes"],
        "source": "Local verified export; source rights unchanged",
    }
    return runtime


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("configs/model_probe.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = export_and_verify(args.manifest, args.output)
    print(
        json.dumps({key: result[key] for key in ("status", "derived_sha256", "parity")}, indent=2)
    )


if __name__ == "__main__":
    main()
