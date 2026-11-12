"""Hash-checked ONNX Runtime sessions with an explicit, reported execution device.

`cuda` requires CUDA for every graph node (no silent CPU fallback); `cpu` uses only the
CPU provider; `auto` tries strict CUDA first and otherwise uses CPU. The device actually
used is returned so the application can display it.
"""

import json
from pathlib import Path

from face_surveillance.classical.detection import ROOT
from face_surveillance.vision.model_probe import verify_artifact

DEVICES = ("auto", "cuda", "cpu")


def load_asset(name: str, manifest: Path | None = None) -> tuple[Path, str, dict]:
    """Verify one model listed in the ONNX manifest; return path, SHA-256 and dimension overrides.

    Only the requested asset is checked, so SCRFD/ArcFace do not depend on the FaceNet export.
    """
    manifest = manifest or ROOT / "configs/onnx_models.json"
    spec = json.loads(manifest.read_text())
    asset = spec["assets"][name]
    path = ROOT / asset["path"]
    if not path.exists():
        raise ValueError(
            f"{name} model missing at {asset['path']}; acquire it as in docs/ENVIRONMENT.md"
        )
    verify_artifact(path, asset["sha256"])
    return path, asset["sha256"], spec.get("onnx_dimension_overrides", {}).get(name, {})


def create_session(path: Path, overrides: dict, device: str):
    """Create an inference session on exactly `device` ("cuda" or "cpu")."""
    import onnxruntime as ort

    options = ort.SessionOptions()
    for dimension, size in overrides.items():
        options.add_free_dimension_override_by_name(dimension, size)
    if device == "cpu":
        session = ort.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"]
        )
    elif device == "cuda":
        ort.preload_dlls()
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        session = ort.InferenceSession(
            str(path),
            sess_options=options,
            providers=[("CUDAExecutionProvider", {"use_tf32": "0"})],
        )
        if session.get_providers()[0] != "CUDAExecutionProvider":
            raise RuntimeError(f"{path.name}: CUDA provider is not active")
    else:
        raise ValueError(f"Unknown execution device {device!r}")
    session.disable_fallback()
    return session


def create_sessions(paths: dict[str, tuple[Path, dict]], device: str) -> tuple[dict, str]:
    """Create all sessions on one device. `auto` uses CUDA only if every session succeeds."""
    if device not in DEVICES:
        raise ValueError(f"Device must be one of {', '.join(DEVICES)}")
    if device == "auto":
        try:
            return create_sessions(paths, "cuda")
        except Exception:
            return create_sessions(paths, "cpu")
    return {
        name: create_session(path, overrides, device) for name, (path, overrides) in paths.items()
    }, device
