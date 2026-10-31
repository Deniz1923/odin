"""Explicit CPU regression for the three existing neural models after OpenCV changes."""

import argparse
import json
from importlib import metadata
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from insightface.model_zoo.arcface_onnx import ArcFaceONNX
from insightface.model_zoo.scrfd import SCRFD
from insightface.utils.face_align import norm_crop
from PIL import Image
from skimage import data

from face_surveillance.classical.detection import HaarDetector
from face_surveillance.vision.model_probe import facenet_input, normalize_embedding
from face_surveillance.vision.onnx_probe import verify_runtime_manifest

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
report = dict(
    status="running",
    device="explicit CPU; no GPU verification",
    opencv_distribution=metadata.version("opencv-contrib-python"),
)
try:
    try:
        metadata.version("opencv-python")
    except metadata.PackageNotFoundError:
        pass
    else:
        raise ValueError("Conflicting main OpenCV distribution is installed")
    manifest, paths = verify_runtime_manifest(Path("configs/onnx_models.json"))
    sessions = {}
    for name, path in paths.items():
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        for dimension, size in manifest["onnx_dimension_overrides"].get(name, {}).items():
            options.add_free_dimension_override_by_name(dimension, size)
        sessions[name] = ort.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        sessions[name].disable_fallback()
    report["providers"] = {name: session.get_providers() for name, session in sessions.items()}
    rgb = data.astronaut()
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    detector = SCRFD(str(paths["detector"]), session=sessions["detector"])
    detector.prepare(ctx_id=-1, input_size=(640, 640), det_thresh=0.5)
    boxes, landmarks = detector.detect(bgr)
    blank, _ = detector.detect(np.zeros_like(bgr))
    multiple, _ = detector.detect(np.concatenate([bgr, bgr], axis=1))
    assert (len(boxes), len(blank), len(multiple)) == (1, 0, 2)
    arc = ArcFaceONNX(str(paths["arcface"]), session=sessions["arcface"])
    arc_crop = norm_crop(bgr, landmarks[0], image_size=112)
    x1, y1, x2, y2 = boxes[0, :4].astype(int)
    fn_crop = (
        Image.fromarray(rgb).crop((x1, y1, x2, y2)).resize((160, 160), Image.Resampling.BILINEAR)
    )
    outputs = {
        "arcface": arc.get_feat(arc_crop),
        "facenet": sessions["facenet"].run(
            ["embedding"], {"face": facenet_input(np.asarray(fn_crop))}
        )[0],
    }
    report["embeddings"] = {}
    for name, output in outputs.items():
        value = normalize_embedding(output)
        assert output.shape == (1, 512)
        reference_path = Path("runs/shared-onnx-final-merge") / f"{name}_embedding.npy"
        entry = dict(
            shape=list(output.shape),
            finite=bool(np.isfinite(output).all()),
            normalized_norm=float(np.linalg.norm(value)),
        )
        if reference_path.exists():
            delta = float(np.max(np.abs(value - np.load(reference_path))))
            assert delta < 1e-4
            entry["max_difference_from_historical_cuda"] = delta
        report["embeddings"][name] = entry
    haar = HaarDetector()
    assert len(haar.detect(bgr)) == 1
    assert len(haar.detect(np.zeros_like(bgr))) == 0
    report.update(
        status="passed",
        scrfd_face_counts=[1, 0, 2],
        haar_face_counts=[1, 0],
        claim="Execution/preprocessing regression only; no identity accuracy claim",
    )
except Exception as exc:
    report.update(status="failed", error=str(exc))
    raise
finally:
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
