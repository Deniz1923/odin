"""Pinned Haar detection and explicit classical preprocessing/quality checks."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[4]


class HaarDetector:
    def __init__(self, manifest: Path | None = None):
        manifest = manifest or ROOT / "configs/haar_model.json"
        spec = json.loads(manifest.read_text())
        path = ROOT / spec["path"]
        if not path.exists():
            raise ValueError("Haar model missing; run .venv/bin/python scripts/acquire_haar.py")
        if hashlib.sha256(path.read_bytes()).hexdigest() != spec["sha256"]:
            raise ValueError("Haar model SHA-256 mismatch")
        self.model_hash = spec["sha256"]
        self.cascade = cv2.CascadeClassifier(str(path))
        if self.cascade.empty():
            raise ValueError("Cannot load Haar cascade")

    def detect(self, bgr: np.ndarray) -> list[list[int]]:
        gray = cv2.equalizeHist(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
        # Bound detector work while mapping boxes back to original pixels.
        scale = min(1.0, 800 / max(gray.shape))
        if scale < 1:
            gray = cv2.resize(gray, None, fx=scale, fy=scale)
        boxes = self.cascade.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
        return [[int(v / scale) for v in box] for box in boxes]


def prepare_face(bgr: np.ndarray, box: list[int]) -> tuple[np.ndarray | None, str]:
    x, y, w, h = box
    height, width = bgr.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(width, x + w), min(height, y + h)
    if min(x2 - x1, y2 - y1) < 40:
        return None, "face_too_small"
    gray = cv2.cvtColor(bgr[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)
    if not 25 <= float(resized.mean()) <= 230:
        return None, "poor_exposure"
    if cv2.Laplacian(resized, cv2.CV_64F).var() < 30:
        return None, "blur_or_low_texture"
    return cv2.equalizeHist(resized), "usable"
