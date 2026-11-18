"""SCRFD-10GF face detection with five landmarks, adapted to the shared detector contract."""

import numpy as np


class SCRFDDetector:
    """Wraps an InsightFace SCRFD model; returns boxes and landmarks in original-frame pixels."""

    def __init__(self, model, model_hash: str, *, device: str, threshold: float = 0.5):
        if not 0 < threshold < 1:
            raise ValueError("SCRFD score threshold must lie between zero and one")
        self.model = model
        self.model_hash = model_hash
        self.threshold = threshold
        self.device = device
        self.name = f"SCRFD-10GF · {device.upper()}"

    def detect(self, bgr: np.ndarray) -> list[dict]:
        boxes, landmarks = self.model.detect(bgr, det_thresh=self.threshold)
        height, width = bgr.shape[:2]
        detections = []
        for index in np.argsort(-boxes[:, 4], kind="stable") if len(boxes) else []:
            x1, y1, x2, y2, score = (float(value) for value in boxes[index])
            if not np.isfinite([x1, y1, x2, y2, score]).all():
                continue
            left, top = max(0, round(x1)), max(0, round(y1))
            right, bottom = min(width, round(x2)), min(height, round(y2))
            if right <= left or bottom <= top:
                continue
            detection = dict(bbox=[left, top, right - left, bottom - top], score=score)
            if landmarks is not None:
                detection["landmarks"] = landmarks[index].astype(float).tolist()
            detections.append(detection)
        return detections


def load_scrfd(session, path, model_hash: str, *, device: str, threshold: float = 0.5):
    from insightface.model_zoo.scrfd import SCRFD

    model = SCRFD(str(path), session=session)
    # The pinned session fixes SCRFD's symbolic spatial dimensions to 640x640.
    model.prepare(ctx_id=0, input_size=(640, 640), det_thresh=threshold)
    return SCRFDDetector(model, model_hash, device=device, threshold=threshold)
