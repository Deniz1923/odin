"""ArcFace-family recognition: landmark alignment, L2-normalized embeddings, cosine matching."""

import warnings

import numpy as np

# InsightFace's norm_crop calls a scikit-image API deprecated for 2.2; behaviour is unchanged.
warnings.filterwarnings("ignore", message="`estimate` is deprecated", category=FutureWarning)

MIN_FACE_PIXELS = 40
EMBEDDING_SIZE = 512


class ArcFaceRecognizer:
    """Recognizer contract implementation for the buffalo_l w600k_r50 ArcFace encoder.

    `embed` maps an aligned 112x112 BGR crop to a raw 512-value embedding. Scores are cosine
    similarities between unit vectors: higher is closer. Defaults are uncalibrated.
    """

    name = "ArcFace R50 (w600k)"
    score_kind = "cosine_similarity"
    score_range = (-1.0, 1.0)
    higher_is_better = True
    default_threshold = 0.40
    default_margin = 0.05

    def __init__(self, embed, model_hash: str, *, device: str):
        self.embed = embed
        self.model_hash = model_hash
        self.device = device
        self.name = f"{ArcFaceRecognizer.name} · {device.upper()}"
        # Space covers weights, landmark source/alignment, crop size and normalization.
        self.space = f"arcface-w600k_r50-{model_hash[:16]}-scrfd5pt-normcrop112-l2-v1"

    def extract(self, frame: np.ndarray, detection: dict) -> tuple[np.ndarray | None, str]:
        from insightface.utils.face_align import norm_crop

        landmarks = detection.get("landmarks")
        if landmarks is None:
            return None, "landmarks_required"
        points = np.asarray(landmarks, dtype=np.float32)
        if points.shape != (5, 2) or not np.isfinite(points).all():
            return None, "invalid_landmarks"
        if min(detection["bbox"][2:]) < MIN_FACE_PIXELS:
            return None, "face_too_small"
        crop = norm_crop(frame, points, image_size=112)
        value = np.asarray(self.embed(crop), dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(value))
        if value.size != EMBEDDING_SIZE or not np.isfinite(value).all() or norm < 1e-12:
            return None, "invalid_embedding"
        return value / norm, "usable"

    def validate_descriptor(self, value: np.ndarray):
        if (
            value.shape != (EMBEDDING_SIZE,)
            or not np.isfinite(value).all()
            or abs(float(np.linalg.norm(value)) - 1) > 1e-3
        ):
            raise ValueError("Invalid ArcFace embedding")

    def validate_settings(self, threshold: float, margin: float):
        if not -1 <= threshold <= 1 or not 0 <= margin <= 2:
            raise ValueError("Similarity threshold must lie in [-1, 1] and margin in [0, 2]")

    def match(self, descriptor, gallery, threshold, margin) -> dict:
        self.validate_settings(threshold, margin)
        # Best similarity per person, highest first; ties are ordered by person ID.
        scores = sorted(
            (
                (float(np.max(np.stack(samples) @ descriptor)), person)
                for person, samples in gallery.items()
                if len(samples)
            ),
            key=lambda item: (-item[0], item[1]),
        )
        if not scores:
            return dict(state="insufficient", person_id=None, score=None, margin=None)
        best, person = scores[0]
        gap = best - scores[1][0] if len(scores) > 1 else None
        state = "candidate"
        if best < threshold:
            state = "unknown"
        elif gap is not None and (gap <= 1e-12 or gap < margin):
            state = "insufficient"
        return dict(
            state=state,
            person_id=person if state == "candidate" else None,
            score=best,
            margin=gap,
        )


def load_arcface(session, path, model_hash: str, *, device: str) -> ArcFaceRecognizer:
    from insightface.model_zoo.arcface_onnx import ArcFaceONNX

    model = ArcFaceONNX(str(path), session=session)
    return ArcFaceRecognizer(model.get_feat, model_hash, device=device)
