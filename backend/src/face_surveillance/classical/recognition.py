"""LBP-specific preparation, descriptor validation and observation matching."""

import numpy as np

from .detection import prepare_face
from .lbp import SPACE, describe, match


class LBPRecognizer:
    name = "LBP histograms"
    space = SPACE
    score_kind = "chi_square_distance"
    score_range = (0.0, 1.0)
    higher_is_better = False
    default_threshold = 0.35
    default_margin = 0.02

    def extract(self, frame: np.ndarray, detection: dict) -> tuple[np.ndarray | None, str]:
        gray, reason = prepare_face(frame, detection["bbox"])
        return (describe(gray) if gray is not None else None), reason

    def validate_descriptor(self, value: np.ndarray):
        if (
            value.shape != (16384,)
            or not np.isfinite(value).all()
            or (value < 0).any()
            or value.sum() <= 0
        ):
            raise ValueError("Invalid LBP descriptor")

    def validate_settings(self, threshold: float, margin: float):
        if not 0 <= threshold <= 1 or not 0 <= margin <= 1:
            raise ValueError("Distance threshold and margin must lie between zero and one")

    def match(self, descriptor, gallery, threshold, margin) -> dict:
        result = match(descriptor, gallery, threshold, margin)
        result["score"] = result.pop("distance")
        return result
