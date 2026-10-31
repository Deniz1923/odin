"""Our radius-one LBP, spatial histogram descriptor and chi-square matcher.

Bits 0..7 visit NW, N, NE, E, SE, S, SW, W. Equal neighbors set the bit.
The one-pixel border is excluded. This is basic 256-bin LBP, not uniform LBP.
"""

import numpy as np

SPACE = "lbp-r1-p8-clockwise-nw-ge-grid8-equalized128-v1"


def lbp_codes(gray: np.ndarray) -> np.ndarray:
    if gray.ndim != 2 or gray.dtype != np.uint8 or min(gray.shape) < 3:
        raise ValueError("LBP requires a uint8 grayscale image at least 3x3")
    center = gray[1:-1, 1:-1]
    neighbors = (
        gray[:-2, :-2],
        gray[:-2, 1:-1],
        gray[:-2, 2:],
        gray[1:-1, 2:],
        gray[2:, 2:],
        gray[2:, 1:-1],
        gray[2:, :-2],
        gray[1:-1, :-2],
    )
    codes = np.zeros_like(center)
    for bit, neighbor in enumerate(neighbors):
        codes |= (neighbor >= center).astype(np.uint8) << bit
    return codes


def describe(gray: np.ndarray, grid: int = 8) -> np.ndarray:
    codes = lbp_codes(gray)
    if grid < 1 or min(codes.shape) < grid:
        raise ValueError("Image too small for histogram grid")
    cells = []
    for row in np.array_split(codes, grid, axis=0):
        for cell in np.array_split(row, grid, axis=1):
            counts = np.bincount(cell.ravel(), minlength=256).astype(np.float32)
            cells.append(counts / counts.sum())
    return np.concatenate(cells)


def distance(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a), np.asarray(b)
    if (
        a.ndim != 1
        or a.shape != b.shape
        or not a.size
        or not np.isfinite(a).all()
        or not np.isfinite(b).all()
        or (a < 0).any()
        or (b < 0).any()
        or a.sum() <= 0
        or b.sum() <= 0
    ):
        raise ValueError("Invalid histogram descriptors")
    # Mean cell distance, approximately [0,1] for our normalized descriptors.
    return float(0.5 * np.sum((a - b) ** 2 / (a + b + 1e-12)) / a.sum())


def match(query: np.ndarray, templates: dict, threshold: float, margin: float) -> dict:
    if not 0 <= threshold <= 1 or not 0 <= margin <= 1:
        raise ValueError("Distance threshold and margin must lie between zero and one")
    scores = sorted(
        (min(distance(query, sample) for sample in samples), person)
        for person, samples in templates.items()
        if len(samples)
    )
    if not scores:
        return dict(state="insufficient", person_id=None, distance=None, margin=None)
    best, person = scores[0]
    gap = scores[1][0] - best if len(scores) > 1 else None
    state = "candidate"
    if best > threshold:
        state = "unknown"
    elif gap is not None and (gap <= 1e-12 or gap < margin):
        state = "insufficient"
    return dict(
        state=state, person_id=person if state == "candidate" else None, distance=best, margin=gap
    )
