"""Named detector/recognizer compositions shared by the API, evaluation runner and scripts.

| Method      | Detector      | Recognizer                         |
|-------------|---------------|------------------------------------|
| `lbp`       | Haar          | our LBP histograms (default)       |
| `scrfd-lbp` | SCRFD-10GF    | our LBP histograms (matched boxes) |
| `arcface`   | SCRFD-10GF    | ArcFace R50, landmark-aligned      |

Each method keeps its own default data directory so galleries and settings never mix.
"""

from pathlib import Path

from face_surveillance.classical.detection import ROOT

METHODS = ("lbp", "scrfd-lbp", "arcface")


def default_data_dir(method: str) -> Path:
    # The original classical gallery location is kept for the default method.
    return ROOT / "data" / ("classical" if method == "lbp" else method)


def build(method: str = "lbp", device: str = "auto", manifest: Path | None = None):
    """Return `(detector, recognizer)` for a named method; neural models load hash-checked."""
    if method not in METHODS:
        raise ValueError(f"Method must be one of {', '.join(METHODS)}")
    if method == "lbp":
        from face_surveillance.classical.detection import HaarDetector
        from face_surveillance.classical.recognition import LBPRecognizer

        return HaarDetector(), LBPRecognizer()

    from face_surveillance.classical.recognition import LBPRecognizer
    from face_surveillance.neural.detection import load_scrfd
    from face_surveillance.neural.recognition import load_arcface
    from face_surveillance.neural.runtime import create_sessions, load_asset

    names = ("detector", "arcface") if method == "arcface" else ("detector",)
    assets = {name: load_asset(name, manifest) for name in names}
    sessions, used = create_sessions(
        {name: (path, overrides) for name, (path, _, overrides) in assets.items()}, device
    )
    path, digest, _ = assets["detector"]
    detector = load_scrfd(sessions["detector"], path, digest, device=used)
    if method == "scrfd-lbp":
        return detector, LBPRecognizer()
    path, digest, _ = assets["arcface"]
    return detector, load_arcface(sessions["arcface"], path, digest, device=used)
