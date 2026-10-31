"""Explicitly acquire the hash-pinned OpenCV Haar model; never runs at API startup."""

import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

root = Path(__file__).resolve().parents[1]
spec = json.loads((root / "configs/haar_model.json").read_text())
with urlopen(spec["source"], timeout=30) as response:
    content = response.read(2_000_000)
if hashlib.sha256(content).hexdigest() != spec["sha256"]:
    raise ValueError("Haar download SHA-256 mismatch")
target = root / spec["path"]
target.parent.mkdir(parents=True, exist_ok=True)
target.write_bytes(content)
print(target)
