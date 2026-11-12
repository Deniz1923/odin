"""Generate a public still-image video fixture and exercise the real classical pipeline.

This is a plumbing regression, not independent identity/robustness evaluation.
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from skimage import data

from face_surveillance.classical.evaluate import run
from face_surveillance.classical.service import Service
from face_surveillance.methods import METHODS, build
from face_surveillance.neural.runtime import DEVICES

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--method", choices=METHODS, default="lbp")
parser.add_argument("--device", choices=DEVICES, default="auto")
args = parser.parse_args()
detector, recognizer = build(args.method, args.device)
args.output.mkdir(parents=True, exist_ok=False, mode=0o700)
rgb = data.astronaut()
Image.fromarray(rgb).save(args.output / "reference.png")
video = args.output / "synthetic.avi"
writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (512, 512))
if not writer.isOpened():
    raise ValueError("MJPG video writer unavailable")
for _ in range(20):
    writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
for _ in range(5):
    writer.write(np.zeros((512, 512, 3), dtype=np.uint8))
writer.release()
manifest = dict(
    schema_version=1,
    split="smoke",
    enrollment=[
        dict(person_id="fixture", images=["reference.png"], authorized=True, alert_enabled=True)
    ],
    videos=[
        dict(
            path="synthetic.avi",
            condition="repeated-public-still",
            annotations=[
                dict(
                    frame=i,
                    faces=[dict(person_id="fixture", bbox=[177, 67, 93, 93])] if i < 20 else [],
                )
                for i in range(25)
            ],
            encounters=[dict(person_id="fixture", start=0, end=1.9)],
        )
    ],
)
path = args.output / "manifest.json"
path.write_text(json.dumps(manifest, indent=2) + "\n")
report = run(path, args.output / "evaluation", detector=detector, recognizer=recognizer)
assert report["videos"][0]["frames"] == 25
assert report["videos"][0]["event_metrics"]["correct"] == 1
assert report["videos"][0]["event_metrics"]["duplicates"] == 0
service = Service(args.output / "private-service", detector=detector, recognizer=recognizer)
try:
    service.enroll("Public fixture", [(args.output / "reference.png").read_bytes()], True, True)
    service.start(str(video), "smoke:public-still")
    deadline = time.monotonic() + 15
    while service.status()["state"] not in ("ended", "failed") and time.monotonic() < deadline:
        time.sleep(0.05)
    assert service.status()["state"] == "ended", service.status()
    assert len(service.store.events()) == 1
    person = service.store.people()[0]
    service.delete(person["id"])
    assert service.store.people() == [] and service.store.events() == []
    assert service.preview() is None
finally:
    service.close()
print(
    json.dumps(
        dict(
            status="passed",
            detector=detector.name,
            recognizer=recognizer.name,
            offline_frames=25,
            persisted_events=1,
            deletion_checked=True,
            claim="Repeated still fixture; NOT recognition accuracy",
        ),
        indent=2,
    )
)
