import time

import cv2
import numpy as np
import pytest

from face_surveillance.classical.detection import ROOT
from face_surveillance.neural.detection import SCRFDDetector
from face_surveillance.neural.recognition import ArcFaceRecognizer

LANDMARKS = [[30.0, 40.0], [70.0, 40.0], [50.0, 60.0], [35.0, 80.0], [65.0, 80.0]]


class FakeSCRFD:
    def __init__(self, boxes, landmarks):
        self.boxes, self.landmarks = np.asarray(boxes, np.float32), landmarks
        self.thresholds = []

    def detect(self, image, det_thresh=None):
        self.thresholds.append(det_thresh)
        return self.boxes, self.landmarks


def unit(index, size=512):
    value = np.zeros(size, np.float32)
    value[index] = 1
    return value


def test_scrfd_adapter_converts_clips_and_orders_detections():
    landmarks = np.arange(20, dtype=np.float32).reshape(2, 5, 2)
    model = FakeSCRFD([[-5, 10, 50, 90, 0.6], [100, 20, 260, 140, 0.9]], landmarks)
    detector = SCRFDDetector(model, "abc", device="cpu", threshold=0.55)
    first, second = detector.detect(np.zeros((120, 200, 3), np.uint8))
    assert model.thresholds == [0.55]
    assert first == dict(
        bbox=[100, 20, 100, 100], score=pytest.approx(0.9), landmarks=landmarks[1].tolist()
    )
    assert second["bbox"] == [0, 10, 50, 80]
    assert detector.name == "SCRFD-10GF · CPU" and detector.model_hash == "abc"
    empty = SCRFDDetector(FakeSCRFD(np.empty((0, 5)), None), "abc", device="cpu")
    assert empty.detect(np.zeros((10, 10, 3), np.uint8)) == []
    outside = SCRFDDetector(FakeSCRFD([[300, 300, 400, 400, 0.9]], None), "abc", device="cpu")
    assert outside.detect(np.zeros((120, 200, 3), np.uint8)) == []


def test_arcface_extract_aligns_with_landmarks_and_normalizes():
    crops = []

    def embed(crop):
        crops.append(crop)
        return unit(0) * 7

    recognizer = ArcFaceRecognizer(embed, "f" * 64, device="cuda")
    frame = np.full((120, 100, 3), 90, np.uint8)
    detection = dict(bbox=[10, 10, 80, 100], landmarks=LANDMARKS)
    value, reason = recognizer.extract(frame, detection)
    assert reason == "usable" and crops[0].shape == (112, 112, 3)
    np.testing.assert_allclose(value, unit(0))
    recognizer.validate_descriptor(value)
    assert recognizer.extract(frame, dict(bbox=[0, 0, 80, 80]))[1] == "landmarks_required"
    assert recognizer.extract(frame, dict(bbox=[0, 0, 30, 80], landmarks=LANDMARKS))[1] == (
        "face_too_small"
    )
    bad = dict(bbox=[0, 0, 80, 80], landmarks=[[np.nan, 0]] * 5)
    assert recognizer.extract(frame, bad)[1] == "invalid_landmarks"
    zero = ArcFaceRecognizer(lambda crop: np.zeros(512), "f" * 64, device="cpu")
    assert zero.extract(frame, detection)[1] == "invalid_embedding"
    assert recognizer.space.startswith("arcface-w600k_r50-" + "f" * 16)
    assert recognizer.name.endswith("CUDA") and recognizer.higher_is_better


def test_arcface_match_is_cosine_higher_is_better_with_rejection_and_margin():
    recognizer = ArcFaceRecognizer(None, "f" * 64, device="cpu")
    query = unit(0)
    near = (unit(0) * 0.9 + unit(1) * np.sqrt(1 - 0.81)).astype(np.float32)
    assert recognizer.match(query, {}, 0.4, 0.05)["state"] == "insufficient"
    result = recognizer.match(query, {"a": [unit(2), near], "b": [unit(1)]}, 0.4, 0.05)
    assert result["state"] == "candidate" and result["person_id"] == "a"
    assert result["score"] == pytest.approx(0.9) and result["margin"] == pytest.approx(0.9)
    assert recognizer.match(query, {"a": [unit(1)]}, 0.4, 0.05)["state"] == "unknown"
    tie = recognizer.match(query, {"a": [near], "b": [near]}, 0.4, 0.05)
    assert tie["state"] == "insufficient" and tie["person_id"] is None
    with pytest.raises(ValueError):
        recognizer.match(query, {"a": [near]}, 1.5, 0.05)
    with pytest.raises(ValueError):
        recognizer.validate_descriptor(unit(0) * 2)
    with pytest.raises(ValueError):
        recognizer.validate_descriptor(unit(0, size=128))


def test_service_keeps_arcface_gallery_separate_from_lbp(tmp_path):
    from face_surveillance.classical.service import Service

    class OneFace:
        name, model_hash = "fake detector", "fake"

        def detect(self, frame):
            return [dict(bbox=[10, 10, 80, 100], landmarks=LANDMARKS, score=0.9)]

    _, png = cv2.imencode(".png", np.full((120, 100, 3), 90, np.uint8))
    recognizer = ArcFaceRecognizer(lambda crop: unit(3), "f" * 64, device="cpu")
    service = Service(tmp_path / "arcface", detector=OneFace(), recognizer=recognizer)
    person = service.enroll("A", [png.tobytes()], True, True)
    status = service.status()
    assert status["policy"]["threshold"] == ArcFaceRecognizer.default_threshold
    assert status["higher_is_better"] and status["score_kind"] == "cosine_similarity"
    np.testing.assert_allclose(service.store.gallery()[person["id"]][0], unit(3))
    service.close()
    with pytest.raises(ValueError, match="descriptor space"):
        Service(tmp_path / "arcface", detector=OneFace())


def test_unknown_method_and_device_are_rejected():
    from face_surveillance.methods import build, default_data_dir
    from face_surveillance.neural.runtime import create_sessions

    with pytest.raises(ValueError, match="Method"):
        build("facenet")
    with pytest.raises(ValueError, match="Device"):
        create_sessions({}, "tpu")
    assert default_data_dir("lbp").name == "classical"
    assert default_data_dir("arcface").name == "arcface"


MODELS_PRESENT = all(
    (ROOT / path).exists()
    for path in ("models/feasibility/det_10g.onnx", "models/feasibility/w600k_r50.onnx")
)
needs_models = pytest.mark.skipif(
    not MODELS_PRESENT, reason="SCRFD/ArcFace ONNX files not acquired"
)


@needs_models
def test_real_scrfd_arcface_on_cpu_detects_and_recognizes_public_fixture():
    from skimage import data

    from face_surveillance.methods import build

    detector, recognizer = build("arcface", "cpu")
    assert detector.name.endswith("CPU") and recognizer.name.endswith("CPU")
    bgr = cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR)
    [face] = detector.detect(bgr)
    assert len(face["landmarks"]) == 5
    assert detector.detect(np.zeros_like(bgr)) == []
    assert len(detector.detect(np.concatenate([bgr, bgr], axis=1))) == 2
    reference, reason = recognizer.extract(bgr, face)
    assert reason == "usable"
    flipped = cv2.flip(bgr, 1)
    query, _ = recognizer.extract(flipped, detector.detect(flipped)[0])
    # Same public photograph, mirrored: a plumbing check, not recognition accuracy.
    assert float(query @ reference) > 0.8


@needs_models
def test_real_arcface_replay_confirms_enrolled_fixture_and_persists_event(tmp_path):
    from skimage import data

    from face_surveillance.classical.service import Service
    from face_surveillance.methods import build

    detector, recognizer = build("arcface", "cpu")
    bgr = cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR)
    video = tmp_path / "fixture.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (512, 512))
    for _ in range(10):
        writer.write(bgr)
    writer.release()
    service = Service(tmp_path / "private", detector=detector, recognizer=recognizer)
    try:
        service.enroll("Fixture", [cv2.imencode(".png", bgr)[1].tobytes()], True, True)
        service.start(str(video), "replay:fixture")
        deadline = time.monotonic() + 30
        while service.status()["state"] not in ("ended", "failed") and time.monotonic() < deadline:
            time.sleep(0.05)
        assert service.status()["state"] == "ended", service.status()
        [event] = service.store.events()
        assert event["score_kind"] == "cosine_similarity" and event["score"] > 0.9
    finally:
        service.close()
