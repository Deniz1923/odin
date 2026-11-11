import asyncio
import io
import json
import sqlite3
import time

import cv2
import numpy as np
import pytest
from PIL import Image

from face_surveillance.api.main import create_app
from face_surveillance.classical.evaluate import run, select_threshold
from face_surveillance.classical.pipeline import Pipeline, Policy
from face_surveillance.classical.recognition import LBPRecognizer
from face_surveillance.classical.service import Service
from face_surveillance.classical.storage import Store


class LandmarkDetector:
    name = "Test landmark detector"
    model_hash = "test-detector"

    def detect(self, frame):
        return [dict(bbox=[0, 0, 9, 7], landmarks=[[1, 2]] * 5)]


class SimilarityRecognizer:
    """Test double with non-LBP preparation, signed vectors and higher-is-better scores."""

    name = "Test similarity"
    space = "test-signed-2d-v1"
    score_kind = "cosine_similarity"
    score_range = (-1.0, 1.0)
    higher_is_better = True
    default_threshold = 0.8
    default_margin = 0.1

    def __init__(self):
        self.prepared = []

    def extract(self, frame, detection):
        assert detection["landmarks"] == [[1, 2]] * 5
        x, y, w, h = detection["bbox"]
        rgb_chw = frame[y : y + h, x : x + w, ::-1].transpose(2, 0, 1)
        self.prepared.append(rgb_chw.copy())
        if not rgb_chw[1].any():
            return None, "test_quality_rejection"
        return np.array([1.0 if rgb_chw[0].mean() > 127 else -1.0, 0.0]), "usable"

    def validate_descriptor(self, value):
        if (
            value.shape != (2,)
            or not np.isfinite(value).all()
            or not np.isclose(np.linalg.norm(value), 1)
        ):
            raise ValueError("Invalid test embedding")

    def validate_settings(self, threshold, margin):
        if not -1 <= threshold <= 1 or not 0 <= margin <= 2:
            raise ValueError("Invalid test matching settings")

    def match(self, descriptor, gallery, threshold, margin):
        self.validate_settings(threshold, margin)
        scores = sorted(
            (
                (max(float(descriptor @ sample) for sample in samples), person)
                for person, samples in gallery.items()
            ),
            reverse=True,
        )
        if not scores:
            return dict(state="insufficient", person_id=None, score=None, margin=None)
        score, person = scores[0]
        gap = score - scores[1][0] if len(scores) > 1 else None
        state = "candidate"
        if score < threshold:
            state = "unknown"
        elif gap is not None and (gap <= 1e-12 or gap < margin):
            state = "insufficient"
        return dict(
            state=state,
            person_id=person if state == "candidate" else None,
            score=score,
            margin=gap,
        )


def reference_png():
    output = io.BytesIO()
    Image.new("RGB", (80, 80), (0, 30, 60)).save(output, format="PNG")
    return output.getvalue()


def test_pipeline_passes_original_crop_and_landmarks_to_recognizer():
    recognizer = SimilarityRecognizer()
    pipeline = Pipeline(
        {"A": [np.array([-1.0, 0.0])]},
        {"A"},
        Policy(threshold=0.8),
        LandmarkDetector(),
        recognizer=recognizer,
    )
    frame = np.full((12, 16, 3), [60, 30, 0], dtype=np.uint8)
    for stamp in (0, 0.3, 0.6):
        result = pipeline.process(frame, stamp)
    assert recognizer.prepared[0].shape == (3, 7, 9)
    assert recognizer.prepared[0][:, 0, 0].tolist() == [0, 30, 60]
    assert result["tracks"][0]["state"] == "confirmed"
    assert result["events"][0]["score"] == 1
    assert result["events"][0]["score_kind"] == "cosine_similarity"
    frame[:, :, 2] = 255
    assert pipeline.process(frame, 0.7)["tracks"][0]["state"] == "unknown"
    frame[:] = 0
    result = pipeline.process(frame, 2)
    assert result["tracks"][0]["state"] == "insufficient"
    assert result["tracks"][0]["reason"] == "test_quality_rejection"


def test_gallery_uses_recognizer_validation_and_rejects_another_space(tmp_path):
    recognizer = SimilarityRecognizer()
    store = Store(tmp_path, recognizer=recognizer)
    person = store.enroll("A", [np.array([-1.0, 0.0])], authorized=True)
    np.testing.assert_array_equal(store.gallery()[person["id"]][0], [-1, 0])
    for invalid in (np.zeros(2), np.array([np.nan, 0]), np.ones(3)):
        with pytest.raises(ValueError):
            store.enroll("bad", [invalid], authorized=True)
    with pytest.raises(ValueError, match="descriptor space"):
        Store(tmp_path, recognizer=LBPRecognizer()).gallery()
    assert len(store.people()) == 1


def test_legacy_event_migration_preserves_values_acknowledgement_and_key(tmp_path):
    recognizer = LBPRecognizer()
    store = Store(tmp_path, recognizer=recognizer)
    descriptor = np.zeros((64, 256), dtype=np.float32)
    descriptor[:, 0] = 1
    person = store.enroll("A", [descriptor.ravel()], authorized=True, alert_enabled=True)
    event_id = store.add_event(
        person["id"],
        "camera",
        "session",
        1,
        0.123,
        3,
        {"legacy": True},
        score_kind="chi_square_distance",
    )
    store.acknowledge(event_id)
    key = (tmp_path / "templates.key").read_bytes()
    with sqlite3.connect(store.path) as db:
        db.execute("ALTER TABLE events DROP COLUMN score_kind")
        db.execute("ALTER TABLE events RENAME COLUMN score TO distance")
    for _ in range(2):
        reopened = Store(tmp_path, recognizer=recognizer)
        event = reopened.events()[0]
        assert event["id"] == event_id and event["acknowledged"] == 1
        assert event["score"] == 0.123 and event["score_kind"] == "chi_square_distance"
        assert json.loads(event["policy"]) == {"legacy": True}
        assert (tmp_path / "templates.key").read_bytes() == key
        np.testing.assert_array_equal(reopened.gallery()[person["id"]][0], descriptor.ravel())


def test_api_enrollment_and_policy_use_selected_recognizer(tmp_path):
    from httpx2 import ASGITransport, AsyncClient

    async def exercise():
        recognizer = SimilarityRecognizer()
        app = create_app(tmp_path, detector=LandmarkDetector(), recognizer=recognizer)
        async with app.router.lifespan_context(app):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://testserver"
            ) as client:
                token = (await client.get("/api/session")).json()["token"]
                headers = {"Authorization": f"Bearer {token}"}
                response = await client.post(
                    "/api/people",
                    headers=headers,
                    data={"name": "A", "authorized": "true"},
                    files={"images": ("reference.png", reference_png(), "image/png")},
                )
                assert response.status_code == 201, response.text
                status = (await client.get("/api/status")).json()
                assert status["recognizer"] == recognizer.name
                assert status["descriptor_space"] == recognizer.space
                assert status["score_kind"] == "cosine_similarity"
                assert status["score_range"] == [-1, 1] and status["higher_is_better"]
                assert status["policy"]["threshold"] == 0.8
                assert recognizer.prepared[0].shape == (3, 7, 9)
                assert (
                    await client.put("/api/policy", headers=headers, json={"threshold": -0.2})
                ).status_code == 200
                assert (
                    await client.put("/api/policy", headers=headers, json={"threshold": 1.2})
                ).status_code == 422
                person_id = response.json()["id"]
        restarted = Service(tmp_path, detector=LandmarkDetector(), recognizer=recognizer)
        try:
            assert restarted.policy.threshold == -0.2
            assert person_id in restarted.store.gallery()
        finally:
            restarted.close()

    asyncio.run(exercise())


def test_calibration_respects_higher_is_better_scores():
    result = select_threshold(
        [
            dict(score=score, margin=0.2, correct=known, known=known)
            for score, known in ((0.9, True), (0.8, True), (0.85, False))
        ],
        recognizer=SimilarityRecognizer(),
        split="validation",
        max_unknown_fpir=0,
    )
    assert result["threshold"] == 0.9
    assert result["known_correct"] == 1 and result["unknown_false_accept"] == 0
    assert result["higher_is_better"] and result["score_kind"] == "cosine_similarity"


def test_evaluation_uses_same_extraction_and_matching_for_enrollment_and_video(tmp_path):
    (tmp_path / "reference.png").write_bytes(reference_png())
    video = tmp_path / "study.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (80, 80))
    assert writer.isOpened()
    for i in range(10):
        writer.write(np.full((80, 80, 3), [60, 30, 0 if i < 7 else 255], dtype=np.uint8))
    writer.release()
    manifest = dict(
        schema_version=1,
        split="validation",
        enrollment=[
            dict(person_id="A", authorized=True, alert_enabled=True, images=["reference.png"])
        ],
        videos=[
            dict(
                path="study.avi",
                annotations=[
                    dict(frame=i, faces=[dict(person_id="A" if i < 7 else None, bbox=[0, 0, 9, 7])])
                    for i in range(10)
                ],
                encounters=[dict(person_id="A", start=0, end=0.6)],
            )
        ],
    )
    manifest_path = tmp_path / "study.json"
    manifest_path.write_text(json.dumps(manifest))
    recognizer = SimilarityRecognizer()
    report = run(
        manifest_path,
        tmp_path / "results",
        calibrate=True,
        detector=LandmarkDetector(),
        recognizer=recognizer,
    )
    assert report["model"] == recognizer.name and report["descriptor_space"] == recognizer.space
    assert report["metrics"]["unknown_rejected"] == 3
    assert report["videos"][0]["event_metrics"]["correct"] == 1
    assert report["calibration"]["threshold"] == 1
    assert report["calibration"]["unknown_false_accept"] == 0
    assert len(recognizer.prepared) == 21  # enrollment, processing, calibration
    predictions = [
        json.loads(line)
        for line in (tmp_path / "results/predictions.jsonl").read_text().splitlines()
    ]
    assert predictions[6]["tracks"][0]["score_kind"] == "cosine_similarity"
    service = Service(tmp_path / "private", detector=LandmarkDetector(), recognizer=recognizer)
    try:
        person = service.enroll("A", [reference_png()], True, True)
        service.start(str(video), "test-replay")
        deadline = time.monotonic() + 5
        while service.status()["state"] not in ("ended", "failed"):
            assert time.monotonic() < deadline
            time.sleep(0.02)
        assert service.status()["state"] == "ended"
        events = service.store.events()
        assert len(events) == 1 and events[0]["score"] == 1
        assert events[0]["score_kind"] == "cosine_similarity"
        service.delete(person["id"])
        assert service.store.gallery() == {} and service.store.events() == []
    finally:
        service.close()


def test_legacy_policy_loads_as_lbp_and_cannot_be_reused_for_another_recognizer(tmp_path):
    (tmp_path / "policy.json").write_text(Policy(threshold=0.123).model_dump_json())
    service = Service(tmp_path)
    try:
        assert service.policy.threshold == 0.123
        service.configure(service.policy)
    finally:
        service.close()
    with pytest.raises(ValueError, match="decision settings descriptor space"):
        Service(tmp_path, detector=LandmarkDetector(), recognizer=SimilarityRecognizer())


@pytest.mark.parametrize("threshold,margin", [(-0.1, 0.02), (1.1, 0.02), (0.35, 1.1)])
def test_shared_policy_still_enforces_lbp_matching_limits(threshold, margin):
    with pytest.raises(ValueError):
        Pipeline({}, set(), Policy(threshold=threshold, margin=margin), recognizer=LBPRecognizer())
