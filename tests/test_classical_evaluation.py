import pytest

from face_surveillance.classical.recognition import LBPRecognizer


def test_metrics_count_detection_misses_and_wrong_known_identities():
    from face_surveillance.classical.evaluate import score_frame

    truth = [
        {"bbox": [0, 0, 80, 80], "person_id": "A"},
        {"bbox": [200, 0, 80, 80], "person_id": "B"},
        {"bbox": [400, 0, 80, 80], "person_id": None},
    ]
    tracks = [
        {"bbox": [0, 0, 80, 80], "person_id": "B", "state": "confirmed", "missing": False},
        {"bbox": [400, 0, 80, 80], "person_id": "A", "state": "confirmed", "missing": False},
    ]
    result = score_frame(tracks, truth, {"A", "B"})
    assert result["faces"] == 3
    assert result["detected"] == 2
    assert result["known"] == 2
    assert result["known_correct"] == 0
    assert result["known_wrong"] == 1
    assert result["known_detection_missed"] == 1
    assert result["unknown_false_accept"] == 1


def test_calibration_respects_unknown_constraint_and_rejects_test_split():
    from face_surveillance.classical.evaluate import select_threshold

    rows = [
        {"score": 0.1, "margin": 0.2, "correct": True, "known": True},
        {"score": 0.2, "margin": 0.2, "correct": True, "known": True},
        {"score": 0.15, "margin": 0.2, "correct": False, "known": False},
    ]
    result = select_threshold(
        rows, split="validation", max_unknown_fpir=0, margin=0.02, recognizer=LBPRecognizer()
    )
    assert result["threshold"] == pytest.approx(0.1)
    assert result["known_correct"] == 1
    assert result["unknown_false_accept"] == 0
    with pytest.raises(ValueError, match="validation"):
        select_threshold(
            rows, split="test", max_unknown_fpir=0, margin=0.02, recognizer=LBPRecognizer()
        )


def test_event_metrics_match_encounters_once_and_count_duplicates():
    from face_surveillance.classical.evaluate import score_events

    encounters = [{"person_id": "A", "start": 0, "end": 5}]
    events = [
        {"person_id": "A", "time": 1},
        {"person_id": "A", "time": 2},
        {"person_id": "B", "time": 3},
    ]
    result = score_events(events, encounters)
    assert result["correct"] == 1
    assert result["duplicates"] == 1
    assert result["false"] == 1
    assert result["missed"] == 0
