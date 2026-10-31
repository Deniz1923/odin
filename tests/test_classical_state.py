import sqlite3

import numpy as np
import pytest


def template(index=0):
    value = np.zeros((64, 256), dtype=np.float32)
    value[:, index] = 1
    return value.ravel()


def test_gallery_encrypts_templates_and_deletion_removes_events(tmp_path):
    from face_surveillance.classical.storage import Store

    store = Store(tmp_path / "private")
    with pytest.raises(ValueError):
        store.enroll("A", [template()], authorized=False)
    person = store.enroll("A", [template()], authorized=True, alert_enabled=True)
    np.testing.assert_array_equal(store.gallery()[person["id"]][0], template())
    store.add_event(person["id"], "webcam:0", "session", 1, 0.1, 3, {}, now=1000)
    assert len(store.events(now=1001)) == 1
    db = sqlite3.connect(store.path)
    payload = db.execute("select payload from templates").fetchone()[0]
    assert template().tobytes() not in payload
    db.close()
    store.delete(person["id"])
    assert store.gallery() == {}
    assert store.events(now=1001) == []
    assert Store(tmp_path / "private").people() == []


def test_retention_and_cross_track_cooldown(tmp_path):
    from face_surveillance.classical.storage import Store

    store = Store(tmp_path / "private", retention_days=1)
    p = store.enroll("A", [template()], authorized=True, alert_enabled=True)["id"]
    assert store.add_event(p, "camera", "s", 1, 0.1, 3, {}, now=1000)
    assert store.add_event(p, "camera", "s", 2, 0.1, 3, {}, now=1001) is None
    assert store.events(now=100000) == []


def test_temporal_confirmation_unknown_conflict_and_expiry():
    from face_surveillance.classical.pipeline import Pipeline, Policy

    pipeline = Pipeline({"A": [template()], "B": [template(1)]}, {"A"}, Policy())

    def observation(value):
        return [{"bbox": [0, 0, 80, 80], "descriptor": value, "reason": "usable"}]

    result = pipeline.update(observation(template()), 0)
    assert result["tracks"][0]["state"] == "candidate"
    pipeline.update(observation(template()), 0.3)
    result = pipeline.update(observation(template()), 0.6)
    assert result["tracks"][0]["state"] == "confirmed"
    assert len(result["events"]) == 1
    assert pipeline.update(observation(template()), 0.9)["events"] == []
    result = pipeline.update(observation(template(1)), 1.2)
    assert result["tracks"][0]["state"] == "candidate"
    assert result["tracks"][0]["person_id"] == "B"
    assert result["events"] == []
    result = pipeline.update([{"bbox": [0, 0, 80, 80], "descriptor": None, "reason": "blur"}], 2.3)
    assert result["tracks"][0]["state"] == "insufficient"
    unknown = Pipeline({"A": [template()]}, {"A"}, Policy())
    assert unknown.update(observation(template(2)), 0)["tracks"][0]["state"] == "unknown"


def test_ambiguous_association_does_not_transfer_confirmed_identity():
    from face_surveillance.classical.pipeline import Pipeline, Policy

    pipeline = Pipeline({"A": [template()]}, {"A"}, Policy(min_observations=1, min_span=0))
    one = [{"bbox": [0, 0, 80, 80], "descriptor": template(), "reason": "usable"}]
    assert pipeline.update(one, 0)["events"]
    two = one + [{"bbox": [10, 0, 80, 80], "descriptor": template(), "reason": "usable"}]
    result = pipeline.update(two, 0.3)
    assert result["events"] == []
    assert all(t["state"] == "insufficient" for t in result["tracks"])


def test_new_identity_needs_new_confirmation_and_can_emit_its_own_event():
    from face_surveillance.classical.pipeline import Pipeline, Policy

    pipeline = Pipeline(
        {"A": [template()], "B": [template(1)]}, {"A", "B"}, Policy(min_observations=1, min_span=0)
    )

    def observation(value):
        return [{"bbox": [0, 0, 80, 80], "descriptor": value, "reason": "usable"}]

    assert pipeline.update(observation(template()), 0)["events"][0]["person_id"] == "A"
    assert pipeline.update(observation(template(1)), 0.3)["events"][0]["person_id"] == "B"


def test_impossible_temporal_policy_is_rejected():
    from face_surveillance.classical.pipeline import Policy

    with pytest.raises(ValueError):
        Policy(min_span=2, evidence_window=1)


def test_contradiction_between_sample_times_clears_confirmation():
    from face_surveillance.classical.pipeline import Pipeline, Policy

    pipeline = Pipeline({"A": [template()], "B": [template(1)]}, {"A"}, Policy())

    def obs(value):
        return [{"bbox": [0, 0, 80, 80], "descriptor": value, "reason": "usable"}]

    for timestamp in (0, 0.3, 0.6):
        pipeline.update(obs(template()), timestamp)
    result = pipeline.update(obs(template(1)), 0.7)
    assert result["tracks"][0]["state"] != "confirmed"
    assert result["tracks"][0]["person_id"] != "A"


def test_merge_resets_every_track_involved_in_ambiguous_assignment():
    from face_surveillance.classical.pipeline import Pipeline, Policy

    pipeline = Pipeline({"A": [template()]}, {"A"}, Policy(min_observations=1, min_span=0))
    pipeline.update(
        [
            {"bbox": [0, 0, 80, 80], "descriptor": template(), "reason": "usable"},
            {"bbox": [90, 0, 80, 80], "descriptor": template(), "reason": "usable"},
        ],
        0,
    )
    result = pipeline.update(
        [{"bbox": [40, 0, 80, 80], "descriptor": template(), "reason": "usable"}], 0.3
    )
    assert all(t["state"] == "insufficient" for t in result["tracks"])
