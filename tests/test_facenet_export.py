import hashlib
import json

import numpy as np
import pytest

from face_surveillance.vision.export_facenet import compare_embeddings, export_and_verify


def test_identical_embeddings_pass_with_all_rows_checked():
    values = np.eye(512, dtype=np.float32)[:3]
    result = compare_embeddings(values, values.copy())
    assert result["cases"] == 3
    assert result["max_component_difference"] == 0
    assert result["max_pairwise_cosine_difference"] == 0


@pytest.mark.parametrize("bad", [0, np.nan, np.inf])
def test_invalid_middle_output_rejected(bad):
    reference = np.eye(512, dtype=np.float32)[:3]
    candidate = reference.copy()
    candidate[1] = bad
    with pytest.raises(ValueError):
        compare_embeddings(reference, candidate)


def test_conversion_cannot_hide_output_scaling_by_normalizing_first():
    reference = np.eye(512, dtype=np.float32)[:3]
    with pytest.raises(ValueError, match="unit"):
        compare_embeddings(reference, reference * 2)


def test_wrong_output_dimension_rejected():
    with pytest.raises(ValueError, match="shape"):
        compare_embeddings(np.ones((3, 512)), np.ones((3, 128)))


def test_wrong_embedding_direction_rejected():
    reference = np.eye(512, dtype=np.float32)[:3]
    with pytest.raises(ValueError, match="component"):
        compare_embeddings(reference, reference[::-1].copy())


def test_small_component_changes_can_still_fail_pairwise_score_gate():
    reference = np.eye(512, dtype=np.float32)[:2]
    candidate = reference.copy()
    candidate[0, 1] = 7.5e-6
    candidate[1, 0] = 7.5e-6
    with pytest.raises(ValueError, match="cosine"):
        compare_embeddings(reference, candidate)


def source_manifest(tmp_path):
    asset = tmp_path / "asset"
    asset.write_bytes(b"source")
    specification = {"path": str(asset), "sha256": hashlib.sha256(b"source").hexdigest()}
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "assets": {
                    name: specification
                    for name in ("facenet", "facenet_wheel", "detector", "arcface")
                },
                "overlay_path": str(tmp_path),
                "overlay_files": {},
            }
        )
    )
    return path


def test_failed_export_leaves_failure_report_without_runtime_manifest(tmp_path, monkeypatch):
    from face_surveillance.vision import export_facenet

    def fail(*args):
        raise RuntimeError("parity failed")

    monkeypatch.setattr(export_facenet, "_export", fail)
    output = tmp_path / "output"
    with pytest.raises(RuntimeError, match="parity failed"):
        export_and_verify(source_manifest(tmp_path), output)
    assert json.loads((output / "export_report.json").read_text())["status"] == "failed"
    assert not (output / "runtime_manifest.json").exists()


def test_runtime_manifest_links_finalized_successful_report(tmp_path, monkeypatch):
    from face_surveillance.vision import export_facenet

    monkeypatch.setattr(export_facenet, "_export", lambda *args: {"schema_version": 1})
    output = tmp_path / "output"
    export_and_verify(source_manifest(tmp_path), output)
    result = json.loads((output / "runtime_manifest.json").read_text())
    report = output / "export_report.json"
    assert result["validation"]["report_sha256"] == hashlib.sha256(report.read_bytes()).hexdigest()
    assert json.loads(report.read_text())["status"] == "passed"
