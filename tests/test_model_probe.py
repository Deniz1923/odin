import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from face_surveillance.vision.model_probe import (
    create_run_directory,
    facenet_input,
    normalize_embedding,
    run_probe,
    summarize_profile,
    validate_embedding_series,
    verify_artifact,
)


def test_artifact_hash_rejects_missing_and_changed_weights(tmp_path):
    path = tmp_path / "weights"
    digest = hashlib.sha256(b"original").hexdigest()
    with pytest.raises(FileNotFoundError):
        verify_artifact(path, digest)
    path.write_bytes(b"original")
    assert verify_artifact(path, digest) == digest
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA-256"):
        verify_artifact(path, digest)


@pytest.mark.parametrize(
    "value",
    [np.zeros(512), np.full(512, np.nan), np.full(512, np.inf), np.ones(128)],
)
def test_invalid_embedding_cannot_become_valid_evidence(value):
    with pytest.raises(ValueError):
        normalize_embedding(value)


def test_embedding_is_normalized_without_changing_direction():
    value = np.arange(1, 513, dtype=np.float32)
    result = normalize_embedding(value)
    assert result.shape == (512,)
    assert result.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(result), 1, atol=1e-6)
    np.testing.assert_allclose(result / result[0], value, rtol=1e-6)


@pytest.mark.parametrize(
    "middle",
    [np.zeros(512), np.full(512, np.nan), np.ones(128), np.arange(512)],
)
def test_bad_middle_embedding_fails_even_when_endpoints_agree(middle):
    reference = np.ones(512, dtype=np.float32)
    with pytest.raises(ValueError):
        validate_embedding_series([reference, middle, reference])


def test_repeated_embeddings_report_all_samples_and_maximum_difference():
    reference = np.ones(512, dtype=np.float32)
    middle = reference.copy()
    middle[0] += 1e-5
    final, result = validate_embedding_series([reference, middle, reference])
    assert result["checked_outputs"] == 3
    assert result["repeat_max_absolute_difference"] > 0
    np.testing.assert_allclose(final, normalize_embedding(reference))


def test_facenet_preprocessing_preserves_rgb_and_uses_fixed_standardization():
    rgb = np.zeros((160, 160, 3), dtype=np.uint8)
    rgb[:, :, 0] = 255
    rgb[:, :, 1] = 128
    result = facenet_input(rgb)
    assert result.shape == (1, 3, 160, 160)
    assert result.dtype == np.float32
    np.testing.assert_allclose(result[0, :, 0, 0], (np.array([255, 128, 0]) - 127.5) / 128)


def test_facenet_preprocessing_rejects_wrong_shape():
    with pytest.raises(ValueError):
        facenet_input(np.zeros((112, 112, 3), dtype=np.uint8))


def test_gpu_profile_requires_executed_cuda_nodes_and_rejects_cpu(tmp_path):
    path = tmp_path / "profile.json"
    path.write_text(json.dumps([{"cat": "Session", "args": {}}]))
    with pytest.raises(ValueError, match="CUDA"):
        summarize_profile(path)
    cuda = {"cat": "Node", "args": {"provider": "CUDAExecutionProvider"}}
    path.write_text(json.dumps([cuda, cuda]))
    assert summarize_profile(path) == {"CUDAExecutionProvider": 2}
    path.write_text(
        json.dumps(
            [
                cuda,
                {
                    "cat": "Node",
                    "args": {
                        "provider": "CPUExecutionProvider",
                    },
                },
            ]
        )
    )
    with pytest.raises(ValueError, match="CPU"):
        summarize_profile(path)


def test_run_directory_does_not_overwrite_prior_results(tmp_path):
    path = tmp_path / "run"
    create_run_directory(path)
    (path / "report.json").write_text("original")
    with pytest.raises(FileExistsError):
        create_run_directory(path)
    assert (path / "report.json").read_text() == "original"


def test_failed_probe_records_reason_without_claiming_success(tmp_path, monkeypatch):
    from face_surveillance.vision import model_probe

    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"schema_version": 1, "assets": {}, "overlay_path": ".", "overlay_files": {}})
    )
    output = tmp_path / "run"

    def unavailable(*args):
        raise RuntimeError("CUDA unavailable")

    monkeypatch.setattr(model_probe, "_execute_probe", unavailable)
    with pytest.raises(RuntimeError, match="CUDA unavailable"):
        run_probe(manifest, output)
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "failed"
    assert report["error"] == "CUDA unavailable"
    assert report["finished_utc"]
    assert (
        report["probe_code_sha256"]
        == hashlib.sha256(Path(model_probe.__file__).read_bytes()).hexdigest()
    )


def test_probe_rejects_unsupported_manifest_before_creating_outputs(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"schema_version": 99}')
    with pytest.raises(ValueError, match="schema_version"):
        run_probe(manifest, tmp_path / "run")
    assert not (tmp_path / "run").exists()
