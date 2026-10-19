import hashlib
import importlib
import json
import sys
from types import ModuleType

import pytest

from face_surveillance.vision.onnx_probe import runtime_only_imports, verify_runtime_manifest


@pytest.mark.parametrize("name", ["facenet_pytorch", "onnxscript", "onnx_ir"])
def test_runtime_explicitly_blocks_export_only_imports(name):
    with runtime_only_imports():
        with pytest.raises(RuntimeError, match="Export-only"):
            importlib.import_module(name)


def test_runtime_rejects_export_library_already_in_process(monkeypatch):
    monkeypatch.setitem(sys.modules, "facenet_pytorch", ModuleType("facenet_pytorch"))
    with pytest.raises(RuntimeError, match="already imported"):
        with runtime_only_imports():
            pass


def manifest_fixture(tmp_path):
    model = tmp_path / "model.onnx"
    model.write_bytes(b"pinned bytes")
    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    report = tmp_path / "export_report.json"
    report.write_text(
        json.dumps(
            {"status": "passed", "derived_sha256": digest, "source_checkpoint_sha256": "original"}
        )
    )
    manifest = {
        "schema_version": 1,
        "assets": {
            name: {"path": str(model), "sha256": digest}
            for name in ("detector", "arcface", "facenet")
        },
        "derived_from": {"sha256": "original"},
        "validation": {
            "report_path": str(report),
            "report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
        },
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path, report, model, manifest


def test_runtime_manifest_requires_matching_passed_export_and_model_hashes(tmp_path):
    path, report, model, manifest = manifest_fixture(tmp_path)
    loaded, paths = verify_runtime_manifest(path)
    assert loaded == manifest
    assert paths["facenet"] == model
    model.write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA-256"):
        verify_runtime_manifest(path)


def test_failed_export_cannot_be_activated_even_with_a_correct_report_hash(tmp_path):
    path, report, model, manifest = manifest_fixture(tmp_path)
    report.write_text(json.dumps({"status": "failed"}))
    manifest["validation"]["report_sha256"] = hashlib.sha256(report.read_bytes()).hexdigest()
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="passed"):
        verify_runtime_manifest(path)


def test_manifest_cannot_substitute_a_different_validated_encoder(tmp_path):
    path, report, model, manifest = manifest_fixture(tmp_path)
    result = json.loads(report.read_text())
    result["derived_sha256"] = "different"
    report.write_text(json.dumps(result))
    manifest["validation"]["report_sha256"] = hashlib.sha256(report.read_bytes()).hexdigest()
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="derived"):
        verify_runtime_manifest(path)
