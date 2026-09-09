import hashlib
import sys
import tarfile
from pathlib import Path

from scripts.qualification import package_evidence


def _run(monkeypatch, evidence: Path, out: Path, label: str) -> int:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "package-evidence",
            "--evidence-root",
            str(evidence),
            "--output",
            str(out),
            "--label",
            label,
        ],
    )
    return package_evidence.main()


def test_packager_preserves_full_v1_0_0_basename(tmp_path, monkeypatch):
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    (evidence / "result.json").write_text('{"status":"passed"}\n', encoding="utf-8")
    out = tmp_path / "out"

    assert _run(monkeypatch, evidence, out, "kaggle-q8-cpu") == 0

    archive = out / "mage-flow-turbo-native-inference-v1.0.0-kaggle-q8-cpu-evidence.tar.gz"
    sidecar = archive.with_name(archive.name + ".sha256")
    assert archive.is_file()
    assert sidecar.is_file()
    digest, name = sidecar.read_text(encoding="utf-8").strip().split(maxsplit=1)
    assert digest == hashlib.sha256(archive.read_bytes()).hexdigest()
    assert name == archive.name


def test_packager_rejects_model_weights(tmp_path, monkeypatch):
    suffixes = [".gguf", ".safetensors", ".ckpt", ".pt", ".pth", ".onnx", ".bin"]
    for index, suffix in enumerate(suffixes):
        evidence = tmp_path / f"evidence-{index}"
        evidence.mkdir()
        (evidence / f"model{suffix}").write_bytes(b"weight")
        out = tmp_path / f"out-{index}"
        assert _run(monkeypatch, evidence, out, f"cell-{index}") == 3
        assert not list(out.glob("*.tar.gz")) if out.exists() else True


def test_archive_paths_and_internal_manifest_are_safe(tmp_path, monkeypatch):
    evidence = tmp_path / "evidence"
    (evidence / "nested").mkdir(parents=True)
    (evidence / "nested" / "result.json").write_text("{}\n", encoding="utf-8")
    out = tmp_path / "out"

    assert _run(monkeypatch, evidence, out, "kaggle-bf16-t4") == 0
    archive = out / "mage-flow-turbo-native-inference-v1.0.0-kaggle-bf16-t4-evidence.tar.gz"

    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        assert members
        assert all(not Path(member.name).is_absolute() for member in members)
        assert all(".." not in Path(member.name).parts for member in members)
        roots = {Path(member.name).parts[0] for member in members if member.name}
        assert roots == {"mage-flow-turbo-native-inference-v1.0.0-kaggle-bf16-t4-evidence"}
        tar.extractall(tmp_path / "extract", filter="data")

    root = tmp_path / "extract" / "mage-flow-turbo-native-inference-v1.0.0-kaggle-bf16-t4-evidence"
    manifest = root / "MANIFEST.sha256"
    assert manifest.is_file()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, rel = line.split("  ", 1)
        target = root / rel
        assert target.is_file()
        assert hashlib.sha256(target.read_bytes()).hexdigest() == digest
