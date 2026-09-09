import subprocess
import sys
from pathlib import Path

from integrations.kaggle import qualification_matrix as qm
from integrations.kaggle.profiles import (
    BF16_SAFETENSORS_PROFILE,
    Q8_REFERENCE_PROFILE,
    get_profile,
)


def test_run_canonical_requires_backend():
    p = subprocess.run(
        [sys.executable, "scripts/qualification/run-canonical.py"],
        capture_output=True,
        text=True,
    )
    assert p.returncode != 0


def test_run_canonical_unknown_backend_fails():
    p = subprocess.run(
        [
            sys.executable,
            "scripts/qualification/run-canonical.py",
            "--backend",
            "gpu9",
        ],
        capture_output=True,
        text=True,
    )
    assert p.returncode != 0


def test_matrix_release_profiles_and_backends_are_exact():
    assert qm.SUPPORTED_MATRIX_BACKENDS == ("cpu", "cuda0")
    assert qm.MATRIX_RESOLUTIONS == [512, 640, 768, 1024]
    assert get_profile(Q8_REFERENCE_PROFILE).allowed_backends == ("cpu", "cuda0")
    assert get_profile(BF16_SAFETENSORS_PROFILE).allowed_backends == ("cpu", "cuda0")


def test_matrix_runtime_hashes_are_backend_specific_and_frozen():
    assert qm.runtime_spec_for_backend("cpu").sha256 == (
        "7539d90b99eaf2b6279eec4f9006a68ae53e87bfe0c9c325ff3f329220468a5c"
    )
    assert qm.runtime_spec_for_backend("cuda0").sha256 == (
        "3fae6c1991ad0ac764c36495f688817c8a3d295d7651369bf74b7fd33743c3d0"
    )
    assert qm.runtime_spec_for_backend("cpu").env_var == "MAGE_CPU_PREBUILT_SD_CLI"
    assert qm.runtime_spec_for_backend("cuda0").env_var == "MAGE_CUDA_PREBUILT_SD_CLI"


def test_matrix_source_enforces_cuda0_telemetry_and_mask():
    source = Path("integrations/kaggle/qualification_matrix.py").read_text(encoding="utf-8")
    assert 'CUDA_VISIBLE_DEVICES") != "0"' in source
    assert 'collect_cuda=(backend == "cuda0")' in source
    assert "cuda0 qualification requires positive gpu_peak_mib" in source
    assert 'BackendSpec(backend=backend)' in source


def test_release_qualification_paths_do_not_build_cuda_from_source():
    matrix_source = Path("integrations/kaggle/qualification_matrix.py").read_text(encoding="utf-8")
    single_source = Path("integrations/kaggle/qualification.py").read_text(encoding="utf-8")
    for source in (matrix_source, single_source):
        assert "RuntimeBuildBackend" not in source
        assert "manager.build(" not in source
    assert "MAGE_CUDA_PREBUILT_SD_CLI" in matrix_source
    assert "prebuilt runtime is required" in single_source


def test_package_evidence_rejects_present_secret(tmp_path: Path):
    evidence = tmp_path / "ev"
    evidence.mkdir()
    secret = "token=" + "g" + "hp_" + "1234567890abcdef\n"
    (evidence / "note.txt").write_text(secret)
    output = tmp_path / "pkg"
    p = subprocess.run(
        [
            sys.executable,
            "scripts/qualification/package-evidence.py",
            "--evidence-root",
            str(evidence),
            "--output",
            str(output),
            "--label",
            "linux-cpu",
        ],
        capture_output=True,
        text=True,
    )
    assert p.returncode == 2
    assert "SECRET_SCAN=FAIL" in p.stdout


def test_package_evidence_rejects_model_weight(tmp_path: Path):
    evidence = tmp_path / "ev"
    evidence.mkdir()
    (evidence / "model.gguf").write_bytes(b"weight")
    output = tmp_path / "pkg"
    p = subprocess.run(
        [
            sys.executable,
            "scripts/qualification/package-evidence.py",
            "--evidence-root",
            str(evidence),
            "--output",
            str(output),
            "--label",
            "linux-cpu",
        ],
        capture_output=True,
        text=True,
    )
    assert p.returncode == 3
    assert "MODEL_WEIGHT_SCAN=FAIL" in p.stdout


def test_package_evidence_ok_with_no_secrets(tmp_path: Path):
    evidence = tmp_path / "ev"
    evidence.mkdir()
    (evidence / "result.json").write_text('{"ok": true}\n')
    output = tmp_path / "pkg"
    p = subprocess.run(
        [
            sys.executable,
            "scripts/qualification/package-evidence.py",
            "--evidence-root",
            str(evidence),
            "--output",
            str(output),
            "--label",
            "linux-cpu",
        ],
        capture_output=True,
        text=True,
    )
    assert p.returncode == 0
    assert "MODEL_WEIGHT_SCAN=PASS" in p.stdout
    assert "SECRET_SCAN=PASS" in p.stdout
    assert "ARCHIVE_PATH_SAFETY=PASS" in p.stdout
    assert "v1.0.0-linux-cpu-evidence.tar.gz" in p.stdout


def test_local_layout_proof_requires_all_three():
    p = subprocess.run(
        ["bash", "scripts/qualification/run-local-layout-proof.sh", "--dit", "a"],
        capture_output=True,
        text=True,
    )
    assert p.returncode != 0


def test_local_layout_proof_syntax_is_valid():
    p = subprocess.run(
        ["bash", "-n", "scripts/qualification/run-local-layout-proof.sh"],
        capture_output=True,
        text=True,
    )
    assert p.returncode == 0


def test_qualification_evidence_records_filename_and_sha256_explicitly():
    producers = (
        (
            Path("integrations/kaggle/qualification.py"),
            "manifest_with_root",
        ),
        (
            Path("scripts/qualification/run-canonical.py"),
            "manifest",
        ),
    )

    for path, manifest_name in producers:
        source = path.read_text(encoding="utf-8")
        assert '"model_hashes"' not in source
        assert '"models": {' in source
        assert '"filename": verified["diffusion"].name' in source
        assert '"filename": verified["text_encoder"].name' in source
        assert '"filename": verified["vae"].name' in source
        assert f'"sha256": {manifest_name}.diffusion.sha256' in source
        assert f'"sha256": {manifest_name}.text_encoder.sha256' in source
        assert f'"sha256": {manifest_name}.vae.sha256' in source
