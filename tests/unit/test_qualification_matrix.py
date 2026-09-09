import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from integrations.kaggle import qualification_matrix as qm
from integrations.kaggle.profiles import (
    BF16_SAFETENSORS_PROFILE,
    ProfilePreflightError,
    Q8_REFERENCE_PROFILE,
    get_profile,
)
from mageflow_native.constants import (
    CANONICAL_CFG,
    CANONICAL_PROMPT,
    CANONICAL_SEED,
    CANONICAL_STEPS,
    CANONICAL_THREADS,
)
from mageflow_native.inference.runner import ArtifactInfo, GenerationResult
from mageflow_native.models.manifest import ModelComponent, ModelManifest

DIT_SHA = "4c3dafc143ee64121692b6b63563a4f5288bf6183c4870e1d65f1566519ba7f0"
BF16_SHA = "6df47df3d7efc9ebdad075b87b3e9e4f74d09dca672d592271788f0ee27ab97d"
QWEN_SHA = "66358cb18bb6b3b1b6675aa412c7a88ef01d228f481184d13668e5201c730a0a"
VAE_SHA = "34e076dc1e8a15321e1e07be5111d59cf16dd10b804b7c7e20b4de29013427e0"


def _write_executable(tmp_path: Path, name: str) -> Path:
    path = tmp_path / name
    path.write_bytes(b"#!/bin/sh\necho sd-cli 6b3edaa fake\n")
    path.chmod(0o755)
    return path


def _install_runtime(monkeypatch, tmp_path: Path, backend: str) -> Path:
    cli = _write_executable(tmp_path, f"sd-cli-{backend}")
    spec = qm.runtime_spec_for_backend(backend)
    monkeypatch.setenv(spec.env_var, str(cli))
    monkeypatch.setitem(
        qm._RUNTIME_SPECS,
        backend,
        qm.RuntimeSpec(
            env_var=spec.env_var,
            sha256=hashlib.sha256(cli.read_bytes()).hexdigest(),
        ),
    )
    return cli


def _fake_manifest(tmp_path: Path, profile: str) -> ModelManifest:
    diffusion_sha = DIT_SHA if profile == Q8_REFERENCE_PROFILE else BF16_SHA
    diffusion_format = "gguf" if profile == Q8_REFERENCE_PROFILE else "safetensors"
    diffusion_quant = "Q8_0" if profile == Q8_REFERENCE_PROFILE else None
    return ModelManifest(
        schema_version=1,
        model_family="Mage-Flow-Turbo",
        diffusion=ModelComponent(
            tmp_path / "dit.model",
            diffusion_sha,
            diffusion_format,
            diffusion_quant,
        ),
        text_encoder=ModelComponent(
            tmp_path / "qwen.gguf",
            QWEN_SHA,
            "gguf",
            "Q4_K_M",
        ),
        vae=ModelComponent(
            tmp_path / "vae.safetensors",
            VAE_SHA,
            "safetensors",
        ),
    )


def _fake_result(width: int, *, backend: str, gpu_peak_mib=None, min_mem=None):
    if gpu_peak_mib is None and backend == "cuda0":
        gpu_peak_mib = 8192
    if min_mem is None and backend == "cpu":
        min_mem = 4 * 1024 * 1024
    return GenerationResult(
        request_id=f"fake-{backend}-{width}",
        seed=CANONICAL_SEED,
        exit_code=0,
        elapsed_ms=1234,
        peak_sd_cli_rss_kb=1000,
        minimum_mem_available_kb=min_mem,
        gpu_peak_mib=gpu_peak_mib,
        artifact=ArtifactInfo(
            filename=f"fake-{width}.png",
            bytes=1024,
            sha256="d" * 64,
            width=width,
            height=width,
        ),
        stdout_path="/tmp/out.log",
        stderr_path="/tmp/err.log",
    )


class _FakeRuntimeManager:
    verifies = None

    def __init__(self, runtime_root, *, explicit_sd_cli=None):
        self.runtime_root = runtime_root
        self.explicit_sd_cli = explicit_sd_cli

    def verify(self, sd_cli_path, requested_backend):
        self.verifies.append((str(sd_cli_path), requested_backend))
        from mageflow_native.runtime.manager import RuntimeIdentity

        devices = (
            "cuda0 NVIDIA Tesla T4" if requested_backend == "cuda0"
            else "CPU Intel Xeon"
        )
        return RuntimeIdentity(
            path=str(sd_cli_path),
            version_output="sd-cli 6b3edaa (fake)",
            devices_output=devices,
            pinned_commit="6b3edaaf32cc19e5bb2d819c788bd557eddc8eba",
        )


def _setup(monkeypatch, tmp_path: Path, profile: str, backend: str):
    input_root = tmp_path / "input"
    work_root = tmp_path / "work"
    manifest = _fake_manifest(tmp_path, profile)
    build_calls = []
    verify_calls = []
    generation_calls = []

    monkeypatch.setattr(qm, "_mem_total_kb", lambda: 64 * 1024 * 1024)
    monkeypatch.setattr(qm, "read_mem_available_kb", lambda: 10 * 1024 * 1024)
    _install_runtime(monkeypatch, tmp_path, backend)

    def fake_build(input_root, output, *, profile):
        build_calls.append((input_root, output, profile))
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text('{"schema_version": 1}\n', encoding="utf-8")
        return output

    def fake_load(path, *, model_root=None):
        return manifest

    def fake_verify(m):
        verify_calls.append(m)
        return {
            "diffusion": manifest.diffusion.path,
            "text_encoder": manifest.text_encoder.path,
            "vae": manifest.vae.path,
        }

    _FakeRuntimeManager.verifies = []
    monkeypatch.setattr(qm, "build_kaggle_manifest", fake_build)
    monkeypatch.setattr(qm, "load_manifest", fake_load)
    monkeypatch.setattr(qm, "verify_manifest", fake_verify)
    monkeypatch.setattr(qm, "RuntimeManager", _FakeRuntimeManager)

    def fake_generation(sd_cli, manifest, backend_spec, **kwargs):
        generation_calls.append((backend_spec.backend, kwargs))
        return _fake_result(kwargs["width"], backend=backend_spec.backend)

    monkeypatch.setattr(qm, "run_generation", fake_generation)
    if backend == "cuda0":
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    else:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    return {
        "input_root": input_root,
        "work_root": work_root,
        "build_calls": build_calls,
        "verify_calls": verify_calls,
        "generation_calls": generation_calls,
        "runtime_verifies": _FakeRuntimeManager.verifies,
    }


def _run(env, *, profile, backend, repo_dir=None, resolutions=None):
    return qm.run_matrix(
        input_root=env["input_root"],
        work_root=env["work_root"],
        backend=backend,
        profile=profile,
        repo_dir=repo_dir,
        resolutions=resolutions,
    )


def test_matrix_default_resolution_order_is_exact():
    assert qm.MATRIX_RESOLUTIONS == [512, 640, 768, 1024]


def test_matrix_backends_are_cpu_and_cuda0_only():
    assert qm.SUPPORTED_MATRIX_BACKENDS == ("cpu", "cuda0")


def test_runtime_specs_are_frozen_per_backend():
    assert qm.runtime_spec_for_backend("cpu").env_var == "MAGE_CPU_PREBUILT_SD_CLI"
    assert qm.runtime_spec_for_backend("cpu").sha256 == (
        "7539d90b99eaf2b6279eec4f9006a68ae53e87bfe0c9c325ff3f329220468a5c"
    )
    assert qm.runtime_spec_for_backend("cuda0").env_var == "MAGE_CUDA_PREBUILT_SD_CLI"
    assert qm.runtime_spec_for_backend("cuda0").sha256 == (
        "3fae6c1991ad0ac764c36495f688817c8a3d295d7651369bf74b7fd33743c3d0"
    )


def test_unknown_matrix_backend_fails_closed():
    with pytest.raises(ValueError, match="unsupported matrix backend"):
        qm.runtime_spec_for_backend("auto")


def test_parse_resolutions_rejects_invalid_values():
    for value in ("512,512", "512,a", "0,512", "512,-4", "1.5,512"):
        with pytest.raises(ValueError):
            qm.parse_resolutions(value)


def test_parse_resolutions_preserves_order():
    assert qm.parse_resolutions("768,512,1024") == [768, 512, 1024]


@pytest.mark.parametrize(
    "profile,backend",
    [
        (Q8_REFERENCE_PROFILE, "cpu"),
        (BF16_SAFETENSORS_PROFILE, "cpu"),
        (Q8_REFERENCE_PROFILE, "cuda0"),
        (BF16_SAFETENSORS_PROFILE, "cuda0"),
    ],
)
def test_all_four_cells_run_ordered_matrix(monkeypatch, tmp_path, profile, backend):
    env = _setup(monkeypatch, tmp_path, profile, backend)
    code, aggregate = _run(env, profile=profile, backend=backend)
    assert code == 0
    assert aggregate["status"] == "passed"
    assert aggregate["profile"] == profile
    assert aggregate["backend"] == backend
    assert aggregate["completed_resolutions"] == [512, 640, 768, 1024]
    assert [call[1]["width"] for call in env["generation_calls"]] == [512, 640, 768, 1024]
    assert len(env["build_calls"]) == 1
    assert len(env["verify_calls"]) == 1
    assert len(env["runtime_verifies"]) == 1


def test_canonical_request_is_stable_for_all_runs(monkeypatch, tmp_path):
    env = _setup(monkeypatch, tmp_path, Q8_REFERENCE_PROFILE, "cpu")
    code, _ = _run(env, profile=Q8_REFERENCE_PROFILE, backend="cpu")
    assert code == 0
    for _, kwargs in env["generation_calls"]:
        assert kwargs["prompt"] == CANONICAL_PROMPT
        assert kwargs["seed"] == CANONICAL_SEED
        assert kwargs["steps"] == CANONICAL_STEPS
        assert kwargs["cfg_scale"] == CANONICAL_CFG
        assert kwargs["threads"] == CANONICAL_THREADS
        assert kwargs["width"] == kwargs["height"]
        assert kwargs["collect_cuda"] is False


def test_cuda_enables_cuda_collection_and_positive_vram(monkeypatch, tmp_path):
    env = _setup(monkeypatch, tmp_path, Q8_REFERENCE_PROFILE, "cuda0")
    code, aggregate = _run(env, profile=Q8_REFERENCE_PROFILE, backend="cuda0")
    assert code == 0
    assert all(call[0] == "cuda0" for call in env["generation_calls"])
    assert all(call[1]["collect_cuda"] is True for call in env["generation_calls"])
    assert all(record["gpu_peak_mib"] > 0 for record in aggregate["matrix"])
    assert aggregate["session"]["cuda_visible_devices"] == "0"


def test_cuda_requires_exact_visible_device_mask(monkeypatch, tmp_path):
    env = _setup(monkeypatch, tmp_path, Q8_REFERENCE_PROFILE, "cuda0")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    code, aggregate = _run(env, profile=Q8_REFERENCE_PROFILE, backend="cuda0")
    assert code != 0
    assert aggregate["error"]["phase"] == "setup"
    assert "CUDA_VISIBLE_DEVICES=0" in aggregate["error"]["message"]
    assert env["generation_calls"] == []


def test_cuda_missing_gpu_telemetry_fails_at_512(monkeypatch, tmp_path):
    env = _setup(monkeypatch, tmp_path, Q8_REFERENCE_PROFILE, "cuda0")

    def fake_generation(sd_cli, manifest, backend_spec, **kwargs):
        env["generation_calls"].append((backend_spec.backend, kwargs))
        return _fake_result(kwargs["width"], backend="cuda0", gpu_peak_mib=0)

    monkeypatch.setattr(qm, "run_generation", fake_generation)
    code, aggregate = _run(env, profile=Q8_REFERENCE_PROFILE, backend="cuda0")
    assert code != 0
    assert len(env["generation_calls"]) == 1
    assert aggregate["failed_resolution"]["resolution"] == 512
    assert aggregate["matrix"][0]["error"]["type"] == "MissingCudaTelemetryError"


def test_failure_stops_without_retry_or_backend_fallback(monkeypatch, tmp_path):
    env = _setup(monkeypatch, tmp_path, BF16_SAFETENSORS_PROFILE, "cuda0")

    def fail(sd_cli, manifest, backend_spec, **kwargs):
        env["generation_calls"].append((backend_spec.backend, kwargs))
        raise RuntimeError("oom")

    monkeypatch.setattr(qm, "run_generation", fail)
    code, aggregate = _run(env, profile=BF16_SAFETENSORS_PROFILE, backend="cuda0")
    assert code != 0
    assert len(env["generation_calls"]) == 1
    assert env["generation_calls"][0][0] == "cuda0"
    assert aggregate["failed_resolution"]["resolution"] == 512


def test_failure_at_768_prevents_1024(monkeypatch, tmp_path):
    env = _setup(monkeypatch, tmp_path, BF16_SAFETENSORS_PROFILE, "cpu")

    def fake_generation(sd_cli, manifest, backend_spec, **kwargs):
        env["generation_calls"].append((backend_spec.backend, kwargs))
        if kwargs["width"] == 768:
            raise RuntimeError("boom")
        return _fake_result(kwargs["width"], backend="cpu")

    monkeypatch.setattr(qm, "run_generation", fake_generation)
    code, aggregate = _run(env, profile=BF16_SAFETENSORS_PROFILE, backend="cpu")
    assert code != 0
    assert [call[1]["width"] for call in env["generation_calls"]] == [512, 640, 768]
    assert aggregate["completed_resolutions"] == [512, 640]


def test_bf16_cpu_headroom_failure_stops_matrix(monkeypatch, tmp_path):
    env = _setup(monkeypatch, tmp_path, BF16_SAFETENSORS_PROFILE, "cpu")
    state = {"calls": 0}

    def validate(name, *, backend, minimum_mem_available_kb):
        state["calls"] += 1
        if state["calls"] == 2:
            raise ProfilePreflightError("headroom below 3 GiB")

    monkeypatch.setattr(qm, "validate_profile_result", validate)
    code, aggregate = _run(env, profile=BF16_SAFETENSORS_PROFILE, backend="cpu")
    assert code != 0
    assert aggregate["failed_resolution"]["resolution"] == 640
    assert [call[1]["width"] for call in env["generation_calls"]] == [512, 640]


def test_bf16_cuda_does_not_use_cpu_headroom_gate(monkeypatch, tmp_path):
    env = _setup(monkeypatch, tmp_path, BF16_SAFETENSORS_PROFILE, "cuda0")
    monkeypatch.setattr(qm, "_mem_total_kb", lambda: 16 * 1024 * 1024)
    code, aggregate = _run(env, profile=BF16_SAFETENSORS_PROFILE, backend="cuda0")
    assert code == 0
    assert aggregate["status"] == "passed"


def test_runtime_sha_mismatch_fails_before_generation(monkeypatch, tmp_path):
    env = _setup(monkeypatch, tmp_path, Q8_REFERENCE_PROFILE, "cpu")
    spec = qm.runtime_spec_for_backend("cpu")
    monkeypatch.setitem(
        qm._RUNTIME_SPECS,
        "cpu",
        qm.RuntimeSpec(env_var=spec.env_var, sha256="0" * 64),
    )
    code, aggregate = _run(env, profile=Q8_REFERENCE_PROFILE, backend="cpu")
    assert code != 0
    assert aggregate["error"]["phase"] == "setup"
    assert "runtime sha256 mismatch" in aggregate["error"]["message"]
    assert env["generation_calls"] == []


def test_source_identity_records_head_and_tree(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    (repo / "a.txt").write_text("a\n", encoding="utf-8")
    subprocess.run(["git", "add", "a.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo, check=True, capture_output=True)

    head, tree = qm._source_identity(repo)
    expected_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True, capture_output=True, check=True
    ).stdout.strip()
    expected_tree = subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"], cwd=repo, text=True, capture_output=True, check=True
    ).stdout.strip()
    assert (head, tree) == (expected_head, expected_tree)


def test_evidence_repeats_source_head_tree(monkeypatch, tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    (repo / "a.txt").write_text("a\n", encoding="utf-8")
    subprocess.run(["git", "add", "a.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo, check=True, capture_output=True)

    env = _setup(monkeypatch, tmp_path, Q8_REFERENCE_PROFILE, "cpu")
    code, aggregate = _run(env, profile=Q8_REFERENCE_PROFILE, backend="cpu", repo_dir=repo)
    assert code == 0
    assert aggregate["source_head"] != "unknown"
    assert aggregate["source_tree"] != "unknown"
    assert all(r["source_head"] == aggregate["source_head"] for r in aggregate["matrix"])
    assert all(r["source_tree"] == aggregate["source_tree"] for r in aggregate["matrix"])


def test_per_resolution_files_use_new_profile_name(monkeypatch, tmp_path):
    env = _setup(monkeypatch, tmp_path, BF16_SAFETENSORS_PROFILE, "cpu")
    code, _ = _run(env, profile=BF16_SAFETENSORS_PROFILE, backend="cpu")
    assert code == 0
    output = env["work_root"] / "output"
    for resolution in (512, 640, 768, 1024):
        path = output / f"qualification-bf16-safetensors-cpu-{resolution:04d}.json"
        assert path.is_file()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["profile"] == "bf16-safetensors"
        assert data["schema_version"] == 2
        assert data["release_version"] == "1.0.0"


def test_q8_reference_contract_remains_cpu_and_cuda0():
    profile = get_profile(Q8_REFERENCE_PROFILE)
    assert profile.allowed_backends == ("cpu", "cuda0")
