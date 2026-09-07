import hashlib
import json
from pathlib import Path

import pytest

from integrations.kaggle import qualification_matrix as qm
from integrations.kaggle.profiles import (
    BF16_HIGH_MEMORY_CPU_PROFILE,
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
QWEN_SHA = "66358cb18bb6b3b1b6675aa412c7a88ef01d228f481184d13668e5201c730a0a"
VAE_SHA = "34e076dc1e8a15321e1e07be5111d59cf16dd10b804b7c7e20b4de29013427e0"


def _write_executable(tmp_path: Path, name: str = "sd-cli") -> Path:
    path = tmp_path / name
    path.write_bytes(b"#!/bin/sh\necho sd-cli 6b3edaa fake\n")
    path.chmod(0o755)
    return path


def _install_runtime(monkeypatch, tmp_path: Path) -> Path:
    cli = _write_executable(tmp_path)
    monkeypatch.setenv("MAGE_CPU_PREBUILT_SD_CLI", str(cli))
    monkeypatch.setattr(
        qm,
        "BF16_CPU_RUNTIME_SHA256",
        hashlib.sha256(cli.read_bytes()).hexdigest(),
    )
    return cli


def _fake_manifest(tmp_path: Path) -> ModelManifest:
    return ModelManifest(
        schema_version=1,
        model_family="Mage-Flow-Turbo",
        diffusion=ModelComponent(tmp_path / "dit.safetensors", DIT_SHA, "safetensors"),
        text_encoder=ModelComponent(tmp_path / "qwen.gguf", QWEN_SHA, "gguf", "Q4_K_M"),
        vae=ModelComponent(tmp_path / "vae.safetensors", VAE_SHA, "safetensors"),
    )


def _fake_result(width: int, height: int, **overrides) -> GenerationResult:
    values = dict(
        request_id=f"fake-{width}",
        seed=CANONICAL_SEED,
        exit_code=0,
        elapsed_ms=1234,
        peak_sd_cli_rss_kb=1000,
        minimum_mem_available_kb=4 * 1024 * 1024,
        gpu_peak_mib=None,
        artifact=ArtifactInfo(
            filename=f"fake-{width}.png",
            bytes=1024,
            sha256="d" * 64,
            width=width,
            height=height,
        ),
        stdout_path="/tmp/out.log",
        stderr_path="/tmp/err.log",
    )
    values.update(overrides)
    return GenerationResult(**values)


def _patch_manifest_build(monkeypatch, manifest, calls):
    def fake_build(input_root, output, *, profile):
        calls.append((input_root, output, profile))
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
        return output

    monkeypatch.setattr(qm, "build_kaggle_manifest", fake_build)


def _patch_manifest_verify(monkeypatch, manifest, calls):
    def fake_load(path, *, model_root=None):
        return manifest

    def fake_verify(m):
        calls.append(m)
        return {
            "diffusion": manifest.diffusion.path,
            "text_encoder": manifest.text_encoder.path,
            "vae": manifest.vae.path,
        }

    monkeypatch.setattr(qm, "load_manifest", fake_load)
    monkeypatch.setattr(qm, "verify_manifest", fake_verify)


class _FakeRuntimeManager:
    verifies = None

    def __init__(self, runtime_root, *, explicit_sd_cli=None):
        self.runtime_root = runtime_root
        self.explicit_sd_cli = explicit_sd_cli

    def verify(self, sd_cli_path, requested_backend):
        self.verifies.append((sd_cli_path, requested_backend))
        from mageflow_native.runtime.manager import RuntimeIdentity

        return RuntimeIdentity(
            path=str(sd_cli_path),
            version_output="sd-cli 6b3edaa (fake)",
            devices_output="CPU Intel(R) Xeon(R) CPU @ 2.00GHz",
            pinned_commit="6b3edaaf32cc19e5bb2d819c788bd557eddc8eba",
        )


def _patch_runtime_manager(monkeypatch) -> list:
    calls: list = []
    _FakeRuntimeManager.verifies = calls
    monkeypatch.setattr(qm, "RuntimeManager", _FakeRuntimeManager)
    return calls


def _patch_run_generation(monkeypatch, calls, *, raise_at=None) -> None:
    def fake(sd_cli, manifest, backend_spec, **kwargs):
        calls.append(kwargs)
        width = kwargs["width"]
        if raise_at is not None and width == raise_at:
            raise RuntimeError(f"boom at {width}")
        return _fake_result(width=width, height=kwargs.get("height", width))

    monkeypatch.setattr(qm, "run_generation", fake)


@pytest.fixture
def matrix_env(monkeypatch, tmp_path):
    work_root = tmp_path / "work"
    input_root = tmp_path / "input"
    manifest = _fake_manifest(tmp_path)

    build_calls: list = []
    verify_calls: list = []
    gen_calls: list = []

    _install_runtime(monkeypatch, tmp_path)
    _patch_manifest_build(monkeypatch, manifest, build_calls)
    _patch_manifest_verify(monkeypatch, manifest, verify_calls)
    runtime_verify_calls = _patch_runtime_manager(monkeypatch)
    _patch_run_generation(monkeypatch, gen_calls)

    return {
        "work_root": work_root,
        "input_root": input_root,
        "build_calls": build_calls,
        "verify_calls": verify_calls,
        "runtime_verify_calls": runtime_verify_calls,
        "gen_calls": gen_calls,
    }


def _run(matrix_env, **overrides) -> tuple[int, dict]:
    code, aggregate = qm.run_matrix(
        input_root=matrix_env["input_root"],
        work_root=matrix_env["work_root"],
        backend="cpu",
        profile=BF16_HIGH_MEMORY_CPU_PROFILE,
        repo_dir=None,
        **overrides,
    )
    return code, aggregate


def test_matrix_default_resolution_order_is_exact():
    assert qm.MATRIX_RESOLUTIONS == [512, 640, 768, 1024]


def test_parse_resolutions_rejects_invalid_values():
    with pytest.raises(ValueError):
        qm.parse_resolutions("512,512")
    with pytest.raises(ValueError):
        qm.parse_resolutions("512,a,768")
    with pytest.raises(ValueError):
        qm.parse_resolutions("512,-4")
    with pytest.raises(ValueError):
        qm.parse_resolutions("0,512")
    with pytest.raises(ValueError):
        qm.parse_resolutions("1.5,512")


def test_parse_resolutions_preserves_order():
    assert qm.parse_resolutions("768,512,1024") == [768, 512, 1024]


def test_successful_matrix_runs_all_four_resolutions_in_order(matrix_env):
    code, aggregate = _run(matrix_env)
    assert code == 0
    assert aggregate["status"] == "passed"
    assert aggregate["completed_resolutions"] == [512, 640, 768, 1024]
    dims = [(c["width"], c["height"]) for c in matrix_env["gen_calls"]]
    assert dims == [(512, 512), (640, 640), (768, 768), (1024, 1024)]
    assert [r["resolution"]["width"] for r in aggregate["matrix"]] == [512, 640, 768, 1024]


def test_setup_happens_exactly_once_per_matrix_process(matrix_env):
    code, _ = _run(matrix_env)
    assert code == 0
    assert len(matrix_env["build_calls"]) == 1
    assert len(matrix_env["verify_calls"]) == 1
    assert len(matrix_env["runtime_verify_calls"]) == 1


def test_generation_params_stay_canonical_for_all_runs(matrix_env):
    _run(matrix_env)
    for kwargs in matrix_env["gen_calls"]:
        assert kwargs["prompt"] == CANONICAL_PROMPT
        assert kwargs["seed"] == CANONICAL_SEED
        assert kwargs["steps"] == CANONICAL_STEPS
        assert kwargs["cfg_scale"] == CANONICAL_CFG
        assert kwargs["threads"] == CANONICAL_THREADS
    assert all(c["width"] == c["height"] for c in matrix_env["gen_calls"])


def test_headroom_validation_invoked_after_every_generation(matrix_env, monkeypatch):
    from integrations.kaggle.profiles import validate_profile_result as real_validate

    post_calls: list = []

    def spy(name, *, minimum_mem_available_kb):
        post_calls.append(minimum_mem_available_kb)
        return real_validate(
            name, minimum_mem_available_kb=minimum_mem_available_kb
        )

    monkeypatch.setattr(qm, "validate_profile_result", spy)
    _run(matrix_env)
    assert len(post_calls) == 4


def test_failure_at_768_generation_prevents_1024(matrix_env, monkeypatch):
    gen_calls = matrix_env["gen_calls"]
    gen_calls.clear()

    def fake(sd_cli, manifest, backend_spec, **kwargs):
        gen_calls.append(kwargs)
        if kwargs["width"] == 768:
            raise RuntimeError("sd-cli exploded at 768")
        return _fake_result(width=kwargs["width"], height=kwargs.get("height", kwargs["width"]))

    monkeypatch.setattr(qm, "run_generation", fake)
    code, aggregate = _run(matrix_env)
    assert code != 0
    assert [c["width"] for c in gen_calls] == [512, 640, 768]
    assert aggregate["status"] == "failed"
    assert aggregate["failed_resolution"]["resolution"] == 768
    assert aggregate["completed_resolutions"] == [512, 640]


def test_headroom_failure_stops_matrix_and_writes_partial_evidence(matrix_env, monkeypatch):
    gen_calls = matrix_env["gen_calls"]
    gen_calls.clear()

    def fake(sd_cli, manifest, backend_spec, **kwargs):
        gen_calls.append(kwargs)
        return _fake_result(width=kwargs["width"], height=kwargs.get("height", kwargs["width"]))

    state = {"calls": 0}

    def failing_validate(name, *, minimum_mem_available_kb):
        state["calls"] += 1
        if state["calls"] == 3:
            raise ProfilePreflightError("headroom below 3 GiB at 768")
        return None

    monkeypatch.setattr(qm, "run_generation", fake)
    monkeypatch.setattr(qm, "validate_profile_result", failing_validate)
    code, aggregate = _run(matrix_env)
    assert code != 0
    assert [c["width"] for c in gen_calls] == [512, 640, 768]
    assert aggregate["status"] == "failed"
    assert aggregate["failed_resolution"]["resolution"] == 768
    per_res = aggregate["matrix"][-1]
    assert per_res["status"] == "failed"
    aggregate_path = (
        matrix_env["work_root"]
        / "output"
        / "qualification-matrix-bf16-high-memory-cpu-cpu.json"
    )
    assert aggregate_path.is_file()
    saved = json.loads(aggregate_path.read_text(encoding="utf-8"))
    assert saved["status"] == "failed"
    assert saved["completed_resolutions"] == [512, 640]


def test_cpu_bf16_rejects_cuda_backend_before_any_generation(matrix_env):
    code, aggregate = (
        qm.run_matrix(
            input_root=matrix_env["input_root"],
            work_root=matrix_env["work_root"],
            backend="cuda0",
            profile=BF16_HIGH_MEMORY_CPU_PROFILE,
            repo_dir=None,
        )
    )
    assert code != 0
    assert aggregate["status"] == "failed"
    assert matrix_env["gen_calls"] == []


def test_missing_prebuilt_runtime_fails_before_any_generation(monkeypatch, tmp_path):
    gen_calls: list = []
    _patch_run_generation(monkeypatch, gen_calls)
    work_root = tmp_path / "work"
    code, aggregate = qm.run_matrix(
        input_root=tmp_path / "input",
        work_root=work_root,
        backend="cpu",
        profile=BF16_HIGH_MEMORY_CPU_PROFILE,
        repo_dir=None,
    )
    assert code != 0
    assert aggregate["status"] == "failed"
    assert aggregate["error"]["phase"] == "setup"
    assert gen_calls == []


def test_runtime_sha_mismatch_fails_before_any_generation(monkeypatch, tmp_path):
    cli = _write_executable(tmp_path)
    monkeypatch.setenv("MAGE_CPU_PREBUILT_SD_CLI", str(cli))
    monkeypatch.setattr(qm, "BF16_CPU_RUNTIME_SHA256", "0" * 64)
    manifest = _fake_manifest(tmp_path)
    _patch_manifest_build(monkeypatch, manifest, [])
    _patch_manifest_verify(monkeypatch, manifest, [])
    _patch_runtime_manager(monkeypatch)
    gen_calls: list = []
    _patch_run_generation(monkeypatch, gen_calls)
    code, aggregate = qm.run_matrix(
        input_root=tmp_path / "input",
        work_root=tmp_path / "work",
        backend="cpu",
        profile=BF16_HIGH_MEMORY_CPU_PROFILE,
        repo_dir=None,
    )
    assert code != 0
    assert aggregate["status"] == "failed"
    assert "mismatch" in aggregate["error"]["message"]
    assert gen_calls == []


def test_missing_telemetry_stops_matrix(monkeypatch, tmp_path):
    gen_calls: list = []

    def fake(sd_cli, manifest, backend_spec, **kwargs):
        gen_calls.append(kwargs)
        return _fake_result(
            width=kwargs["width"],
            height=kwargs.get("height", kwargs["width"]),
            minimum_mem_available_kb=None,
        )

    _install_runtime(monkeypatch, tmp_path)
    manifest = _fake_manifest(tmp_path)
    _patch_manifest_build(monkeypatch, manifest, [])
    _patch_manifest_verify(monkeypatch, manifest, [])
    _patch_runtime_manager(monkeypatch)
    monkeypatch.setattr(qm, "run_generation", fake)
    code, aggregate = qm.run_matrix(
        input_root=tmp_path / "input",
        work_root=tmp_path / "work",
        backend="cpu",
        profile=BF16_HIGH_MEMORY_CPU_PROFILE,
        repo_dir=None,
    )
    assert code != 0
    assert [c["width"] for c in gen_calls] == [512]
    assert aggregate["failed_resolution"]["resolution"] == 512


def test_successful_matrix_writes_per_resolution_evidence(matrix_env):
    code, _ = _run(matrix_env)
    assert code == 0
    output = matrix_env["work_root"] / "output"
    for resolution in (512, 640, 768, 1024):
        path = output / f"qualification-bf16-high-memory-cpu-cpu-{resolution:04d}.json"
        assert path.is_file(), path
        record = json.loads(path.read_text(encoding="utf-8"))
        assert record["status"] == "passed"
        assert record["resolution"]["width"] == resolution
        assert record["resolution"]["height"] == resolution
        assert record["request"]["cfg"] == CANONICAL_CFG
        assert record["request"]["threads"] == CANONICAL_THREADS


def test_q8_reference_contract_remains_unchanged():
    profile = get_profile(Q8_REFERENCE_PROFILE)
    assert profile.name == "q8-reference"
    assert profile.diffusion.filename == "Mage-Flow-Turbo-DiT-Q8_0.gguf"
    assert profile.diffusion.format == "gguf"
    assert profile.diffusion.quantization == "Q8_0"
    assert profile.allowed_backends == ("cpu", "cuda0")