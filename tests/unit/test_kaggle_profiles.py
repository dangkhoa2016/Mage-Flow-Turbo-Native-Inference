import json
from pathlib import Path

import pytest

from integrations.kaggle import input_adapter
from integrations.kaggle.profiles import (
    BF16_HIGH_MEMORY_CPU_PROFILE,
    BF16_MIN_HEADROOM_KB,
    BF16_MIN_RAM_KB,
    BF16_TRANSFORMER_SHA256,
    Q8_REFERENCE_PROFILE,
    ProfilePreflightError,
    get_profile,
    validate_profile_environment,
    validate_profile_result,
)


def test_q8_reference_profile_remains_default_portable_contract():
    profile = get_profile(Q8_REFERENCE_PROFILE)
    assert profile.name == "q8-reference"
    assert profile.diffusion.filename == "Mage-Flow-Turbo-DiT-Q8_0.gguf"
    assert profile.diffusion.format == "gguf"
    assert profile.diffusion.quantization == "Q8_0"
    assert profile.allowed_backends == ("cpu", "cuda0")
    assert profile.min_ram_kb == 0


def test_bf16_high_memory_profile_uses_exact_transformer_identity():
    profile = get_profile(BF16_HIGH_MEMORY_CPU_PROFILE)
    assert profile.name == "bf16-high-memory-cpu"
    assert profile.diffusion.filename == "diffusion_pytorch_model.safetensors"
    assert profile.diffusion.sha256 == BF16_TRANSFORMER_SHA256
    assert profile.diffusion.sha256 == "6df47df3d7efc9ebdad075b87b3e9e4f74d09dca672d592271788f0ee27ab97d"
    assert profile.diffusion.format == "safetensors"
    assert profile.diffusion.quantization is None
    assert profile.allowed_backends == ("cpu",)
    assert BF16_MIN_RAM_KB == 27 * 1024 * 1024
    assert BF16_MIN_HEADROOM_KB == 3 * 1024 * 1024
    assert profile.min_ram_kb == BF16_MIN_RAM_KB


def test_bf16_profile_uses_single_pytorch_default_mirror_for_transformer_and_vae():
    bf16 = get_profile(BF16_HIGH_MEMORY_CPU_PROFILE)
    q8 = get_profile(Q8_REFERENCE_PROFILE)
    default_fragment = "mage-flow-community-mage-flow-turbo/pytorch/default"
    assert bf16.diffusion.required_fragment == default_fragment
    assert bf16.vae.required_fragment == default_fragment
    assert q8.vae.required_fragment == "mage-flow-community-mage-flow-turbo/pytorch/vae-only"


def test_bf16_profile_rejects_cuda_without_q8_fallback():
    with pytest.raises(ProfilePreflightError, match="CPU-only"):
        validate_profile_environment(
            BF16_HIGH_MEMORY_CPU_PROFILE,
            backend="cuda0",
            mem_total_kb=64 * 1024 * 1024,
        )


def test_bf16_profile_rejects_16_gib_ram():
    with pytest.raises(ProfilePreflightError, match="27 GiB"):
        validate_profile_environment(
            BF16_HIGH_MEMORY_CPU_PROFILE,
            backend="cpu",
            mem_total_kb=16 * 1024 * 1024,
        )


def test_bf16_profile_accepts_30_gib_cpu():
    profile = validate_profile_environment(
        BF16_HIGH_MEMORY_CPU_PROFILE,
        backend="cpu",
        mem_total_kb=30 * 1024 * 1024,
    )
    assert profile.name == BF16_HIGH_MEMORY_CPU_PROFILE


def test_bf16_result_requires_three_gib_headroom():
    with pytest.raises(ProfilePreflightError, match="headroom"):
        validate_profile_result(
            BF16_HIGH_MEMORY_CPU_PROFILE,
            minimum_mem_available_kb=2 * 1024 * 1024,
        )
    validate_profile_result(
        BF16_HIGH_MEMORY_CPU_PROFILE,
        minimum_mem_available_kb=3 * 1024 * 1024,
    )


def test_unknown_profile_fails_closed():
    with pytest.raises(ProfilePreflightError, match="unsupported model profile"):
        validate_profile_environment(
            "bf16-auto-magic",
            backend="cpu",
            mem_total_kb=30 * 1024 * 1024,
        )


def test_build_kaggle_manifest_emits_bf16_diffusion_component(tmp_path: Path, monkeypatch):
    input_root = tmp_path / "input"
    output = tmp_path / "manifest.json"
    bf16 = input_root / "bf16" / "transformer" / "diffusion_pytorch_model.safetensors"
    qwen = input_root / "qwen" / "Qwen3VL-4B-Instruct-Q4_K_M.gguf"
    vae = input_root / "vae" / "diffusion_pytorch_model.safetensors"
    for path in (bf16, qwen, vae):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"placeholder")

    def fake_discover(root, *, filename, required_fragment, expected_sha256):
        del root, required_fragment, expected_sha256
        if filename == "Qwen3VL-4B-Instruct-Q4_K_M.gguf":
            return qwen.resolve()
        return vae.resolve()

    def fake_discover_by_sha(root, *, filename, expected_sha256):
        del root, filename
        assert expected_sha256 == BF16_TRANSFORMER_SHA256
        return bf16.resolve()

    monkeypatch.setattr(input_adapter, "discover_input", fake_discover)
    monkeypatch.setattr(input_adapter, "discover_input_by_sha", fake_discover_by_sha)
    input_adapter.build_kaggle_manifest(
        input_root=input_root,
        output=output,
        profile=BF16_HIGH_MEMORY_CPU_PROFILE,
    )

    data = json.loads(output.read_text(encoding="utf-8"))
    diffusion = data["components"]["diffusion"]
    assert diffusion["format"] == "safetensors"
    assert diffusion["sha256"] == BF16_TRANSFORMER_SHA256
    assert "quantization" not in diffusion
