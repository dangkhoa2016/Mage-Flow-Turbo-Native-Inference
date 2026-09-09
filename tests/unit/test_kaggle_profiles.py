import json
from pathlib import Path

import pytest

from integrations.kaggle import input_adapter
from integrations.kaggle.input_adapter import (
    InputAttachmentPolicyError,
    detect_attached_diffusion_families,
    validate_kaggle_input_attachment_policy,
)
from integrations.kaggle.profiles import (
    BF16_MIN_HEADROOM_KB,
    BF16_MIN_RAM_KB,
    BF16_SAFETENSORS_PROFILE,
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


def test_bf16_safetensors_profile_uses_exact_transformer_identity():
    profile = get_profile(BF16_SAFETENSORS_PROFILE)
    assert profile.name == "bf16-safetensors"
    assert profile.diffusion.filename == "diffusion_pytorch_model.safetensors"
    assert profile.diffusion.sha256 == BF16_TRANSFORMER_SHA256
    assert profile.diffusion.sha256 == "6df47df3d7efc9ebdad075b87b3e9e4f74d09dca672d592271788f0ee27ab97d"
    assert profile.diffusion.format == "safetensors"
    assert profile.diffusion.quantization is None
    assert profile.allowed_backends == ("cpu", "cuda0")
    assert BF16_MIN_RAM_KB == 27 * 1024 * 1024
    assert BF16_MIN_HEADROOM_KB == 3 * 1024 * 1024
    assert profile.min_ram_kb == BF16_MIN_RAM_KB


def test_bf16_profile_uses_single_pytorch_default_mirror_for_transformer_and_vae():
    bf16 = get_profile(BF16_SAFETENSORS_PROFILE)
    q8 = get_profile(Q8_REFERENCE_PROFILE)
    default_fragment = "mage-flow-community-mage-flow-turbo/pytorch/default"
    assert bf16.diffusion.required_fragment == default_fragment
    assert bf16.vae.required_fragment == default_fragment
    assert q8.vae.required_fragment == "mage-flow-community-mage-flow-turbo/pytorch/vae-only"


def test_bf16_safetensors_profile_supports_cpu_and_cuda0():
    profile = get_profile(BF16_SAFETENSORS_PROFILE)
    assert profile.allowed_backends == ("cpu", "cuda0")


def test_bf16_cuda_does_not_apply_cpu_minimum_ram_gate():
    profile = validate_profile_environment(
        BF16_SAFETENSORS_PROFILE,
        backend="cuda0",
        mem_total_kb=16 * 1024 * 1024,
    )
    assert profile.name == BF16_SAFETENSORS_PROFILE


def test_bf16_cpu_rejects_16_gib_ram():
    with pytest.raises(ProfilePreflightError, match="27 GiB"):
        validate_profile_environment(
            BF16_SAFETENSORS_PROFILE,
            backend="cpu",
            mem_total_kb=16 * 1024 * 1024,
        )


def test_bf16_cpu_accepts_30_gib():
    profile = validate_profile_environment(
        BF16_SAFETENSORS_PROFILE,
        backend="cpu",
        mem_total_kb=30 * 1024 * 1024,
    )
    assert profile.name == BF16_SAFETENSORS_PROFILE


def test_bf16_cpu_result_requires_three_gib_headroom():
    with pytest.raises(ProfilePreflightError, match="headroom"):
        validate_profile_result(
            BF16_SAFETENSORS_PROFILE,
            backend="cpu",
            minimum_mem_available_kb=2 * 1024 * 1024,
        )
    validate_profile_result(
        BF16_SAFETENSORS_PROFILE,
        backend="cpu",
        minimum_mem_available_kb=3 * 1024 * 1024,
    )


def test_bf16_cuda_result_does_not_require_cpu_headroom_gate():
    validate_profile_result(
        BF16_SAFETENSORS_PROFILE,
        backend="cuda0",
        minimum_mem_available_kb=None,
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
        profile=BF16_SAFETENSORS_PROFILE,
    )

    data = json.loads(output.read_text(encoding="utf-8"))
    diffusion = data["components"]["diffusion"]
    assert diffusion["format"] == "safetensors"
    assert diffusion["sha256"] == BF16_TRANSFORMER_SHA256
    assert "quantization" not in diffusion


def _write_input_file(path: Path, data: bytes = b"placeholder") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _mixed_diffusion_tree(tmp_path):
    input_root = tmp_path / "input"
    _write_input_file(
        input_root
        / "mage-flow-community-mage-flow-turbo"
        / "gguf"
        / "q8-0"
        / "Mage-Flow-Turbo-DiT-Q8_0.gguf"
    )
    _write_input_file(
        input_root
        / "mage-flow-community-mage-flow-turbo"
        / "pytorch"
        / "default"
        / "transformer"
        / "diffusion_pytorch_model.safetensors"
    )
    return input_root


def test_policy_normal_mode_rejects_both_mage_diffusion_families(tmp_path):
    input_root = _mixed_diffusion_tree(tmp_path)
    families = detect_attached_diffusion_families(input_root)
    assert families == (Q8_REFERENCE_PROFILE, BF16_SAFETENSORS_PROFILE)
    with pytest.raises(InputAttachmentPolicyError) as exc_info:
        validate_kaggle_input_attachment_policy(input_root)
    msg = str(exc_info.value).lower()
    for fact in (
        "normal Kaggle inference",
        "both Mage-Flow-Turbo diffusion families detected",
        "q8-reference",
        "GGUF",
        "bf16-safetensors",
        "PyTorch/Transformers SafeTensors",
        "detach one family",
        "restart the Kaggle session",
        "controlled benchmark",
    ):
        assert fact.lower() in msg, f"policy error must state: {fact!r}"


def test_policy_q8_plus_vae_only_safetensors_is_valid(tmp_path):
    input_root = tmp_path / "input"
    _write_input_file(
        input_root
        / "mage-flow-community-mage-flow-turbo"
        / "gguf"
        / "q8-0"
        / "Mage-Flow-Turbo-DiT-Q8_0.gguf"
    )
    _write_input_file(
        input_root
        / "mage-flow-community-mage-flow-turbo"
        / "pytorch"
        / "vae-only"
        / "diffusion_pytorch_model.safetensors"
    )
    families = validate_kaggle_input_attachment_policy(input_root)
    assert families == (Q8_REFERENCE_PROFILE,)


def test_policy_bf16_only_input_is_valid(tmp_path):
    input_root = tmp_path / "input"
    _write_input_file(
        input_root
        / "mage-flow-community-mage-flow-turbo"
        / "pytorch"
        / "default"
        / "transformer"
        / "diffusion_pytorch_model.safetensors"
    )
    families = validate_kaggle_input_attachment_policy(input_root)
    assert families == (BF16_SAFETENSORS_PROFILE,)


def test_policy_benchmark_override_permits_both_families(tmp_path):
    input_root = _mixed_diffusion_tree(tmp_path)
    detected = detect_attached_diffusion_families(input_root)
    assert detected == (Q8_REFERENCE_PROFILE, BF16_SAFETENSORS_PROFILE)
    families = validate_kaggle_input_attachment_policy(
        input_root,
        allow_mixed_diffusion_families=True,
    )
    assert families == (Q8_REFERENCE_PROFILE, BF16_SAFETENSORS_PROFILE)


def test_policy_empty_input_root_returns_no_families(tmp_path):
    assert validate_kaggle_input_attachment_policy(tmp_path / "input") == ()
