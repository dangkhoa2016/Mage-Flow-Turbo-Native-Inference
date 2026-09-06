from __future__ import annotations

from dataclasses import dataclass

from mageflow_native.constants import (
    DIT_FILENAME,
    DIT_SHA256,
    QWEN_FILENAME,
    QWEN_SHA256,
    VAE_FILENAME,
    VAE_SHA256,
)

Q8_REFERENCE_PROFILE = "q8-reference"
BF16_HIGH_MEMORY_CPU_PROFILE = "bf16-high-memory-cpu"
BF16_MIN_RAM_KB = 27 * 1024 * 1024
BF16_MIN_HEADROOM_KB = 3 * 1024 * 1024
BF16_TRANSFORMER_FILENAME = "diffusion_pytorch_model.safetensors"
BF16_TRANSFORMER_SHA256 = "6df47df3d7efc9ebdad075b87b3e9e4f74d09dca672d592271788f0ee27ab97d"
MAGE_PYTORCH_DEFAULT_FRAGMENT = "mage-flow-community-mage-flow-turbo/pytorch/default"


class ProfilePreflightError(RuntimeError):
    pass


@dataclass(frozen=True)
class ComponentProfile:
    filename: str
    sha256: str
    format: str
    quantization: str | None = None
    required_fragment: str | None = None


@dataclass(frozen=True)
class ModelProfile:
    name: str
    diffusion: ComponentProfile
    text_encoder: ComponentProfile
    vae: ComponentProfile
    allowed_backends: tuple[str, ...]
    min_ram_kb: int = 0


_QWEN = ComponentProfile(
    filename=QWEN_FILENAME,
    sha256=QWEN_SHA256,
    format="gguf",
    quantization="Q4_K_M",
    required_fragment="qwen-qwen3-vl-4b-instruct-gguf/gguf/q4-k-m",
)
_Q8_VAE = ComponentProfile(
    filename=VAE_FILENAME,
    sha256=VAE_SHA256,
    format="safetensors",
    required_fragment="mage-flow-community-mage-flow-turbo/pytorch/vae-only",
)
_BF16_VAE = ComponentProfile(
    filename=VAE_FILENAME,
    sha256=VAE_SHA256,
    format="safetensors",
    required_fragment=MAGE_PYTORCH_DEFAULT_FRAGMENT,
)

_PROFILES = {
    Q8_REFERENCE_PROFILE: ModelProfile(
        name=Q8_REFERENCE_PROFILE,
        diffusion=ComponentProfile(
            filename=DIT_FILENAME,
            sha256=DIT_SHA256,
            format="gguf",
            quantization="Q8_0",
            required_fragment="mage-flow-community-mage-flow-turbo/gguf/q8-0",
        ),
        text_encoder=_QWEN,
        vae=_Q8_VAE,
        allowed_backends=("cpu", "cuda0"),
    ),
    BF16_HIGH_MEMORY_CPU_PROFILE: ModelProfile(
        name=BF16_HIGH_MEMORY_CPU_PROFILE,
        diffusion=ComponentProfile(
            filename=BF16_TRANSFORMER_FILENAME,
            sha256=BF16_TRANSFORMER_SHA256,
            format="safetensors",
            quantization=None,
            required_fragment=MAGE_PYTORCH_DEFAULT_FRAGMENT,
        ),
        text_encoder=_QWEN,
        vae=_BF16_VAE,
        allowed_backends=("cpu",),
        min_ram_kb=BF16_MIN_RAM_KB,
    ),
}


def get_profile(name: str) -> ModelProfile:
    try:
        return _PROFILES[name]
    except KeyError as exc:
        raise ProfilePreflightError(f"unsupported model profile: {name}") from exc


def validate_profile_environment(
    name: str,
    *,
    backend: str,
    mem_total_kb: int,
) -> ModelProfile:
    profile = get_profile(name)
    if backend not in profile.allowed_backends:
        if profile.name == BF16_HIGH_MEMORY_CPU_PROFILE:
            raise ProfilePreflightError(
                "bf16-high-memory-cpu is CPU-only; no backend or Q8 fallback is permitted"
            )
        raise ProfilePreflightError(
            f"profile {profile.name} does not allow backend {backend}"
        )
    if mem_total_kb < profile.min_ram_kb:
        required_gib = profile.min_ram_kb // (1024 * 1024)
        raise ProfilePreflightError(
            f"profile {profile.name} requires at least {required_gib} GiB visible RAM; "
            f"detected {mem_total_kb / (1024 * 1024):.2f} GiB"
        )
    return profile


def validate_profile_result(
    name: str,
    *,
    minimum_mem_available_kb: int | None,
) -> None:
    profile = get_profile(name)
    if profile.name != BF16_HIGH_MEMORY_CPU_PROFILE:
        return
    if minimum_mem_available_kb is None:
        raise ProfilePreflightError(
            "bf16-high-memory-cpu requires measurable RAM headroom telemetry"
        )
    if minimum_mem_available_kb < BF16_MIN_HEADROOM_KB:
        raise ProfilePreflightError(
            "bf16-high-memory-cpu failed RAM headroom gate: "
            f"minimum available {minimum_mem_available_kb / (1024 * 1024):.2f} GiB < 3 GiB"
        )
