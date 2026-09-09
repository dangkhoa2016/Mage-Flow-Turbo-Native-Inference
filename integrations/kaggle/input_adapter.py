from __future__ import annotations

import json
from pathlib import Path

from mageflow_native.models.manifest import sha256_file
from integrations.kaggle.profiles import (
    BF16_HIGH_MEMORY_CPU_PROFILE,
    Q8_REFERENCE_PROFILE,
    ComponentProfile,
    get_profile,
)


class InputResolutionError(RuntimeError):
    pass


class InputAttachmentPolicyError(InputResolutionError):
    pass


_MAGE_DIFFUSION_FAMILY_LABELS = {
    Q8_REFERENCE_PROFILE: "GGUF q8-0",
    BF16_HIGH_MEMORY_CPU_PROFILE: "PyTorch/Transformers SafeTensors default",
}


def _norm(p: Path) -> str:
    return p.as_posix().lower()


def discover_input(
    root: Path,
    *,
    filename: str,
    required_fragment: str,
    expected_sha256: str,
) -> Path:
    root = Path(root)
    required = required_fragment.lower().strip("/")
    candidates = []
    for p in root.rglob(filename):
        if required in _norm(p):
            candidates.append(p)
    if len(candidates) == 1:
        matches = candidates
    else:
        matches = [p for p in candidates if sha256_file(p) == expected_sha256]
        if len(matches) != 1:
            raise InputResolutionError(
                f"expected exactly one {filename} under *{required_fragment}* "
                f"with SHA256 {expected_sha256}, found {len(matches)} of "
                f"{len(candidates)} candidates"
            )
    p = matches[0].resolve()
    digest = sha256_file(p)
    if digest != expected_sha256:
        raise InputResolutionError(f"SHA256 mismatch for {p.name}: {digest}")
    return p


def discover_input_by_sha(
    root: Path,
    *,
    filename: str,
    expected_sha256: str,
) -> Path:
    root = Path(root)
    matches: list[Path] = []
    for candidate in root.rglob(filename):
        if sha256_file(candidate) == expected_sha256:
            matches.append(candidate.resolve())
    if len(matches) != 1:
        raise InputResolutionError(
            f"expected exactly one {filename} with SHA256 {expected_sha256}, found {len(matches)}"
        )
    return matches[0]


def _discover_component(input_root: Path, component: ComponentProfile) -> Path:
    if component.required_fragment:
        return discover_input(
            input_root,
            filename=component.filename,
            required_fragment=component.required_fragment,
            expected_sha256=component.sha256,
        )
    return discover_input_by_sha(
        input_root,
        filename=component.filename,
        expected_sha256=component.sha256,
    )


def _relative_or_abs(path: Path, base: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(path.resolve())


def _manifest_component(path: Path, input_root: Path, component: ComponentProfile) -> dict:
    data = {
        "path": _relative_or_abs(path, input_root),
        "sha256": component.sha256,
        "format": component.format,
    }
    if component.quantization is not None:
        data["quantization"] = component.quantization
    return data


def detect_attached_diffusion_families(input_root: Path) -> tuple[str, ...]:
    input_root = Path(input_root)
    detected: list[str] = []
    for profile_name in (Q8_REFERENCE_PROFILE, BF16_HIGH_MEMORY_CPU_PROFILE):
        profile = get_profile(profile_name)
        fragment = (profile.diffusion.required_fragment or "").strip("/").lower()
        for candidate in input_root.rglob(profile.diffusion.filename):
            if fragment and fragment in _norm(candidate):
                detected.append(profile_name)
                break
    return tuple(detected)


def validate_kaggle_input_attachment_policy(
    input_root: Path,
    *,
    allow_mixed_diffusion_families: bool = False,
) -> tuple[str, ...]:
    families = detect_attached_diffusion_families(input_root)
    if len(families) > 1 and not allow_mixed_diffusion_families:
        described = ", ".join(
            f"{name} ({_MAGE_DIFFUSION_FAMILY_LABELS[name]})" for name in families
        )
        raise InputAttachmentPolicyError(
            "normal Kaggle inference requires exactly one Mage-Flow-Turbo "
            "diffusion family; both Mage-Flow-Turbo diffusion families "
            f"detected: {described}. Detach one family, then restart the "
            "Kaggle session. The controlled benchmark is the only "
            "mixed-family exception."
        )
    return families


def build_kaggle_manifest(
    input_root: Path,
    output: str | Path,
    *,
    profile: str = Q8_REFERENCE_PROFILE,
    allow_mixed_diffusion_families: bool = False,
) -> Path:
    input_root = Path(input_root)
    validate_kaggle_input_attachment_policy(
        input_root,
        allow_mixed_diffusion_families=allow_mixed_diffusion_families,
    )
    selected = get_profile(profile)
    dit = _discover_component(input_root, selected.diffusion)
    qwen = _discover_component(input_root, selected.text_encoder)
    vae = _discover_component(input_root, selected.vae)
    manifest = {
        "schema_version": 1,
        "model_family": "Mage-Flow-Turbo",
        "components": {
            "diffusion": _manifest_component(dit, input_root, selected.diffusion),
            "text_encoder": _manifest_component(qwen, input_root, selected.text_encoder),
            "vae": _manifest_component(vae, input_root, selected.vae),
        },
    }
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return output_path
