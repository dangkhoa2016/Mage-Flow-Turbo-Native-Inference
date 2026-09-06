from __future__ import annotations

import json
from pathlib import Path

from mageflow_native.models.manifest import sha256_file
from integrations.kaggle.profiles import (
    Q8_REFERENCE_PROFILE,
    ComponentProfile,
    get_profile,
)


class InputResolutionError(RuntimeError):
    pass


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


def build_kaggle_manifest(
    input_root: Path,
    output: str | Path,
    *,
    profile: str = Q8_REFERENCE_PROFILE,
) -> Path:
    input_root = Path(input_root)
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
