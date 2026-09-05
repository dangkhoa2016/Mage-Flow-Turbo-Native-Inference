from __future__ import annotations

import json
from pathlib import Path

EXPECTED_VERSION = "1.0.0"
EXPECTED_SDCPP = "6b3edaaf32cc19e5bb2d819c788bd557eddc8eba"
EXPECTED_TARGETS = [
    "linux_x86_64_cpu",
    "nvidia_cuda_cuda0",
    "kaggle_cpu_adapter",
    "kaggle_cuda0_adapter",
]
REQUIRED_DOCS = [
    "README.md",
    "README.vi.md",
    "CHANGELOG.md",
    "CHANGELOG.vi.md",
    "docs/RELEASE-NOTES-v1.0.0.md",
    "docs/RELEASE-NOTES-v1.0.0.vi.md",
]
FORBIDDEN_WORDING = [
    "qualification pending",
    "pending publication",
    "planned first public release",
    "not yet released",
    "đang chờ qualification",
    "đang chờ publish",
    "bản phát hành công khai đầu tiên dự kiến",
    "qualified release target",
    "qualified integration target",
    "qualified release targets",
    "release target đã qualification",
    "integration target đã qualification",
]


def _text(root: Path, rel: str) -> str:
    return (root / rel).read_text(encoding="utf-8")


def audit_release_contract(root: Path) -> list[str]:
    errors: list[str] = []

    for rel in REQUIRED_DOCS:
        path = root / rel
        if not path.is_file():
            errors.append(f"required release document missing: {rel}")

    if errors:
        return errors

    public_text = "\n".join(_text(root, rel).lower() for rel in REQUIRED_DOCS)
    for needle in FORBIDDEN_WORDING:
        if needle in public_text:
            errors.append(f"stale or premature release wording remains: {needle}")

    pyproject = _text(root, "pyproject.toml")
    if f'version = "{EXPECTED_VERSION}"' not in pyproject:
        errors.append("pyproject version is not 1.0.0")

    provenance_path = root / "runtime" / "RUNTIME-PROVENANCE.json"
    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"runtime provenance is unreadable: {exc}")
    else:
        if provenance.get("schema_version") != 2:
            errors.append("runtime provenance schema_version must be 2")
        if provenance.get("release_version") != EXPECTED_VERSION:
            errors.append("runtime provenance release_version mismatch")
        if provenance.get("commit") != EXPECTED_SDCPP:
            errors.append("runtime commit mismatch")
        if provenance.get("qualification_targets") != EXPECTED_TARGETS:
            errors.append("runtime qualification_targets mismatch")
        if "qualification_state" in provenance or "qualified_backends" in provenance:
            errors.append("runtime provenance contains mutable qualification state")

    readme = _text(root, "README.md")
    readme_vi = _text(root, "README.vi.md")
    if "[Tiếng Việt](README.vi.md)" not in readme:
        errors.append("README English language switch is missing")
    if "[English](README.md)" not in readme_vi:
        errors.append("README Vietnamese language switch is missing")

    changelog = _text(root, "CHANGELOG.md")
    changelog_vi = _text(root, "CHANGELOG.vi.md")
    if "[Tiếng Việt](CHANGELOG.vi.md)" not in changelog:
        errors.append("CHANGELOG English language switch is missing")
    if "[English](CHANGELOG.md)" not in changelog_vi:
        errors.append("CHANGELOG Vietnamese language switch is missing")

    ci = _text(root, ".github/workflows/ci.yml")
    if "fetch-depth: 0" not in ci:
        errors.append("CI must check out full history with fetch-depth: 0")
    if "verify_history.py" not in ci:
        errors.append("CI does not enforce public history invariants")
    if "audit_release_contract.py" not in ci:
        errors.append("CI does not enforce release contract audit")

    native = _text(root, ".github/workflows/native-runtime.yml")
    if "tags:" not in native or "'v*'" not in native:
        errors.append("Native Runtime workflow does not run for v* tags")

    return errors


def main() -> int:
    errors = audit_release_contract(Path("."))
    if errors:
        print("RELEASE_CONTRACT_AUDIT=FAIL")
        for error in errors:
            print("  - " + error)
        return 1
    print("RELEASE_CONTRACT_AUDIT=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
