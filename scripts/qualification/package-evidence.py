from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tarfile
from pathlib import Path

_SECRET_PATTERNS = [
    re.compile(r"(ghp_|gho_|ghu_|github_pat_)[A-Za-z0-9_]+"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"(?i)authorization:\s*Bearer\s+\S+"),
    re.compile(r"(?i)(password|passwd|secret|token)\s*=\s*\S+"),
]
_MODEL_WEIGHT_SUFFIXES = {
    ".gguf",
    ".safetensors",
    ".ckpt",
    ".pt",
    ".pth",
    ".onnx",
    ".bin",
}


def _scan_secrets(root: Path) -> list[str]:
    findings = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() in _MODEL_WEIGHT_SUFFIXES or p.suffix == ".png":
            continue
        text = p.read_text(encoding="utf-8", errors="ignore")
        for pattern in _SECRET_PATTERNS:
            if pattern.search(text):
                findings.append(f"{p}: {pattern.pattern}")
    return findings


def _find_model_weights(root: Path) -> list[Path]:
    return sorted(
        p
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in _MODEL_WEIGHT_SUFFIXES
    )


def _member_is_safe(name: str) -> bool:
    path = Path(name)
    return not path.is_absolute() and ".." not in path.parts


def main() -> int:
    parser = argparse.ArgumentParser(prog="package-evidence")
    parser.add_argument("--evidence-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()

    src = Path(args.evidence_root).resolve()
    if not src.is_dir():
        print(f"EVIDENCE_ROOT=FAIL missing directory: {src}")
        return 4

    weights = _find_model_weights(src)
    if weights:
        print("MODEL_WEIGHT_SCAN=FAIL")
        for path in weights:
            print(f"  {path}")
        return 3
    print("MODEL_WEIGHT_SCAN=PASS")

    out = Path(args.output)
    out = out / f"mage-flow-turbo-native-inference-v1.0.0-{args.label}-evidence"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    shutil.copytree(src, out / "evidence", dirs_exist_ok=True)

    findings = _scan_secrets(out)
    if findings:
        print("SECRET_SCAN=FAIL")
        for finding in findings:
            print("  " + finding)
        shutil.rmtree(out, ignore_errors=True)
        return 2

    manifest = {}
    for p in sorted(out.rglob("*")):
        if p.is_file():
            digest = hashlib.sha256(p.read_bytes()).hexdigest()
            rel = str(p.relative_to(out))
            manifest[rel] = digest
    (out / "MANIFEST.sha256").write_text(
        "".join(f"{value}  {key}\n" for key, value in sorted(manifest.items())),
        encoding="utf-8",
    )

    archive = out.with_name(out.name + ".tar.gz")
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(out, arcname=out.name)

    with tarfile.open(archive, "r:gz") as tar:
        unsafe = [member.name for member in tar.getmembers() if not _member_is_safe(member.name)]
    if unsafe:
        print("ARCHIVE_PATH_SAFETY=FAIL")
        for name in unsafe:
            print(f"  {name}")
        archive.unlink(missing_ok=True)
        shutil.rmtree(out, ignore_errors=True)
        return 5

    sidecar = archive.with_name(archive.name + ".sha256")
    sidecar.write_text(
        f"{hashlib.sha256(archive.read_bytes()).hexdigest()}  {archive.name}\n",
        encoding="utf-8",
    )
    print("SECRET_SCAN=PASS")
    print("ARCHIVE_PATH_SAFETY=PASS")
    print(f"ARCHIVE={archive}")
    print(f"SIDE_CAR={sidecar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
