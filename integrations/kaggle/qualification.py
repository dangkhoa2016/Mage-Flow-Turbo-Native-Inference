from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from mageflow_native.constants import (
    CANONICAL_CFG,
    CANONICAL_HEIGHT,
    CANONICAL_PROMPT,
    CANONICAL_SEED,
    CANONICAL_STEPS,
    CANONICAL_THREADS,
    CANONICAL_WIDTH,
)
from mageflow_native.models.manifest import load_manifest, verify_manifest
from mageflow_native.runtime.manager import RuntimeManager
from mageflow_native.runtime.spec import BackendSpec, RuntimeBuildBackend
from mageflow_native.telemetry import read_mem_available_kb
from integrations.kaggle.input_adapter import build_kaggle_manifest
from integrations.kaggle.profiles import (
    BF16_HIGH_MEMORY_CPU_PROFILE,
    Q8_REFERENCE_PROFILE,
    validate_profile_environment,
    validate_profile_result,
)
from integrations.kaggle.runtime_adapter import kaggle_cache_root, runtime_hint


def _source_head(repo_dir: Path) -> str:
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=str(repo_dir),
                capture_output=True,
                text=True,
                timeout=30,
            )
            .stdout.strip()
        )
    except Exception:
        return "unknown"


def _mem_total_kb() -> int:
    page_size = int(os.sysconf("SC_PAGE_SIZE"))
    phys_pages = int(os.sysconf("SC_PHYS_PAGES"))
    return (page_size * phys_pages) // 1024


def run_qualification(
    *,
    input_root: Path,
    work_root: Path,
    backend: str,
    profile: str = Q8_REFERENCE_PROFILE,
    repo_dir: Path | None = None,
) -> dict:
    mem_total_kb = _mem_total_kb()
    mem_available_before_kb = read_mem_available_kb()
    selected_profile = validate_profile_environment(
        profile,
        backend=backend,
        mem_total_kb=mem_total_kb,
    )

    manifest_path = build_kaggle_manifest(
        input_root=input_root,
        output=work_root / "output" / f"manifest-{profile}.json",
        profile=profile,
    )
    manifest_with_root = load_manifest(manifest_path, model_root=input_root)
    verified = verify_manifest(manifest_with_root)

    runtime_root = kaggle_cache_root()
    sd_cli_hint = runtime_hint(backend)
    manager = RuntimeManager(runtime_root, explicit_sd_cli=sd_cli_hint)

    if backend == "cuda0":
        if not sd_cli_hint:
            print("building pinned CUDA runtime from source...", flush=True)
            sd_cli = manager.build(RuntimeBuildBackend("cuda"))
        else:
            sd_cli = Path(sd_cli_hint)
    else:
        sd_cli = manager.resolve()

    identity = manager.verify(sd_cli, requested_backend=backend)

    from mageflow_native.inference.runner import run_generation

    backend_spec = BackendSpec(backend=backend)
    output_dir = work_root / "output"
    runs_dir = work_root / "output" / ".runs"
    output_dir.mkdir(parents=True, exist_ok=True)
    request_id = f"qual-{profile}-{backend}"

    print(f"SOURCE_HEAD={_source_head(repo_dir) if repo_dir else 'n/a'}", flush=True)
    print(f"QUALIFICATION_PROFILE={profile}", flush=True)
    print(f"QUALIFICATION_BACKEND={backend}", flush=True)
    print(f"MEM_TOTAL_KB={mem_total_kb}", flush=True)
    print(f"MEM_AVAILABLE_BEFORE_KB={mem_available_before_kb}", flush=True)

    result = run_generation(
        sd_cli,
        manifest_with_root,
        backend_spec,
        prompt=CANONICAL_PROMPT,
        seed=CANONICAL_SEED,
        width=CANONICAL_WIDTH,
        height=CANONICAL_HEIGHT,
        steps=CANONICAL_STEPS,
        cfg_scale=CANONICAL_CFG,
        threads=CANONICAL_THREADS,
        output_dir=output_dir,
        runs_dir=runs_dir,
        client_request_id=request_id,
        timeout_seconds=2700,
        collect_cuda=(backend == "cuda0"),
    )

    validate_profile_result(
        profile,
        minimum_mem_available_kb=result.minimum_mem_available_kb,
    )

    evidence = {
        "source_head": _source_head(repo_dir) if repo_dir else None,
        "runtime_commit": identity.pinned_commit,
        "runtime_version": identity.version_output,
        "devices": identity.devices_output,
        "profile": selected_profile.name,
        "backend": backend,
        "memory": {
            "mem_total_kb": mem_total_kb,
            "mem_available_before_kb": mem_available_before_kb,
            "minimum_mem_available_kb": result.minimum_mem_available_kb,
            "peak_sd_cli_rss_kb": result.peak_sd_cli_rss_kb,
        },
        "models": {
            "diffusion": {
                "filename": verified["diffusion"].name,
                "sha256": manifest_with_root.diffusion.sha256,
                "format": manifest_with_root.diffusion.format,
                "quantization": manifest_with_root.diffusion.quantization,
            },
            "text_encoder": {
                "filename": verified["text_encoder"].name,
                "sha256": manifest_with_root.text_encoder.sha256,
                "format": manifest_with_root.text_encoder.format,
                "quantization": manifest_with_root.text_encoder.quantization,
            },
            "vae": {
                "filename": verified["vae"].name,
                "sha256": manifest_with_root.vae.sha256,
                "format": manifest_with_root.vae.format,
                "quantization": manifest_with_root.vae.quantization,
            },
        },
        "artifact": {
            "filename": result.artifact.filename,
            "bytes": result.artifact.bytes,
            "sha256": result.artifact.sha256,
            "width": result.artifact.width,
            "height": result.artifact.height,
        },
        "elapsed_ms": result.elapsed_ms,
        "gpu_peak_mib": result.gpu_peak_mib,
    }
    evidence_path = work_root / "output" / f"qualification-{profile}-{backend}.json"
    evidence_path.write_text(
        json.dumps(evidence, indent=2) + "\n", encoding="utf-8"
    )
    return evidence


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mageflow-kaggle-qualification")
    parser.add_argument("--backend", choices=["cpu", "cuda0"], required=True)
    parser.add_argument(
        "--profile",
        choices=[Q8_REFERENCE_PROFILE, BF16_HIGH_MEMORY_CPU_PROFILE],
        default=Q8_REFERENCE_PROFILE,
    )
    parser.add_argument("--input-root", default="/kaggle/input")
    parser.add_argument("--work-root", default="/kaggle/working/mageflow-qualification")
    parser.add_argument("--repo-dir", default=None)
    args = parser.parse_args(argv)
    evidence = run_qualification(
        input_root=Path(args.input_root),
        work_root=Path(args.work_root),
        backend=args.backend,
        profile=args.profile,
        repo_dir=Path(args.repo_dir) if args.repo_dir else None,
    )
    print(json.dumps(evidence, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
