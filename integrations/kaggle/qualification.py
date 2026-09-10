from __future__ import annotations

import argparse
import json
import os
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
from mageflow_native.models.manifest import load_manifest, sha256_file, verify_manifest
from mageflow_native.runtime.manager import RuntimeManager
from mageflow_native.runtime.spec import BackendSpec
from mageflow_native.telemetry import read_mem_available_kb
from integrations.kaggle.input_adapter import build_kaggle_manifest
from integrations.kaggle.profiles import (
    BF16_SAFETENSORS_PROFILE,
    Q8_REFERENCE_PROFILE,
    validate_profile_environment,
    validate_profile_result,
)
from integrations.kaggle.qualification_matrix import (
    _source_identity,
    _validate_cuda_session_policy,
    _validate_runtime_cuda_devices,
    runtime_spec_for_backend,
)
from integrations.kaggle.runtime_adapter import kaggle_cache_root, runtime_hint


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
    physical_gpus = _validate_cuda_session_policy(backend)
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

    sd_cli_hint = runtime_hint(backend)
    if not sd_cli_hint:
        env_name = runtime_spec_for_backend(backend).env_var
        raise FileNotFoundError(
            f"prebuilt runtime is required for {backend}; set {env_name}"
        )
    sd_cli = Path(sd_cli_hint)
    runtime_spec = runtime_spec_for_backend(backend)
    runtime_sha256 = sha256_file(sd_cli)
    if runtime_sha256 != runtime_spec.sha256:
        raise ValueError(
            f"runtime sha256 mismatch for {backend}: "
            f"expected {runtime_spec.sha256}, got {runtime_sha256}"
        )
    manager = RuntimeManager(kaggle_cache_root(), explicit_sd_cli=str(sd_cli))
    identity = manager.verify(sd_cli, requested_backend=backend)
    _validate_runtime_cuda_devices(backend, identity.devices_output)

    from mageflow_native.inference.runner import run_generation

    backend_spec = BackendSpec(backend=backend)
    output_dir = work_root / "output"
    runs_dir = work_root / "output" / ".runs"
    output_dir.mkdir(parents=True, exist_ok=True)
    request_id = f"qual-{profile}-{backend}"
    source_head, source_tree = _source_identity(repo_dir)

    print(f"SOURCE_HEAD={source_head}", flush=True)
    print(f"SOURCE_TREE={source_tree}", flush=True)
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

    if backend == "cpu" and result.minimum_mem_available_kb is None:
        raise ValueError("CPU qualification requires minimum memory available telemetry")
    if backend == "cuda0" and (result.gpu_peak_mib is None or result.gpu_peak_mib <= 0):
        raise ValueError("cuda0 qualification requires positive gpu_peak_mib")

    validate_profile_result(
        profile,
        backend=backend,
        minimum_mem_available_kb=result.minimum_mem_available_kb,
    )

    evidence = {
        "schema_version": 2,
        "release_version": "1.0.0",
        "source_head": source_head,
        "source_tree": source_tree,
        "runtime": {
            "commit": identity.pinned_commit,
            "version": identity.version_output,
            "devices": identity.devices_output,
            "path": identity.path,
            "sha256": runtime_sha256,
        },
        "session": {
            "hostname": os.uname().nodename,
            "backend": backend,
            "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "physical_gpus": physical_gpus,
        },
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
        choices=[Q8_REFERENCE_PROFILE, BF16_SAFETENSORS_PROFILE],
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
