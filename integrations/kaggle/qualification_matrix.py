from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from mageflow_native.constants import (
    CANONICAL_CFG,
    CANONICAL_PROMPT,
    CANONICAL_SEED,
    CANONICAL_STEPS,
    CANONICAL_THREADS,
)
from mageflow_native.inference.runner import run_generation
from mageflow_native.models.manifest import load_manifest, sha256_file, verify_manifest
from mageflow_native.runtime.manager import RuntimeManager
from mageflow_native.runtime.spec import BackendSpec
from mageflow_native.telemetry import read_mem_available_kb
from integrations.kaggle.input_adapter import build_kaggle_manifest
from integrations.kaggle.profiles import (
    BF16_HIGH_MEMORY_CPU_PROFILE,
    Q8_REFERENCE_PROFILE,
    ProfilePreflightError,
    validate_profile_environment,
    validate_profile_result,
)
from integrations.kaggle.runtime_adapter import kaggle_cache_root

MATRIX_RESOLUTIONS = [512, 640, 768, 1024]
BF16_CPU_RUNTIME_ENV = "MAGE_CPU_PREBUILT_SD_CLI"
BF16_CPU_RUNTIME_SHA256 = (
    "7539d90b99eaf2b6279eec4f9006a68ae53e87bfe0c9c325ff3f329220468a5c"
)


def _mem_total_kb() -> int:
    page_size = int(os.sysconf("SC_PAGE_SIZE"))
    phys_pages = int(os.sysconf("SC_PHYS_PAGES"))
    return (page_size * phys_pages) // 1024


def _source_head(repo_dir: Path | None) -> str:
    if repo_dir is None:
        return "unknown"
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


def _elapsed_ms(start: float) -> int:
    return int((time.monotonic() - start) * 1000)


def parse_resolutions(text: str) -> list[int]:
    if not text or not text.strip():
        raise ValueError("resolution list must not be empty")
    parsed: list[int] = []
    for raw in text.split(","):
        token = raw.strip()
        if not token or not token.isdigit():
            raise ValueError(f"resolution must be a positive integer, got {token!r}")
        value = int(token)
        if value <= 0:
            raise ValueError(f"resolution must be a positive integer, got {value}")
        if value in parsed:
            raise ValueError(f"duplicate resolution: {value}")
        parsed.append(value)
    return parsed


def validate_resolutions(resolutions: list[int]) -> list[int]:
    if not resolutions:
        raise ValueError("resolution list must not be empty")
    if any(not isinstance(value, int) or value <= 0 for value in resolutions):
        raise ValueError("all resolutions must be positive integers")
    if len(set(resolutions)) != len(resolutions):
        raise ValueError("resolution list contains duplicates")
    return list(resolutions)


def resolve_prebuilt_runtime() -> Path:
    hint = os.environ.get(BF16_CPU_RUNTIME_ENV)
    if not hint:
        raise FileNotFoundError(
            f"{BF16_CPU_RUNTIME_ENV} is not set; a prebuilt sd-cli is required"
        )
    path = Path(hint).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"prebuilt sd-cli is not a file: {path}")
    if not os.access(path, os.X_OK):
        raise PermissionError(f"prebuilt sd-cli is not executable: {path}")
    return path.resolve()


def _models_evidence(verified: dict, manifest) -> dict:
    return {
        "diffusion": {
            "filename": verified["diffusion"].name,
            "sha256": manifest.diffusion.sha256,
            "format": manifest.diffusion.format,
            "quantization": manifest.diffusion.quantization,
        },
        "text_encoder": {
            "filename": verified["text_encoder"].name,
            "sha256": manifest.text_encoder.sha256,
            "format": manifest.text_encoder.format,
            "quantization": manifest.text_encoder.quantization,
        },
        "vae": {
            "filename": verified["vae"].name,
            "sha256": manifest.vae.sha256,
            "format": manifest.vae.format,
            "quantization": manifest.vae.quantization,
        },
    }


def run_matrix(
    *,
    input_root: Path,
    work_root: Path,
    backend: str,
    profile: str = BF16_HIGH_MEMORY_CPU_PROFILE,
    repo_dir: Path | None = None,
    resolutions: list[int] | None = None,
    timeout_seconds: int = 2700,
) -> tuple[int, dict]:
    if resolutions is None:
        resolutions = list(MATRIX_RESOLUTIONS)
    resolutions = validate_resolutions(resolutions)

    wall_start = time.monotonic()
    timing: dict[str, int] = {}
    records: list[dict] = []
    completed: list[int] = []
    failed: dict | None = None
    error: dict | None = None
    source_head: str | None = "unknown"
    matrix_id: str | None = None
    runtime_evidence: dict = {}
    models_evidence: dict = {}
    manifest_evidence: dict = {}
    mem_total_kb: int | None = None
    mem_available_before_kb: int | None = None
    matrix_start: float | None = None
    output_dir = work_root / "output"

    try:
        mem_total_kb = _mem_total_kb()
        mem_available_before_kb = read_mem_available_kb()
        validate_profile_environment(profile, backend=backend, mem_total_kb=mem_total_kb)

        runs_dir = output_dir / ".matrix-runs"
        output_dir.mkdir(parents=True, exist_ok=True)

        build_start = time.monotonic()
        manifest_path = build_kaggle_manifest(
            input_root=input_root,
            output=output_dir / f"manifest-{profile}.json",
            profile=profile,
        )
        timing["manifest_build_elapsed_ms"] = _elapsed_ms(build_start)

        verify_start = time.monotonic()
        manifest = load_manifest(manifest_path, model_root=input_root)
        verified = verify_manifest(manifest)
        timing["manifest_verify_elapsed_ms"] = _elapsed_ms(verify_start)

        runtime_start = time.monotonic()
        sd_cli = resolve_prebuilt_runtime()
        runtime_sha256 = sha256_file(sd_cli)
        if runtime_sha256 != BF16_CPU_RUNTIME_SHA256:
            raise ValueError(
                "runtime sha256 mismatch: "
                f"expected {BF16_CPU_RUNTIME_SHA256}, got {runtime_sha256}"
            )
        manager = RuntimeManager(kaggle_cache_root(), explicit_sd_cli=str(sd_cli))
        identity = manager.verify(sd_cli, requested_backend=backend)
        timing["runtime_verify_elapsed_ms"] = _elapsed_ms(runtime_start)

        source_head = _source_head(repo_dir)
        matrix_id = f"matrix-{profile}-{backend}-{int(time.time())}"
        timing["setup_elapsed_ms"] = _elapsed_ms(wall_start)

        runtime_evidence = {
            "commit": identity.pinned_commit,
            "version": identity.version_output,
            "devices": identity.devices_output,
            "path": identity.path,
            "sha256": runtime_sha256,
        }
        manifest_evidence = {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
        }
        models_evidence = _models_evidence(verified, manifest)

        context = {
            "source_head": source_head,
            "profile": profile,
            "backend": backend,
            "matrix_id": matrix_id,
            "runtime": runtime_evidence,
            "manifest": manifest_evidence,
            "models": models_evidence,
            "mem_total_kb": mem_total_kb,
            "mem_available_before_kb": mem_available_before_kb,
        }

        print(f"MATRIX_SOURCE_HEAD={source_head}", flush=True)
        print(f"MATRIX_PROFILE={profile}", flush=True)
        print(f"MATRIX_BACKEND={backend}", flush=True)
        print(f"MATRIX_ID={matrix_id}", flush=True)
        print(
            f"MATRIX_RESOLUTIONS={','.join(str(r) for r in resolutions)}",
            flush=True,
        )

        matrix_start = time.monotonic()
        for resolution in resolutions:
            width = height = resolution
            request_id = f"qual-{profile}-{backend}-{resolution:04d}"
            print(f"MATRIX_GENERATION {resolution} START", flush=True)
            try:
                result = run_generation(
                    sd_cli,
                    manifest,
                    BackendSpec(backend=backend),
                    prompt=CANONICAL_PROMPT,
                    seed=CANONICAL_SEED,
                    width=width,
                    height=height,
                    steps=CANONICAL_STEPS,
                    cfg_scale=CANONICAL_CFG,
                    threads=CANONICAL_THREADS,
                    output_dir=output_dir,
                    runs_dir=runs_dir,
                    client_request_id=request_id,
                    timeout_seconds=timeout_seconds,
                    collect_cuda=False,
                )
            except Exception as exc:
                error = {
                    "phase": "generation",
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
                failed = {"resolution": resolution}
                records.append(
                    _write_record(
                        context, output_dir, resolution, status="failed", error=error
                    )
                )
                print(f"MATRIX_GENERATION {resolution} FAIL {error['type']}", flush=True)
                break

            if result.minimum_mem_available_kb is None:
                error = {
                    "phase": "telemetry",
                    "type": "MissingTelemetryError",
                    "message": "minimum memory available telemetry is missing",
                }
                failed = {"resolution": resolution}
                records.append(
                    _write_record(
                        context,
                        output_dir,
                        resolution,
                        status="failed",
                        error=error,
                        result=result,
                    )
                )
                print(f"MATRIX_GENERATION {resolution} FAIL telemetry", flush=True)
                break

            try:
                validate_profile_result(
                    profile,
                    minimum_mem_available_kb=result.minimum_mem_available_kb,
                )
            except ProfilePreflightError as exc:
                error = {
                    "phase": "headroom",
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
                failed = {"resolution": resolution}
                records.append(
                    _write_record(
                        context,
                        output_dir,
                        resolution,
                        status="failed",
                        error=error,
                        result=result,
                    )
                )
                print(f"MATRIX_GENERATION {resolution} FAIL headroom", flush=True)
                break

            completed.append(resolution)
            records.append(
                _write_record(
                    context,
                    output_dir,
                    resolution,
                    status="passed",
                    result=result,
                )
            )
            print(
                f"MATRIX_GENERATION {resolution} PASS "
                f"elapsed_ms={result.elapsed_ms} "
                f"min_available_kb={result.minimum_mem_available_kb}",
                flush=True,
            )
    except Exception as exc:
        error = {
            "phase": "setup",
            "type": type(exc).__name__,
            "message": str(exc),
        }

    if matrix_start is not None:
        timing["matrix_generation_elapsed_ms"] = _elapsed_ms(matrix_start)
    timing["matrix_wall_elapsed_ms"] = _elapsed_ms(wall_start)

    aggregate = {
        "status": "passed" if error is None else "failed",
        "source_head": source_head,
        "profile": profile,
        "backend": backend,
        "matrix_id": matrix_id,
        "matrix_resolutions": list(resolutions),
        "completed_resolutions": completed,
        "failed_resolution": failed,
        "error": error,
        "runtime": runtime_evidence,
        "models": models_evidence,
        "setup": {
            "mem_total_kb": mem_total_kb,
            "mem_available_before_kb": mem_available_before_kb,
            **timing,
        },
        "matrix": records,
    }
    _write_aggregate(output_dir, aggregate, profile=profile, backend=backend)
    return (0 if error is None else 1, aggregate)


def _write_record(
    context: dict,
    output_dir: Path,
    resolution: int,
    *,
    status: str,
    error: dict | None = None,
    result=None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "status": status,
        "source_head": context["source_head"],
        "profile": context["profile"],
        "backend": context["backend"],
        "matrix_id": context["matrix_id"],
        "resolution": {"width": resolution, "height": resolution},
        "request": {
            "prompt": CANONICAL_PROMPT,
            "seed": CANONICAL_SEED,
            "steps": CANONICAL_STEPS,
            "cfg": CANONICAL_CFG,
            "threads": CANONICAL_THREADS,
        },
        "memory": {
            "mem_total_kb": context["mem_total_kb"],
            "mem_available_before_run_kb": context["mem_available_before_kb"],
            "minimum_mem_available_kb": (
                result.minimum_mem_available_kb if result is not None else None
            ),
            "peak_sd_cli_rss_kb": (
                result.peak_sd_cli_rss_kb if result is not None else None
            ),
        },
        "elapsed_ms": result.elapsed_ms if result is not None else None,
        "gpu_peak_mib": result.gpu_peak_mib if result is not None else None,
        "artifact": (
            {
                "filename": result.artifact.filename,
                "bytes": result.artifact.bytes,
                "sha256": result.artifact.sha256,
                "width": result.artifact.width,
                "height": result.artifact.height,
            }
            if result is not None
            else {}
        ),
        "runtime": context["runtime"],
        "manifest": context["manifest"],
        "models": context["models"],
    }
    if error is not None:
        record["error"] = error
    path = output_dir / f"qualification-{context['profile']}-{context['backend']}-{resolution:04d}.json"
    path.write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return record


def _write_aggregate(output_dir: Path, aggregate: dict, *, profile: str, backend: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"qualification-matrix-{profile}-{backend}.json"
    path.write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mageflow-kaggle-qualification-matrix")
    parser.add_argument("--backend", choices=["cpu", "cuda0"], required=True)
    parser.add_argument(
        "--profile",
        choices=[Q8_REFERENCE_PROFILE, BF16_HIGH_MEMORY_CPU_PROFILE],
        default=BF16_HIGH_MEMORY_CPU_PROFILE,
    )
    parser.add_argument("--input-root", default="/kaggle/input")
    parser.add_argument("--work-root", default="/kaggle/working/mageflow-bf16-matrix")
    parser.add_argument("--repo-dir", default=None)
    parser.add_argument("--resolutions", default=None)
    args = parser.parse_args(argv)
    if args.resolutions is None:
        resolutions = list(MATRIX_RESOLUTIONS)
    else:
        resolutions = parse_resolutions(args.resolutions)
    code, aggregate = run_matrix(
        input_root=Path(args.input_root),
        work_root=Path(args.work_root),
        backend=args.backend,
        profile=args.profile,
        repo_dir=Path(args.repo_dir) if args.repo_dir else None,
        resolutions=resolutions,
    )
    print(json.dumps(aggregate, indent=2, ensure_ascii=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())