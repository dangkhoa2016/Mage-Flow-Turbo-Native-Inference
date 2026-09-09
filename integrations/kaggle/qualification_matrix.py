from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from dataclasses import dataclass
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
    BF16_SAFETENSORS_PROFILE,
    Q8_REFERENCE_PROFILE,
    ProfilePreflightError,
    validate_profile_environment,
    validate_profile_result,
)
from integrations.kaggle.runtime_adapter import kaggle_cache_root

MATRIX_RESOLUTIONS = [512, 640, 768, 1024]
SUPPORTED_MATRIX_BACKENDS = ("cpu", "cuda0")


@dataclass(frozen=True)
class RuntimeSpec:
    env_var: str
    sha256: str


_RUNTIME_SPECS = {
    "cpu": RuntimeSpec(
        env_var="MAGE_CPU_PREBUILT_SD_CLI",
        sha256="7539d90b99eaf2b6279eec4f9006a68ae53e87bfe0c9c325ff3f329220468a5c",
    ),
    "cuda0": RuntimeSpec(
        env_var="MAGE_CUDA_PREBUILT_SD_CLI",
        sha256="3fae6c1991ad0ac764c36495f688817c8a3d295d7651369bf74b7fd33743c3d0",
    ),
}


def runtime_spec_for_backend(backend: str) -> RuntimeSpec:
    try:
        return _RUNTIME_SPECS[backend]
    except KeyError as exc:
        raise ValueError(f"unsupported matrix backend: {backend!r}") from exc


def _mem_total_kb() -> int:
    page_size = int(os.sysconf("SC_PAGE_SIZE"))
    phys_pages = int(os.sysconf("SC_PHYS_PAGES"))
    return (page_size * phys_pages) // 1024


def _git_rev(repo_dir: Path, expression: str) -> str:
    return subprocess.run(
        ["git", "rev-parse", expression],
        cwd=str(repo_dir),
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    ).stdout.strip()


def _source_identity(repo_dir: Path | None) -> tuple[str, str]:
    if repo_dir is None:
        return "unknown", "unknown"
    try:
        return _git_rev(repo_dir, "HEAD"), _git_rev(repo_dir, "HEAD^{tree}")
    except Exception:
        return "unknown", "unknown"


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


def resolve_prebuilt_runtime(backend: str) -> Path:
    spec = runtime_spec_for_backend(backend)
    hint = os.environ.get(spec.env_var)
    if not hint:
        raise FileNotFoundError(
            f"{spec.env_var} is not set; a prebuilt sd-cli is required for {backend}"
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


def _validate_cuda_session_policy(backend: str) -> None:
    if backend != "cuda0":
        return
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise ValueError("cuda0 qualification requires CUDA_VISIBLE_DEVICES=0")


def run_matrix(
    *,
    input_root: Path,
    work_root: Path,
    backend: str,
    profile: str = BF16_SAFETENSORS_PROFILE,
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
    source_head = "unknown"
    source_tree = "unknown"
    matrix_id: str | None = None
    runtime_evidence: dict = {}
    models_evidence: dict = {}
    manifest_evidence: dict = {}
    mem_total_kb: int | None = None
    mem_available_before_kb: int | None = None
    matrix_start: float | None = None
    output_dir = work_root / "output"
    session_evidence = {
        "hostname": os.uname().nodename,
        "backend": backend,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }

    try:
        if backend not in SUPPORTED_MATRIX_BACKENDS:
            raise ValueError(f"unsupported matrix backend: {backend!r}")
        _validate_cuda_session_policy(backend)

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
        runtime_spec = runtime_spec_for_backend(backend)
        sd_cli = resolve_prebuilt_runtime(backend)
        runtime_sha256 = sha256_file(sd_cli)
        if runtime_sha256 != runtime_spec.sha256:
            raise ValueError(
                f"runtime sha256 mismatch for {backend}: "
                f"expected {runtime_spec.sha256}, got {runtime_sha256}"
            )
        manager = RuntimeManager(kaggle_cache_root(), explicit_sd_cli=str(sd_cli))
        identity = manager.verify(sd_cli, requested_backend=backend)
        timing["runtime_verify_elapsed_ms"] = _elapsed_ms(runtime_start)

        source_head, source_tree = _source_identity(repo_dir)
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
            "schema_version": 2,
            "release_version": "1.0.0",
            "source_head": source_head,
            "source_tree": source_tree,
            "profile": profile,
            "backend": backend,
            "matrix_id": matrix_id,
            "session": session_evidence,
            "runtime": runtime_evidence,
            "manifest": manifest_evidence,
            "models": models_evidence,
            "mem_total_kb": mem_total_kb,
            "mem_available_before_kb": mem_available_before_kb,
        }

        print(f"MATRIX_SOURCE_HEAD={source_head}", flush=True)
        print(f"MATRIX_SOURCE_TREE={source_tree}", flush=True)
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

            mem_available_before_run_kb = read_mem_available_kb()
            if mem_available_before_run_kb is None:
                error = {
                    "phase": "telemetry",
                    "type": "MissingTelemetryError",
                    "message": "per-run memory available telemetry (before) is missing",
                }
                failed = {"resolution": resolution}
                records.append(
                    _write_record(
                        context,
                        output_dir,
                        resolution,
                        status="failed",
                        error=error,
                        mem_available_before_run_kb=None,
                        mem_available_after_run_kb=None,
                    )
                )
                print(f"MATRIX_GENERATION {resolution} FAIL telemetry", flush=True)
                break

            print(f"MATRIX_GENERATION {resolution} START", flush=True)
            result = None
            error = None
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
                    collect_cuda=(backend == "cuda0"),
                )
            except Exception as exc:
                error = {
                    "phase": "generation",
                    "type": type(exc).__name__,
                    "message": str(exc),
                }

            mem_available_after_run_kb = read_mem_available_kb()

            if error is None and backend == "cpu" and result.minimum_mem_available_kb is None:
                error = {
                    "phase": "telemetry",
                    "type": "MissingTelemetryError",
                    "message": "minimum memory available telemetry is missing",
                }
            if error is None and backend == "cuda0":
                if result.gpu_peak_mib is None or result.gpu_peak_mib <= 0:
                    error = {
                        "phase": "telemetry",
                        "type": "MissingCudaTelemetryError",
                        "message": "cuda0 qualification requires positive gpu_peak_mib",
                    }
            if error is None:
                try:
                    validate_profile_result(
                        profile,
                        backend=backend,
                        minimum_mem_available_kb=result.minimum_mem_available_kb,
                    )
                except ProfilePreflightError as exc:
                    error = {
                        "phase": "headroom",
                        "type": type(exc).__name__,
                        "message": str(exc),
                    }

            if error is not None:
                failed = {"resolution": resolution}
                records.append(
                    _write_record(
                        context,
                        output_dir,
                        resolution,
                        status="failed",
                        error=error,
                        result=result,
                        mem_available_before_run_kb=mem_available_before_run_kb,
                        mem_available_after_run_kb=mem_available_after_run_kb,
                    )
                )
                print(f"MATRIX_GENERATION {resolution} FAIL {error['type']}", flush=True)
                break

            completed.append(resolution)
            records.append(
                _write_record(
                    context,
                    output_dir,
                    resolution,
                    status="passed",
                    result=result,
                    mem_available_before_run_kb=mem_available_before_run_kb,
                    mem_available_after_run_kb=mem_available_after_run_kb,
                )
            )
            print(
                f"MATRIX_GENERATION {resolution} PASS "
                f"elapsed_ms={result.elapsed_ms} "
                f"min_available_kb={result.minimum_mem_available_kb} "
                f"gpu_peak_mib={result.gpu_peak_mib}",
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
        "schema_version": 2,
        "release_version": "1.0.0",
        "status": "passed" if error is None else "failed",
        "source_head": source_head,
        "source_tree": source_tree,
        "profile": profile,
        "backend": backend,
        "matrix_id": matrix_id,
        "session": session_evidence,
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
    mem_available_before_run_kb: int | None = None,
    mem_available_after_run_kb: int | None = None,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": context["schema_version"],
        "release_version": context["release_version"],
        "status": status,
        "source_head": context["source_head"],
        "source_tree": context["source_tree"],
        "profile": context["profile"],
        "backend": context["backend"],
        "matrix_id": context["matrix_id"],
        "session": context["session"],
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
            "mem_available_before_run_kb": mem_available_before_run_kb,
            "mem_available_after_run_kb": mem_available_after_run_kb,
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
    parser.add_argument(
        "--backend", choices=list(SUPPORTED_MATRIX_BACKENDS), required=True
    )
    parser.add_argument(
        "--profile",
        choices=[Q8_REFERENCE_PROFILE, BF16_SAFETENSORS_PROFILE],
        default=BF16_SAFETENSORS_PROFILE,
    )
    parser.add_argument("--input-root", default="/kaggle/input")
    parser.add_argument("--work-root", default="/kaggle/working/mageflow-matrix")
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
