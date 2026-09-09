from __future__ import annotations

import argparse
import json
from pathlib import Path

Q8_PROFILE = "q8-reference"
BF16_PROFILE = "bf16-safetensors"
PINNED_SDCPP_COMMIT = "6b3edaaf32cc19e5bb2d819c788bd557eddc8eba"
CPU_RUNTIME_SHA256 = "7539d90b99eaf2b6279eec4f9006a68ae53e87bfe0c9c325ff3f329220468a5c"
CUDA_RUNTIME_SHA256 = "3fae6c1991ad0ac764c36495f688817c8a3d295d7651369bf74b7fd33743c3d0"
REQUEST_FIELDS = ("prompt", "seed", "steps", "cfg", "threads")
EXPECTED_RESOLUTIONS = (512, 640, 768, 1024)

Q8_DIFFUSION_SHA256 = "4c3dafc143ee64121692b6b63563a4f5288bf6183c4870e1d65f1566519ba7f0"
BF16_DIFFUSION_SHA256 = "6df47df3d7efc9ebdad075b87b3e9e4f74d09dca672d592271788f0ee27ab97d"
SHARED_TEXT_ENCODER_SHA256 = (
    "66358cb18bb6b3b1b6675aa412c7a88ef01d228f481184d13668e5201c730a0a"
)
SHARED_VAE_SHA256 = "34e076dc1e8a15321e1e07be5111d59cf16dd10b804b7c7e20b4de29013427e0"
ALLOWED_T4_NAMES = {"Tesla T4", "NVIDIA T4", "NVIDIA Tesla T4"}
CELL_KEYS = ("q8_cpu", "bf16_cpu", "q8_cuda0", "bf16_cuda0")
_EXPECTED_CELL = {
    "q8_cpu": (Q8_PROFILE, "cpu"),
    "bf16_cpu": (BF16_PROFILE, "cpu"),
    "q8_cuda0": (Q8_PROFILE, "cuda0"),
    "bf16_cuda0": (BF16_PROFILE, "cuda0"),
}


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _record_resolution(record: dict) -> int | None:
    resolution = record.get("resolution") or {}
    width = resolution.get("width")
    height = resolution.get("height")
    if isinstance(width, int) and width == height:
        return width
    return None


def _canonical_request(data: dict, label: str) -> tuple[dict | None, list[str]]:
    errors: list[str] = []
    matrix = data.get("matrix") or []
    if not matrix:
        return None, [f"{label}: no matrix records"]
    canonical: dict | None = None
    for record in matrix:
        request = record.get("request")
        if not isinstance(request, dict):
            return None, [f"{label}: missing or invalid request in matrix record"]
        normalized = {field: request.get(field) for field in REQUEST_FIELDS}
        if any(value is None for value in normalized.values()):
            return None, [f"{label}: incomplete request evidence"]
        if canonical is None:
            canonical = normalized
        elif normalized != canonical:
            return None, [f"{label}: inconsistent request evidence across matrix records"]
    return canonical, errors


def _records_by_resolution(data: dict) -> dict[int, dict]:
    result: dict[int, dict] = {}
    for record in data.get("matrix") or []:
        resolution = _record_resolution(record)
        if resolution is not None:
            result[resolution] = record
    return result


def _model_sha(data: dict, role: str) -> str | None:
    return ((data.get("models") or {}).get(role) or {}).get("sha256")


def _validate_t4_session(label: str, data: dict) -> list[str]:
    errors: list[str] = []
    session = data.get("session") or {}
    if session.get("cuda_device_order") != "PCI_BUS_ID":
        errors.append(f"{label}: CUDA_DEVICE_ORDER must be PCI_BUS_ID")
    if session.get("cuda_visible_devices") != "0":
        errors.append(f"{label}: CUDA_VISIBLE_DEVICES must be 0")

    physical_gpus = session.get("physical_gpus")
    if not isinstance(physical_gpus, list) or len(physical_gpus) not in (1, 2):
        errors.append(f"{label}: physical_gpus must describe T4 or T4x2")
    else:
        indices = [gpu.get("index") for gpu in physical_gpus if isinstance(gpu, dict)]
        names = [gpu.get("name") for gpu in physical_gpus if isinstance(gpu, dict)]
        if len(indices) != len(physical_gpus) or indices != list(range(len(physical_gpus))):
            errors.append(f"{label}: physical GPU indices are invalid")
        if len(names) != len(physical_gpus) or any(name not in ALLOWED_T4_NAMES for name in names):
            errors.append(f"{label}: physical GPU identity is not T4/T4x2")

    devices = str((data.get("runtime") or {}).get("devices") or "").lower()
    if "cuda0" not in devices:
        errors.append(f"{label}: runtime devices did not expose cuda0")
    if "cuda1" in devices:
        errors.append(f"{label}: runtime devices must not expose cuda1")
    return errors


def _validate_record_identity(label: str, data: dict, record: dict) -> list[str]:
    errors: list[str] = []
    resolution = _record_resolution(record)
    tag = f"{label}/{resolution if resolution is not None else 'invalid-resolution'}"
    expected = {
        "schema_version": data.get("schema_version"),
        "release_version": data.get("release_version"),
        "source_head": data.get("source_head"),
        "source_tree": data.get("source_tree"),
        "profile": data.get("profile"),
        "backend": data.get("backend"),
    }
    for field, value in expected.items():
        if record.get(field) != value:
            errors.append(f"{tag}: record {field} mismatch")
    for field in ("session", "runtime", "models"):
        if record.get(field) != data.get(field):
            errors.append(f"{tag}: record {field} mismatch")
    if record.get("status") == "passed":
        elapsed = record.get("elapsed_ms")
        if not isinstance(elapsed, (int, float)) or elapsed <= 0:
            errors.append(f"{tag}: passed record requires positive elapsed_ms")
        artifact = record.get("artifact") or {}
        if artifact.get("width") != resolution or artifact.get("height") != resolution:
            errors.append(f"{tag}: passed artifact dimensions mismatch")
        digest = artifact.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            errors.append(f"{tag}: passed artifact sha256 missing or invalid")
    return errors


def _validate_matrix_structure(label: str, data: dict) -> list[str]:
    errors: list[str] = []
    matrix = data.get("matrix") or []
    resolutions = [_record_resolution(record) for record in matrix]
    if any(resolution is None for resolution in resolutions):
        errors.append(f"{label}: matrix contains invalid non-square resolution")
        return errors
    if len(set(resolutions)) != len(resolutions):
        errors.append(f"{label}: matrix contains duplicate resolution records")
    if tuple(resolutions) != EXPECTED_RESOLUTIONS[: len(resolutions)]:
        errors.append(f"{label}: matrix records are not an ordered contiguous prefix")

    for record in matrix:
        errors.extend(_validate_record_identity(label, data, record))

    status = data.get("status")
    completed = data.get("completed_resolutions") or []
    failed_resolution = (data.get("failed_resolution") or {}).get("resolution")
    if status == "passed":
        if tuple(resolutions) != EXPECTED_RESOLUTIONS:
            errors.append(f"{label}: passed matrix must contain all four resolutions")
        if any(record.get("status") != "passed" for record in matrix):
            errors.append(f"{label}: passed matrix contains non-passed record")
        if tuple(completed) != EXPECTED_RESOLUTIONS:
            errors.append(f"{label}: passed matrix completed_resolutions mismatch")
        if data.get("failed_resolution") is not None or data.get("error") is not None:
            errors.append(f"{label}: passed matrix must not record failure state")
    elif status == "failed":
        if len(matrix) < 2:
            errors.append(f"{label}: failed matrix must occur after canonical 512 pass")
        elif any(record.get("status") != "passed" for record in matrix[:-1]):
            errors.append(f"{label}: failed matrix has a non-passed record before terminal failure")
        if not matrix or matrix[-1].get("status") != "failed":
            errors.append(f"{label}: failed matrix must end with exactly one failed record")
        expected_completed = resolutions[:-1]
        if completed != expected_completed:
            errors.append(f"{label}: failed matrix completed_resolutions mismatch")
        terminal = resolutions[-1] if resolutions else None
        if failed_resolution != terminal:
            errors.append(f"{label}: failed_resolution does not match terminal record")
        if data.get("error") is None:
            errors.append(f"{label}: failed matrix requires aggregate error")
    else:
        errors.append(f"{label}: unsupported matrix status {status!r}")
    return errors


def _validate_cell(label: str, data: dict) -> list[str]:
    errors: list[str] = []
    expected_profile, expected_backend = _EXPECTED_CELL[label]
    if data.get("schema_version") != 2:
        errors.append(f"{label}: schema_version must be 2")
    if data.get("release_version") != "1.0.0":
        errors.append(f"{label}: release_version must be 1.0.0")
    if data.get("profile") != expected_profile:
        errors.append(
            f"{label}: profile mismatch: expected {expected_profile}, got {data.get('profile')}"
        )
    if data.get("backend") != expected_backend:
        errors.append(
            f"{label}: backend mismatch: expected {expected_backend}, got {data.get('backend')}"
        )

    runtime = data.get("runtime") or {}
    if runtime.get("commit") != PINNED_SDCPP_COMMIT:
        errors.append(f"{label}: runtime commit mismatch")
    expected_runtime_sha = (
        CPU_RUNTIME_SHA256 if expected_backend == "cpu" else CUDA_RUNTIME_SHA256
    )
    if runtime.get("sha256") != expected_runtime_sha:
        errors.append(f"{label}: runtime sha mismatch")

    if _model_sha(data, "text_encoder") != SHARED_TEXT_ENCODER_SHA256:
        errors.append(f"{label}: text encoder sha mismatch")
    if _model_sha(data, "vae") != SHARED_VAE_SHA256:
        errors.append(f"{label}: vae sha mismatch")
    expected_diffusion = (
        Q8_DIFFUSION_SHA256 if expected_profile == Q8_PROFILE else BF16_DIFFUSION_SHA256
    )
    if _model_sha(data, "diffusion") != expected_diffusion:
        errors.append(f"{label}: diffusion sha mismatch")

    matrix = data.get("matrix") or []
    if not matrix:
        errors.append(f"{label}: no matrix records")
    else:
        first = matrix[0]
        if _record_resolution(first) != 512:
            errors.append(f"{label}: canonical 512 record must be first")
        if first.get("status") != "passed":
            errors.append(f"{label}: canonical 512 record must pass")
    errors.extend(_validate_matrix_structure(label, data))

    if expected_backend == "cuda0":
        errors.extend(_validate_t4_session(label, data))
        for record in matrix:
            if record.get("status") == "passed":
                gpu_peak = record.get("gpu_peak_mib")
                if not isinstance(gpu_peak, (int, float)) or gpu_peak <= 0:
                    errors.append(f"{label}: successful CUDA record lacks positive gpu_peak_mib")
                    break
    return errors


def check_comparability(cells: dict[str, dict]) -> tuple[list[str], dict | None]:
    errors: list[str] = []
    for key in CELL_KEYS:
        if key not in cells:
            errors.append(f"missing cell: {key}")
            continue
        errors.extend(_validate_cell(key, cells[key]))
    if errors:
        return errors, None

    heads = {cells[key].get("source_head") for key in CELL_KEYS}
    trees = {cells[key].get("source_tree") for key in CELL_KEYS}
    if len(heads) != 1 or None in heads or "unknown" in heads:
        errors.append(f"source_head mismatch or unknown: {sorted(str(v) for v in heads)}")
    if len(trees) != 1 or None in trees or "unknown" in trees:
        errors.append(f"source_tree mismatch or unknown: {sorted(str(v) for v in trees)}")

    resolution_lists = {tuple(cells[key].get("matrix_resolutions") or []) for key in CELL_KEYS}
    if len(resolution_lists) != 1:
        errors.append("resolution list mismatch across cells")
    elif next(iter(resolution_lists), ()) != EXPECTED_RESOLUTIONS:
        errors.append("resolution list must be exactly 512,640,768,1024")

    canonical: dict | None = None
    for key in CELL_KEYS:
        request, request_errors = _canonical_request(cells[key], key)
        errors.extend(request_errors)
        if request is not None:
            if canonical is None:
                canonical = request
            elif request != canonical:
                errors.append(f"{key}: canonical request mismatch")
    return errors, canonical


def _passed(record: dict | None) -> bool:
    return bool(record and record.get("status") == "passed")


def _safe_ratio(numerator: int | float | None, denominator: int | float | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return round(numerator / denominator, 4)


def _cell_metrics(record: dict | None) -> dict:
    if record is None:
        return {"status": "missing", "elapsed_ms": None, "peak_rss_kb": None, "gpu_peak_mib": None}
    memory = record.get("memory") or {}
    return {
        "status": record.get("status"),
        "elapsed_ms": record.get("elapsed_ms"),
        "peak_rss_kb": memory.get("peak_sd_cli_rss_kb"),
        "gpu_peak_mib": record.get("gpu_peak_mib"),
        "error": record.get("error"),
        "artifact": record.get("artifact"),
    }


def _ratio_if_passed(a: dict | None, b: dict | None) -> float | None:
    if not (_passed(a) and _passed(b)):
        return None
    return _safe_ratio(a.get("elapsed_ms"), b.get("elapsed_ms"))


def build_2x2_comparison(
    q8_cpu_path: Path,
    bf16_cpu_path: Path,
    q8_cuda0_path: Path,
    bf16_cuda0_path: Path,
) -> dict:
    cells = {
        "q8_cpu": _load(q8_cpu_path),
        "bf16_cpu": _load(bf16_cpu_path),
        "q8_cuda0": _load(q8_cuda0_path),
        "bf16_cuda0": _load(bf16_cuda0_path),
    }
    errors, canonical_request = check_comparability(cells)
    if errors:
        return {
            "status": "failed",
            "comparability": "failed",
            "errors": errors,
        }

    by_resolution = {key: _records_by_resolution(value) for key, value in cells.items()}
    rows: list[dict] = []
    for resolution in EXPECTED_RESOLUTIONS:
        q8_cpu = by_resolution["q8_cpu"].get(resolution)
        bf16_cpu = by_resolution["bf16_cpu"].get(resolution)
        q8_cuda = by_resolution["q8_cuda0"].get(resolution)
        bf16_cuda = by_resolution["bf16_cuda0"].get(resolution)

        q8_cpu_m = _cell_metrics(q8_cpu)
        q8_cuda_m = _cell_metrics(q8_cuda)
        bf16_cpu_m = _cell_metrics(bf16_cpu)
        bf16_cuda_m = _cell_metrics(bf16_cuda)

        q8_speedup = _ratio_if_passed(q8_cpu, q8_cuda)
        bf16_speedup = _ratio_if_passed(bf16_cpu, bf16_cuda)
        cpu_ratio = _ratio_if_passed(bf16_cpu, q8_cpu)
        cuda_ratio = _ratio_if_passed(bf16_cuda, q8_cuda)

        rows.append(
            {
                "resolution": resolution,
                "q8": {
                    "cpu": q8_cpu_m,
                    "cuda0": q8_cuda_m,
                    "cpu_to_cuda_speedup": q8_speedup,
                },
                "bf16": {
                    "cpu": bf16_cpu_m,
                    "cuda0": bf16_cuda_m,
                    "cpu_to_cuda_speedup": bf16_speedup,
                },
                "cpu": {
                    "bf16_over_q8_latency_ratio": cpu_ratio,
                    "bf16_minus_q8_peak_rss_kb": (
                        bf16_cpu_m["peak_rss_kb"] - q8_cpu_m["peak_rss_kb"]
                        if _passed(bf16_cpu)
                        and _passed(q8_cpu)
                        and bf16_cpu_m["peak_rss_kb"] is not None
                        and q8_cpu_m["peak_rss_kb"] is not None
                        else None
                    ),
                },
                "cuda0": {
                    "bf16_over_q8_latency_ratio": cuda_ratio,
                    "bf16_minus_q8_gpu_peak_mib": (
                        bf16_cuda_m["gpu_peak_mib"] - q8_cuda_m["gpu_peak_mib"]
                        if _passed(bf16_cuda)
                        and _passed(q8_cuda)
                        and bf16_cuda_m["gpu_peak_mib"] is not None
                        and q8_cuda_m["gpu_peak_mib"] is not None
                        else None
                    ),
                },
            }
        )

    return {
        "schema_version": 1,
        "release_version": "1.0.0",
        "status": "passed",
        "comparability": "passed",
        "source_head": cells["q8_cpu"]["source_head"],
        "source_tree": cells["q8_cpu"]["source_tree"],
        "request": canonical_request,
        "resolutions": list(EXPECTED_RESOLUTIONS),
        "cells": cells,
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mageflow-compare-matrix-evidence")
    parser.add_argument("--q8-cpu-aggregate", required=True)
    parser.add_argument("--bf16-cpu-aggregate", required=True)
    parser.add_argument("--q8-cuda0-aggregate", required=True)
    parser.add_argument("--bf16-cuda0-aggregate", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    result = build_2x2_comparison(
        Path(args.q8_cpu_aggregate),
        Path(args.bf16_cpu_aggregate),
        Path(args.q8_cuda0_aggregate),
        Path(args.bf16_cuda0_aggregate),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["comparability"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
