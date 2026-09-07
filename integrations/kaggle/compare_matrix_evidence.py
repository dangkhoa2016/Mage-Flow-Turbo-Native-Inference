from __future__ import annotations

import argparse
import json
from pathlib import Path

Q8_PROFILE = "q8-reference"
BF16_PROFILE = "bf16-high-memory-cpu"
CSS_BACKEND = "cpu"
REQUEST_FIELDS = ("prompt", "seed", "steps", "cfg", "threads")

Q8_DIFFUSION_SHA256 = "4c3dafc143ee64121692b6b63563a4f5288bf6183c4870e1d65f1566519ba7f0"
BF16_DIFFUSION_SHA256 = "6df47df3d7efc9ebdad075b87b3e9e4f74d09dca672d592271788f0ee27ab97d"
SHARED_TEXT_ENCODER_SHA256 = (
    "66358cb18bb6b3b1b6675aa412c7a88ef01d228f481184d13668e5201c730a0a"
)
SHARED_VAE_SHA256 = "34e076dc1e8a15321e1e07be5111d59cf16dd10b804b7c7e20b4de29013427e0"


def _load(path: Path) -> tuple[str, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    profile = data.get("profile")
    return profile, data


def _verify_memory_record(record: dict) -> None:
    if record.get("status") != "passed":
        raise ValueError("record must be passed to be comparable")
    memory = record.get("memory") or {}
    if memory.get("mem_available_before_run_kb") is None:
        raise ValueError("per-run pre-memory telemetry is missing")
    if memory.get("minimum_mem_available_kb") is None:
        raise ValueError("minimum memory available telemetry is missing")


def _extract_canonical_request(data: dict, label: str) -> tuple[dict | None, list[str]]:
    errors: list[str] = []
    matrix = data.get("matrix") or []
    if not matrix:
        errors.append(f"{label}: no matrix records")
        return None, errors

    canonical: dict | None = None
    for record in matrix:
        req = record.get("request")
        if not isinstance(req, dict):
            errors.append(f"{label}: missing or invalid request in matrix record")
            return None, errors
        normalized = {f: req.get(f) for f in REQUEST_FIELDS}
        if any(v is None for v in normalized.values()):
            errors.append(
                f"{label}: incomplete request evidence in matrix record, "
                f"fields={REQUEST_FIELDS}"
            )
            return None, errors
        if canonical is None:
            canonical = normalized
        elif normalized != canonical:
            errors.append(
                f"{label}: inconsistent request evidence across matrix records"
            )
            return None, errors

    if canonical is None:
        errors.append(f"{label}: no valid request evidence found")
    return canonical, errors


def check_comparability(q8: dict, bf16: dict) -> tuple[list[str], dict | None, dict | None]:
    errors: list[str] = []

    if q8.get("source_head") != bf16.get("source_head"):
        errors.append(
            f"source_head mismatch: q8={q8.get('source_head')} "
            f"bf16={bf16.get('source_head')}"
        )
    if q8.get("backend") != CSS_BACKEND or bf16.get("backend") != CSS_BACKEND:
        errors.append(
            f"backend must be {CSS_BACKEND}: q8={q8.get('backend')} "
            f"bf16={bf16.get('backend')}"
        )

    q8_runtime = q8.get("runtime") or {}
    bf16_runtime = bf16.get("runtime") or {}
    if q8_runtime.get("sha256") != bf16_runtime.get("sha256"):
        errors.append("runtime sha mismatch")
    if q8_runtime.get("commit") != bf16_runtime.get("commit"):
        errors.append("runtime commit mismatch")

    q8_res = q8.get("matrix_resolutions") or []
    bf16_res = bf16.get("matrix_resolutions") or []
    if q8_res != bf16_res:
        errors.append(
            f"resolution list mismatch: q8={q8_res} bf16={bf16_res}"
        )

    q8_canonical, q8_req_errors = _extract_canonical_request(q8, "q8")
    errors.extend(q8_req_errors)
    bf16_canonical, bf16_req_errors = _extract_canonical_request(bf16, "bf16")
    errors.extend(bf16_req_errors)

    if q8_canonical is not None and bf16_canonical is not None:
        if q8_canonical != bf16_canonical:
            errors.append("request params mismatch (cross-profile)")

    q8_models = q8.get("models") or {}
    bf16_models = bf16.get("models") or {}
    q8_te = (q8_models.get("text_encoder") or {}).get("sha256")
    bf16_te = (bf16_models.get("text_encoder") or {}).get("sha256")
    if q8_te != SHARED_TEXT_ENCODER_SHA256 or bf16_te != SHARED_TEXT_ENCODER_SHA256:
        errors.append("text encoder sha mismatch")
    q8_vae = (q8_models.get("vae") or {}).get("sha256")
    bf16_vae = (bf16_models.get("vae") or {}).get("sha256")
    if q8_vae != SHARED_VAE_SHA256 or bf16_vae != SHARED_VAE_SHA256:
        errors.append("vae sha mismatch")

    q8_dit = (q8_models.get("diffusion") or {}).get("sha256")
    bf16_dit = (bf16_models.get("diffusion") or {}).get("sha256")
    if q8_dit != Q8_DIFFUSION_SHA256:
        errors.append("q8 diffusion identity mismatch")
    if bf16_dit != BF16_DIFFUSION_SHA256:
        errors.append("bf16 diffusion identity mismatch")

    if q8.get("status") != "passed" or bf16.get("status") != "passed":
        errors.append("matrix status must be passed for both profiles")

    return errors, q8_canonical, bf16_canonical


def _records_by_resolution(data: dict) -> dict[int, dict]:
    by_res: dict[int, dict] = {}
    for record in data.get("matrix") or []:
        width = (record.get("resolution") or {}).get("width")
        by_res[width] = record
    return by_res


def compute_comparison(q8: dict, bf16: dict) -> dict:
    q8_by = _records_by_resolution(q8)
    bf16_by = _records_by_resolution(bf16)

    rows: list[dict] = []
    q8_total = 0
    bf16_total = 0
    for resolution in q8.get("matrix_resolutions") or []:
        q8_rec = q8_by.get(resolution)
        bf16_rec = bf16_by.get(resolution)
        if q8_rec is None or bf16_rec is None:
            continue

        _verify_memory_record(q8_rec)
        _verify_memory_record(bf16_rec)

        q8_elapsed = q8_rec.get("elapsed_ms")
        bf16_elapsed = bf16_rec.get("elapsed_ms")
        speedup = bf16_elapsed / q8_elapsed if q8_elapsed else None
        slowdown = (speedup - 1) * 100 if speedup is not None else None

        q8_rss = (q8_rec.get("memory") or {}).get("peak_sd_cli_rss_kb")
        bf16_rss = (bf16_rec.get("memory") or {}).get("peak_sd_cli_rss_kb")

        rows.append(
            {
                "resolution": resolution,
                "q8_elapsed_ms": q8_elapsed,
                "bf16_elapsed_ms": bf16_elapsed,
                "q8_speedup_vs_bf16": (
                    round(speedup, 4) if speedup is not None else None
                ),
                "bf16_slowdown_percent": (
                    round(slowdown, 2) if slowdown is not None else None
                ),
                "q8_peak_rss_kb": q8_rss,
                "bf16_peak_rss_kb": bf16_rss,
                "bf16_peak_rss_delta_kb": (
                    (bf16_rss - q8_rss)
                    if q8_rss is not None and bf16_rss is not None
                    else None
                ),
                "q8_min_mem_available_kb": (q8_rec.get("memory") or {}).get(
                    "minimum_mem_available_kb"
                ),
                "bf16_min_mem_available_kb": (bf16_rec.get("memory") or {}).get(
                    "minimum_mem_available_kb"
                ),
            }
        )
        if q8_elapsed is not None:
            q8_total += q8_elapsed
        if bf16_elapsed is not None:
            bf16_total += bf16_elapsed

    aggregate_speedup = bf16_total / q8_total if q8_total else None
    aggregate = {
        "sum_q8_elapsed_ms": q8_total,
        "sum_bf16_elapsed_ms": bf16_total,
        "aggregate_q8_speedup_vs_bf16": (
            round(aggregate_speedup, 4) if aggregate_speedup is not None else None
        ),
        "aggregate_bf16_slowdown_percent": (
            round((aggregate_speedup - 1) * 100, 2)
            if aggregate_speedup is not None
            else None
        ),
    }
    return {"comparison": rows, "aggregate": aggregate}


def build_comparison(
    q8_path: Path, bf16_path: Path
) -> dict:
    q8_profile, q8 = _load(q8_path)
    bf16_profile, bf16 = _load(bf16_path)

    if q8_profile != Q8_PROFILE or bf16_profile != BF16_PROFILE:
        raise ValueError("profiles must be q8-reference and bf16-high-memory-cpu")

    errors, q8_canonical, bf16_canonical = check_comparability(q8, bf16)
    if errors:
        return {
            "status": "failed",
            "comparability": "failed",
            "errors": errors,
        }

    canonical_request = q8_canonical
    metrics = compute_comparison(q8, bf16)
    return {
        "status": "passed",
        "comparability": "passed",
        "source_head": q8.get("source_head"),
        "backend": CSS_BACKEND,
        "runtime": q8.get("runtime"),
        "request": canonical_request,
        "resolutions": q8.get("matrix_resolutions"),
        "q8": q8,
        "bf16": bf16,
        "comparison": metrics["comparison"],
        "aggregate": metrics["aggregate"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mageflow-compare-matrix-evidence")
    parser.add_argument("--q8-aggregate", required=True)
    parser.add_argument("--bf16-aggregate", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    result = build_comparison(
        Path(args.q8_aggregate), Path(args.bf16_aggregate)
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
