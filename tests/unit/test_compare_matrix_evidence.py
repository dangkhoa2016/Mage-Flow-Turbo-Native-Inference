import json
from pathlib import Path

from integrations.kaggle import compare_matrix_evidence as cmp

HEAD = "f901b6d6221c107107945e4c49a2596da9d30791"
TREE = "1" * 40
EXPECTED_REQUEST = {
    "prompt": "a fox",
    "seed": 42,
    "steps": 4,
    "cfg": 1.0,
    "threads": 4,
}


def _model(profile):
    return {
        "diffusion": {
            "filename": profile,
            "sha256": (
                cmp.Q8_DIFFUSION_SHA256
                if profile == cmp.Q8_PROFILE
                else cmp.BF16_DIFFUSION_SHA256
            ),
        },
        "text_encoder": {
            "filename": "Qwen3VL-4B-Instruct-Q4_K_M.gguf",
            "sha256": cmp.SHARED_TEXT_ENCODER_SHA256,
        },
        "vae": {
            "filename": "diffusion_pytorch_model.safetensors",
            "sha256": cmp.SHARED_VAE_SHA256,
        },
    }


def _record(resolution, *, elapsed_ms, backend, status="passed"):
    record = {
        "status": status,
        "source_head": HEAD,
        "source_tree": TREE,
        "profile": None,
        "backend": backend,
        "resolution": {"width": resolution, "height": resolution},
        "request": dict(EXPECTED_REQUEST),
        "memory": {
            "mem_available_before_run_kb": 1_000_000,
            "mem_available_after_run_kb": 900_000,
            "minimum_mem_available_kb": 800_000,
            "peak_sd_cli_rss_kb": 1000,
        },
        "elapsed_ms": elapsed_ms if status == "passed" else None,
        "gpu_peak_mib": 8192 if backend == "cuda0" and status == "passed" else None,
        "artifact": {"sha256": "a" * 64} if status == "passed" else {},
    }
    if status != "passed":
        record["error"] = {"type": "RuntimeError", "message": "oom"}
    return record


def _aggregate(profile, backend, *, elapsed, source_head=HEAD, source_tree=TREE):
    records = []
    for resolution, elapsed_ms in zip([512, 640, 768, 1024], elapsed):
        rec = _record(resolution, elapsed_ms=elapsed_ms, backend=backend)
        rec["profile"] = profile
        rec["source_head"] = source_head
        rec["source_tree"] = source_tree
        records.append(rec)
    return {
        "schema_version": 2,
        "release_version": "1.0.0",
        "status": "passed",
        "source_head": source_head,
        "source_tree": source_tree,
        "profile": profile,
        "backend": backend,
        "session": {
            "hostname": "fake",
            "backend": backend,
            "cuda_visible_devices": "0" if backend == "cuda0" else None,
        },
        "matrix_resolutions": [512, 640, 768, 1024],
        "completed_resolutions": [512, 640, 768, 1024],
        "failed_resolution": None,
        "error": None,
        "runtime": {
            "commit": cmp.PINNED_SDCPP_COMMIT,
            "sha256": (
                cmp.CPU_RUNTIME_SHA256 if backend == "cpu" else cmp.CUDA_RUNTIME_SHA256
            ),
        },
        "models": _model(profile),
        "matrix": records,
    }


def _write(tmp_path: Path, name: str, data: dict) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _paths(tmp_path, cells):
    return {
        key: _write(tmp_path, f"{key}.json", value)
        for key, value in cells.items()
    }


def _valid_cells():
    return {
        "q8_cpu": _aggregate(cmp.Q8_PROFILE, "cpu", elapsed=[100, 200, 300, 400]),
        "bf16_cpu": _aggregate(cmp.BF16_PROFILE, "cpu", elapsed=[200, 400, 600, 800]),
        "q8_cuda0": _aggregate(cmp.Q8_PROFILE, "cuda0", elapsed=[10, 20, 30, 40]),
        "bf16_cuda0": _aggregate(cmp.BF16_PROFILE, "cuda0", elapsed=[20, 40, 60, 80]),
    }


def _build(tmp_path, cells=None):
    cells = cells or _valid_cells()
    paths = _paths(tmp_path, cells)
    return cmp.build_2x2_comparison(
        paths["q8_cpu"],
        paths["bf16_cpu"],
        paths["q8_cuda0"],
        paths["bf16_cuda0"],
    )


def test_four_cells_produce_strict_2x2_rows(tmp_path):
    result = _build(tmp_path)
    assert result["status"] == "passed"
    assert result["comparability"] == "passed"
    assert result["source_head"] == HEAD
    assert result["source_tree"] == TREE
    row512 = next(row for row in result["rows"] if row["resolution"] == 512)
    assert row512["q8"]["cpu"]["elapsed_ms"] == 100
    assert row512["q8"]["cuda0"]["elapsed_ms"] == 10
    assert row512["q8"]["cpu_to_cuda_speedup"] == 10.0
    assert row512["bf16"]["cpu_to_cuda_speedup"] == 10.0
    assert row512["cpu"]["bf16_over_q8_latency_ratio"] == 2.0
    assert row512["cuda0"]["bf16_over_q8_latency_ratio"] == 2.0


def test_source_head_mismatch_fails(tmp_path):
    cells = _valid_cells()
    cells["bf16_cuda0"]["source_head"] = "deadbeef"
    for record in cells["bf16_cuda0"]["matrix"]:
        record["source_head"] = "deadbeef"
    result = _build(tmp_path, cells)
    assert result["comparability"] == "failed"
    assert any("source_head" in error for error in result["errors"])


def test_source_tree_mismatch_fails(tmp_path):
    cells = _valid_cells()
    cells["q8_cuda0"]["source_tree"] = "2" * 40
    result = _build(tmp_path, cells)
    assert result["comparability"] == "failed"
    assert any("source_tree" in error for error in result["errors"])


def test_wrong_model_identity_fails(tmp_path):
    cells = _valid_cells()
    cells["bf16_cpu"]["models"]["diffusion"]["sha256"] = "0" * 64
    result = _build(tmp_path, cells)
    assert result["comparability"] == "failed"
    assert any("diffusion sha" in error for error in result["errors"])


def test_wrong_backend_fails(tmp_path):
    cells = _valid_cells()
    cells["q8_cuda0"]["backend"] = "cpu"
    result = _build(tmp_path, cells)
    assert result["comparability"] == "failed"
    assert any("backend mismatch" in error for error in result["errors"])


def test_wrong_runtime_sha_fails(tmp_path):
    cells = _valid_cells()
    cells["bf16_cuda0"]["runtime"]["sha256"] = "0" * 64
    result = _build(tmp_path, cells)
    assert result["comparability"] == "failed"
    assert any("runtime sha" in error for error in result["errors"])


def test_cuda_mask_must_be_zero(tmp_path):
    cells = _valid_cells()
    cells["q8_cuda0"]["session"]["cuda_visible_devices"] = "0,1"
    result = _build(tmp_path, cells)
    assert result["comparability"] == "failed"
    assert any("CUDA_VISIBLE_DEVICES" in error for error in result["errors"])


def test_cuda_success_requires_positive_gpu_peak(tmp_path):
    cells = _valid_cells()
    cells["q8_cuda0"]["matrix"][0]["gpu_peak_mib"] = None
    result = _build(tmp_path, cells)
    assert result["comparability"] == "failed"
    assert any("gpu_peak_mib" in error for error in result["errors"])


def test_request_mismatch_fails(tmp_path):
    cells = _valid_cells()
    cells["bf16_cpu"]["matrix"][0]["request"]["seed"] = 43
    result = _build(tmp_path, cells)
    assert result["comparability"] == "failed"
    assert any("request" in error for error in result["errors"])


def test_resolution_order_mismatch_fails(tmp_path):
    cells = _valid_cells()
    cells["q8_cpu"]["matrix_resolutions"] = [512, 768, 640, 1024]
    result = _build(tmp_path, cells)
    assert result["comparability"] == "failed"
    assert any("resolution" in error for error in result["errors"])


def test_partial_later_resolution_keeps_identity_comparable_but_omits_ratios(tmp_path):
    cells = _valid_cells()
    cell = cells["bf16_cuda0"]
    failed = _record(768, elapsed_ms=0, backend="cuda0", status="failed")
    failed["profile"] = cmp.BF16_PROFILE
    cell["matrix"] = [cell["matrix"][0], cell["matrix"][1], failed]
    cell["status"] = "failed"
    cell["completed_resolutions"] = [512, 640]
    cell["failed_resolution"] = {"resolution": 768}
    cell["error"] = {"type": "RuntimeError", "message": "oom"}

    result = _build(tmp_path, cells)
    assert result["comparability"] == "passed"
    row768 = next(row for row in result["rows"] if row["resolution"] == 768)
    assert row768["bf16"]["cuda0"]["status"] == "failed"
    assert row768["bf16"]["cpu_to_cuda_speedup"] is None
    assert row768["cuda0"]["bf16_over_q8_latency_ratio"] is None


def test_canonical_512_failure_is_not_comparable(tmp_path):
    cells = _valid_cells()
    cells["bf16_cuda0"]["matrix"][0]["status"] = "failed"
    result = _build(tmp_path, cells)
    assert result["comparability"] == "failed"
    assert any("canonical 512" in error for error in result["errors"])
