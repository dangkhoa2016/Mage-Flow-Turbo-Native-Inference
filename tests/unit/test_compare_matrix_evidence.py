import json

import pytest

from integrations.kaggle import compare_matrix_evidence as cmp

HEAD = "f901b6d6221c107107945e4c49a2596da9d30791"


def _model(dit_sha, name):
    return {
        "diffusion": {
            "filename": name,
            "sha256": dit_sha,
            "format": "safetensors",
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


def _record(resolution, *, elapsed_ms, rss, min_mem, before_run, after_run):
    return {
        "status": "passed",
        "resolution": {"width": resolution, "height": resolution},
        "request": {
            "prompt": "a fox",
            "seed": 42,
            "steps": 4,
            "cfg": 1.0,
            "threads": 4,
        },
        "memory": {
            "mem_available_before_run_kb": before_run,
            "mem_available_after_run_kb": after_run,
            "minimum_mem_available_kb": min_mem,
            "peak_sd_cli_rss_kb": rss,
        },
        "elapsed_ms": elapsed_ms,
        "gpu_peak_mib": None,
    }


def _aggregate(profile, *, dit_sha, elapsed, source_head=HEAD):
    return {
        "status": "passed",
        "source_head": source_head,
        "profile": profile,
        "backend": "cpu",
        "matrix_resolutions": [512, 640, 768, 1024],
        "runtime": {
            "commit": "6b3edaaf32cc19e5bb2d819c788bd557eddc8eba",
            "sha256": "7539d90b99eaf2b6279eec4f9006a68ae53e87bfe0c9c325ff3f329220468a5c",
        },
        "models": _model(dit_sha, profile),
        "matrix": [_record(r, elapsed_ms=e, rss=1000, min_mem=4 * 1024 * 1024,
                            before_run=1_000_000, after_run=2_000_000)
                    for r, e in zip([512, 640, 768, 1024], elapsed)],
    }


def test_comparable_inputs_yield_passed_comparison(tmp_path):
    q8 = _aggregate(cmp.Q8_PROFILE, dit_sha=cmp.Q8_DIFFUSION_SHA256,
                    elapsed=[10, 20, 30, 40])
    bf16 = _aggregate(cmp.BF16_PROFILE, dit_sha=cmp.BF16_DIFFUSION_SHA256,
                      elapsed=[20, 40, 60, 80])
    q8p = tmp_path / "q8.json"
    bf16p = tmp_path / "bf16.json"
    q8p.write_text(json.dumps(q8), encoding="utf-8")
    bf16p.write_text(json.dumps(bf16), encoding="utf-8")

    result = cmp.build_comparison(q8p, bf16p)
    assert result["status"] == "passed"
    assert result["comparability"] == "passed"
    assert result["backend"] == "cpu"

    rows = {r["resolution"]: r for r in result["comparison"]}
    assert rows[512]["q8_speedup_vs_bf16"] == 2.0
    assert rows[512]["bf16_slowdown_percent"] == 100.0
    assert rows[1024]["q8_elapsed_ms"] == 40
    assert rows[1024]["bf16_elapsed_ms"] == 80

    agg = result["aggregate"]
    assert agg["sum_q8_elapsed_ms"] == 100
    assert agg["sum_bf16_elapsed_ms"] == 200
    assert agg["aggregate_q8_speedup_vs_bf16"] == 2.0


def test_source_head_mismatch_fails_comparability(tmp_path):
    q8 = _aggregate(cmp.Q8_PROFILE, dit_sha=cmp.Q8_DIFFUSION_SHA256,
                    elapsed=[10, 20, 30, 40], source_head="deadbeef")
    bf16 = _aggregate(cmp.BF16_PROFILE, dit_sha=cmp.BF16_DIFFUSION_SHA256,
                      elapsed=[10, 20, 30, 40])
    q8p = tmp_path / "q8.json"
    bf16p = tmp_path / "bf16.json"
    q8p.write_text(json.dumps(q8), encoding="utf-8")
    bf16p.write_text(json.dumps(bf16), encoding="utf-8")
    result = cmp.build_comparison(q8p, bf16p)
    assert result["comparability"] == "failed"
    assert result["status"] == "failed"
    assert any("source_head" in e for e in result["errors"])


def test_runtime_sha_mismatch_fails_comparability(tmp_path):
    q8 = _aggregate(cmp.Q8_PROFILE, dit_sha=cmp.Q8_DIFFUSION_SHA256,
                    elapsed=[10, 20, 30, 40])
    bf16 = _aggregate(cmp.BF16_PROFILE, dit_sha=cmp.BF16_DIFFUSION_SHA256,
                      elapsed=[10, 20, 30, 40])
    bf16["runtime"]["sha256"] = "0" * 64
    q8p = tmp_path / "q8.json"
    bf16p = tmp_path / "bf16.json"
    q8p.write_text(json.dumps(q8), encoding="utf-8")
    bf16p.write_text(json.dumps(bf16), encoding="utf-8")
    result = cmp.build_comparison(q8p, bf16p)
    assert result["comparability"] == "failed"
    assert any("runtime sha" in e for e in result["errors"])


def test_missing_per_run_pre_memory_fails_closed(tmp_path):
    q8 = _aggregate(cmp.Q8_PROFILE, dit_sha=cmp.Q8_DIFFUSION_SHA256,
                    elapsed=[10, 20, 30, 40])
    bf16 = _aggregate(cmp.BF16_PROFILE, dit_sha=cmp.BF16_DIFFUSION_SHA256,
                      elapsed=[10, 20, 30, 40])
    bf16["matrix"][0]["memory"]["mem_available_before_run_kb"] = None
    q8p = tmp_path / "q8.json"
    bf16p = tmp_path / "bf16.json"
    q8p.write_text(json.dumps(q8), encoding="utf-8")
    bf16p.write_text(json.dumps(bf16), encoding="utf-8")
    with pytest.raises(ValueError, match="pre-memory"):
        cmp.build_comparison(q8p, bf16p)
