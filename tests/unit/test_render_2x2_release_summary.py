from scripts.publication.render_2x2_release_summary import render_summary


def _data():
    return {
        "comparability": "passed",
        "source_head": "a" * 40,
        "source_tree": "b" * 40,
        "cells": {},
        "rows": [
            {
                "resolution": 512,
                "q8": {
                    "cpu": {"status": "passed", "elapsed_ms": 200000, "peak_rss_kb": 9 * 1024 * 1024},
                    "cuda0": {"status": "passed", "elapsed_ms": 8000, "gpu_peak_mib": 8000},
                    "cpu_to_cuda_speedup": 25.0,
                },
                "bf16": {
                    "cpu": {"status": "passed", "elapsed_ms": 300000, "peak_rss_kb": 12 * 1024 * 1024},
                    "cuda0": {"status": "passed", "elapsed_ms": 12000, "gpu_peak_mib": 12000},
                    "cpu_to_cuda_speedup": 25.0,
                },
                "cpu": {"bf16_over_q8_latency_ratio": 1.5},
                "cuda0": {"bf16_over_q8_latency_ratio": 1.5},
            }
        ],
    }


def test_renderer_emits_four_cell_table_and_source_identity():
    text = render_summary(_data())
    assert "# v1.0.0 Strict 2x2 Native Benchmark" in text
    assert "512×512" in text
    assert "25.00×" in text
    assert "Source HEAD" in text
    assert "Transformers inference backend" in text
    assert "8,000 MiB" in text
    assert "12.00 GiB" in text


def test_renderer_rejects_non_comparable_evidence():
    data = _data()
    data["comparability"] = "failed"
    try:
        render_summary(data)
    except ValueError as exc:
        assert "non-comparable" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_renderer_marks_missing_ratios_as_na():
    data = _data()
    data["rows"][0]["bf16"]["cpu_to_cuda_speedup"] = None
    assert "N/A" in render_summary(data)
