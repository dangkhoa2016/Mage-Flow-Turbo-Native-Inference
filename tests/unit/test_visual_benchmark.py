import json
from pathlib import Path

import pytest

from integrations.kaggle import visual_benchmark as vb

HEAD = "bc31e7ba18343ca33bb389b6f1dfc4b1f32d72ab"


class _FakeRuntime:
    def __init__(self, path):
        self.path = str(path)
        self.pinned_commit = vb.RUNTIME_COMMIT
        self.version_output = f"sd-cli {vb.RUNTIME_COMMIT[:7]}"
        self.devices_output = "cpu"


def _make_template(tmp_path) -> Path:
    import shutil
    src = Path("configs/visual-benchmark-manifest-768.template.json")
    assert src.is_file(), f"missing {src}"
    data = json.loads(src.read_text(encoding="utf-8"))
    data["source_head"] = HEAD
    out = tmp_path / "exec-manifest.json"
    out.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return out


def _make_harness(tmp_path):
    manifest = _make_template(tmp_path)
    bench = vb.VisualBenchmark(
        manifest,
        repo_dir=tmp_path,
        work_root=tmp_path / "work",
        fake=True,
    )
    bench.fingerprint = vb.HostFingerprint(
        hostname="host-a",
        linux_boot_id="boot-1",
        cpu_model="Fake CPU",
        mem_total_kb=32768 * 1024,
    )
    return bench


def _write_passed_record(bench, prompt_id, profile, *, png_sha=None):
    rec = {
        "status": "passed",
        "benchmark_id": bench.manifest.benchmark_id,
        "manifest_sha256": bench.manifest.sha256,
        "source_head": bench.manifest.source_head,
        "host_fingerprint": bench.fingerprint.to_dict(),
        "prompt_id": prompt_id,
        "profile": profile,
        "request": {
            "prompt": next(
                p["prompt"] for p in bench.manifest.prompts if p["id"] == prompt_id
            ),
            "seed": next(
                p["seed"] for p in bench.manifest.prompts if p["id"] == prompt_id
            ),
            "width": 768,
            "height": 768,
            "steps": 4,
            "cfg": 1.0,
            "threads": 4,
        },
        "elapsed_ms": 100,
        "memory": {"peak_sd_cli_rss_kb": 1000},
        "png_path": str(bench.root.image_path(prompt_id, profile)),
        "png_sha256": png_sha,
    }
    path = bench.root.record_path(prompt_id, profile)
    path.parent.mkdir(parents=True, exist_ok=True)
    img = bench.root.image_path(prompt_id, profile)
    img.parent.mkdir(parents=True, exist_ok=True)
    img.write_bytes(b"FAKEPNG")
    if png_sha is None:
        import hashlib
        rec["png_sha256"] = hashlib.sha256(b"FAKEPNG").hexdigest()
    path.write_text(json.dumps(rec), encoding="utf-8")
    return rec


def test_H1_manifest_identity_validation(tmp_path):
    manifest = _make_template(tmp_path)
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data["provenance"]["runtime"]["sha256"] = "0" * 64
    manifest.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(vb.VisualBenchmarkError, match="runtime sha256"):
        vb.VisualBenchmark(manifest, repo_dir=tmp_path, work_root=tmp_path / "w")


def test_H2_source_head_validation(tmp_path):
    manifest = _make_template(tmp_path)
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data["source_head"] = "1" * 40
    manifest.write_text(json.dumps(data), encoding="utf-8")
    bench = vb.VisualBenchmark(manifest, repo_dir=tmp_path, work_root=tmp_path / "w")
    with pytest.raises(vb.VisualBenchmarkError, match="source_head"):
        bench.validate()


def test_H3_prompt_seed_contract(tmp_path):
    manifest = _make_template(tmp_path)
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data["prompts"][0]["id"] = "P01"
    data["prompts"].append(dict(data["prompts"][0]))
    data["prompts"][-1]["id"] = "P11"
    manifest.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(vb.VisualBenchmarkError, match="P01..P10"):
        vb.VisualBenchmark(manifest, repo_dir=tmp_path, work_root=tmp_path / "w")


def test_H4_plan_order(tmp_path):
    bench = _make_harness(tmp_path)
    plan = bench.plan()
    assert plan["run_count"] == 20
    runs = plan["runs"]
    for prompt_id in {f"P{i:02d}" for i in range(1, 11)}:
        pair = [r for r in runs if r["prompt_id"] == prompt_id]
        assert len(pair) == 2
        expected = _expected_order(prompt_id)
        assert [r["profile"] for r in pair] == expected
        assert pair[0]["execution_index"] % 2 == 0


def _expected_order(prompt_id):
    num = int(prompt_id[1:])
    if num % 2 == 1:
        return [vb.Q8_PROFILE, vb.BF16_PROFILE]
    return [vb.BF16_PROFILE, vb.Q8_PROFILE]


def test_H5_same_host_resume_safe_skip(tmp_path, monkeypatch):
    bench = _make_harness(tmp_path)
    item = bench.run_plan[0]
    _write_passed_record(bench, item.prompt_id, item.profile)
    assert bench._is_resume_safe(item) is True


def test_H6_cross_host_resume_fail_closed(tmp_path):
    bench = _make_harness(tmp_path)
    item = bench.run_plan[0]
    rec = _write_passed_record(bench, item.prompt_id, item.profile)
    other = vb.HostFingerprint(
        hostname="host-B",
        linux_boot_id="boot-2",
        cpu_model="Other CPU",
        mem_total_kb=16 * 1024 * 1024,
    )
    bench.fingerprint = other
    assert bench._is_resume_safe(item) is False


def test_H7_corrupted_png_not_skipped(tmp_path):
    bench = _make_harness(tmp_path)
    rec = _write_passed_record(
        bench, bench.run_plan[0].prompt_id, bench.run_plan[0].profile, png_sha="sha"
    )
    img = bench.root.image_path(bench.run_plan[0].prompt_id, bench.run_plan[0].profile)
    img.write_bytes(b"CHANGED")
    assert bench._is_resume_safe(bench.run_plan[0]) is False


def test_H8_aggregate_rebuild_nondestructive(tmp_path, monkeypatch):
    bench = _make_harness(tmp_path)
    monkeypatch.setattr(bench, "_resolve_runtime", lambda: tmp_path / "nonexistent")
    full = _write_passed_record(bench, "P01", vb.Q8_PROFILE)
    full2 = _write_passed_record(bench, "P01", vb.BF16_PROFILE)
    agg = bench.rebuild_aggregate([full, full2, bench._failed_record(
        next(i for i in bench.run_plan if i.prompt_id == "P02"), RuntimeError("x"))])
    assert agg["total_pairs"] == 1
    assert agg["pairs"][0]["prompt_id"] == "P01"


def test_H9_atomic_persistence(tmp_path):
    path = tmp_path / "out.json"
    vb.atomic_write_json(path, {"a": 1})
    assert path.is_file()
    assert json.loads(path.read_text()) == {"a": 1}
    leftovers = [p for p in tmp_path.iterdir() if ".tmp" in p.name]
    assert leftovers == []


def test_H10_blinding(tmp_path):
    bench = _make_harness(tmp_path)
    for prompt in bench.manifest.prompts:
        _write_passed_record(bench, prompt["id"], vb.Q8_PROFILE)
        _write_passed_record(bench, prompt["id"], vb.BF16_PROFILE)
    out = tmp_path / "pkg"
    result = bench.finalize_blind_package(out)
    pub = json.loads((out / "public-blind-map.json").read_text())
    pub_text = (out / "public-blind-map.json").read_text()
    assert vb.Q8_PROFILE not in pub_text
    assert vb.BF16_PROFILE not in pub_text
    commitment_file = out / "private-truth-map.json.sha256"
    with open(commitment_file) as f:
        commitment = f.read().strip()
    private_bytes = json.dumps(
        {
            pid: {"A": vb.Q8_PROFILE, "B": vb.BF16_PROFILE}
            for pid in sorted({f"P{i:02d}" for i in range(1, 11)})
        },
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")
    import hashlib
    assert commitment == hashlib.sha256(private_bytes).hexdigest()


def test_H11_no_real_inference_in_unit_tests(tmp_path, monkeypatch):
    called = []
    def fake_gen(sd_cli, manifest, backend_spec, **kw):
        called.append(kw["prompt"])
        return None  # not used because fake run_generation is real
    monkeypatch.setattr(vb, "run_generation", fake_gen)
    bench = _make_harness(tmp_path)
    assert bench.fake is True
    plan = bench.plan()
    assert plan["run_count"] == 20


def test_e2e_run_with_fake_does_not_require_real_runtime(tmp_path, monkeypatch):
    bench = _make_harness(tmp_path)
    monkeypatch.setattr(vb, "_source_head", lambda repo_dir: HEAD)
    monkeypatch.setattr(
        bench,
        "_resolve_runtime",
        lambda: (tmp_path / "work" / ".runs-mock"),
    )
    monkeypatch.setattr(
        bench,
        "_verify_runtime",
        lambda p: _FakeRuntime(p),
    )
    agg = bench.run()
    assert agg["total_pairs"] == 10
    for pair in agg["pairs"]:
        assert pair["q8"]["status"] == "passed"
        assert pair["bf16"]["status"] == "passed"
        assert pair["q8"]["png_sha256"]
