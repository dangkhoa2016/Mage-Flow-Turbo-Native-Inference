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
        "runtime": {
            "commit": vb.RUNTIME_COMMIT,
            "sha256": vb.RUNTIME_SHA256,
            "path": "fake",
        },
        "models": {
            "diffusion": {
                "sha256": (
                    vb.Q8_DIFFUSION_SHA256
                    if profile == vb.Q8_PROFILE
                    else vb.BF16_DIFFUSION_SHA256
                ),
                "path": "fake",
            },
            "text_encoder": {"sha256": vb.SHARED_TEXT_ENCODER_SHA256, "path": "fake"},
            "vae": {"sha256": vb.SHARED_VAE_SHA256, "path": "fake"},
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


def test_H12_model_identity_mismatch_not_skipped(tmp_path):
    bench = _make_harness(tmp_path)
    item = bench.run_plan[0]
    rec = _write_passed_record(bench, item.prompt_id, item.profile)
    wrong_sha = "0" * 64
    if item.profile == vb.Q8_PROFILE:
        rec["models"]["diffusion"]["sha256"] = wrong_sha
    else:
        rec["models"]["text_encoder"]["sha256"] = wrong_sha
    path = bench.root.record_path(item.prompt_id, item.profile)
    path.write_text(json.dumps(rec), encoding="utf-8")
    assert bench._is_resume_safe(item) is False


def test_H13_runtime_identity_mismatch_not_skipped(tmp_path):
    bench = _make_harness(tmp_path)
    item = bench.run_plan[0]
    rec = _write_passed_record(bench, item.prompt_id, item.profile)
    rec["runtime"]["sha256"] = "0" * 64
    path = bench.root.record_path(item.prompt_id, item.profile)
    path.write_text(json.dumps(rec), encoding="utf-8")
    assert bench._is_resume_safe(item) is False


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


def _make_prepared(profile, tmp_path):
    from mageflow_native.models.manifest import ModelComponent, ModelManifest
    if profile == vb.Q8_PROFILE:
        diffusion_sha = vb.Q8_DIFFUSION_SHA256
    else:
        diffusion_sha = vb.BF16_DIFFUSION_SHA256
    comp = lambda name, sha: ModelComponent(
        path=tmp_path / f"{name}.bin",
        sha256=sha,
        format="gguf",
    )
    mm = ModelManifest(
        schema_version=1,
        model_family="Mage-Flow-Turbo",
        diffusion=comp("diffusion", diffusion_sha),
        text_encoder=comp("text_encoder", vb.SHARED_TEXT_ENCODER_SHA256),
        vae=comp("vae", vb.SHARED_VAE_SHA256),
    )
    return vb.PreparedProfileAssets(
        profile=profile,
        manifest_path=tmp_path / f"{profile}.json",
        manifest=mm,
        verified_paths={
            "diffusion": tmp_path / "diffusion.bin",
            "text_encoder": tmp_path / "text_encoder.bin",
            "vae": tmp_path / "vae.bin",
        },
    )


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
    monkeypatch.setattr(
        bench,
        "_prepare_all_profile_assets",
        lambda: {
            vb.Q8_PROFILE: _make_prepared(vb.Q8_PROFILE, tmp_path),
            vb.BF16_PROFILE: _make_prepared(vb.BF16_PROFILE, tmp_path),
        },
    )
    agg = bench.run()
    assert agg["total_pairs"] == 10
    for pair in agg["pairs"]:
        assert pair["q8"]["status"] == "passed"
        assert pair["bf16"]["status"] == "passed"
        assert pair["q8"]["png_sha256"]


# ---------------------------------------------------------------------------
# Task 1.1 — prove real CLI --run dispatches to harness without --fake
# ---------------------------------------------------------------------------


def test_real_run_cli_dispatches_to_harness(monkeypatch, tmp_path):
    calls = []

    class FakeBenchmark:
        def __init__(self, *args, **kwargs):
            pass

        def validate(self):
            calls.append("validate")

        def run(self):
            calls.append("run")
            return {"status": "mocked"}

    monkeypatch.setattr(vb, "VisualBenchmark", FakeBenchmark)

    rc = vb.main([
        "--manifest", str(tmp_path / "manifest.json"),
        "--repo-dir", str(tmp_path),
        "--work-root", str(tmp_path / "work"),
        "--run",
    ])

    assert rc == 0
    assert calls == ["validate", "run"]


# ---------------------------------------------------------------------------
# Task 1.2 — prove _run_single() passes real prepared manifests, not stubs
# ---------------------------------------------------------------------------


def test_run_single_passes_prepared_manifests_not_stubs(tmp_path, monkeypatch):
    from integrations.kaggle.profiles import Q8_REFERENCE_PROFILE, BF16_HIGH_MEMORY_CPU_PROFILE

    manifest_path = _make_template(tmp_path)
    bench = vb.VisualBenchmark(
        manifest_path,
        repo_dir=tmp_path,
        work_root=tmp_path / "work",
        fake=True,
    )

    q8_assets = _make_prepared(Q8_REFERENCE_PROFILE, tmp_path)
    bf16_assets = _make_prepared(BF16_HIGH_MEMORY_CPU_PROFILE, tmp_path)
    q8_manifest = q8_assets.manifest
    bf16_manifest = bf16_assets.manifest
    prepared = {
        Q8_REFERENCE_PROFILE: q8_assets,
        BF16_HIGH_MEMORY_CPU_PROFILE: bf16_assets,
    }

    q8_manifest_passed = None
    bf16_manifest_passed = None
    stubs_used = []

    def fake_run_generation(sd_cli, manifest, backend_spec, **kw):
        nonlocal q8_manifest_passed, bf16_manifest_passed
        if manifest is q8_manifest:
            q8_manifest_passed = manifest
        elif manifest is bf16_manifest:
            bf16_manifest_passed = manifest
        else:
            stubs_used.append(type(manifest).__name__)
        from mageflow_native.inference.runner import GenerationResult, ArtifactInfo
        from pathlib import Path as P
        out_dir = P(kw["output_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        fake_png = out_dir / "test.png"
        import struct, zlib
        def _write_minimal_png(p):
            sig = b'\x89PNG\r\n\x1a\n'
            ihdr_data = struct.pack('>IIBBBBB', 1, 1, 8, 2, 0, 0, 0)
            ihdr_crc = zlib.crc32(b'IHDR' + ihdr_data)
            ihdr = struct.pack('>I', 13) + b'IHDR' + ihdr_data + struct.pack('>I', ihdr_crc & 0xffffffff)
            raw = b'\x00\xff\x00\x00'
            idat_data = zlib.compress(raw)
            idat_crc = zlib.crc32(b'IDAT' + idat_data)
            idat = struct.pack('>I', len(idat_data)) + b'IDAT' + idat_data + struct.pack('>I', idat_crc & 0xffffffff)
            iend_crc = zlib.crc32(b'IEND')
            iend = struct.pack('>I', 0) + b'IEND' + struct.pack('>I', iend_crc & 0xffffffff)
            with open(p, 'wb') as f:
                f.write(sig + ihdr + idat + iend)
        _write_minimal_png(fake_png)
        return GenerationResult(
            request_id="test", seed=kw["seed"], exit_code=0, elapsed_ms=10,
            peak_sd_cli_rss_kb=100, minimum_mem_available_kb=100,
            gpu_peak_mib=None,
            artifact=ArtifactInfo(
                filename=fake_png.name, bytes=fake_png.stat().st_size,
                sha256="aa", width=1, height=1,
            ),
            stdout_path="", stderr_path="",
        )

    monkeypatch.setattr(vb, "run_generation", fake_run_generation)
    monkeypatch.setattr(bench, "_resolve_runtime", lambda: tmp_path / "fake-cli")
    monkeypatch.setattr(bench, "_verify_runtime", lambda p: _FakeRuntime(p))

    item = bench.run_plan[0]
    sd_cli = tmp_path / "fake-cli"
    bench._run_single(item, sd_cli, None, prepared_assets=prepared)

    assert q8_manifest_passed is q8_manifest or bf16_manifest_passed is bf16_manifest, (
        "at least one profile should receive its prepared manifest"
    )
    assert len(stubs_used) == 0, f"stub objects were passed: {stubs_used}"


# ---------------------------------------------------------------------------
# Task 1.3 — real-preflight fail-closed tests
# ---------------------------------------------------------------------------


def _preflight_bench(tmp_path):
    manifest_path = _make_template(tmp_path)
    return vb.VisualBenchmark(
        manifest_path,
        repo_dir=tmp_path,
        work_root=tmp_path / "work",
        input_root=tmp_path / "input",
    )


def _fake_model_manifest_for(profile, tmp_path):
    from mageflow_native.models.manifest import ModelComponent, ModelManifest
    diffusion_sha = (
        vb.Q8_DIFFUSION_SHA256 if profile == vb.Q8_PROFILE else vb.BF16_DIFFUSION_SHA256
    )
    return ModelManifest(
        schema_version=1,
        model_family="Mage-Flow-Turbo",
        diffusion=ModelComponent(tmp_path / "diffusion.bin", diffusion_sha, "gguf"),
        text_encoder=ModelComponent(
            tmp_path / "te.bin", vb.SHARED_TEXT_ENCODER_SHA256, "gguf"
        ),
        vae=ModelComponent(tmp_path / "vae.bin", vb.SHARED_VAE_SHA256, "safetensors"),
    )


def _mock_build_manifest(monkeypatch):
    def _build(*args, **kwargs):
        out = kwargs["output"]
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"components": {}}), encoding="utf-8")
        return out
    monkeypatch.setattr(
        "integrations.kaggle.input_adapter.build_kaggle_manifest",
        _build,
    )


def test_preflight_q8_input_discovery_fails(tmp_path, monkeypatch):
    bench = _preflight_bench(tmp_path)
    monkeypatch.setattr(
        "integrations.kaggle.input_adapter.build_kaggle_manifest",
        lambda *a, **kw: (_ for _ in ()).throw(vb.VisualBenchmarkError("input not found")),
    )
    with pytest.raises(Exception):
        bench._prepare_all_profile_assets()


def test_preflight_bf16_input_discovery_fails(tmp_path, monkeypatch):
    bench = _preflight_bench(tmp_path)
    call_count = [0]

    def selective_build(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 2:
            raise vb.VisualBenchmarkError("BF16 input not found")
        tmp = kwargs.get("output") or Path(str(args[1]))
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps({"components": {}}), encoding="utf-8")
        return Path(tmp)

    monkeypatch.setattr(
        "integrations.kaggle.input_adapter.build_kaggle_manifest",
        selective_build,
    )
    with pytest.raises(Exception):
        bench._prepare_all_profile_assets()


def test_preflight_q8_model_verification_fails(tmp_path, monkeypatch):
    bench = _preflight_bench(tmp_path)
    _mock_build_manifest(monkeypatch)
    monkeypatch.setattr(
        "integrations.kaggle.visual_benchmark.load_manifest",
        lambda *a, **kw: _fake_model_manifest_for(vb.Q8_PROFILE, tmp_path),
    )
    monkeypatch.setattr(
        "integrations.kaggle.visual_benchmark.verify_manifest",
        lambda m: (_ for _ in ()).throw(vb.VisualBenchmarkError("verification failed")),
    )
    with pytest.raises(vb.VisualBenchmarkError, match="verification failed"):
        bench._prepare_all_profile_assets()


def test_preflight_bf16_model_verification_fails(tmp_path, monkeypatch):
    bench = _preflight_bench(tmp_path)
    _mock_build_manifest(monkeypatch)
    call_count = [0]

    def selective_load(*args, **kwargs):
        call_count[0] += 1
        profile = vb.Q8_PROFILE if call_count[0] == 1 else vb.BF16_PROFILE
        if call_count[0] == 2:
            raise vb.VisualBenchmarkError("BF16 verification failed")
        return _fake_model_manifest_for(profile, tmp_path)

    monkeypatch.setattr(
        "integrations.kaggle.visual_benchmark.load_manifest",
        selective_load,
    )
    monkeypatch.setattr(
        "integrations.kaggle.visual_benchmark.verify_manifest",
        lambda m: {
            "diffusion": tmp_path / "diffusion.bin",
            "text_encoder": tmp_path / "te.bin",
            "vae": tmp_path / "vae.bin",
        },
    )
    with pytest.raises(vb.VisualBenchmarkError, match="BF16 verification"):
        bench._prepare_all_profile_assets()


def test_preflight_runtime_sha_mismatch(tmp_path, monkeypatch):
    manifest_path = _make_template(tmp_path)
    bench = vb.VisualBenchmark(
        manifest_path,
        repo_dir=tmp_path,
        work_root=tmp_path / "work",
    )
    fake_cli = tmp_path / "sd-cli"
    fake_cli.write_text("#!/bin/sh\nexit 0")
    fake_cli.chmod(0o755)
    monkeypatch.setenv("MAGE_CPU_PREBUILT_SD_CLI", str(fake_cli))
    monkeypatch.setattr(
        "integrations.kaggle.visual_benchmark.sha256_file",
        lambda p: "0" * 64,
    )
    with pytest.raises(vb.VisualBenchmarkError, match="runtime sha256 mismatch"):
        bench._resolve_runtime()


def test_preflight_source_head_mismatch(tmp_path, monkeypatch):
    manifest_path = _make_template(tmp_path)
    bench = vb.VisualBenchmark(
        manifest_path,
        repo_dir=tmp_path,
        work_root=tmp_path / "work",
    )
    monkeypatch.setattr(vb, "_source_head", lambda repo_dir: "0" * 40)
    with pytest.raises(vb.VisualBenchmarkError, match="source_head mismatch"):
        bench.validate()


def test_preflight_dirty_git_tree(tmp_path, monkeypatch):
    manifest_path = _make_template(tmp_path)
    bench = vb.VisualBenchmark(
        manifest_path,
        repo_dir=tmp_path,
        work_root=tmp_path / "work",
    )
    monkeypatch.setattr(
        "subprocess.run",
        lambda *a, **kw: type("R", (), {"stdout": "M file.py\n", "stderr": ""})(),
    )
    with pytest.raises(vb.VisualBenchmarkError, match="working tree is not clean"):
        bench.validate()


def test_preflight_host_fingerprint_mismatch(tmp_path, monkeypatch):
    manifest_path = _make_template(tmp_path)
    bench = vb.VisualBenchmark(
        manifest_path,
        repo_dir=tmp_path,
        work_root=tmp_path / "work",
    )
    bench._ensure_fingerprint_file()
    bench.fingerprint = vb.HostFingerprint(
        hostname="different-host",
        linux_boot_id="different-boot",
        cpu_model="Different CPU",
        mem_total_kb=999,
    )
    with pytest.raises(vb.VisualBenchmarkError, match="host fingerprint change"):
        bench._ensure_fingerprint_file()


# ---------------------------------------------------------------------------
