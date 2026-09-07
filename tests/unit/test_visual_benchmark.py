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


def _run_item(bench, prompt_id, profile):
    return next(
        i for i in bench.run_plan if i.prompt_id == prompt_id and i.profile == profile
    )


def _write_passed_record(bench, prompt_id, profile, *, png_sha=None):
    item = _run_item(bench, prompt_id, profile)
    attempt_id = f"vis-{prompt_id}-{profile}-attempt-001"
    rec = {
        "status": "passed",
        "benchmark_id": bench.manifest.benchmark_id,
        "manifest_sha256": bench.manifest.sha256,
        "source_head": bench.manifest.source_head,
        "host_fingerprint": bench.fingerprint.to_dict(),
        "prompt_id": prompt_id,
        "prompt_class": item.prompt_class,
        "profile": profile,
        "execution_index": item.execution_index,
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
            "backend": vb.CPU_BACKEND,
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
        "attempt": {
            "attempt_id": attempt_id,
            "run_dir": str(bench.work_root / ".runs" / attempt_id),
            "previous_attempts": [],
        },
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
    records = []
    for prompt in bench.manifest.prompts:
        records.append(_write_passed_record(bench, prompt["id"], vb.Q8_PROFILE))
        records.append(_write_passed_record(bench, prompt["id"], vb.BF16_PROFILE))
    records.append(bench._failed_record(
        next(i for i in bench.run_plan if i.prompt_id == "P02"), RuntimeError("x")))
    agg = bench.rebuild_aggregate(records)
    assert agg["total_pairs"] == 10
    assert [p["prompt_id"] for p in agg["pairs"]] == [f"P{i:02d}" for i in range(1, 11)]
    for pair in agg["pairs"]:
        assert pair["q8"]["status"] == "passed"
        assert pair["bf16"]["status"] == "passed"


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
    truth = tmp_path / "truth"
    result = bench.finalize_blind_package(out, truth_dir=truth)
    assert (truth / "private-truth-map.json").is_file()
    pub = json.loads((out / "public-blind-map.json").read_text())
    pub_text = (out / "public-blind-map.json").read_text()
    assert vb.Q8_PROFILE not in pub_text
    assert vb.BF16_PROFILE not in pub_text
    commitment_file = out / "private-truth-map.json.sha256"
    with open(commitment_file) as f:
        commitment = f.read().strip()
    assert len(commitment) == 64
    # The public package should NOT contain the private truth map
    assert not (out / "private-truth-map.json").exists()


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
# Task 1.2 — prove _run_single() passes real prepared manifests for Q8 AND BF16
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
        request_id = kw["client_request_id"]
        out_dir = P(kw["output_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        fake_png = out_dir / f"{request_id}.png"
        fake_png.write_bytes(b"FAKEPNG")
        return GenerationResult(
            request_id=request_id, seed=kw["seed"], exit_code=0, elapsed_ms=10,
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

    sd_cli = tmp_path / "fake-cli"
    q8_item = next(i for i in bench.run_plan if i.profile == vb.Q8_PROFILE)
    bf16_item = next(i for i in bench.run_plan if i.profile == vb.BF16_PROFILE)

    bench._run_single(q8_item, sd_cli, None, prepared_assets=prepared)
    bench._run_single(bf16_item, sd_cli, None, prepared_assets=prepared)

    assert q8_manifest_passed is q8_assets.manifest, "Q8 must receive its prepared manifest"
    assert bf16_manifest_passed is bf16_assets.manifest, "BF16 must receive its prepared manifest"
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


def _mock_env_gate(monkeypatch):
    monkeypatch.setattr(
        "integrations.kaggle.profiles.validate_profile_environment",
        lambda *a, **kw: object(),
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
    _mock_env_gate(monkeypatch)
    monkeypatch.setattr(
        "integrations.kaggle.input_adapter.build_kaggle_manifest",
        lambda *a, **kw: (_ for _ in ()).throw(vb.VisualBenchmarkError("input not found")),
    )
    with pytest.raises(Exception):
        bench._prepare_all_profile_assets()


def test_preflight_bf16_input_discovery_fails(tmp_path, monkeypatch):
    bench = _preflight_bench(tmp_path)
    _mock_env_gate(monkeypatch)
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
    _mock_env_gate(monkeypatch)
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
    _mock_env_gate(monkeypatch)
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
# Task 4 — true randomized blind A/B assignment tests
# ---------------------------------------------------------------------------


def _seeded_bench_for_blinding(tmp_path):
    manifest_path = _make_template(tmp_path)
    bench = vb.VisualBenchmark(
        manifest_path,
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


def test_BLD1_public_map_hides_identities(tmp_path, monkeypatch):
    bench = _seeded_bench_for_blinding(tmp_path)
    for prompt in bench.manifest.prompts:
        _write_passed_record(bench, prompt["id"], vb.Q8_PROFILE)
        _write_passed_record(bench, prompt["id"], vb.BF16_PROFILE)
    out = tmp_path / "pkg"
    truth = tmp_path / "truth"
    bench.finalize_blind_package(out, truth_dir=truth)
    public_text = (out / "public-blind-map.json").read_text()
    assert "q8-reference" not in public_text
    assert "bf16-high-memory-cpu" not in public_text


def test_BLD2_random_assignment_not_hardcoded(tmp_path, monkeypatch):
    import secrets
    bits = iter([0, 1, 0, 1, 1, 0, 1, 0, 0, 1])
    monkeypatch.setattr(secrets, "randbits", lambda n: next(bits))
    bench = _seeded_bench_for_blinding(tmp_path)
    for prompt in bench.manifest.prompts:
        _write_passed_record(bench, prompt["id"], vb.Q8_PROFILE)
        _write_passed_record(bench, prompt["id"], vb.BF16_PROFILE)
    out = tmp_path / "pkg"
    truth = tmp_path / "truth"
    bench.finalize_blind_package(out, truth_dir=truth)
    private = json.loads((truth / "private-truth-map.json").read_text())["pairs"]
    a_is_q8 = [pid for pid in sorted(private) if private[pid]["A"] == vb.Q8_PROFILE]
    a_is_bf16 = [pid for pid in sorted(private) if private[pid]["A"] == vb.BF16_PROFILE]
    assert a_is_q8, "no pair has A=Q8; assignment is not random/hard-coded"
    assert a_is_bf16, "no pair has A=BF16; assignment is not random/hard-coded"


def test_BLD3_private_truth_map_exists(tmp_path, monkeypatch):
    bench = _seeded_bench_for_blinding(tmp_path)
    for prompt in bench.manifest.prompts:
        _write_passed_record(bench, prompt["id"], vb.Q8_PROFILE)
        _write_passed_record(bench, prompt["id"], vb.BF16_PROFILE)
    out = tmp_path / "pkg"
    truth = tmp_path / "truth"
    bench.finalize_blind_package(out, truth_dir=truth)
    private_path = truth / "private-truth-map.json"
    assert private_path.is_file()
    data = json.loads(private_path.read_text())
    assert data["benchmark_id"] == bench.manifest.benchmark_id
    assert data["manifest_sha256"] == bench.manifest.sha256
    assert data["source_head"] == bench.manifest.source_head
    assert set(data["pairs"].keys()) == {f"P{i:02d}" for i in range(1, 11)}


def test_BLD4_private_map_absent_from_public_output(tmp_path, monkeypatch):
    bench = _seeded_bench_for_blinding(tmp_path)
    for prompt in bench.manifest.prompts:
        _write_passed_record(bench, prompt["id"], vb.Q8_PROFILE)
        _write_passed_record(bench, prompt["id"], vb.BF16_PROFILE)
    out = tmp_path / "pkg"
    truth = tmp_path / "truth"
    bench.finalize_blind_package(out, truth_dir=truth)
    assert not (out / "private-truth-map.json").exists()
    assert (out / "private-truth-map.json.sha256").is_file()


def test_BLD5_commitment_verifies(tmp_path, monkeypatch):
    bench = _seeded_bench_for_blinding(tmp_path)
    for prompt in bench.manifest.prompts:
        _write_passed_record(bench, prompt["id"], vb.Q8_PROFILE)
        _write_passed_record(bench, prompt["id"], vb.BF16_PROFILE)
    out = tmp_path / "pkg"
    truth = tmp_path / "truth"
    bench.finalize_blind_package(out, truth_dir=truth)
    assert vb.verify_truth_map_commitment(
        truth / "private-truth-map.json",
        out / "private-truth-map.json.sha256",
    ) is True


def test_BLD6_modified_truth_map_fails_commitment(tmp_path, monkeypatch):
    bench = _seeded_bench_for_blinding(tmp_path)
    for prompt in bench.manifest.prompts:
        _write_passed_record(bench, prompt["id"], vb.Q8_PROFILE)
        _write_passed_record(bench, prompt["id"], vb.BF16_PROFILE)
    out = tmp_path / "pkg"
    truth = tmp_path / "truth"
    bench.finalize_blind_package(out, truth_dir=truth)
    private = truth / "private-truth-map.json"
    data = json.loads(private.read_text())
    pid = sorted(data["pairs"])[0]
    if data["pairs"][pid]["A"] == vb.Q8_PROFILE:
        data["pairs"][pid]["A"] = vb.BF16_PROFILE
    else:
        data["pairs"][pid]["A"] = vb.Q8_PROFILE
    private.write_text(json.dumps(data), encoding="utf-8")
    assert vb.verify_truth_map_commitment(
        private, out / "private-truth-map.json.sha256"
    ) is False


def test_BLD7_public_image_hashes_match_blind_images(tmp_path, monkeypatch):
    bench = _seeded_bench_for_blinding(tmp_path)
    for prompt in bench.manifest.prompts:
        _write_passed_record(bench, prompt["id"], vb.Q8_PROFILE)
        _write_passed_record(bench, prompt["id"], vb.BF16_PROFILE)
    out = tmp_path / "pkg"
    truth = tmp_path / "truth"
    bench.finalize_blind_package(out, truth_dir=truth)
    public = json.loads((out / "public-blind-map.json").read_text())
    for pid, info in public.items():
        a_img = out / "blind" / pid / info["a"]
        b_img = out / "blind" / pid / info["b"]
        import hashlib
        assert hashlib.sha256(a_img.read_bytes()).hexdigest() == info["a_png_sha256"]
        assert hashlib.sha256(b_img.read_bytes()).hexdigest() == info["b_png_sha256"]


def test_BLD8_unsafe_truth_output_overlap_fails_closed(tmp_path, monkeypatch):
    import secrets
    monkeypatch.setattr(secrets, "randbits", lambda n: 0)
    bench = _seeded_bench_for_blinding(tmp_path)
    for prompt in bench.manifest.prompts:
        _write_passed_record(bench, prompt["id"], vb.Q8_PROFILE)
        _write_passed_record(bench, prompt["id"], vb.BF16_PROFILE)
    same = tmp_path / "pkg"
    with pytest.raises(vb.VisualBenchmarkError):
        bench.finalize_blind_package(same, truth_dir=same)
    truth = tmp_path / "truth"
    out_in_truth = truth / "inner" / "pkg"
    with pytest.raises(vb.VisualBenchmarkError):
        bench.finalize_blind_package(out_in_truth, truth_dir=truth)
    truth_in_out = same / "inner" / "truth"
    with pytest.raises(vb.VisualBenchmarkError):
        bench.finalize_blind_package(same, truth_dir=truth_in_out)


# ---------------------------------------------------------------------------
# contract hardening 1 — full canonical run-record contract (resume identity)
# ---------------------------------------------------------------------------


def _item_for_profile(bench, profile):
    return next(i for i in bench.run_plan if i.profile == profile)


def _tamper_and_reload(bench, item, mutator):
    rec = _write_passed_record(bench, item.prompt_id, item.profile)
    mutator(rec)
    path = bench.root.record_path(item.prompt_id, item.profile)
    path.write_text(json.dumps(rec), encoding="utf-8")
    return rec


CONTRACT_TAMPERS = [
    (
        "q8-diffusion",
        vb.Q8_PROFILE,
        lambda r: r["models"]["diffusion"].update({"sha256": "1" * 64}),
    ),
    (
        "bf16-diffusion",
        vb.BF16_PROFILE,
        lambda r: r["models"]["diffusion"].update({"sha256": "1" * 64}),
    ),
    (
        "text-encoder",
        vb.Q8_PROFILE,
        lambda r: r["models"]["text_encoder"].update({"sha256": "1" * 64}),
    ),
    (
        "vae",
        vb.Q8_PROFILE,
        lambda r: r["models"]["vae"].update({"sha256": "1" * 64}),
    ),
    (
        "runtime-sha",
        vb.Q8_PROFILE,
        lambda r: r["runtime"].update({"sha256": "1" * 64}),
    ),
    (
        "runtime-commit",
        vb.Q8_PROFILE,
        lambda r: r["runtime"].update({"commit": "1" * 40}),
    ),
    (
        "backend",
        vb.Q8_PROFILE,
        lambda r: r["request"].update({"backend": "gpu"}),
    ),
    (
        "request-seed",
        vb.Q8_PROFILE,
        lambda r: r["request"].update({"seed": r["request"]["seed"] + 1}),
    ),
    (
        "missing-png-sha",
        vb.Q8_PROFILE,
        lambda r: r.pop("png_sha256"),
    ),
    (
        "canonical-png-path-mismatch",
        vb.Q8_PROFILE,
        lambda r: r.update({"png_path": "/tmp/not-the-canonical.png"}),
    ),
]


@pytest.mark.parametrize(
    "case,profile,mutator",
    CONTRACT_TAMPERS,
    ids=[c[0] for c in CONTRACT_TAMPERS],
)
def test_CONTRACT_tampered_field_not_resumable(tmp_path, case, profile, mutator):
    bench = _make_harness(tmp_path)
    item = _item_for_profile(bench, profile)
    _tamper_and_reload(bench, item, mutator)
    assert bench._is_resume_safe(item) is False


def test_CONTRACT_untampered_record_is_resumable_for_both_profiles(tmp_path):
    bench = _make_harness(tmp_path)
    for profile in (vb.Q8_PROFILE, vb.BF16_PROFILE):
        item = _item_for_profile(bench, profile)
        _write_passed_record(bench, item.prompt_id, item.profile)
        assert bench._is_resume_safe(item) is True


# ---------------------------------------------------------------------------
# contract hardening 2 — blind package must use the canonical contract validator
# ---------------------------------------------------------------------------


def _seed_all_pair_records(bench):
    for prompt in bench.manifest.prompts:
        _write_passed_record(bench, prompt["id"], vb.Q8_PROFILE)
        _write_passed_record(bench, prompt["id"], vb.BF16_PROFILE)


def test_BLIND_reject_tampered_shared_vae_sha(tmp_path):
    bench = _make_harness(tmp_path)
    _seed_all_pair_records(bench)
    rec_path = bench.root.record_path("P01", vb.Q8_PROFILE)
    rec = json.loads(rec_path.read_text())
    rec["models"]["vae"]["sha256"] = "1" * 64
    rec_path.write_text(json.dumps(rec), encoding="utf-8")
    with pytest.raises(vb.VisualBenchmarkError, match="19 valid passed runs; need 20"):
        bench.finalize_blind_package(tmp_path / "pkg", truth_dir=tmp_path / "truth")


def test_BLIND_reject_tampered_request_backend(tmp_path):
    bench = _make_harness(tmp_path)
    _seed_all_pair_records(bench)
    rec_path = bench.root.record_path("P05", vb.BF16_PROFILE)
    rec = json.loads(rec_path.read_text())
    rec["request"]["backend"] = "gpu"
    rec_path.write_text(json.dumps(rec), encoding="utf-8")
    with pytest.raises(vb.VisualBenchmarkError, match="19 valid passed runs; need 20"):
        bench.finalize_blind_package(tmp_path / "pkg", truth_dir=tmp_path / "truth")


def test_BLIND_accepts_fully_valid_records(tmp_path):
    bench = _make_harness(tmp_path)
    _seed_all_pair_records(bench)
    result = bench.finalize_blind_package(tmp_path / "pkg", truth_dir=tmp_path / "truth")
    assert (tmp_path / "pkg" / "public-blind-map.json").is_file()


# ---------------------------------------------------------------------------
# contract hardening 3 — aggregate must use the canonical contract validator
# ---------------------------------------------------------------------------


def test_AGGREGATE_rejects_tampered_shared_text_encoder_sha(tmp_path):
    bench = _make_harness(tmp_path)
    q8 = _write_passed_record(bench, "P01", vb.Q8_PROFILE)
    bf16 = _write_passed_record(bench, "P01", vb.BF16_PROFILE)
    q8["models"]["text_encoder"]["sha256"] = "1" * 64
    with pytest.raises(vb.VisualBenchmarkError, match="incomplete pairs"):
        bench.rebuild_aggregate([q8, bf16])


def test_AGGREGATE_rejects_tampered_request_seed(tmp_path):
    bench = _make_harness(tmp_path)
    q8 = _write_passed_record(bench, "P02", vb.Q8_PROFILE)
    bf16 = _write_passed_record(bench, "P02", vb.BF16_PROFILE)
    q8["request"]["seed"] += 1
    with pytest.raises(vb.VisualBenchmarkError, match="incomplete pairs"):
        bench.rebuild_aggregate([q8, bf16])


def test_AGGREGATE_rejects_prompt_when_both_profiles_are_invalid(tmp_path):
    bench = _make_harness(tmp_path)
    records = []
    for prompt in bench.manifest.prompts:
        records.append(_write_passed_record(bench, prompt["id"], vb.Q8_PROFILE))
        records.append(_write_passed_record(bench, prompt["id"], vb.BF16_PROFILE))
    for rec in records:
        if rec["prompt_id"] != "P05":
            continue
        if rec["profile"] == vb.Q8_PROFILE:
            rec["models"]["text_encoder"]["sha256"] = "1" * 64
        else:
            rec["models"]["vae"]["sha256"] = "1" * 64
    with pytest.raises(vb.VisualBenchmarkError, match="incomplete pairs"):
        bench.rebuild_aggregate(records)


def test_AGGREGATE_complete_20_record_set_succeeds(tmp_path):
    bench = _make_harness(tmp_path)
    records = []
    for prompt in bench.manifest.prompts:
        records.append(_write_passed_record(bench, prompt["id"], vb.Q8_PROFILE))
        records.append(_write_passed_record(bench, prompt["id"], vb.BF16_PROFILE))
    agg = bench.rebuild_aggregate(records)
    assert agg["total_pairs"] == 10
    assert {p["prompt_id"] for p in agg["pairs"]} == {
        f"P{i:02d}" for i in range(1, 11)
    }
    for pair in agg["pairs"]:
        assert pair["q8"]["profile"] == vb.Q8_PROFILE
        assert pair["bf16"]["profile"] == vb.BF16_PROFILE
        assert pair["q8"]["status"] == "passed"
        assert pair["bf16"]["status"] == "passed"


# ---------------------------------------------------------------------------
# contract hardening 4 — retry attempt isolation regression
# ---------------------------------------------------------------------------


def _retry_prepared_assets(tmp_path):
    return {
        vb.Q8_PROFILE: _make_prepared(vb.Q8_PROFILE, tmp_path),
        vb.BF16_PROFILE: _make_prepared(vb.BF16_PROFILE, tmp_path),
    }


def test_RETRY_failed_attempt_then_new_run_dir_succeeds(tmp_path, monkeypatch):
    from mageflow_native.inference.runner import GenerationResult, ArtifactInfo

    bench = _make_harness(tmp_path)
    monkeypatch.setattr(vb, "_source_head", lambda repo_dir: HEAD)
    monkeypatch.setattr(
        bench,
        "_resolve_runtime",
        lambda: (tmp_path / "work" / ".runs-mock"),
    )
    monkeypatch.setattr(bench, "_verify_runtime", lambda p: _FakeRuntime(p))
    monkeypatch.setattr(
        bench,
        "_prepare_all_profile_assets",
        lambda: _retry_prepared_assets(tmp_path),
    )

    target = next(
        i for i in bench.run_plan
        if i.prompt_id == "P05" and i.profile == vb.Q8_PROFILE
    )
    target_prefix = f"vis-{target.prompt_id}-{target.profile}"
    real_run_gen = vb.run_generation
    fail_state = {"count": 0}

    def flaky_gen(sd_cli, manifest, backend_spec, **kw):
        request_id = kw["client_request_id"]
        runs_dir = Path(kw["runs_dir"])
        if not request_id.startswith(target_prefix):
            return real_run_gen(sd_cli, manifest, backend_spec, **kw)
        run_dir = runs_dir / request_id
        if run_dir.exists():
            raise FileExistsError(f"run dir already exists: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=False)
        if fail_state["count"] == 0:
            fail_state["count"] += 1
            raise RuntimeError("boom: first attempt failed")
        fail_state["count"] += 1
        out_dir = Path(kw["output_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{request_id}.png"
        out_path.write_bytes(b"FAKEPNG")
        return GenerationResult(
            request_id=request_id, seed=kw["seed"], exit_code=0, elapsed_ms=10,
            peak_sd_cli_rss_kb=100, minimum_mem_available_kb=100,
            gpu_peak_mib=None,
            artifact=ArtifactInfo(
                filename=out_path.name, bytes=out_path.stat().st_size,
                sha256="aa", width=1, height=1,
            ),
            stdout_path="", stderr_path="",
        )

    monkeypatch.setattr(vb, "run_generation", flaky_gen)

    # First run: target attempt-001 fails; other 19 items pass.
    with pytest.raises(vb.VisualBenchmarkError, match="incomplete pairs"):
        bench.run()

    attempt_001_name = f"vis-{target.prompt_id}-{target.profile}-attempt-001"
    run_dir_001 = tmp_path / "work" / ".runs" / attempt_001_name
    assert run_dir_001.is_dir(), "attempt-001 physical run dir must exist"

    failed_rec = json.loads(
        bench.root.record_path(target.prompt_id, target.profile).read_text()
    )
    assert failed_rec["status"] == "failed"
    assert failed_rec["attempt"]["attempt_id"] == attempt_001_name

    # Second run: retry must allocate attempt-002 and succeed.
    agg = bench.run()
    assert agg["total_pairs"] == 10

    passed_rec = json.loads(
        bench.root.record_path(target.prompt_id, target.profile).read_text()
    )
    assert passed_rec["status"] == "passed"
    attempt = passed_rec["attempt"]
    assert attempt["attempt_id"] == (
        f"vis-{target.prompt_id}-{target.profile}-attempt-002"
    )
    assert attempt_001_name in attempt["previous_attempts"]
    assert attempt["run_dir"] != str(run_dir_001)
    assert run_dir_001.is_dir(), "attempt-001 directory must be preserved"
    attempt_002_dir = tmp_path / "work" / ".runs" / attempt["attempt_id"]
    assert attempt_002_dir.is_dir(), "retry must create a distinct physical run dir"
