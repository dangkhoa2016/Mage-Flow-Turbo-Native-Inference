from pathlib import Path


def test_single_qualification_reuses_frozen_runtime_and_t4_authority():
    source = Path("integrations/kaggle/qualification.py").read_text(encoding="utf-8")
    assert "runtime_spec_for_backend" in source
    assert "runtime sha256 mismatch for" in source
    assert "_validate_cuda_session_policy" in source
    assert "_validate_runtime_cuda_devices" in source
    assert 'collect_cuda=(backend == "cuda0")' in source
    assert "cuda0 qualification requires positive gpu_peak_mib" in source


def test_single_qualification_records_exact_source_tree_and_session():
    source = Path("integrations/kaggle/qualification.py").read_text(encoding="utf-8")
    assert "source_head, source_tree = _source_identity(repo_dir)" in source
    assert '"source_tree": source_tree' in source
    assert '"cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER")' in source
    assert '"physical_gpus": physical_gpus' in source
    assert '"schema_version": 2' in source
    assert '"release_version": "1.0.0"' in source


def test_single_qualification_stays_prebuilt_only():
    source = Path("integrations/kaggle/qualification.py").read_text(encoding="utf-8")
    assert "RuntimeBuildBackend" not in source
    assert "manager.build(" not in source
    assert "prebuilt runtime is required" in source
