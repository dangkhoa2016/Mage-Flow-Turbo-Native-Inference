import json
from pathlib import Path

from scripts.publication.audit_release_contract import audit_release_contract

ROOT = Path(__file__).resolve().parents[2]
EXPECTED_MATRIX = [
    {"profile": "q8-reference", "backend": "cpu"},
    {"profile": "bf16-safetensors", "backend": "cpu"},
    {"profile": "q8-reference", "backend": "cuda0"},
    {"profile": "bf16-safetensors", "backend": "cuda0"},
]


def test_release_contract_has_no_errors():
    assert audit_release_contract(ROOT) == []


def test_runtime_provenance_declares_four_immutable_cells():
    data = json.loads(
        (ROOT / "runtime" / "RUNTIME-PROVENANCE.json").read_text(encoding="utf-8")
    )
    assert data["schema_version"] == 3
    assert data["release_version"] == "1.0.0"
    assert data["qualification_matrix"] == EXPECTED_MATRIX
    assert data["commit"] == "6b3edaaf32cc19e5bb2d819c788bd557eddc8eba"
    assert "qualification_results" not in data


def _public_release_text() -> str:
    paths = (
        ROOT / "README.md",
        ROOT / "README.vi.md",
        ROOT / "CHANGELOG.md",
        ROOT / "CHANGELOG.vi.md",
        ROOT / "docs" / "RELEASE-NOTES-v1.0.0.md",
        ROOT / "docs" / "RELEASE-NOTES-v1.0.0.vi.md",
        ROOT / "docs" / "kaggle.md",
        ROOT / "docs" / "kaggle.vi.md",
        ROOT / "docs" / "BENCHMARKS-v1.0.0.md",
        ROOT / "docs" / "BENCHMARKS-v1.0.0.vi.md",
    )
    return "\n".join(p.read_text(encoding="utf-8") for p in paths)


def test_public_docs_have_no_pending_release_wording():
    needles = (
        "qualification pending",
        "pending publication",
        "planned first public release",
        "not yet released",
        "đang chờ qualification",
        "đang chờ publish",
        "bản phát hành công khai đầu tiên dự kiến",
    )
    text = _public_release_text().lower()
    for needle in needles:
        assert needle not in text


def test_public_docs_use_new_profile_name_and_native_runtime_wording():
    text = _public_release_text()
    lower = text.lower()
    assert "q8-reference" in text
    assert "bf16-safetensors" in text
    assert "bf16-high-memory-cpu" not in text
    assert "stable-diffusion.cpp" in text
    assert "hugging face transformers inference backend" in lower
    assert (
        "does not provide a hugging face transformers" in lower
        or "không có hugging face transformers" in lower
        or "không cung cấp hugging face transformers" in lower
    )


def test_kaggle_docs_explain_profile_attachments_and_supplemental_reproduction():
    english = (ROOT / "docs" / "kaggle.md").read_text(encoding="utf-8")
    vietnamese = (ROOT / "docs" / "kaggle.vi.md").read_text(encoding="utf-8")
    combined = english + "\n" + vietnamese

    assert "bf16-high-memory-cpu" not in combined
    assert "bf16-safetensors" in english
    assert "bf16-safetensors" in vietnamese

    assert "Kaggle Dataset" in combined
    assert "Kaggle Model" in combined
    assert "qualification_matrix" in combined
    assert "same-session full-reset recovery" in combined
    assert "supplemental reproduction" in combined


def test_kaggle_docs_preserve_exact_add_input_onboarding_contract():
    english = (ROOT / "docs" / "kaggle.md").read_text(encoding="utf-8")
    vietnamese = (ROOT / "docs" / "kaggle.vi.md").read_text(encoding="utf-8")
    combined = english + "\n" + vietnamese

    for slug in (
        "dangkhoa2016/stable-diffusion-cpp-6b3edaa-portable-cpu-runtime",
        "dangkhoa2016/stable-diffusion-cpp-6b3edaa-cuda-t4-runtime",
        "dangkhoa2016/mage-flow-community-mage-flow-turbo",
        "dangkhoa2016/qwen-qwen3-vl-4b-instruct-gguf",
    ):
        assert slug in combined
    for variation in (
        "GGUF / q8-0",
        "GGUF / q4-k-m",
        "PyTorch / vae-only",
        "PyTorch / default",
    ):
        assert variation in combined

    assert "Session options → Accelerator" in combined
    assert "Input → Add Input" in combined
    assert "DO NOT ATTACH Mage-Flow **GGUF / q8-0**" in combined
    assert "PyTorch / default** already contains the BF16 diffusion model and VAE" in combined
    assert "Run → Run All" in combined


def test_notebook_preserves_exact_add_input_onboarding_contract():
    notebook = (ROOT / "notebooks" / "kaggle-production-demo.ipynb").read_text(
        encoding="utf-8"
    )
    for slug in (
        "dangkhoa2016/stable-diffusion-cpp-6b3edaa-portable-cpu-runtime",
        "dangkhoa2016/stable-diffusion-cpp-6b3edaa-cuda-t4-runtime",
        "dangkhoa2016/mage-flow-community-mage-flow-turbo",
        "dangkhoa2016/qwen-qwen3-vl-4b-instruct-gguf",
    ):
        assert slug in notebook
    for text in (
        "Session options → Accelerator",
        "Input → Add Input",
        "GGUF / q8-0",
        "GGUF / q4-k-m",
        "PyTorch / vae-only",
        "PyTorch / default",
        "GPU1_NOT_USED=PASS",
        "Run → Run All",
    ):
        assert text in notebook


def test_readme_exposes_strict_two_by_two_matrix():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "`q8-reference`" in readme
    assert "`bf16-safetensors`" in readme
    assert "CUDA `cuda0`" in readme
    assert "512 → 640 → 768 → 1024" in readme
    assert "CUDA_VISIBLE_DEVICES=0" in readme
    assert "no `cuda1`" in readme
    assert "no `auto-fit`" in readme


def test_release_docs_preserve_measured_evidence_and_equivalence_provenance():
    text = _public_release_text()
    assert "all four retained benchmark cells are fresh" not in text
    assert "measured benchmark evidence source" in text
    assert "final publication source" in text
    assert "qualification-equivalence manifest" in text
    assert "byte-identical" in text
    assert "same-session full-reset recovery" in text
    assert "supplemental reproduction" in text
    for name in (
        "mage-flow-turbo-native-inference-v1.0.0-q8-reference-cpu-same-session-recovery-evidence.tar.gz",
        "mage-flow-turbo-native-inference-v1.0.0-q8-reference-cuda0-evidence.tar.gz",
        "mage-flow-turbo-native-inference-v1.0.0-bf16-safetensors-cpu-evidence.tar.gz",
        "mage-flow-turbo-native-inference-v1.0.0-bf16-safetensors-cuda0-evidence.tar.gz",
        "mage-flow-v1.0.0-strict-2x2-comparison.json",
        "mage-flow-v1.0.0-strict-2x2-release-summary.md",
        "SHA256SUMS.txt",
    ):
        assert name in text
