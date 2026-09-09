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


def test_readme_exposes_strict_two_by_two_matrix():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "`q8-reference`" in readme
    assert "`bf16-safetensors`" in readme
    assert "CUDA `cuda0`" in readme
    assert "512 → 640 → 768 → 1024" in readme
    assert "CUDA_VISIBLE_DEVICES=0" in readme
    assert "no `cuda1`" in readme
    assert "no `auto-fit`" in readme
