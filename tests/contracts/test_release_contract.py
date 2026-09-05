from pathlib import Path

from scripts.publication.audit_release_contract import audit_release_contract

ROOT = Path(__file__).resolve().parents[2]


def test_release_contract_has_no_errors():
    assert audit_release_contract(ROOT) == []


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
    paths = (
        ROOT / "README.md",
        ROOT / "README.vi.md",
        ROOT / "CHANGELOG.md",
        ROOT / "CHANGELOG.vi.md",
        ROOT / "docs" / "RELEASE-NOTES-v1.0.0.md",
        ROOT / "docs" / "RELEASE-NOTES-v1.0.0.vi.md",
    )
    text = "\n".join(p.read_text(encoding="utf-8").lower() for p in paths)
    for needle in needles:
        assert needle not in text
