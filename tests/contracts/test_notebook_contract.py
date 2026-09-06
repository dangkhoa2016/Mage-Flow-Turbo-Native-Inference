import json
from pathlib import Path

NOTEBOOK = Path("notebooks/kaggle-production-demo.ipynb")


def test_notebook_is_valid_json():
    data = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert data["nbformat"] == 4
    assert data["nbformat_minor"] >= 4
    assert len(data["cells"]) > 0


def test_notebook_outputs_empty():
    data = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for cell in data["cells"]:
        assert cell.get("outputs") == [] or cell.get("outputs") is None


def test_notebook_execution_counts_null():
    data = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for cell in data["cells"]:
        if cell["cell_type"] == "code":
            assert cell.get("execution_count") is None


def test_notebook_clones_native_inference_repo():
    text = NOTEBOOK.read_text(encoding="utf-8")
    assert "Mage-Flow-Turbo-Native-Inference.git" in text
    assert "integrations.kaggle.input_adapter" in text
    assert "from integrations.kaggle.input_adapter import build_kaggle_manifest" in text


def test_notebook_references_generic_manifest_location():
    text = NOTEBOOK.read_text(encoding="utf-8")
    assert "build_kaggle_manifest" in text
    assert "ALLOW_SOURCE_BUILD = False" in text
