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


def _notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _cell_source(cell: dict) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else source


def test_notebook_rest_startup_budget_and_failed_start_cleanup_contract():
    data = _notebook()
    code_source = "\n".join(
        _cell_source(cell) for cell in data["cells"] if cell["cell_type"] == "code"
    )
    config_cells = [cell for cell in data["cells"] if cell.get("id") == "mf-02-50857551"]
    startup_cells = [cell for cell in data["cells"] if cell.get("id") == "mf-18-2bb9d53d"]
    assert len(config_cells) == 1
    assert len(startup_cells) == 1
    config = config_cells[0]
    startup = _cell_source(startup_cells[0])

    assert isinstance(config.get("source"), str)
    assert "REST_STARTUP_TIMEOUT_SECONDS = 300" in code_source
    assert "deadline = time.monotonic() + REST_STARTUP_TIMEOUT_SECONDS" in startup
    assert "deadline = time.monotonic() + 30" not in startup
    assert "SERVER_PROC.terminate()" in startup
    assert "SERVER_PROC.wait(timeout=10)" in startup
    assert "SERVER_PROC.kill()" in startup
    assert "SERVER_LOG_HANDLE.flush()" in startup
    assert "SERVER_LOG_HANDLE.close()" in startup
    assert "startup_elapsed" in startup
    assert "pre_cleanup_returncode" in startup
