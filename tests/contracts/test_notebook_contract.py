import json
from pathlib import Path

NOTEBOOK = Path("notebooks/kaggle-production-demo.ipynb")


def _notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _cell_source(cell: dict) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else source


def test_notebook_is_valid_json():
    data = _notebook()
    assert data["nbformat"] == 4
    assert data["nbformat_minor"] >= 4
    assert len(data["cells"]) > 0


def test_notebook_outputs_empty():
    data = _notebook()
    for cell in data["cells"]:
        assert cell.get("outputs") == [] or cell.get("outputs") is None


def test_notebook_execution_counts_null():
    data = _notebook()
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


def test_notebook_exposes_profile_specific_kaggle_attachment_contract():
    data = _notebook()
    source = "\n".join(_cell_source(cell) for cell in data["cells"])
    markdown = "\n".join(
        _cell_source(cell) for cell in data["cells"] if cell["cell_type"] == "markdown"
    )

    assert 'MODEL_PROFILE = "q8-reference"' in source
    assert 'assert MODEL_PROFILE in {"q8-reference", "bf16-safetensors"}' in source
    assert "build_kaggle_manifest(" in source
    assert "profile=MODEL_PROFILE" in source

    assert "Kaggle Datasets" in markdown
    assert "Kaggle Models" in markdown
    assert "GGUF / q8-0" in markdown
    assert "GGUF / q4-k-m" in markdown
    assert "PyTorch / vae-only" in markdown
    assert "PyTorch / default" in markdown
    assert "VAE is already included" in markdown

    assert "AUTO_RUN_LABEL" not in markdown
    assert "RUN_LABEL" not in markdown
    assert "CPU_REFERENCE_SECONDS" not in source
    assert "215.816" not in source


NOTEBOOK_CODE_CELL_SOURCE_SHA256 = "479de5417d0b07f3641f1458d5d639de18b83f966a0bab82e6dc2c0de43357b6"


def _code_cell_digest(data: dict) -> str:
    import hashlib

    payload = [
        {"id": cell.get("id"), "cell_type": cell["cell_type"], "source": cell.get("source", [])}
        for cell in data["cells"]
        if cell["cell_type"] == "code"
    ]
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _markdown_cells(data: dict) -> list[str]:
    return [_cell_source(cell) for cell in data["cells"] if cell["cell_type"] == "markdown"]


def test_notebook_bilingual_publication_contract_and_code_freeze():
    import re

    data = _notebook()
    markdown_cells = _markdown_cells(data)
    markdown = "\n".join(markdown_cells)
    code_cells = [cell for cell in data["cells"] if cell["cell_type"] == "code"]

    assert len(data["cells"]) == 32
    assert len(markdown_cells) == 17
    assert len(code_cells) == 15
    assert _code_cell_digest(data) == NOTEBOOK_CODE_CELL_SOURCE_SHA256
    assert all("### English" in cell and "### Tiếng Việt" in cell for cell in markdown_cells)

    steps = [int(value) for cell in markdown_cells for value in re.findall(r"^## (\d+)\.", cell, re.MULTILINE)]
    assert steps == list(range(1, 16))

    model_guide = markdown_cells[1]
    for needle in (
        "q8-reference",
        "bf16-safetensors",
        "GGUF / q8-0",
        "PyTorch / vae-only",
        "PyTorch / default",
        "GGUF / q4-k-m",
        "VAE is already included",
        "do not attach `GGUF / q8-0`",
        "Khi chạy `bf16-safetensors`, không gắn",
        "từ chối an toàn",
    ):
        assert needle in model_guide

    configuration = markdown_cells[2]
    for setting in (
        "RUN_MODE",
        "MODEL_PROFILE",
        "RESOLUTION_PRESET",
        "RUN_FAIR_COMPARISON_BENCHMARK",
        "ALLOW_SOURCE_BUILD",
        "ENABLE_REST_DEMO",
        "RUN_REST_GENERATION",
        "ENABLE_QUICK_TUNNEL",
        "I_UNDERSTAND_QUICK_TUNNEL_IS_PUBLIC",
        "fresh BF16 T4x2",
        "BF16 T4x2 mới",
    ):
        assert setting in configuration

    for needle in (
        "DISABLED BY DEFAULT",
        "PUBLIC URL",
        "UNAUTHENTICATED DEMO",
        "NOT REQUIRED FOR QUALIFICATION",
        "MẶC ĐỊNH BỊ TẮT",
        "Input → Add Input",
        "Khắc phục sự cố",
        "ORPHAN_PROCESS_GATE=PASS",
        "DEMO_CLEANUP=PASS",
        "FAIR_COMPARISON_BENCHMARK_STATUS=SKIPPED",
        "demo-session-summary.json",
    ):
        assert needle in markdown

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
