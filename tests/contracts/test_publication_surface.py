import builtins
import json
import subprocess
import symtable
import sys
from pathlib import Path

REPO = Path(".")


def test_public_notebook_has_no_unresolved_module_globals():
    notebook_path = REPO / "notebooks" / "kaggle-production-demo.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))

    code = "\n\n".join(
        "".join(cell.get("source", []))
        if isinstance(cell.get("source", ""), list)
        else str(cell.get("source", ""))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    )

    table = symtable.symtable(code, str(notebook_path), "exec")
    builtins_set = set(dir(builtins))

    unresolved = sorted(
        symbol.get_name()
        for symbol in table.get_symbols()
        if symbol.is_referenced()
        and not (
            symbol.is_assigned()
            or symbol.is_imported()
            or symbol.is_parameter()
            or symbol.is_namespace()
        )
        and symbol.get_name() not in builtins_set
    )

    assert unresolved == [], f"unresolved notebook module globals: {unresolved}"


def test_publication_audit_passes_on_clean_tree():
    p = subprocess.run(
        [sys.executable, "scripts/publication/audit_public_surface.py", "--root", "."],
        capture_output=True,
        text=True,
    )
    text = p.stdout + p.stderr
    assert p.returncode == 0, text


def test_core_has_no_kaggle_paths():
    text = ""
    for p in (REPO / "mageflow_native").rglob("*.py"):
        text += p.read_text(encoding="utf-8", errors="ignore")
    assert "/kaggle/input" not in text
    assert "/kaggle/working" not in text


def test_workflows_use_v7():
    for wf in (REPO / ".github" / "workflows").glob("*.yml"):
        content = wf.read_text(encoding="utf-8")
        for line in content.splitlines():
            line = line.strip()
            if line.startswith("uses:"):
                action = line.split()[1]
                if action.startswith(("actions/checkout", "actions/setup-python", "actions/upload-artifact")):
                    tag = action.split("@")[-1]
                    assert tag.startswith("v7"), f"{wf.name}: {action}"


def test_no_model_weights_tracked():
    for p in REPO.rglob("*"):
        if p.is_file() and ".git" not in p.parts and p.suffix in (".gguf", ".safetensors", ".ckpt", ".pt", ".pth", ".onnx"):
            raise AssertionError(f"model weight tracked: {p}")


def test_script_syntax():
    for p in (REPO / "scripts").rglob("*.sh"):
        subprocess.run(["bash", "-n", str(p)], check=True)


def test_pending_qualification_rejects_positive_public_claims(tmp_path: Path):
    (tmp_path / "mageflow_native").mkdir()
    (tmp_path / "runtime").mkdir()
    (tmp_path / "README.md").write_text(
        "Mage-Flow-Turbo-Native-Inference stable-diffusion.cpp Q8_0 Q4_K_M CPU CUDA Kaggle\n"
        "![Linux CPU](https://img.shields.io/badge/Linux-CPU-qualified-success)\n"
        "| Linux x86-64 | CPU | **Qualified** |\n",
        encoding="utf-8",
    )
    (tmp_path / "runtime" / "RUNTIME-PROVENANCE.json").write_text(
        '{"qualified_backends": [], "qualification_state": {"linux_cpu": "pending", "nvidia_cuda_cuda0": "pending"}}\n',
        encoding="utf-8",
    )
    p = subprocess.run(
        [sys.executable, "scripts/publication/audit_public_surface.py", "--root", str(tmp_path)],
        capture_output=True,
        text=True,
    )
    text = p.stdout + p.stderr
    assert p.returncode == 1, text
    assert "qualification claim contradicts pending evidence state" in text
