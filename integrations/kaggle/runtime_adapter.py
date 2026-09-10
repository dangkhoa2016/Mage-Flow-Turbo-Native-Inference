from __future__ import annotations

import os
from pathlib import Path

from mageflow_native.config import default_runtime_root


def kaggle_cache_root() -> Path:
    if os.environ.get("MAGE_RUNTIME_ROOT"):
        return Path(os.environ["MAGE_RUNTIME_ROOT"]).expanduser()
    work = Path("/kaggle/working")
    if work.is_dir():
        return work / ".mageflow-native"
    return default_runtime_root()


def runtime_hint(backend: str) -> str | None:
    hint = os.environ.get("MAGE_SD_CLI")
    if hint and Path(hint).is_file():
        return hint
    env_name = {
        "cpu": "MAGE_CPU_PREBUILT_SD_CLI",
        "cuda0": "MAGE_CUDA_PREBUILT_SD_CLI",
    }.get(backend)
    if env_name is None:
        return None
    probe = os.environ.get(env_name)
    if probe and Path(probe).is_file():
        return probe
    return None
