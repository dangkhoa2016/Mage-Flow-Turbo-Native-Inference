import subprocess

import pytest

from integrations.kaggle import qualification_matrix as qm


class _Completed:
    def __init__(self, stdout: str):
        self.stdout = stdout


def test_t4_policy_accepts_single_t4(monkeypatch):
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(
        qm.subprocess,
        "run",
        lambda *args, **kwargs: _Completed("0, Tesla T4\n"),
    )
    assert qm._validate_cuda_session_policy("cuda0") == [
        {"index": 0, "name": "Tesla T4"}
    ]


def test_t4_policy_accepts_t4x2_but_keeps_slot_zero_visible(monkeypatch):
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(
        qm.subprocess,
        "run",
        lambda *args, **kwargs: _Completed("0, Tesla T4\n1, Tesla T4\n"),
    )
    assert qm._validate_cuda_session_policy("cuda0") == [
        {"index": 0, "name": "Tesla T4"},
        {"index": 1, "name": "Tesla T4"},
    ]


def test_t4_policy_rejects_wrong_device_order(monkeypatch):
    monkeypatch.delenv("CUDA_DEVICE_ORDER", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(ValueError, match="CUDA_DEVICE_ORDER=PCI_BUS_ID"):
        qm._validate_cuda_session_policy("cuda0")


def test_t4_policy_rejects_non_t4(monkeypatch):
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(
        qm.subprocess,
        "run",
        lambda *args, **kwargs: _Completed("0, NVIDIA L4\n"),
    )
    with pytest.raises(ValueError, match="requires only NVIDIA T4"):
        qm._validate_cuda_session_policy("cuda0")


def test_t4_policy_rejects_more_than_two_physical_gpus(monkeypatch):
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(
        qm.subprocess,
        "run",
        lambda *args, **kwargs: _Completed(
            "0, Tesla T4\n1, Tesla T4\n2, Tesla T4\n"
        ),
    )
    with pytest.raises(ValueError, match="detected 3 GPUs"):
        qm._validate_cuda_session_policy("cuda0")


def test_runtime_devices_must_expose_only_cuda0():
    qm._validate_runtime_cuda_devices("cuda0", "cuda0 NVIDIA Tesla T4")
    with pytest.raises(ValueError, match="did not expose cuda0"):
        qm._validate_runtime_cuda_devices("cuda0", "CPU Intel Xeon")
    with pytest.raises(ValueError, match="must not expose cuda1"):
        qm._validate_runtime_cuda_devices(
            "cuda0", "cuda0 NVIDIA Tesla T4\ncuda1 NVIDIA Tesla T4"
        )


def test_cpu_policy_does_not_probe_nvidia(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("nvidia-smi must not be called for CPU qualification")

    monkeypatch.setattr(qm.subprocess, "run", fail)
    assert qm._validate_cuda_session_policy("cpu") == []
