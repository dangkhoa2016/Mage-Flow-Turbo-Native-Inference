# Mage-Flow-Turbo-Native-Inference v1.0.0 Release Notes

[![CI](https://github.com/dangkhoa2016/Mage-Flow-Turbo-Native-Inference/actions/workflows/ci.yml/badge.svg)](https://github.com/dangkhoa2016/Mage-Flow-Turbo-Native-Inference/actions/workflows/ci.yml)
[![License](https://img.shields.io/github/license/dangkhoa2016/Mage-Flow-Turbo-Native-Inference)](LICENSE)

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](RELEASE-NOTES-v1.0.0.vi.md)

## What this is

Mage-Flow-Turbo-Native-Inference is a **portable native inference and deployment stack for Mage-Flow-Turbo**. Python provides configuration, validation, CLI/REST orchestration, lifecycle control and evidence collection; the actual inference is executed by the native `stable-diffusion.cpp` `sd-cli`. It is not a new model and does not train or fine-tune anything.

v1.0.0 is the planned first public release of this lineage. It is not tagged or released until the exact public source head passes the required CPU/CUDA qualification gates and final evidence is verified.

## Qualification targets (pending)

- **Linux x86-64 CPU** — qualification pending.
- **NVIDIA CUDA** (`cuda0`) — qualification pending.
- **Kaggle CPU notebook** — integration qualification pending.
- **Kaggle T4/T4x2 notebook** — integration qualification pending using `cuda0`.

Vulkan, Metal, ROCm, SYCL, Windows and multi-GPU are documented upstream capabilities but are **not** v1.0.0 qualification targets.

## Highlights

- Generic JSON model manifest with SHA-256 verification of every canonical component, failing closed on any missing/ambiguous/mismatched component.
- Portable runtime manager that resolves, builds and verifies the pinned `stable-diffusion.cpp` `sd-cli` for CPU and CUDA.
- Configurable backend/parameter placement (`cpu`, `cuda0`, validated placement strings, `max_vram`, `split_mode`, `auto_fit`).
- `mageflow-native` CLI: `doctor`, `verify`, `generate`, `serve`, `runtime build`.
- Loopback REST API with stable endpoints and single-flight execution.
- Generic core with no `/kaggle/*` hard dependency; Kaggle is a thin adapter.
- Generic local-layout portability proof and CPU/CUDA qualification harnesses.
- Model-weight-free CI and publication-surface audits (Actions v7).

## Frozen technical contract

```text
stable-diffusion.cpp = 6b3edaaf32cc19e5bb2d819c788bd557eddc8eba
DiT Q8 SHA256         = 4c3dafc143ee64121692b6b63563a4f5288bf6183c4870e1d65f1566519ba7f0
Qwen Q4_K_M SHA256    = 66358cb18bb6b3b1b6675aa412c7a88ef01d228f481184d13668e5201c730a0a
VAE-only SHA256       = 34e076dc1e8a15321e1e07be5111d59cf16dd10b804b7c7e20b4de29013427e0
qualification          = pending: Linux CPU, NVIDIA CUDA (cuda0)
steps / CFG / threads  = 4 / 1.0 / 4
```

## Evidence

After the required external gates pass, CPU/CUDA acceptance evidence, executed notebooks, acceptance PNGs, the final model manifest and runtime provenance will be published as release assets. No model weights will be included. Measured wall times, peak RAM/VRAM and accepted PNG SHA-256 values are recorded only from verified final qualification evidence.
