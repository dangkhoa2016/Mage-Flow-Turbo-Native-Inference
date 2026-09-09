# Mage-Flow-Turbo-Native-Inference v1.0.0 Release Notes

[![CI](https://github.com/dangkhoa2016/Mage-Flow-Turbo-Native-Inference/actions/workflows/ci.yml/badge.svg)](https://github.com/dangkhoa2016/Mage-Flow-Turbo-Native-Inference/actions/workflows/ci.yml)
[![License](https://img.shields.io/github/license/dangkhoa2016/Mage-Flow-Turbo-Native-Inference)](../LICENSE)

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](RELEASE-NOTES-v1.0.0.vi.md)

## Overview

Mage-Flow-Turbo-Native-Inference is a **portable native inference and deployment stack for Mage-Flow-Turbo**. Python provides configuration, validation, CLI/REST orchestration, lifecycle control and evidence collection; actual inference is executed by the native `stable-diffusion.cpp` `sd-cli`. The project does not define, train or fine-tune model weights.

v1.0.0 is the first public release of this lineage.

## Release qualification targets

The following targets are required to pass exact-head qualification before the v1.0.0 GitHub Release is published:

- **Linux x86-64 CPU** — canonical native CPU target.
- **NVIDIA CUDA `cuda0`** — canonical single-GPU CUDA target.
- **Kaggle CPU notebook** — CPU adapter/reference target.
- **Kaggle T4/T4x2 notebook** — CUDA adapter/reference target using physical GPU 0 only.

Multi-GPU inference, Vulkan, Metal, ROCm, SYCL and Windows are outside the v1.0.0 release qualification scope.

## Highlights

- JSON model manifest with fail-closed SHA-256 verification of every canonical component.
- Portable runtime manager for pinned `stable-diffusion.cpp` `sd-cli` on CPU and CUDA.
- Explicit backend/parameter placement and deterministic qualification placement.
- `mageflow-native` CLI for diagnostics, verification, generation, serving and runtime lifecycle.
- Loopback REST API with stable health/readiness/info/generation/artifact endpoints.
- Kaggle integration isolated to `integrations/kaggle/`; no hard Kaggle path dependency in the generic core.
- Portable CPU/CUDA qualification harnesses with structured RAM/VRAM telemetry and exact argv capture.
- Sanitized clean-room-verifiable evidence archives with manifest and sidecar checksums.
- Model-weight-free CI, public-surface audit, public-history invariant verification and release-contract auditing.

## Frozen technical contract

```text
stable-diffusion.cpp = 6b3edaaf32cc19e5bb2d819c788bd557eddc8eba
DiT Q8 SHA256         = 4c3dafc143ee64121692b6b63563a4f5288bf6183c4870e1d65f1566519ba7f0
Qwen Q4_K_M SHA256    = 66358cb18bb6b3b1b6675aa412c7a88ef01d228f481184d13668e5201c730a0a
VAE-only SHA256       = 34e076dc1e8a15321e1e07be5111d59cf16dd10b804b7c7e20b4de29013427e0
canonical size        = 512x512
seed / steps / CFG    = 42 / 4 / 1.0
threads               = 4
```

## Evidence and provenance

After all required exact-head gates pass, CPU/CUDA qualification artifacts are published as GitHub Release assets. They include evidence archives and sidecars, executed notebooks, acceptance PNGs, release provenance and a `SHA256SUMS` file covering every custom release asset. No model weights are included.

Measured wall times, peak RAM/VRAM, runtime binary SHA-256 values, acceptance PNG hashes, PASS/FAIL results and GitHub Actions run IDs are intentionally recorded in release assets and the GitHub Release body rather than in mutable source documentation.
