# Changelog

[![CI](https://github.com/dangkhoa2016/Mage-Flow-Turbo-Native-Inference/actions/workflows/ci.yml/badge.svg)](https://github.com/dangkhoa2016/Mage-Flow-Turbo-Native-Inference/actions/workflows/ci.yml)
[![License](https://img.shields.io/github/license/dangkhoa2016/Mage-Flow-Turbo-Native-Inference)](LICENSE)

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](CHANGELOG.vi.md)

## v1.0.0

First public release of **Mage-Flow-Turbo-Native-Inference**, a portable native inference and deployment stack for Mage-Flow-Turbo.

- JSON model manifest with SHA-256 verification for every canonical component.
- Portable runtime manager that resolves, builds and verifies pinned `stable-diffusion.cpp` (`sd-cli`) for Linux CPU and NVIDIA CUDA.
- Explicit backend and parameter placement, including `cpu`, `cuda0`, validated placement strings, `max_vram`, `split_mode` and `auto_fit` for non-release experimentation.
- `mageflow-native` CLI: `doctor`, `verify`, `generate`, `serve`, and `runtime build --backend cpu|cuda`.
- Loopback REST API with stable health, readiness, info, generation and artifact endpoints.
- Kaggle as a thin adapter under `integrations/kaggle/`; the generic core has no hard `/kaggle/*` dependency.
- Portable CPU/CUDA qualification harnesses with sanitized, clean-room-verifiable evidence packaging.
- Model-weight-free CI, publication-surface audit, public-history invariants and release-contract checks.
- Exact-head CPU and CUDA evidence, executed notebooks, acceptance PNGs, release provenance and SHA-256 checksums published with the GitHub Release.

Frozen technical contract:

```text
stable-diffusion.cpp = 6b3edaaf32cc19e5bb2d819c788bd557eddc8eba
DiT Q8 SHA256         = 4c3dafc143ee64121692b6b63563a4f5288bf6183c4870e1d65f1566519ba7f0
Qwen Q4_K_M SHA256    = 66358cb18bb6b3b1b6675aa412c7a88ef01d228f481184d13668e5201c730a0a
VAE-only SHA256       = 34e076dc1e8a15321e1e07be5111d59cf16dd10b804b7c7e20b4de29013427e0
canonical request     = 512x512, seed 42, 4 steps, CFG 1.0, 4 threads
```
