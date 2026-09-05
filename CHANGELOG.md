# Changelog

[![CI](https://github.com/dangkhoa2016/Mage-Flow-Turbo-Native-Inference/actions/workflows/ci.yml/badge.svg)](https://github.com/dangkhoa2016/Mage-Flow-Turbo-Native-Inference/actions/workflows/ci.yml)
[![License](https://img.shields.io/github/license/dangkhoa2016/Mage-Flow-Turbo-Native-Inference)](LICENSE)

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](CHANGELOG.vi.md)

## v1.0.0 — pending publication

Planned first public release of **Mage-Flow-Turbo-Native-Inference**, a portable native inference stack. Publication remains gated on exact-head CPU/CUDA qualification and verified evidence.

- Generic core with a JSON model manifest and SHA-256 verification of every canonical component.
- Portable runtime manager that resolves, builds and verifies the pinned `stable-diffusion.cpp` (`sd-cli`) for Linux CPU and NVIDIA CUDA.
- Configurable backend/parameter placement (`cpu`, `cuda0`, validated placement strings, `max_vram`, `split_mode`, `auto_fit`).
- `mageflow-native` CLI: `doctor`, `verify`, `generate`, `serve`, `runtime build --backend cpu|cuda`.
- Loopback REST API with the stable endpoints (`/healthz`, `/readyz`, `/v1/info`, `/v1/images/generate`, `/v1/artifacts/<png>`).
- Kaggle moved to a thin adapter (`integrations/kaggle/`); the generic core has no `/kaggle/*` hard dependency.
- Generic local-layout portability proof and CPU/CUDA qualification harnesses with sanitized evidence packaging.
- Model-weight-free CI and publication-surface audits (Actions v7).

Frozen technical contract:

```text
stable-diffusion.cpp = 6b3edaaf32cc19e5bb2d819c788bd557eddc8eba
DiT Q8 SHA256         = 4c3dafc143ee64121692b6b63563a4f5288bf6183c4870e1d65f1566519ba7f0
Qwen Q4_K_M SHA256    = 66358cb18bb6b3b1b6675aa412c7a88ef01d228f481184d13668e5201c730a0a
VAE-only SHA256       = 34e076dc1e8a15321e1e07be5111d59cf16dd10b804b7c7e20b4de29013427e0
qualification          = pending: Linux CPU, NVIDIA CUDA (cuda0)
steps / CFG / threads  = 4 / 1.0 / 4
```
