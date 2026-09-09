# Changelog

[![CI](https://github.com/dangkhoa2016/Mage-Flow-Turbo-Native-Inference/actions/workflows/ci.yml/badge.svg)](https://github.com/dangkhoa2016/Mage-Flow-Turbo-Native-Inference/actions/workflows/ci.yml)
[![License](https://img.shields.io/github/license/dangkhoa2016/Mage-Flow-Turbo-Native-Inference)](LICENSE)

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](CHANGELOG.vi.md)

## v1.0.0

First public release of **Mage-Flow-Turbo-Native-Inference**, a portable native inference and deployment stack for Mage-Flow-Turbo.

- Native inference is executed by pinned `stable-diffusion.cpp` `sd-cli`; there is no Hugging Face Transformers/PyTorch inference backend.
- Two release model profiles:
  - `q8-reference` — Mage-Flow-Turbo GGUF Q8_0 diffusion, shared Qwen3-VL-4B GGUF Q4_K_M text encoder and SafeTensors VAE; canonical/default.
  - `bf16-safetensors` — Mage-Flow-Turbo BF16 SafeTensors diffusion, shared Qwen3-VL-4B GGUF Q4_K_M text encoder and SafeTensors VAE; supported alternative.
- Both release profiles support explicit `cpu` and `cuda0` qualification backends.
- Strict four-cell qualification matrix: Q8/CPU, BF16/CPU, Q8/T4 CUDA0 and BF16/T4 CUDA0.
- Every cell uses one fresh Kaggle session, one selected Mage diffusion family, the same final source HEAD/TREE and the same ordered `512 → 640 → 768 → 1024` canonical matrix.
- GPU qualification is single-T4 only even on T4x2 hosts: `CUDA_VISIBLE_DEVICES=0`, effective backend `cuda0`, no `cuda1`, no multi-GPU split, no `auto-fit`, and no CPU inference fallback.
- Release qualification is prebuilt-runtime only; source builds remain available for deliberate development outside release evidence sessions.
- JSON model manifests enforce SHA-256 identity for every selected model component.
- Evidence records source HEAD/TREE, profile/backend identity, runtime commit/binary SHA, model identities, canonical request, elapsed time, memory/RSS, CUDA peak VRAM and PNG integrity.
- Evidence packaging rejects model weights and known secret patterns, writes an internal SHA-256 manifest and emits exact `v1.0.0` archive basenames.
- A strict offline four-cell comparator computes ratios only after comparability gates pass; a deterministic renderer produces the release-facing Markdown benchmark summary after source freeze.
- Fresh measured 2×2 numbers are published in checksum-protected GitHub Release assets/body rather than pasted back into source after qualification.
- Loopback REST API, CLI tooling, Kaggle adapter, publication-surface audit, public-history invariants and release-contract checks remain part of the release.

Frozen technical identities before final qualification:

```text
stable-diffusion.cpp = 6b3edaaf32cc19e5bb2d819c788bd557eddc8eba
DiT Q8 SHA256         = 4c3dafc143ee64121692b6b63563a4f5288bf6183c4870e1d65f1566519ba7f0
DiT BF16 SHA256       = 6df47df3d7efc9ebdad075b87b3e9e4f74d09dca672d592271788f0ee27ab97d
Qwen Q4_K_M SHA256    = 66358cb18bb6b3b1b6675aa412c7a88ef01d228f481184d13668e5201c730a0a
VAE SHA256            = 34e076dc1e8a15321e1e07be5111d59cf16dd10b804b7c7e20b4de29013427e0
CPU sd-cli SHA256     = 7539d90b99eaf2b6279eec4f9006a68ae53e87bfe0c9c325ff3f329220468a5c
CUDA sd-cli SHA256    = 3fae6c1991ad0ac764c36495f688817c8a3d295d7651369bf74b7fd33743c3d0
canonical request     = seed 42, 4 steps, CFG 1.0, 4 threads
resolution matrix     = 512, 640, 768, 1024
```
